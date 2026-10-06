// ServiceCore end to end on this host: a real clusterlm-node worker supervised from helper reports that arrive
// over a real local IPC pipe, plus suspend/resume and session notifications. Mock power; no Windows needed.
#include <doctest/doctest.h>

#include <filesystem>
#include <thread>

#include "clusterlm/node/service_core.hpp"
#include "clusterlm/platform/helper_client.hpp"
#include "clusterlm/platform/mock_adapters.hpp"

using namespace clusterlm;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

bool has(const std::vector<node::SupervisorEvent>& evs, node::SupervisorEventKind k) {
  for (const auto& e : evs)
    if (e.kind == k) return true;
  return false;
}

struct Rig {
  fs::path dir;
  platform::MockPowerMonitor power;
  platform::MockActivityMonitor helper_source;  // what the "user session" looks like to the helper
  std::unique_ptr<node::ServiceCore> core;
  std::unique_ptr<platform::HelperClient> helper;
  std::vector<node::SupervisorEvent> sink_events;
  std::mutex sink_mu;

  explicit Rig(const char* name, std::chrono::milliseconds stale = 5000ms) {
    dir = fs::temp_directory_path() / (std::string("clm-core-") + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    node::ServiceCoreConfig cfg;
    cfg.supervisor.worker_binary = CLUSTERLM_NODE_BINARY;
    cfg.supervisor.worker_args = {"--name",    "core-node", "--listen", "127.0.0.1:0", "--staging", (dir / "staging").string(),
                                  "--insecure-loopback", "--log", "warn"};
    cfg.supervisor.policy.idle_seconds_required = 60;
    cfg.supervisor.cooperative_deadline = 1500ms;
    cfg.helper_endpoint.socket_dir = dir / "ipc";
    cfg.helper_report_stale_after = stale;
    cfg.enforce_report_session_match = false;  // the rig has no real terminal-services session to match
    cfg.staging_root = dir / "staging";
    cfg.paired_father = "feedbeef";
    core = std::make_unique<node::ServiceCore>(std::move(cfg), power);
    core->set_event_sink([this](const node::SupervisorEvent& e) {
      std::lock_guard lock(sink_mu);
      sink_events.push_back(e);
    });
    REQUIRE_MESSAGE(core->start().is_ok(), "ServiceCore::start");
    platform::HelperClient::Options o;
    o.endpoint.socket_dir = dir / "ipc";
    o.session_id = 0;
    helper = std::make_unique<platform::HelperClient>(o, helper_source);
  }
  ~Rig() {
    helper.reset();
    core.reset();
    fs::remove_all(dir);
  }
  // The helper reports and the service evaluates policy once.
  std::vector<node::SupervisorEvent> report(std::uint32_t idle, bool locked = false) {
    helper_source.current = {idle, locked};
    auto st = helper->report_once();
    REQUIRE_MESSAGE(st.is_ok(), st.to_string());
    auto r = core->run_once();
    REQUIRE(r.is_ok());
    return std::move(r).value();
  }
};

}  // namespace

TEST_CASE("helper reports drive offer and revoke through the whole stack") {
  Rig r("drive");
  CHECK(r.core->state() == ipc::NodeState::kBusy);

  auto evs = r.report(0);  // user at the keyboard
  CHECK(evs.empty());
  CHECK(r.core->state() == ipc::NodeState::kBusy);

  evs = r.report(120);  // idle beyond the 60 s policy
  CHECK(has(evs, node::SupervisorEventKind::kOffered));
  CHECK(r.core->state() == ipc::NodeState::kOffering);

  evs = r.report(0);  // user is back
  REQUIRE(evs.size() == 1);
  CHECK(evs[0].kind == node::SupervisorEventKind::kRevoked);
  CHECK(evs[0].residual_bytes == 0);
  CHECK(r.core->state() == ipc::NodeState::kBusy);

  // A locked session counts as idle under the default policy, regardless of the idle counter.
  evs = r.report(0, /*locked=*/true);
  CHECK(has(evs, node::SupervisorEventKind::kOffered));
}

TEST_CASE("silence from the helper revokes: stale reports fail closed") {
  Rig r("stale", 300ms);
  auto evs = r.report(500);
  REQUIRE(has(evs, node::SupervisorEventKind::kOffered));
  std::this_thread::sleep_for(450ms);  // no more reports
  auto next = r.core->run_once();
  REQUIRE(next.is_ok());
  REQUIRE(next->size() == 1);
  CHECK(next->front().kind == node::SupervisorEventKind::kRevoked);
  CHECK_FALSE(r.core->helper_activity().reports_fresh());
  // The helper comes back and the Node offers again.
  evs = r.report(500);
  CHECK(has(evs, node::SupervisorEventKind::kOffered));
}

TEST_CASE("suspend revokes synchronously; resume starts Busy and needs a fresh helper report") {
  Rig r("power");
  REQUIRE(has(r.report(500), node::SupervisorEventKind::kOffered));

  r.core->handle_power_event({platform::PowerEventKind::kSuspend, true, false});
  CHECK(r.core->state() == ipc::NodeState::kSuspended);
  {
    std::lock_guard lock(r.sink_mu);
    bool revoked = false;
    for (const auto& e : r.sink_events) revoked = revoked || e.kind == node::SupervisorEventKind::kRevoked;
    CHECK(revoked);  // already done when handle_power_event returned
  }
  // While suspended even a perfectly idle report offers nothing.
  CHECK_FALSE(has(r.report(500), node::SupervisorEventKind::kOffered));
  CHECK(r.core->state() == ipc::NodeState::kSuspended);

  r.core->handle_power_event({platform::PowerEventKind::kResume, true, false});
  CHECK(r.core->state() == ipc::NodeState::kBusy);
  auto evs = r.core->run_once();  // no report since resume: pre-sleep data was discarded
  REQUIRE(evs.is_ok());
  CHECK(evs->empty());
  CHECK(r.core->state() == ipc::NodeState::kBusy);
  CHECK(has(r.report(500), node::SupervisorEventKind::kOffered));  // fresh report satisfies policy again
}

TEST_CASE("status and pause over IPC; pause revokes and resume re-offers") {
  Rig r("status");
  REQUIRE(has(r.report(500), node::SupervisorEventKind::kOffered));
  auto st = r.helper->status();
  REQUIRE(st.is_ok());
  CHECK(st->state == ipc::NodeState::kOffering);
  CHECK(st->paired_father == "feedbeef");
  CHECK(st->helper_reports_fresh);
  CHECK(st->storage_bytes == 0);

  REQUIRE(r.helper->pause(0s).is_ok());
  auto evs = r.report(500);
  REQUIRE(evs.size() == 1);
  CHECK(evs[0].kind == node::SupervisorEventKind::kRevoked);
  st = r.helper->status();
  REQUIRE(st.is_ok());
  CHECK(st->state == ipc::NodeState::kPaused);

  REQUIRE(r.helper->resume().is_ok());
  CHECK(has(r.report(500), node::SupervisorEventKind::kOffered));
}

TEST_CASE("logoff of the reporting session and headless state") {
  Rig r("sessions");
  REQUIRE(has(r.report(500), node::SupervisorEventKind::kOffered));
  r.report(0);  // revoke
  r.core->handle_session_event({platform::SessionEventKind::kLogoff, 0});  // session 0 = the helper's id in this rig
  auto evs = r.core->run_once();
  REQUIRE(evs.is_ok());
  CHECK(evs->empty());  // no reports and no headless knowledge: still in use
  CHECK_FALSE(r.core->helper_activity().reports_fresh());
}

TEST_CASE("a hostile or buggy peer cannot break the service: garbage kinds get an error Ack") {
  Rig r("hostile");
  auto raw = ipc::connect({ipc::kNodeHelperPipeName, r.dir / "ipc"}, {}, 1000ms);
  REQUIRE(raw.is_ok());
  REQUIRE(raw.value()->send({0x7777, 1, Bytes{1, 2, 3}}, 1000ms).is_ok());
  auto reply = raw.value()->receive(2000ms);
  REQUIRE(reply.is_ok());
  auto ack = ipc::decode_ack(reply.value());
  REQUIRE(ack.is_ok());
  CHECK(ack->code == ErrorCode::kProtocolError);
  // Valid kind, malformed payload.
  auto bad = ipc::encode(ipc::ActivityReport{1, false, 0});
  bad.payload.resize(2);
  REQUIRE(raw.value()->send(bad, 1000ms).is_ok());
  ack = ipc::decode_ack(raw.value()->receive(2000ms).value());
  REQUIRE(ack.is_ok());
  CHECK(ack->code != ErrorCode::kOk);
  // The connection and the service are still fine.
  REQUIRE(raw.value()->send(ipc::encode(ipc::StatusRequest{}), 1000ms).is_ok());
  CHECK(ipc::decode_status_reply(raw.value()->receive(2000ms).value()).is_ok());
}
