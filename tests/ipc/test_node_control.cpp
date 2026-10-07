// The Node service's product-integration surface: helper-pipe messages for settings, pairing mode and full lease
// state, and the Father's UnpairNotice. Real worker process, real Unix-socket IPC, real mutual TLS for the unpair
// paths; mock power. Windows differs only in the pipe transport (type-checked, not run).
#include <doctest/doctest.h>

#include <filesystem>
#include <thread>

#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/father/node_notify.hpp"
#include "clusterlm/node/service_core.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/platform/helper_client.hpp"
#include "clusterlm/platform/mock_adapters.hpp"
#include "clusterlm/protocol/wire.hpp"

#ifndef CLUSTERLM_NODE_BINARY
#define CLUSTERLM_NODE_BINARY "clusterlm-node.exe"
#endif

using namespace clusterlm;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

bool has_flag(const std::vector<std::string>& args, const std::string& flag, const std::string& value) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i)
    if (args[i] == flag && args[i + 1] == value) return true;
  return false;
}
bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
  for (const auto& a : args)
    if (a == flag) return true;
  return false;
}

struct RigOptions {
  bool tls = false;
  bool paired = false;                       // (with tls) settings and worker trust name the test Father
  std::string settings_father_fp_override{};  // settings claim a different Father than the worker trusts
  std::chrono::milliseconds settings_interval{0};
  std::chrono::milliseconds pairing_interval{0};
};

struct Rig {
  fs::path dir;
  platform::MockPowerMonitor power;
  platform::MockActivityMonitor helper_source;
  std::shared_ptr<config::NodeSettingsStore> settings;
  std::unique_ptr<node::ServiceCore> core;
  std::unique_ptr<platform::HelperClient> helper;
  std::mutex sink_mu;
  std::vector<node::SupervisorEvent> events;
  std::optional<transport::DeviceIdentity> father;  // set for mutual TLS rigs
  std::string father_fp;

  explicit Rig(const char* name, RigOptions o = {}) {
    dir = fs::temp_directory_path() / (std::string("clm-ctl-") + name);
    { std::error_code ec_rm; fs::remove_all(dir, ec_rm); }
    fs::create_directories(dir);
    auto st = config::NodeSettingsStore::open(dir / "node-settings.json");
    REQUIRE_MESSAGE(st.is_ok(), st.status().to_string());
    settings = std::move(st).value();
    node::ServiceCoreConfig cfg;
    cfg.supervisor.worker_binary = CLUSTERLM_NODE_BINARY;
    cfg.supervisor.worker_args = {"--name", "ctl-node", "--listen", "127.0.0.1:0", "--staging", (dir / "staging").string(),
                                  "--log", "warn"};
    if (o.tls) {
      cfg.supervisor.worker_args.push_back("--identity");
      cfg.supervisor.worker_args.push_back((dir / "identity").string());
      auto id = transport::DeviceIdentity::generate("test-father");
      REQUIRE(id.is_ok());
      father.emplace(std::move(id).value());
      father_fp = father->fingerprint();
    } else {
      cfg.supervisor.worker_args.push_back("--insecure-loopback");
    }
    if (o.paired) {
      config::PairedDevice d;
      d.fingerprint = o.settings_father_fp_override.empty() ? father_fp : o.settings_father_fp_override;
      d.name = "Father";
      d.role = "father";
      REQUIRE(settings
                  ->update([&](config::NodeSettings& s) {
                    s.paired_father = d;
                    s.name = "ctl-node";
                    return Status::ok();
                  })
                  .is_ok());
      if (o.tls) cfg.supervisor.trusted_peers = {father_fp};
      cfg.paired_father = "abcd-ef01-2345";
    }
    cfg.supervisor.policy.idle_seconds_required = 60;
    cfg.supervisor.cooperative_deadline = 1500ms;
    cfg.supervisor.lease_poll_interval = 0ms;  // poll the worker on every tick: tests drive time themselves
    cfg.helper_endpoint.socket_dir = dir / "ipc";
    cfg.enforce_report_session_match = false;
    cfg.staging_root = dir / "staging";
    cfg.settings = settings;
    cfg.settings_min_interval = o.settings_interval;
    cfg.pairing_min_interval = o.pairing_interval;
    core = std::make_unique<node::ServiceCore>(std::move(cfg), power);
    core->set_event_sink([this](const node::SupervisorEvent& e) {
      std::lock_guard lock(sink_mu);
      events.push_back(e);
    });
    REQUIRE_MESSAGE(core->start().is_ok(), "ServiceCore::start");
    platform::HelperClient::Options ho;
    ho.endpoint.socket_dir = dir / "ipc";
    helper = std::make_unique<platform::HelperClient>(ho, helper_source);
  }
  ~Rig() {
    helper.reset();
    core.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  ipc::Endpoint endpoint() const { return {ipc::kNodeHelperPipeName, dir / "ipc"}; }

  // One request/reply on a fresh helper-pipe connection (what the Node UI does).
  ipc::Envelope ask(const ipc::Envelope& request) {
    ipc::ClientOptions co;
    auto c = ipc::connect(endpoint(), co, 1000ms);
    REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
    REQUIRE(c.value()->send(request, 2000ms).is_ok());
    auto r = c.value()->receive(10000ms);  // a settings save may restart the worker
    REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
    return std::move(r).value();
  }
  Status ask_ack(const ipc::Envelope& request) {
    auto a = ipc::decode_ack(ask(request));
    REQUIRE(a.is_ok());
    return a->code == ErrorCode::kOk ? Status::ok() : make_error(a->code, a->message);
  }
  ipc::StatusReply status() {
    auto r = ipc::decode_status_reply(ask(ipc::encode(ipc::StatusRequest{})));
    REQUIRE(r.is_ok());
    return std::move(r).value();
  }
  void offer() {
    helper_source.current = {500, false};
    REQUIRE(helper->report_once().is_ok());
    REQUIRE(core->run_once().is_ok());
  }
  bool saw(node::SupervisorEventKind k, const std::string& detail = {}) {
    std::lock_guard lock(sink_mu);
    for (const auto& e : events)
      if (e.kind == k && (detail.empty() || e.detail == detail)) return true;
    return false;
  }
};

ipc::NodeSettingsView current_view(Rig& r) {
  auto reply = ipc::decode_settings_reply(r.ask(ipc::encode(ipc::SettingsRequest{})));
  REQUIRE(reply.is_ok());
  return reply->settings;
}

}  // namespace

// ---- wire --------------------------------------------------------------------------------------------------------

TEST_CASE("new helper messages round-trip and reject hostile input") {
  ipc::StatusReply s;
  s.state = ipc::NodeState::kOffering;
  s.paired_father = "ab12";
  s.storage_bytes = 99;
  s.helper_reports_fresh = true;
  s.lease_state = ipc::LeaseState::kPreparing;
  s.lease_objects_sealed = 3;
  s.lease_objects_total = 12;
  auto back = ipc::decode_status_reply(ipc::encode(s));
  REQUIRE(back.is_ok());
  CHECK(back->lease_state == ipc::LeaseState::kPreparing);
  CHECK(back->lease_objects_sealed == 3);
  CHECK(back->lease_objects_total == 12);

  // An unknown lease state byte is a protocol error, not a state.
  auto bad = ipc::encode(s);
  bad.payload[bad.payload.size() - 9] = 99;  // the lease-state byte precedes the two u32 counts
  CHECK(ipc::decode_status_reply(bad).status().code() == ErrorCode::kProtocolError);

  ipc::NodeSettingsView v;
  v.allow_when_idle = false;
  v.idle_seconds = 900;
  v.ac_only = false;
  v.temp_storage_limit_gib = 40;
  v.start_with_system = false;
  v.ram_gib = 12;
  v.vram_gib = 6;
  v.threads = 8;
  auto sr = ipc::decode_settings_reply(ipc::encode(ipc::SettingsReply{v}));
  REQUIRE(sr.is_ok());
  CHECK(sr->settings == v);
  auto su = ipc::decode_settings_update(ipc::encode(ipc::SettingsUpdate{v}));
  REQUIRE(su.is_ok());
  CHECK(su->settings == v);
  CHECK(ipc::decode_settings_request(ipc::encode(ipc::SettingsRequest{})).is_ok());
  CHECK(ipc::decode_pairing_mode_request(ipc::encode(ipc::PairingModeRequest{})).is_ok());

  auto pr = ipc::decode_pairing_mode_reply(ipc::encode(ipc::PairingModeReply{"ABCD-EFGH", "10.0.0.5:47601", "ab12-cd34-ef56", 300}));
  REQUIRE(pr.is_ok());
  CHECK(pr->code == "ABCD-EFGH");
  CHECK(pr->window_seconds == 300);

  // Truncated and trailing bytes, wrong kind, and a version-1 peer are all refused.
  auto truncated = ipc::encode(ipc::SettingsUpdate{v});
  truncated.payload.pop_back();
  CHECK_FALSE(ipc::decode_settings_update(truncated).is_ok());
  auto trailing = ipc::encode(ipc::SettingsUpdate{v});
  trailing.payload.push_back(0);
  CHECK_FALSE(ipc::decode_settings_update(trailing).is_ok());
  CHECK_FALSE(ipc::decode_settings_update(ipc::encode(ipc::SettingsReply{v})).is_ok());
  auto old = ipc::encode(ipc::SettingsRequest{});
  old.version = 1;
  CHECK(ipc::decode_settings_request(old).code() == ErrorCode::kVersionMismatch);
}

// ---- settings ----------------------------------------------------------------------------------------------------

TEST_CASE("settings are read, validated, persisted and applied through the helper pipe") {
  Rig r("settings", {.paired = false});
  // Defaults come from the store.
  auto v = current_view(r);
  CHECK(v.allow_when_idle);
  CHECK(v.idle_seconds == 300);
  CHECK(v.ram_gib == 4);
  CHECK(v.threads == 0);

  v.idle_seconds = 120;
  v.ac_only = false;
  v.temp_storage_limit_gib = 40;
  v.ram_gib = 8;
  v.vram_gib = 6;
  v.threads = 4;
  const auto before_restarts = std::count_if(r.events.begin(), r.events.end(), [](const node::SupervisorEvent& e) {
    return e.kind == node::SupervisorEventKind::kWorkerRestarted;
  });
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).is_ok());

  // Persisted: a fresh store over the same file sees the values; fields outside the view are untouched.
  auto reopened = config::NodeSettingsStore::open(r.dir / "node-settings.json");
  REQUIRE(reopened.is_ok());
  const auto doc = reopened.value()->get();
  CHECK(doc.idle_seconds == 120);
  CHECK_FALSE(doc.ac_only);
  CHECK(doc.temp_storage_limit_gib == 40);
  CHECK(doc.caps.ram_gib == 8);
  CHECK(doc.caps.vram_gib == 6);
  CHECK(doc.caps.threads == 4);
  CHECK(doc.name == "node");

  // Applied: policy now, caps through a worker restart with the new launch flags (and nothing else dropped).
  CHECK(r.core->policy().idle_seconds_required == 120);
  CHECK_FALSE(r.core->policy().require_ac_power);
  const auto args = r.core->worker_args();
  CHECK(has_flag(args, "--ram-gib", "8"));
  CHECK(has_flag(args, "--vram-gib", "6"));
  CHECK(has_flag(args, "--disk-gib", "40"));
  CHECK(has_flag(args, "--threads", "4"));
  CHECK(has_flag(args, "--staging", (r.dir / "staging").string()));
  CHECK(has_flag(args, "--insecure-loopback"));
  CHECK(r.saw(node::SupervisorEventKind::kWorkerRestarted, "resource caps changed"));
  const auto after_restarts = std::count_if(r.events.begin(), r.events.end(), [](const node::SupervisorEvent& e) {
    return e.kind == node::SupervisorEventKind::kWorkerRestarted;
  });
  CHECK(after_restarts == before_restarts + 1);

  // The worker is alive again (status answers and the next tick works).
  CHECK(r.status().state == ipc::NodeState::kBusy);
  REQUIRE(r.core->run_once().is_ok());

  // A policy-only change does not restart the worker.
  auto v2 = v;
  v2.idle_seconds = 200;
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v2})).is_ok());
  CHECK(r.core->policy().idle_seconds_required == 200);
  CHECK(std::count_if(r.events.begin(), r.events.end(), [](const node::SupervisorEvent& e) {
          return e.kind == node::SupervisorEventKind::kWorkerRestarted;
        }) == after_restarts);

  // Back to automatic threads and no disk cap: the flags go away rather than becoming 0.
  v2.threads = 0;
  v2.temp_storage_limit_gib = 0;
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v2})).is_ok());
  CHECK_FALSE(has_flag(r.core->worker_args(), "--threads"));
  CHECK_FALSE(has_flag(r.core->worker_args(), "--disk-gib"));
}

TEST_CASE("an invalid settings update is refused and changes nothing") {
  Rig r("settings-bad");
  auto v = current_view(r);
  auto bad = v;
  bad.idle_seconds = 0;  // out of range (config::validate)
  auto st = r.ask_ack(ipc::encode(ipc::SettingsUpdate{bad}));
  CHECK(st.code() == ErrorCode::kInvalidArgument);
  bad = v;
  bad.threads = 100000;
  CHECK(r.ask_ack(ipc::encode(ipc::SettingsUpdate{bad})).code() == ErrorCode::kInvalidArgument);
  bad = v;
  bad.ram_gib = 1u << 30;
  CHECK(r.ask_ack(ipc::encode(ipc::SettingsUpdate{bad})).code() == ErrorCode::kInvalidArgument);
  CHECK(current_view(r) == v);
  CHECK(r.core->policy().idle_seconds_required == 60);  // nothing applied either
  CHECK_FALSE(has_flag(r.core->worker_args(), "--threads"));
  // The view type has no field for name, paired Father, trust or paths, so no message can change them: the
  // document keeps them after a legitimate save.
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).is_ok());
  CHECK(r.settings->get().name == "node");
}

TEST_CASE("settings updates are rate limited") {
  Rig r("settings-rate", {.settings_interval = 400ms});
  auto v = current_view(r);
  v.ram_gib = 6;
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).is_ok());
  v.ram_gib = 7;
  CHECK(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).code() == ErrorCode::kResourceExhausted);
  CHECK(r.settings->get().caps.ram_gib == 6);
  std::this_thread::sleep_for(450ms);
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).is_ok());
  CHECK(r.settings->get().caps.ram_gib == 7);
}

TEST_CASE("allow-when-idle off pauses the Node and on resumes it") {
  Rig r("settings-idle");
  r.offer();
  CHECK(r.status().state == ipc::NodeState::kOffering);
  auto v = current_view(r);
  v.allow_when_idle = false;
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).is_ok());
  CHECK(r.status().state == ipc::NodeState::kPaused);
  v.allow_when_idle = true;
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).is_ok());
  CHECK(r.status().state != ipc::NodeState::kPaused);
}

TEST_CASE("a service without a settings store answers plainly") {
  fs::path dir = fs::temp_directory_path() / "clm-ctl-nostore";
  { std::error_code ec_rm; fs::remove_all(dir, ec_rm); }
  fs::create_directories(dir);
  platform::MockPowerMonitor power;
  node::ServiceCoreConfig cfg;
  cfg.supervisor.worker_binary = CLUSTERLM_NODE_BINARY;
  cfg.supervisor.worker_args = {"--name", "n", "--listen", "127.0.0.1:0", "--staging", (dir / "staging").string(), "--insecure-loopback", "--log", "warn"};
  cfg.helper_endpoint.socket_dir = dir / "ipc";
  cfg.enforce_report_session_match = false;
  node::ServiceCore core(std::move(cfg), power);
  REQUIRE(core.start().is_ok());
  CHECK(core.settings_view().status().code() == ErrorCode::kFailedPrecondition);
  CHECK(core.apply_settings({}).code() == ErrorCode::kFailedPrecondition);
  core.stop();
  { std::error_code ec_rm; fs::remove_all(dir, ec_rm); }
}

// ---- pairing mode ------------------------------------------------------------------------------------------------

TEST_CASE("pairing mode: the helper pipe request reaches the app hook, returns the code line and is rate limited") {
  Rig r("pairing", {.pairing_interval = 400ms});
  // No hook installed: an honest refusal.
  CHECK(r.ask_ack(ipc::encode(ipc::PairingModeRequest{})).code() == ErrorCode::kUnimplemented);

  int starts = 0;
  r.core->set_pairing_starter([&]() -> Result<ipc::PairingModeReply> {
    ++starts;
    return ipc::PairingModeReply{"ABCD-EFGH", r.core->worker_endpoint(), "ab12-cd34-ef56", 300};  // the hook may ask the core
  });
  auto reply = ipc::decode_pairing_mode_reply(r.ask(ipc::encode(ipc::PairingModeRequest{})));
  REQUIRE(reply.is_ok());
  CHECK(reply->code == "ABCD-EFGH");
  CHECK(reply->fingerprint == "ab12-cd34-ef56");
  CHECK(reply->window_seconds == 300);
  CHECK(starts == 1);

  // Again right away: refused without touching the hook (the code is not regenerated by a click storm).
  CHECK(r.ask_ack(ipc::encode(ipc::PairingModeRequest{})).code() == ErrorCode::kResourceExhausted);
  CHECK(starts == 1);
  std::this_thread::sleep_for(450ms);
  REQUIRE(ipc::decode_pairing_mode_reply(r.ask(ipc::encode(ipc::PairingModeRequest{}))).is_ok());
  CHECK(starts == 2);

  // A failing hook is reported without detail.
  std::this_thread::sleep_for(450ms);
  r.core->set_pairing_starter([]() -> Result<ipc::PairingModeReply> { return make_error(ErrorCode::kUnavailable, "port in use 1234"); });
  auto st = r.ask_ack(ipc::encode(ipc::PairingModeRequest{}));
  CHECK(st.code() == ErrorCode::kUnavailable);
  CHECK(st.message().find("1234") == std::string::npos);
}

// ---- full lease state --------------------------------------------------------------------------------------------

TEST_CASE("the status reply carries the worker's lease state and counts while a Father prepares and releases") {
  Rig r("lease");
  r.offer();
  auto s = r.status();
  CHECK(s.state == ipc::NodeState::kOffering);
  CHECK(s.lease_state == ipc::LeaseState::kNone);
  CHECK(s.lease_objects_total == 0);

  const fs::path model = r.dir / "model";
  auto manifest = objects::write_fixture_model(objects::FixtureSpec{}, model);
  REQUIRE(manifest.is_ok());
  coordinator::CoordinatorConfig cc;
  cc.model_dir = model;
  cc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  auto ep = transport::Endpoint::parse(r.core->worker_endpoint());
  REQUIRE(ep.is_ok());
  cc.nodes = {{"node0", ep.value(), ""}};
  cc.request_timeout = 3000ms;
  auto coord = coordinator::Coordinator::create(cc);
  REQUIRE(coord.is_ok());
  REQUIRE(coord.value()->connect().is_ok());
  auto plan = coordinator::ClusterPlan::parse("0-4@father,4-12@0,12-16@father", manifest->geometry.n_layers);
  REQUIRE(plan.is_ok());
  auto report = coord.value()->prepare(plan.value());
  REQUIRE_MESSAGE(report.is_ok(), report.status().to_string());

  REQUIRE(r.core->run_once().is_ok());  // the supervisor polls the worker on every tick in this rig
  s = r.status();
  CHECK(s.state == ipc::NodeState::kOffering);
  CHECK(s.lease_state == ipc::LeaseState::kReady);
  CHECK(s.lease_objects_total == report->nodes[0].objects);
  CHECK(s.lease_objects_sealed == s.lease_objects_total);

  REQUIRE(coord.value()->release().is_ok());
  REQUIRE(r.core->run_once().is_ok());
  s = r.status();
  CHECK(s.lease_state == ipc::LeaseState::kNone);
  CHECK(s.lease_objects_total == 0);
  CHECK(s.storage_bytes == 0);
}

// ---- Father-initiated unpair -------------------------------------------------------------------------------------

namespace {

config::PairedDevice node_device(Rig& r) {
  config::PairedDevice d;
  d.fingerprint = r.core->worker_device_id();
  d.name = "ctl-node";
  d.role = "node";
  d.address = r.core->worker_endpoint();
  return d;
}

bool wait_until_unpaired(Rig& r, std::chrono::milliseconds budget = 6000ms) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (std::chrono::steady_clock::now() < deadline) {
    REQUIRE(r.core->run_once().is_ok());
    if (!r.settings->get().paired_father) return true;
    std::this_thread::sleep_for(20ms);
  }
  return false;
}

}  // namespace

TEST_CASE("the paired Father's UnpairNotice releases the lease, drops trust and clears the paired-Father setting") {
  Rig r("unpair", {.tls = true, .paired = true});
  r.offer();
  REQUIRE(r.settings->get().paired_father.has_value());
  auto father_id = std::make_shared<const transport::DeviceIdentity>(*r.father);

  // A lease is held when the notice arrives: prepare through the real Coordinator over mutual TLS.
  const fs::path model = r.dir / "model";
  auto manifest = objects::write_fixture_model(objects::FixtureSpec{}, model);
  REQUIRE(manifest.is_ok());
  coordinator::CoordinatorConfig cc;
  cc.model_dir = model;
  cc.security.mode = transport::SecurityConfig::Mode::kMutualTls;
  cc.security.identity = father_id;
  cc.security.trusted_peers = {r.core->worker_device_id()};
  auto ep = transport::Endpoint::parse(r.core->worker_endpoint());
  REQUIRE(ep.is_ok());
  cc.nodes = {{"node0", ep.value(), r.core->worker_device_id()}};
  cc.request_timeout = 5000ms;
  auto coord = coordinator::Coordinator::create(cc);
  REQUIRE(coord.is_ok());
  REQUIRE(coord.value()->connect().is_ok());
  auto plan = coordinator::ClusterPlan::parse("0-4@father,4-12@0,12-16@father", manifest->geometry.n_layers);
  REQUIRE(plan.is_ok());
  REQUIRE(coord.value()->prepare(plan.value()).is_ok());
  REQUIRE(r.core->run_once().is_ok());
  CHECK(r.status().lease_state == ipc::LeaseState::kReady);

  // Father's session ends (what pairing.unpair does first), then the notice goes out.
  REQUIRE(coord.value()->release().is_ok());
  coord.value().reset();
  const auto res = father::notify_node_unpaired(father_id, node_device(r), 5000ms);
  CHECK_MESSAGE(res.delivered(), res.detail);
  REQUIRE(res.delivered());

  REQUIRE(wait_until_unpaired(r));
  CHECK_FALSE(r.settings->get().paired_father.has_value());
  CHECK(r.status().paired_father.empty());
  // The worker was restarted trusting nobody; the staged bytes are gone.
  CHECK(r.saw(node::SupervisorEventKind::kWorkerRestarted, "paired device list changed"));
  REQUIRE(r.core->run_once().is_ok());
  auto s = r.status();
  CHECK(s.lease_state == ipc::LeaseState::kNone);
  CHECK(s.storage_bytes == 0);

  // That Father is no longer trusted: its next connection cannot even complete the handshake.
  const auto again = father::notify_node_unpaired(father_id, node_device(r), 1500ms);
  CHECK_FALSE(again.delivered());
  CHECK(again.outcome == father::UnpairNotifyOutcome::kUnreachable);
}

TEST_CASE("an unpair notice from anyone but a pinned Father is not accepted") {
  Rig r("unpair-stranger", {.tls = true, .paired = true});
  r.offer();
  auto stranger = transport::DeviceIdentity::generate("stranger");
  REQUIRE(stranger.is_ok());
  auto stranger_id = std::make_shared<const transport::DeviceIdentity>(std::move(stranger).value());
  const auto res = father::notify_node_unpaired(stranger_id, node_device(r), 1500ms);
  CHECK_FALSE(res.delivered());  // the TLS pin refuses the handshake
  REQUIRE(r.core->run_once().is_ok());
  CHECK(r.settings->get().paired_father.has_value());  // still paired
  CHECK(r.status().paired_father == "abcd-ef01-2345");

  // The paired Father still works afterwards (the worker was not disturbed).
  auto father_id = std::make_shared<const transport::DeviceIdentity>(*r.father);
  const auto ok = father::notify_node_unpaired(father_id, node_device(r), 5000ms);
  CHECK(ok.delivered());
}

TEST_CASE("a notice from a pinned Father that the settings do not record as paired is ignored and trust is restored") {
  Rig r("unpair-mismatch", {.tls = true, .paired = true,
                            .settings_father_fp_override = std::string(64, 'a')});
  r.offer();
  auto father_id = std::make_shared<const transport::DeviceIdentity>(*r.father);
  const auto res = father::notify_node_unpaired(father_id, node_device(r), 5000ms);
  CHECK(res.delivered());  // the worker itself accepted: it pinned this Father
  const auto deadline = std::chrono::steady_clock::now() + 3000ms;
  while (std::chrono::steady_clock::now() < deadline && !r.saw(node::SupervisorEventKind::kWorkerRestarted, "paired device list changed")) {
    REQUIRE(r.core->run_once().is_ok());
    std::this_thread::sleep_for(20ms);
  }
  // The service did not forget the paired Father it has on record, and the restart put the pin back.
  REQUIRE(r.settings->get().paired_father.has_value());
  CHECK(r.settings->get().paired_father->fingerprint == std::string(64, 'a'));
  const auto again = father::notify_node_unpaired(father_id, node_device(r), 5000ms);
  CHECK(again.delivered());
}

TEST_CASE("unpair_father clears the document and the trust list") {
  Rig r("unpair-local", {.tls = true, .paired = true});
  REQUIRE(r.settings->get().paired_father.has_value());
  REQUIRE(r.core->unpair_father().is_ok());
  CHECK_FALSE(r.settings->get().paired_father.has_value());
  CHECK(r.status().paired_father.empty());
  REQUIRE(r.core->run_once().is_ok());
}

TEST_CASE("the thread cap is passed to the worker and reported in its offer") {
  Rig r("threads");
  auto v = current_view(r);
  v.threads = 3;
  REQUIRE(r.ask_ack(ipc::encode(ipc::SettingsUpdate{v})).is_ok());
  REQUIRE(has_flag(r.core->worker_args(), "--threads", "3"));
  r.offer();
  // Read the offer the way a Father does: connect on the control channel and look at OfferResources.
  auto ep = transport::Endpoint::parse(r.core->worker_endpoint());
  REQUIRE(ep.is_ok());
  transport::SecurityConfig sec;
  sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  auto conn = transport::connect(ep.value(), sec, std::nullopt, 2000ms);
  REQUIRE(conn.is_ok());
  protocol::MessageStream stream(std::move(conn).value(), protocol::Channel::kControl);
  protocol::Hello hello;
  hello.role = protocol::NodeRole::kFather;
  hello.channel = protocol::Channel::kControl;
  hello.device_id = "father";
  REQUIRE(stream.send(hello).is_ok());
  REQUIRE(stream.expect<protocol::HelloAck>(2000ms).is_ok());
  auto offer = stream.expect<protocol::OfferResources>(2000ms);
  REQUIRE(offer.is_ok());
  CHECK(offer->cpu_summary.find("threads<=3") != std::string::npos);
  stream.close();
}
