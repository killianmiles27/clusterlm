// NodeSupervisor against a real clusterlm-node worker process, with mocked local-activity/power adapters.
#include <doctest/doctest.h>

#include <filesystem>

#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/node/supervisor.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/platform/mock_adapters.hpp"

using namespace clusterlm;
namespace fs = std::filesystem;

namespace {

struct Fixture {
  fs::path dir;
  platform::MockActivityMonitor activity;
  platform::MockPowerMonitor power;
  explicit Fixture(const char* name) {
    dir = fs::temp_directory_path() / (std::string("clm-supervisor-") + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    activity.current.idle_seconds = 0;
  }
  ~Fixture() { fs::remove_all(dir); }
  node::SupervisorConfig config(std::vector<std::string> extra = {}) {
    node::SupervisorConfig c;
    c.worker_binary = CLUSTERLM_NODE_BINARY;
    c.worker_args = {"--name", "svc-node", "--listen", "127.0.0.1:0", "--staging", (dir / "staging").string(),
                     "--insecure-loopback", "--log", "warn"};
    c.worker_args.insert(c.worker_args.end(), extra.begin(), extra.end());
    c.policy.idle_seconds_required = 60;
    c.cooperative_deadline = std::chrono::milliseconds(1500);
    return c;
  }
};

bool has(const std::vector<node::SupervisorEvent>& evs, node::SupervisorEventKind k) {
  for (const auto& e : evs)
    if (e.kind == k) return true;
  return false;
}

// Prepare a one-Node plan on the supervised worker so it holds staged model objects.
std::unique_ptr<coordinator::Coordinator> lease_worker(const fs::path& dir, const std::string& endpoint) {
  auto m = objects::write_fixture_model(objects::FixtureSpec::tiny(), dir / "model");
  REQUIRE(m.is_ok());
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = dir / "model";
  cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  auto ep = transport::Endpoint::parse(endpoint);
  REQUIRE(ep.is_ok());
  cfg.nodes.push_back({"svc-node", ep.value(), ""});
  auto c = coordinator::Coordinator::create(cfg);
  REQUIRE(c.is_ok());
  REQUIRE(c.value()->connect().is_ok());
  const auto n = m->geometry.n_layers;
  auto plan = coordinator::ClusterPlan::parse(
      "0-" + std::to_string(m->geometry.ple_layer + 1) + "@father," + std::to_string(m->geometry.ple_layer + 1) + "-" +
          std::to_string(n) + "@0," + std::to_string(n) + "-" + std::to_string(n) + "@father",
      n);
  REQUIRE(plan.is_ok());
  auto prep = c.value()->prepare(plan.value());
  REQUIRE_MESSAGE(prep.is_ok(), prep.status().to_string());
  return std::move(c).value();
}

}  // namespace

TEST_CASE("supervisor offers on idle and revokes cooperatively on activity, leaving zero staged bytes") {
  Fixture f("cooperative");
  node::NodeSupervisor sup(f.config(), f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  auto evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK(evs->empty());  // in use: nothing offered

  f.activity.current.idle_seconds = 120;
  evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK(has(evs.value(), node::SupervisorEventKind::kOffered));

  auto coord = lease_worker(f.dir, sup.worker_endpoint());
  CHECK(coord->ready());

  f.activity.current.idle_seconds = 0;  // the user is back
  evs = sup.tick();
  REQUIRE(evs.is_ok());
  REQUIRE(evs->size() == 1);
  CHECK(evs->front().kind == node::SupervisorEventKind::kRevoked);
  CHECK(evs->front().residual_bytes == 0);
  CHECK(evs->front().latency_ms < 1500.0);
  MESSAGE("cooperative revocation latency (development host): " << evs->front().latency_ms << " ms");
}

TEST_CASE("battery power and power saver keep the Node unavailable under the default policy") {
  Fixture f("power");
  node::NodeSupervisor sup(f.config(), f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  f.activity.current.idle_seconds = 1000;
  f.power.current.on_ac_power = false;
  auto evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK_FALSE(sup.eligible());
  f.power.current.on_ac_power = true;
  f.power.current.battery_saver = true;
  evs = sup.tick();
  CHECK_FALSE(sup.eligible());
  f.power.current.battery_saver = false;
  evs = sup.tick();
  CHECK(sup.eligible());
}

TEST_CASE("unreadable activity state fails closed") {
  Fixture f("failclosed");
  node::NodeSupervisor sup(f.config(), f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  f.activity.current.idle_seconds = 1000;
  REQUIRE(sup.tick().is_ok());
  CHECK(sup.eligible());
  f.activity.fail_with = make_error(ErrorCode::kUnavailable, "session helper gone");
  auto evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK_FALSE(sup.eligible());
}

TEST_CASE("a worker hung during cleanup is terminated at the deadline and its staging is recovered") {
  Fixture f("hung");
  node::NodeSupervisor sup(f.config({"--hang-at", "cleanup"}), f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  f.activity.current.idle_seconds = 120;
  REQUIRE(sup.tick().is_ok());
  auto coord = lease_worker(f.dir, sup.worker_endpoint());

  f.activity.current.idle_seconds = 0;
  auto evs = sup.tick();
  REQUIRE(evs.is_ok());
  REQUIRE(evs->size() == 1);
  CHECK(evs->front().kind == node::SupervisorEventKind::kForcedTermination);
  // The relaunched worker ran orphan recovery before listening: nothing staged remains.
  CHECK(evs->front().residual_bytes == 0);
  CHECK(evs->front().latency_ms >= 1500.0);
}

TEST_CASE("a crashed worker is relaunched and comes back Busy") {
  Fixture f("crash");
  node::NodeSupervisor sup(f.config(), f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  sup.worker()->kill();
  (void)sup.worker()->wait(std::chrono::seconds(5));
  auto evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK(has(evs.value(), node::SupervisorEventKind::kWorkerRestarted));
  CHECK_FALSE(sup.eligible());
}
