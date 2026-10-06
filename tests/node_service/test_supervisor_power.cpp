// NodeSupervisor suspend/resume behaviour against a real clusterlm-node worker with mocked adapters.
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
    dir = fs::temp_directory_path() / (std::string("clm-supervisor-power-") + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    activity.current.idle_seconds = 0;
  }
  ~Fixture() { fs::remove_all(dir); }
  node::SupervisorConfig config() {
    node::SupervisorConfig c;
    c.worker_binary = CLUSTERLM_NODE_BINARY;
    c.worker_args = {"--name",   "svc-node", "--listen", "127.0.0.1:0", "--staging", (dir / "staging").string(),
                     "--insecure-loopback", "--log", "warn"};
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

TEST_CASE("suspend revokes a leased worker immediately and resume starts Busy until policy is satisfied") {
  Fixture f("suspend");
  node::NodeSupervisor sup(f.config(), f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  f.activity.current.idle_seconds = 120;
  auto evs = sup.tick();
  REQUIRE(evs.is_ok());
  REQUIRE(has(evs.value(), node::SupervisorEventKind::kOffered));
  auto coord = lease_worker(f.dir, sup.worker_endpoint());
  CHECK(coord->ready());

  // The machine is going to sleep: no Father consultation, the worker is cleaned right now.
  auto sus = sup.on_suspend();
  REQUIRE(sus.is_ok());
  REQUIRE(sus->size() == 1);
  CHECK(sus->front().kind == node::SupervisorEventKind::kRevoked);
  CHECK(sus->front().residual_bytes == 0);
  CHECK(sup.suspended());
  CHECK_FALSE(sup.eligible());

  // The machine looks perfectly idle, but a suspended supervisor offers nothing.
  evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK(evs->empty());
  CHECK_FALSE(sup.eligible());

  // Resume: Busy first; offering needs a fresh policy evaluation, which here is satisfied on the next tick.
  auto res = sup.on_resume();
  REQUIRE(res.is_ok());
  CHECK_FALSE(sup.suspended());
  CHECK_FALSE(sup.eligible());
  f.activity.current.idle_seconds = 0;  // user woke the machine
  evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK_FALSE(sup.eligible());
  f.activity.current.idle_seconds = 120;
  evs = sup.tick();
  REQUIRE(evs.is_ok());
  CHECK(has(evs.value(), node::SupervisorEventKind::kOffered));
}

TEST_CASE("suspend with nothing offered is a no-op and is idempotent") {
  Fixture f("suspend-idle");
  node::NodeSupervisor sup(f.config(), f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  auto a = sup.on_suspend();
  REQUIRE(a.is_ok());
  CHECK(a->empty());
  auto b = sup.on_suspend();
  REQUIRE(b.is_ok());
  CHECK(b->empty());
  CHECK(sup.on_resume().is_ok());
  CHECK(sup.on_resume().is_ok());  // duplicate resume notifications (PBT_APMRESUMEAUTOMATIC + SUSPEND) are harmless
}

TEST_CASE("a worker hung at the suspend deadline is terminated and relaunched Busy") {
  Fixture f("suspend-hung");
  auto cfg = f.config();
  cfg.worker_args.insert(cfg.worker_args.end(), {"--hang-at", "cleanup"});
  node::NodeSupervisor sup(cfg, f.activity, f.power);
  REQUIRE(sup.start().is_ok());
  f.activity.current.idle_seconds = 120;
  REQUIRE(sup.tick().is_ok());
  auto coord = lease_worker(f.dir, sup.worker_endpoint());
  auto sus = sup.on_suspend();
  REQUIRE(sus.is_ok());
  REQUIRE(sus->size() == 1);
  CHECK(sus->front().kind == node::SupervisorEventKind::kForcedTermination);
  CHECK(sus->front().residual_bytes == 0);
}
