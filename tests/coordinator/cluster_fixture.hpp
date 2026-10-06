#pragma once
// In-process test cluster: real NodeWorkers reached over loopback TCP by a real Coordinator. Nothing crosses
// the stage boundary except protocol messages.
#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/node/node_worker.hpp"
#include "clusterlm/objects/fixture_model.hpp"

namespace clusterlm::testing {

inline std::filesystem::path unique_dir(const std::string& name) {
  static std::atomic<int> counter{0};
  auto p = std::filesystem::temp_directory_path() /
           ("clm-tx-" + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(counter++));
  std::filesystem::create_directories(p);
  return p;
}

struct NodeOptions {
  std::shared_ptr<transport::FaultInjector> faults;
  std::optional<transport::NetworkConditions> impairment;
  std::uint64_t ram = 1ull << 30;
  std::uint64_t disk = 0;
  std::function<void(std::string_view)> phase_hook;

  static NodeOptions impaired(transport::NetworkConditions c) {
    NodeOptions o;
    o.impairment = std::move(c);
    return o;
  }
  static NodeOptions faulty(std::shared_ptr<transport::FaultInjector> f) {
    NodeOptions o;
    o.faults = std::move(f);
    return o;
  }
  static NodeOptions with_disk(std::uint64_t disk) {
    NodeOptions o;
    o.disk = disk;
    return o;
  }
};

class TestCluster {
 public:
  explicit TestCluster(const std::string& name, std::size_t nodes = 2, std::vector<NodeOptions> opts = {},
                       objects::FixtureSpec spec = objects::FixtureSpec{})
      : dir_(unique_dir(name)) {
    auto m = objects::write_fixture_model(spec, dir_ / "model");
    REQUIRE(m.is_ok());
    manifest_ = m.value();
    for (std::size_t i = 0; i < nodes; ++i) {
      node::NodeConfig nc;
      nc.name = "node" + std::to_string(i);
      nc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
      nc.staging_root = dir_ / ("staging-" + std::to_string(i));
      NodeOptions o = i < opts.size() ? opts[i] : NodeOptions{};
      nc.ram_allowance = o.ram;
      nc.vram_allowance = o.ram;
      nc.disk_allowance = o.disk;
      nc.faults = o.faults;
      nc.impairment = o.impairment;
      nc.phase_hook = o.phase_hook;
      auto w = node::NodeWorker::start(nc);
      REQUIRE_MESSAGE(w.is_ok(), w.status().to_string());
      endpoints_.push_back({nc.name, w.value()->endpoint(), ""});
      workers_.push_back(std::move(w).value());
    }
  }
  ~TestCluster() {
    workers_.clear();
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  coordinator::CoordinatorConfig config(bool direct = true) const {
    coordinator::CoordinatorConfig c;
    c.model_dir = dir_ / "model";
    c.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    c.nodes = endpoints_;
    c.direct_peer = direct;
    c.request_timeout = std::chrono::milliseconds(3000);
    c.window_timeout = std::chrono::milliseconds(5000);
    return c;
  }

  std::unique_ptr<coordinator::Coordinator> father(coordinator::CoordinatorConfig c, const std::string& plan_text,
                                                   std::uint32_t max_window = 64) {
    auto coord = coordinator::Coordinator::create(std::move(c));
    REQUIRE(coord.is_ok());
    REQUIRE(coord.value()->connect().is_ok());
    auto plan = coordinator::ClusterPlan::parse(plan_text, manifest_.geometry.n_layers);
    REQUIRE_MESSAGE(plan.is_ok(), plan.status().to_string());
    plan.value().max_window = max_window;
    auto prep = coord.value()->prepare(plan.value());
    REQUIRE_MESSAGE(prep.is_ok(), prep.status().to_string());
    last_prepare_ = prep.value();
    return std::move(coord).value();
  }

  // Father-only greedy reference for `prompt`.
  std::vector<std::int32_t> reference(const std::vector<std::int32_t>& prompt, std::uint32_t max_new) {
    auto c = config();
    c.nodes.clear();
    auto f = father(c, "0-" + std::to_string(manifest_.geometry.n_layers) + "@father," +
                           std::to_string(manifest_.geometry.n_layers) + "-" +
                           std::to_string(manifest_.geometry.n_layers) + "@father");
    coordinator::GenerationRequest r;
    r.prompt = prompt;
    r.max_new_tokens = max_new;
    auto g = f->generate(r);
    REQUIRE_MESSAGE(g.is_ok(), g.status().to_string());
    return g->tokens;
  }

  node::NodeWorker& worker(std::size_t i) { return *workers_.at(i); }
  const objects::ModelManifest& manifest() const { return manifest_; }
  const std::filesystem::path& dir() const { return dir_; }
  const coordinator::PrepareReport& last_prepare() const { return last_prepare_; }

 private:
  std::filesystem::path dir_;
  objects::ModelManifest manifest_;
  std::vector<std::unique_ptr<node::NodeWorker>> workers_;
  std::vector<coordinator::NodeEndpoint> endpoints_;
  coordinator::PrepareReport last_prepare_;
};

inline std::vector<std::int32_t> test_prompt(std::size_t n, std::uint32_t vocab, std::uint32_t salt = 0) {
  std::vector<std::int32_t> p;
  for (std::size_t i = 0; i < n; ++i) p.push_back(static_cast<std::int32_t>((i * 7919u + 13u + salt * 31u) % vocab));
  return p;
}

inline constexpr const char* kSplitPlan = "0-4@father,4-10@0,10-13@1,13-16@father";

}  // namespace clusterlm::testing
