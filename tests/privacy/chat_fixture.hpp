#pragma once
// A complete in-process ClusterLM stack for privacy tests: Father service + tokenizer + Coordinator, and two real
// NodeWorkers reached over loopback TCP (Ultra plan: father prefix -> node0 -> node1 -> father tail, direct peer
// forwarding). Every frame Father sends/receives and every frame each Node sends/receives is captured through
// FaultInjector taps, so tests can search the bytes that actually crossed a connection.
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/father/service.hpp"
#include "clusterlm/node/node_worker.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/transport/fault_injector.hpp"
#include "raw_client.hpp"

namespace clusterlm::testing {

struct CapturedFrame {
  std::string party;  // "father", "node0", "node1"
  bool sent = false;  // true: the party sent it; false: the party received it
  std::uint16_t type = 0;
  std::uint8_t channel = 0;
  Bytes payload;
};

// Thread-safe recorder shared by all taps.
class TrafficRecorder {
 public:
  std::shared_ptr<transport::FaultInjector> tap(std::string party) {
    auto inj = std::make_shared<transport::FaultInjector>();
    inj->set_tap([this, party = std::move(party)](transport::FaultDirection d, const transport::Frame& f) {
      std::lock_guard lock(mu_);
      frames_.push_back(CapturedFrame{party, d == transport::FaultDirection::kSend, f.type, f.channel, f.payload});
    });
    return inj;
  }
  std::vector<CapturedFrame> frames() const {
    std::lock_guard lock(mu_);
    return frames_;
  }
  void clear() {
    std::lock_guard lock(mu_);
    frames_.clear();
  }

 private:
  mutable std::mutex mu_;
  std::vector<CapturedFrame> frames_;
};

// int32 little-endian image of a token sequence (what the wire would contain if tokens were ever sent).
inline Bytes int32_le(const std::vector<std::int32_t>& tokens) {
  Bytes out;
  for (std::int32_t t : tokens)
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(static_cast<std::uint32_t>(t) >> (8 * i)));
  return out;
}

inline bool contains(const Bytes& haystack, const Bytes& needle) {
  if (needle.empty() || haystack.size() < needle.size()) return false;
  return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) != haystack.end();
}
inline bool contains(const Bytes& haystack, const std::string& needle) {
  return contains(haystack, Bytes(needle.begin(), needle.end()));
}

// All windows of `k` consecutive tokens of `seq`, as int32 LE byte strings.
inline std::vector<Bytes> token_windows(const std::vector<std::int32_t>& seq, std::size_t k) {
  std::vector<Bytes> out;
  for (std::size_t i = 0; i + k <= seq.size(); ++i)
    out.push_back(int32_le(std::vector<std::int32_t>(seq.begin() + static_cast<std::ptrdiff_t>(i),
                                                     seq.begin() + static_cast<std::ptrdiff_t>(i + k))));
  return out;
}

class PrivacyReadiness final : public father::ReadinessSource {
 public:
  PrivacyReadiness(std::string root, node::NodeWorker* g14, node::NodeWorker* n3060)
      : root_(std::move(root)), g14_(g14), n3060_(n3060) {}

  catalog::ReadinessInputs observe(const catalog::TierEntry& tier, const catalog::TierAssignment& a,
                                   std::uint32_t) override {
    catalog::ReadinessInputs in;
    in.model = {true, true, root_, root_};
    in.backend = {"reference", true};
    for (const auto& role : tier.roles) {
      catalog::MachineInputs m;
      m.role = role;
      m.machine_id = a.machine_for(role).value_or(role);
      m.state = catalog::MachineState::kAvailable;
      if (role == catalog::kRoleLaptop) m.state = map(g14_->status().state);
      if (role == catalog::kRole3060) m.state = map(n3060_->status().state);
      in.machines.push_back(m);
    }
    in.plan.feasible_for_context = true;
    return in;
  }

 private:
  static catalog::MachineState map(node::NodeState s) {
    switch (s) {
      case node::NodeState::kBusy: return catalog::MachineState::kBusy;
      case node::NodeState::kAvailable: return catalog::MachineState::kAvailable;
      case node::NodeState::kPreparing: return catalog::MachineState::kPreparing;
      case node::NodeState::kReady: return catalog::MachineState::kReady;
      case node::NodeState::kInferencing: return catalog::MachineState::kInferencing;
      case node::NodeState::kReleasing: return catalog::MachineState::kReleasing;
      case node::NodeState::kCleanupPending: return catalog::MachineState::kCleanupPending;
    }
    return catalog::MachineState::kOffline;
  }
  std::string root_;
  node::NodeWorker* g14_;
  node::NodeWorker* n3060_;
};

class PrivacyProvider final : public father::DeploymentProvider {
 public:
  PrivacyProvider(std::filesystem::path model, node::NodeWorker* g14, node::NodeWorker* n3060,
                  std::shared_ptr<transport::FaultInjector> father_tap)
      : model_(std::move(model)), g14_(g14), n3060_(n3060), father_tap_(std::move(father_tap)) {}

  Result<father::Deployment> resolve(const catalog::TierEntry& tier, std::uint32_t) override {
    father::Deployment d;
    d.config.model_dir = model_;
    d.config.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    d.config.request_timeout = std::chrono::milliseconds(3000);
    d.config.window_timeout = std::chrono::milliseconds(5000);
    d.config.faults = father_tap_;
    const char* plan = "0-16@father,16-16@father";
    if (tier.id == "ultra") {
      plan = "0-4@father,4-10@0,10-13@1,13-16@father";
      d.config.nodes = {{"g14", g14_->endpoint(), ""}, {"n3060", n3060_->endpoint(), ""}};
    } else if (tier.id == "strong") {
      plan = "0-4@father,4-10@0,10-16@father";
      d.config.nodes = {{"g14", g14_->endpoint(), ""}};
    }
    auto p = coordinator::ClusterPlan::parse(plan, 16);
    if (!p.is_ok()) return p.status();
    d.plan = p.value();
    return d;
  }

 private:
  std::filesystem::path model_;
  node::NodeWorker* g14_;
  node::NodeWorker* n3060_;
  std::shared_ptr<transport::FaultInjector> father_tap_;
};

struct EventCollector {
  std::mutex m;
  std::vector<father::Event> events;
  void operator()(const father::Event& e) {
    std::lock_guard lk(m);
    events.push_back(e);
  }
  std::vector<std::int32_t> all_tokens() {
    std::lock_guard lk(m);
    std::vector<std::int32_t> out;
    for (const auto& e : events)
      if (const auto* t = std::get_if<father::TokensEvent>(&e)) out.insert(out.end(), t->tokens.begin(), t->tokens.end());
    return out;
  }
  std::string all_text() {
    std::lock_guard lk(m);
    std::string out;
    for (const auto& e : events)
      if (const auto* t = std::get_if<father::TokensEvent>(&e)) out += t->text;
    return out;
  }
};

// Father service on the Ultra tier with captured traffic.
struct ChatStack {
  std::filesystem::path dir = privacy_temp_dir("stack");
  TrafficRecorder recorder;
  std::shared_ptr<transport::FaultInjector> father_tap = recorder.tap("father");
  objects::ModelManifest manifest;
  std::unique_ptr<node::NodeWorker> node0, node1;
  std::shared_ptr<father::FixtureByteTokenizer> tokenizer;
  std::unique_ptr<father::FatherService> svc;
  std::shared_ptr<EventCollector> events = std::make_shared<EventCollector>();

  ChatStack() {
    auto m = objects::write_fixture_model(objects::FixtureSpec{}, dir / "model");
    REQUIRE(m.is_ok());
    manifest = m.value();
    auto cat = catalog::Catalog::load(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json");
    REQUIRE(cat.is_ok());
    auto start = [&](const char* name, int i) {
      node::NodeConfig nc;
      nc.name = name;
      nc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
      nc.staging_root = dir / ("staging-" + std::to_string(i));
      nc.ram_allowance = 1ull << 30;
      nc.vram_allowance = 1ull << 30;
      nc.faults = recorder.tap(name);
      auto w = node::NodeWorker::start(nc);
      REQUIRE(w.is_ok());
      return std::move(w).value();
    };
    node0 = start("node0", 0);
    node1 = start("node1", 1);
    auto tok = father::FixtureByteTokenizer::create(256);
    REQUIRE(tok.is_ok());
    tokenizer = tok.value();

    father::ServiceDeps deps;
    deps.catalog = std::move(cat).value();
    deps.assignment.bind("father", "Father");
    deps.assignment.bind("node:laptop-class", "G14");
    deps.assignment.bind("node:designated-3060", "3060");
    deps.tokenizer = tokenizer;
    deps.readiness = std::make_shared<PrivacyReadiness>(manifest.root_hash().hex(), node0.get(), node1.get());
    deps.deployments = std::make_shared<PrivacyProvider>(dir / "model", node0.get(), node1.get(), father_tap);
    deps.options.progress_poll = std::chrono::milliseconds(20);
    auto s = father::make_father_service(std::move(deps));
    REQUIRE(s.is_ok());
    svc = std::move(s).value();
    svc->subscribe([c = events](const father::Event& e) { (*c)(e); });
  }

  ~ChatStack() {
    svc.reset();
    if (node0) node0->stop();
    if (node1) node1->stop();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  // Prepares the Ultra tier and runs one chat turn to completion.
  void run_chat(const std::string& user, const std::string& system, std::uint32_t max_new) {
    using namespace std::chrono_literals;
    REQUIRE(svc->select_tier("ultra").is_ok());
    REQUIRE(svc->prepare_tier("ultra", 4096).is_ok());
    REQUIRE(svc->wait_idle(60s));
    father::ChatRequest req;
    req.user_message = user;
    req.system_prompt = system;
    req.max_new_tokens = max_new;
    REQUIRE(svc->chat(req).is_ok());
    REQUIRE(svc->wait_idle(60s));
  }
};

}  // namespace clusterlm::testing
