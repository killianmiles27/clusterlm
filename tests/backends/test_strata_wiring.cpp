// The Strata backend wired into the product paths, without a GPU: the Father prefix/tail and a real Node worker
// (middle stages) run through StrataDomain over the strata-core fake engine (tests/backends/strata_test_support.hpp),
// driven by a real Coordinator over loopback. What is checked here is the wiring, not Strata's numerics:
//   * the backend factory (names, unknown/unbuilt backends refuse),
//   * end to end generation: boundary transpose, token-free Node stages, commit with partial acceptance, release,
//   * a backend that cannot prepare (kHardwareUnavailable, as the CUDA engine does without a device) reports its real
//     error through prepare()/PlanReady and leaves nothing behind on Father or on the Node.
// The injected adapter is the guarded test seam (CoordinatorConfig/NodeConfig::backend_factory); the fake engine is not
// duplicated.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#include "clusterlm/backends/backend_factory.hpp"
#include "clusterlm/backends/strata/strata_domain.hpp"
#include "clusterlm/common/digest.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/node/node_worker.hpp"
#include "strata_test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::strata_test;
using domain::StageRole;
namespace fs = std::filesystem;

namespace {

// ---- observation of what the engines were asked to do --------------------------------------------------------------

struct RoleStats {
  std::uint64_t prepared = 0, released = 0, runs = 0, commits = 0, aborts = 0, opens = 0, closes = 0;
  std::uint64_t token_positions = 0;      // tokens handed to the engine
  std::uint64_t partial_commits = 0;      // commit(keep) with keep < the window's positions
  std::uint64_t window_positions = 0;
};

struct Observed {
  std::mutex mu;
  std::map<StageRole, RoleStats> by_role;
  RoleStats get(StageRole r) {
    std::lock_guard lock(mu);
    return by_role[r];
  }
};

// FakeEngine plus counters; optionally refuses to prepare like the CUDA engine does without a device.
class ObservedEngine final : public backends::strata::StrataEngine {
 public:
  ObservedEngine(StageRole role, const objects::ModelGeometry& g, std::shared_ptr<Observed> obs, bool refuse_prepare)
      : role_(role), inner_(role, g), obs_(std::move(obs)), refuse_(refuse_prepare) {}

  backends::strata::EngineCaps caps() const override { return inner_.caps(); }
  Result<domain::DomainRequirements> requirements(const std::vector<std::string>& n) const override {
    return inner_.requirements(n);
  }
  Status prepare(const objects::ObjectResolver& r) override {
    if (refuse_) return make_error(ErrorCode::kHardwareUnavailable, "no usable CUDA device (fake engine)");
    bump([](RoleStats& s) { ++s.prepared; });
    return inner_.prepare(r);
  }
  Status open_session(SessionId s) override {
    bump([](RoleStats& st) { ++st.opens; });
    return inner_.open_session(s);
  }
  Status close_session(SessionId s) override {
    bump([](RoleStats& st) { ++st.closes; });
    return inner_.close_session(s);
  }
  Status run(const backends::strata::EngineWindow& w, domain::StageTiming& t) override {
    bump([&](RoleStats& s) {
      ++s.runs;
      s.token_positions += w.tokens.size();
      last_positions_ = w.positions;
    });
    return inner_.run(w, t);
  }
  Status commit(SessionId s, std::uint32_t keep) override {
    bump([&](RoleStats& st) {
      ++st.commits;
      st.window_positions += last_positions_;
      if (keep < last_positions_) ++st.partial_commits;
    });
    return inner_.commit(s, keep);
  }
  Status abort(SessionId s) override {
    bump([](RoleStats& st) { ++st.aborts; });
    return inner_.abort(s);
  }
  Status release() override {
    bump([](RoleStats& st) { ++st.released; });
    return inner_.release();
  }
  backends::strata::EngineCounters counters() const override { return inner_.counters(); }

 private:
  template <class F>
  void bump(F&& f) {
    std::lock_guard lock(obs_->mu);
    f(obs_->by_role[role_]);
  }
  StageRole role_;
  FakeEngine inner_;
  std::shared_ptr<Observed> obs_;
  bool refuse_;
  std::uint32_t last_positions_ = 0;
};

class FakeStrataBackend final : public domain::BackendAdapter {
 public:
  FakeStrataBackend(std::shared_ptr<Observed> obs, std::set<StageRole> refuse) : obs_(std::move(obs)), refuse_(std::move(refuse)) {}
  domain::BackendInfo info() const override {
    domain::BackendInfo i;
    i.name = "strata";
    i.build_hash = "strata-fake-engine";
    i.supports_gpu = true;
    i.hardware_available = true;
    return i;
  }
  Result<std::unique_ptr<domain::ExecutionDomain>> create_domain(const objects::ModelManifest& m,
                                                                 const domain::DomainSpec& spec) override {
    CLM_ASSIGN_OR_RETURN(auto d, backends::strata::StrataDomain::create(
                                     m, spec, std::make_unique<ObservedEngine>(spec.role, m.geometry, obs_, refuse_.contains(spec.role))));
    return std::unique_ptr<domain::ExecutionDomain>(std::move(d));
  }

 private:
  std::shared_ptr<Observed> obs_;
  std::set<StageRole> refuse_;
};

// ---- a model directory with Strata's object names (synthetic manifest, pseudo-random bytes, real digests) ------------

objects::ModelManifest write_model(const fs::path& dir) {
  objects::ModelManifest m = synthetic_manifest(tiny_geometry(), "q8_0");
  Bytes shard(m.shards.at(0).byte_size, 0);
  std::uint64_t x = 0x9E3779B97F4A7C15ull;
  for (auto& b : shard) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    b = static_cast<std::uint8_t>(x);
  }
  for (auto& o : m.objects) {
    Sha256 h;
    for (const auto& r : o.source_ranges) h.update(ByteSpan(shard).subspan(r.offset, r.length));
    o.source_digest = h.finish();
    o.object_digest = o.source_digest;
  }
  m.shards[0].digest = Sha256::of(shard);
  fs::create_directories(dir);
  std::ofstream(dir / m.shards[0].file_name, std::ios::binary).write(reinterpret_cast<const char*>(shard.data()), static_cast<std::streamsize>(shard.size()));
  const std::string json = m.to_json();
  std::ofstream(dir / "manifest.json", std::ios::binary).write(json.data(), static_cast<std::streamsize>(json.size()));
  return m;
}

fs::path unique_dir(const std::string& name) {
  static std::atomic<int> counter{0};
  auto p = fs::temp_directory_path() / ("clm-strata-wiring-" + name + "-" +
                                         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                                         std::to_string(counter++));
  fs::create_directories(p);
  return p;
}

constexpr const char* kPlan = "0-3@father,3-5@0,5-7@1,7-8@father";

struct Cluster {
  fs::path dir = unique_dir("cluster");
  objects::ModelManifest manifest;
  std::shared_ptr<Observed> obs = std::make_shared<Observed>();
  std::vector<std::unique_ptr<node::NodeWorker>> workers;
  std::vector<coordinator::NodeEndpoint> endpoints;

  // `node_refuses` / `father_refuses`: roles whose engine refuses to prepare (kHardwareUnavailable).
  explicit Cluster(std::set<StageRole> node_refuses = {}) {
    manifest = write_model(dir / "model");
    for (int i = 0; i < 2; ++i) {
      node::NodeConfig nc;
      nc.name = "node" + std::to_string(i);
      nc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
      nc.staging_root = dir / ("staging-" + std::to_string(i));
      nc.ram_allowance = nc.vram_allowance = 1ull << 30;
      nc.backend = "strata";
      auto obs_copy = obs;
      nc.backend_factory = [obs_copy, node_refuses]() -> Result<std::unique_ptr<domain::BackendAdapter>> {
        return std::unique_ptr<domain::BackendAdapter>(std::make_unique<FakeStrataBackend>(obs_copy, node_refuses));
      };
      auto w = node::NodeWorker::start(nc);
      REQUIRE_MESSAGE(w.is_ok(), w.status().to_string());
      endpoints.push_back({nc.name, w.value()->endpoint(), ""});
      workers.push_back(std::move(w).value());
    }
  }
  ~Cluster() {
    workers.clear();
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  coordinator::CoordinatorConfig config(std::set<StageRole> father_refuses = {}) const {
    coordinator::CoordinatorConfig c;
    c.model_dir = dir / "model";
    c.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    c.nodes = endpoints;
    c.backend = "strata";
    c.request_timeout = std::chrono::milliseconds(3000);
    c.window_timeout = std::chrono::milliseconds(5000);
    auto obs_copy = obs;
    c.backend_factory = [obs_copy, father_refuses]() -> Result<std::unique_ptr<domain::BackendAdapter>> {
      return std::unique_ptr<domain::BackendAdapter>(std::make_unique<FakeStrataBackend>(obs_copy, father_refuses));
    };
    return c;
  }

  coordinator::ClusterPlan plan() const {
    auto p = coordinator::ClusterPlan::parse(kPlan, manifest.geometry.n_layers);
    REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
    p.value().max_window = 16;
    return p.value();
  }
};

}  // namespace

TEST_CASE("backend factory: names, unknown backends refuse, strata is built only with CLUSTERLM_ENABLE_STRATA") {
  using backends::BackendOptions;
  const auto names = backends::known_backend_names();
  CHECK(std::find(names.begin(), names.end(), "reference") != names.end());
  CHECK(std::find(names.begin(), names.end(), "strata") != names.end());

  BackendOptions ref;
  auto r = backends::make_backend(ref);
  REQUIRE(r.is_ok());
  CHECK(r.value()->info().name == "reference");

  BackendOptions unknown;
  unknown.name = "vulkan-magic";
  CHECK(backends::make_backend(unknown).status().code() == ErrorCode::kInvalidArgument);
  CHECK(backends::check_backend_name("").code() == ErrorCode::kInvalidArgument);

  BackendOptions strata;
  strata.name = "strata";
  auto s = backends::make_backend(strata);
  if (backends::backend_built("strata")) {
    REQUIRE(s.is_ok());
    CHECK(s.value()->info().name == "strata");
  } else {
    CHECK(s.status().code() == ErrorCode::kUnimplemented);
    CHECK(s.status().message().find("CLUSTERLM_ENABLE_STRATA") != std::string::npos);
  }
}

TEST_CASE("Node worker and Coordinator refuse an unknown or unbuilt backend") {
  const fs::path dir = unique_dir("refuse");
  node::NodeConfig nc;
  nc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  nc.staging_root = dir / "staging";
  nc.backend = "vulkan-magic";
  CHECK(node::NodeWorker::start(nc).status().code() == ErrorCode::kInvalidArgument);
  if (!backends::backend_built("strata")) {
    nc.backend = "strata";
    CHECK(node::NodeWorker::start(nc).status().code() == ErrorCode::kUnimplemented);
  }
  const auto m = write_model(dir / "model");
  (void)m;
  coordinator::CoordinatorConfig cc;
  cc.model_dir = dir / "model";
  cc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  cc.backend = "vulkan-magic";
  CHECK(coordinator::Coordinator::create(cc).status().code() == ErrorCode::kInvalidArgument);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("Father + Node + Coordinator end to end through StrataDomain (fake engine): token-free middle, commit, release") {
  Cluster cl;
  auto coord = coordinator::Coordinator::create(cl.config());
  REQUIRE_MESSAGE(coord.is_ok(), coord.status().to_string());
  auto& c = *coord.value();
  REQUIRE(c.connect().is_ok());
  auto prep = c.prepare(cl.plan());
  REQUIRE_MESSAGE(prep.is_ok(), prep.status().to_string());
  CHECK(c.ready());

  const std::uint32_t vocab = cl.manifest.geometry.vocab_size;
  std::vector<std::int32_t> prompt;
  for (std::int32_t i = 0; i < 20; ++i) prompt.push_back(static_cast<std::int32_t>((i * 5 + 3) % static_cast<int>(vocab)));

  SUBCASE("q = 1: greedy tokens, prefill chunked into local sub-batches") {
    coordinator::GenerationRequest r;
    r.prompt = prompt;
    r.max_new_tokens = 6;
    r.prefill_chunk = 16;  // the fake engine's window is 8: sub-batches
    auto g = c.generate(r);
    REQUIRE_MESSAGE(g.is_ok(), g.status().to_string());
    // The fake tail's logits grow with the token index: greedy always picks the last vocabulary entry.
    REQUIRE(g->tokens.size() == 6);
    for (auto t : g->tokens) CHECK(t == static_cast<std::int32_t>(vocab) - 1);
  }

  SUBCASE("q = 3 with a corrupted draft: a partial commit reaches every domain") {
    std::vector<std::int32_t> sequence = prompt;
    sequence.insert(sequence.end(), 16, static_cast<std::int32_t>(vocab) - 1);
    domain::ScriptedDrafter::Config dc;
    dc.vocab = vocab;
    dc.corrupt_draft_index = 2;  // draft 1 accepted, draft 2 rejected: keep 2 of 3
    coordinator::GenerationRequest r;
    r.prompt = prompt;
    r.max_new_tokens = 8;
    r.q = 3;
    r.drafter = std::make_shared<domain::ScriptedDrafter>(sequence, dc);
    auto g = c.generate(r);
    REQUIRE_MESSAGE(g.is_ok(), g.status().to_string());
    REQUIRE(g->tokens.size() == 8);
    for (auto t : g->tokens) CHECK(t == static_cast<std::int32_t>(vocab) - 1);
    CHECK(g->proposed_positions > g->accepted_drafts);
    const auto mid = cl.obs->get(StageRole::kMiddle);
    const auto tail = cl.obs->get(StageRole::kTail);
    CHECK(mid.partial_commits > 0);   // Nodes committed fewer positions than they ran
    CHECK(tail.partial_commits > 0);
  }

  auto rel = c.release();
  REQUIRE(rel.is_ok());
  for (const auto& n : rel->nodes) {
    CHECK(n.resources_released);
    CHECK(n.storage_cleaned);
    CHECK(n.residual_bytes == 0);
  }

  const RoleStats prefix = cl.obs->get(StageRole::kPrefix), mid = cl.obs->get(StageRole::kMiddle),
                  tail = cl.obs->get(StageRole::kTail);
  CHECK(prefix.runs > 0);
  CHECK(prefix.token_positions > 0);  // only the Father prefix embeds tokens
  CHECK(mid.runs > 0);
  CHECK(mid.token_positions == 0);    // Node stages are token-free
  CHECK(tail.runs > 0);
  CHECK(tail.token_positions == 0);
  CHECK(mid.commits > 0);
  // Everything prepared is released: two Node middle domains, Father's prefix and tail (plus the Node-side probes
  // never reach prepare()).
  CHECK(prefix.prepared == prefix.released);
  CHECK(mid.prepared == mid.released);
  CHECK(tail.prepared == tail.released);
  CHECK(mid.prepared >= 2);
  for (auto& w : cl.workers) {
    const auto st = w->status();
    CHECK(st.staging_census_bytes == 0);
    CHECK(st.last_storage_cleaned);
  }
}

TEST_CASE("a Strata backend without a device: Father's own error is reported cleanly by prepare()") {
  Cluster cl;
  // Father's tail engine refuses (the CUDA engine's answer on a host without a GPU); the prefix prepared first and
  // must be released again; no Node was provisioned.
  auto coord = coordinator::Coordinator::create(cl.config({StageRole::kTail}));
  REQUIRE(coord.is_ok());
  auto& c = *coord.value();
  REQUIRE(c.connect().is_ok());
  auto prep = c.prepare(cl.plan());
  REQUIRE_FALSE(prep.is_ok());
  CHECK(prep.status().code() == ErrorCode::kHardwareUnavailable);
  CHECK(prep.status().message().find("CUDA") != std::string::npos);
  CHECK_FALSE(c.ready());
  const auto prefix = cl.obs->get(StageRole::kPrefix);
  CHECK(prefix.prepared == 1);
  CHECK(prefix.released == 1);  // no leak
  CHECK(cl.obs->get(StageRole::kMiddle).prepared == 0);
  for (auto& w : cl.workers) CHECK(w->status().staging_census_bytes == 0);
  // The Coordinator is reusable: a later prepare with a working backend would start from nothing (here: same refusal).
  CHECK(c.prepare(cl.plan()).status().code() == ErrorCode::kHardwareUnavailable);
}

TEST_CASE("a Strata backend without a device on a Node: PlanReady carries the real error, the Node cleans up") {
  Cluster cl({StageRole::kMiddle});
  auto coord = coordinator::Coordinator::create(cl.config());
  REQUIRE(coord.is_ok());
  auto& c = *coord.value();
  REQUIRE(c.connect().is_ok());
  auto prep = c.prepare(cl.plan());
  REQUIRE_FALSE(prep.is_ok());
  CHECK(prep.status().code() == ErrorCode::kHardwareUnavailable);
  CHECK_FALSE(c.ready());
  // Father's own domains were released, the Nodes hold no objects.
  CHECK(cl.obs->get(StageRole::kPrefix).prepared == cl.obs->get(StageRole::kPrefix).released);
  CHECK(cl.obs->get(StageRole::kTail).prepared == cl.obs->get(StageRole::kTail).released);
  for (auto& w : cl.workers) {
    const auto st = w->status();
    CHECK(st.staging_census_bytes == 0);
    CHECK(st.last_storage_cleaned);
  }
}
