// Father service end to end on the fixture model through in-process NodeWorkers (real TCP + protocol):
// tier listing, prepare, streaming chat, cancel, release, redaction, retry and downgrade on node loss.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/father/production.hpp"
#include "clusterlm/father/service.hpp"
#include "clusterlm/node/node_worker.hpp"
#include "clusterlm/objects/fixture_model.hpp"

using namespace clusterlm;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using catalog::MachineState;
using catalog::TierState;

namespace {

fs::path temp_dir(const char* name) {
  auto p = fs::temp_directory_path() / (std::string("clm-father-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

MachineState map_state(node::NodeState s) {
  switch (s) {
    case node::NodeState::kBusy: return MachineState::kBusy;
    case node::NodeState::kAvailable: return MachineState::kAvailable;
    case node::NodeState::kPreparing: return MachineState::kPreparing;
    case node::NodeState::kReady: return MachineState::kReady;
    case node::NodeState::kInferencing: return MachineState::kInferencing;
    case node::NodeState::kReleasing: return MachineState::kReleasing;
    case node::NodeState::kCleanupPending: return MachineState::kCleanupPending;
  }
  return MachineState::kOffline;
}

// Observes the real in-process workers.
class WorkerSource final : public father::ReadinessSource {
 public:
  WorkerSource(std::string root, node::NodeWorker* g14, node::NodeWorker* n3060)
      : root_(std::move(root)), g14_(g14), n3060_(n3060) {}

  catalog::ReadinessInputs observe(const catalog::TierEntry& tier, const catalog::TierAssignment& a, std::uint32_t) override {
    catalog::ReadinessInputs in;
    in.model = {true, true, root_, confirm_.load() ? root_ : std::string()};
    in.backend = {"reference", true};
    for (const auto& role : tier.roles) {
      catalog::MachineInputs m;
      m.role = role;
      m.machine_id = a.machine_for(role).value_or(role);
      m.state = MachineState::kAvailable;
      if (role == catalog::kRoleLaptop) m.state = map_state(g14_->status().state);
      if (role == catalog::kRole3060) m.state = map_state(n3060_->status().state);
      in.machines.push_back(m);
    }
    in.plan.feasible_for_context = true;
    return in;
  }
  std::atomic<bool> confirm_{true};

 private:
  std::string root_;
  node::NodeWorker* g14_;
  node::NodeWorker* n3060_;
};

class FixtureProvider final : public father::DeploymentProvider {
 public:
  FixtureProvider(fs::path model, node::NodeWorker* g14, node::NodeWorker* n3060) : model_(std::move(model)), g14_(g14), n3060_(n3060) {}

  Result<father::Deployment> resolve(const catalog::TierEntry& tier, std::uint32_t) override {
    father::Deployment d;
    d.config.model_dir = model_;
    d.config.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    d.config.request_timeout = 3000ms;
    d.config.window_timeout = 5000ms;
    const char* plan = "0-16@father,16-16@father";
    if (tier.id == "strong") {
      plan = "0-4@father,4-10@0,10-16@father";
      d.config.nodes = {{"g14", g14_->endpoint(), ""}};
    } else if (tier.id == "ultra") {
      plan = "0-4@father,4-10@0,10-13@1,13-16@father";
      d.config.nodes = {{"g14", g14_->endpoint(), ""}, {"n3060", n3060_->endpoint(), ""}};
    }
    auto p = coordinator::ClusterPlan::parse(plan, 16);
    if (!p.is_ok()) return p.status();
    d.plan = p.value();
    resolved.push_back(tier.id);
    return d;
  }
  std::vector<std::string> resolved;

 private:
  fs::path model_;
  node::NodeWorker* g14_;
  node::NodeWorker* n3060_;
};

struct Collector {
  std::mutex m;
  std::vector<father::Event> events;
  void operator()(const father::Event& e) {
    std::lock_guard lk(m);
    events.push_back(e);
  }
  template <typename T>
  std::vector<T> all() {
    std::lock_guard lk(m);
    std::vector<T> out;
    for (const auto& e : events)
      if (const auto* t = std::get_if<T>(&e)) out.push_back(*t);
    return out;
  }
};

struct Fixture {
  fs::path dir = temp_dir("svc");
  objects::ModelManifest manifest;
  std::unique_ptr<node::NodeWorker> g14, n3060;
  std::shared_ptr<WorkerSource> source;
  std::shared_ptr<FixtureProvider> provider;
  std::shared_ptr<father::FixtureByteTokenizer> tokenizer;
  std::unique_ptr<father::FatherService> svc;
  std::shared_ptr<Collector> events = std::make_shared<Collector>();
  catalog::Catalog cat;

  Fixture() {
    auto m = objects::write_fixture_model(objects::FixtureSpec{}, dir / "model");
    REQUIRE(m.is_ok());
    manifest = m.value();
    auto c = catalog::Catalog::load(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json");
    REQUIRE(c.is_ok());
    cat = std::move(c).value();
    auto start = [&](const char* name, int i) {
      node::NodeConfig nc;
      nc.name = name;
      nc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
      nc.staging_root = dir / ("staging-" + std::to_string(i));
      nc.ram_allowance = 1ull << 30;
      nc.vram_allowance = 1ull << 30;
      auto w = node::NodeWorker::start(nc);
      REQUIRE(w.is_ok());
      return std::move(w).value();
    };
    g14 = start("g14", 0);
    n3060 = start("n3060", 1);
    source = std::make_shared<WorkerSource>(manifest.root_hash().hex(), g14.get(), n3060.get());
    provider = std::make_shared<FixtureProvider>(dir / "model", g14.get(), n3060.get());
    auto tok = father::FixtureByteTokenizer::create(256);
    REQUIRE(tok.is_ok());
    tokenizer = tok.value();

    father::ServiceDeps deps;
    deps.catalog = cat;
    deps.assignment.bind("father", "Father");
    deps.assignment.bind("node:laptop-class", "G14");
    deps.assignment.bind("node:designated-3060", "3060");
    deps.tokenizer = tokenizer;
    deps.readiness = source;
    deps.deployments = provider;
    deps.options.progress_poll = 20ms;
    auto s = father::make_father_service(std::move(deps));
    REQUIRE(s.is_ok());
    svc = std::move(s).value();
    svc->subscribe([c = events](const father::Event& e) { (*c)(e); });
  }

  ~Fixture() {
    svc.reset();
    if (g14) g14->stop();
    if (n3060) n3060->stop();
    fs::remove_all(dir);
  }

  const catalog::TierEntry& tier(const char* id) const { return *cat.find(id); }

  void prepare(const char* id) {
    REQUIRE(svc->select_tier(id).is_ok());
    REQUIRE(svc->prepare_tier(id, 4096).is_ok());
    REQUIRE(svc->wait_idle(30s));
  }

  // Greedy reference: Father-only plan driven straight through a Coordinator.
  std::vector<std::int32_t> reference(const std::vector<std::int32_t>& prompt, std::uint32_t n) {
    coordinator::CoordinatorConfig cfg;
    cfg.model_dir = dir / "model";
    cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    auto coord = coordinator::Coordinator::create(cfg);
    REQUIRE(coord.is_ok());
    REQUIRE(coord.value()->connect().is_ok());
    auto plan = coordinator::ClusterPlan::parse("0-16@father,16-16@father", 16);
    REQUIRE(plan.is_ok());
    REQUIRE(coord.value()->prepare(plan.value()).is_ok());
    coordinator::GenerationRequest g;
    g.prompt = prompt;
    g.max_new_tokens = n;
    auto r = coord.value()->generate(g);
    REQUIRE(r.is_ok());
    (void)coord.value()->release();
    auto t = r->tokens;
    t.resize(n);
    return t;
  }

  static std::vector<std::int32_t> flatten(const std::vector<father::TokensEvent>& evs) {
    std::vector<std::int32_t> out;
    for (const auto& e : evs) out.insert(out.end(), e.tokens.begin(), e.tokens.end());
    return out;
  }
};

const catalog::TierReadiness& pick(const std::vector<catalog::TierReadiness>& v, const char* id) {
  for (const auto& r : v)
    if (r.tier_id == id) return r;
  FAIL("tier missing");
  return v.front();
}

}  // namespace

TEST_CASE("tiers list honestly: Available before prepare, Ready only after the plan is prepared") {
  Fixture f;
  auto tiers = f.svc->list_tiers();
  REQUIRE(tiers.size() == 3);
  for (const auto& t : tiers) CHECK(t.state == TierState::kAvailable);
  CHECK(pick(tiers, "ultra").suggested_fallback == std::vector<std::string>{"strong", "fast"});

  f.prepare("ultra");
  tiers = f.svc->list_tiers();
  CHECK(pick(tiers, "ultra").state == TierState::kReady);
  CHECK(pick(tiers, "strong").state != TierState::kReady);  // nodes are leased to the Ultra plan
  CHECK(f.events->all<father::TierReadyEvent>().size() == 1);
  CHECK(f.events->all<father::TierReadyEvent>()[0].model_name == "Qwen3.8-Flash-Next GSQ-RCO IQ3_S");
  CHECK(f.g14->status().state == node::NodeState::kReady);
  CHECK(f.n3060->status().state == node::NodeState::kReady);
  REQUIRE(f.svc->release().is_ok());
  CHECK(pick(f.svc->list_tiers(), "ultra").state == TierState::kAvailable);
  CHECK(f.g14->status().staging_census_bytes == 0);
  CHECK(f.n3060->status().staging_census_bytes == 0);
}

TEST_CASE("an unconfirmed unpinned model blocks preparation") {
  Fixture f;
  f.source->confirm_ = false;
  CHECK(pick(f.svc->list_tiers(), "ultra").state == TierState::kUnavailable);
  REQUIRE(f.svc->prepare_tier("ultra", 4096).is_ok());
  REQUIRE(f.svc->wait_idle(30s));
  auto errs = f.events->all<father::ErrorEvent>();
  REQUIRE(errs.size() == 1);
  CHECK(errs[0].message.find("unpinned") != std::string::npos);
  CHECK(f.events->all<father::TierReadyEvent>().empty());
  CHECK(f.g14->status().state == node::NodeState::kAvailable);  // nothing was provisioned
}

TEST_CASE("chat streams tokens identical to the Father-only reference and keeps history Father-side") {
  Fixture f;
  f.prepare("ultra");
  father::ChatRequest req;
  req.user_message = "hello cluster";
  req.system_prompt = "be brief";
  req.max_new_tokens = 18;
  auto rid = f.svc->chat(req);
  REQUIRE(rid.is_ok());
  REQUIRE(f.svc->wait_idle(60s));

  const auto expected_prompt = f.tokenizer->encode_chat(std::vector<father::ChatMessage>{
      {father::ChatRole::kSystem, "be brief"}, {father::ChatRole::kUser, "hello cluster"}});
  const auto ref = f.reference(expected_prompt, 18);

  auto toks = f.events->all<father::TokensEvent>();
  REQUIRE(toks.size() == 18);  // streamed as each decode round emits (q = 1: one token per round)
  for (const auto& t : toks) {
    CHECK(t.request == rid.value());
    CHECK(t.tier_id == "ultra");
    CHECK(t.model_name == "Qwen3.8-Flash-Next GSQ-RCO IQ3_S");
  }
  CHECK(Fixture::flatten(toks) == ref);

  auto fin = f.events->all<father::FinishedEvent>();
  REQUIRE(fin.size() == 1);
  CHECK(fin[0].reason == father::FinishReason::kCompleted);
  CHECK(fin[0].stats.tokens == 18);
  CHECK(fin[0].stats.ttft_ms > 0);
  CHECK(fin[0].stats.decode_tok_s > 0);
  CHECK(fin[0].stats.accepted_per_round == doctest::Approx(1.0));  // q = 1: one token per round
  CHECK(fin[0].stats.fallbacks == 0);
  CHECK(fin[0].stats.final_model == "Qwen3.8-Flash-Next GSQ-RCO IQ3_S");
  REQUIRE(fin[0].stats.answered_by.size() == 1);
  CHECK(fin[0].stats.answered_by[0].tokens == 18);

  const auto conv = f.svc->conversation();
  REQUIRE(conv.size() == 3);
  CHECK(conv[0].role == father::ChatRole::kSystem);
  CHECK(conv[1].content == "hello cluster");
  CHECK(conv[2].role == father::ChatRole::kAssistant);
  CHECK(conv[2].content == f.tokenizer->decode(ref));

  // Second turn continues the same conversation; the prompt is the whole history.
  req.user_message = "again";
  req.max_new_tokens = 6;
  REQUIRE(f.svc->chat(req).is_ok());
  REQUIRE(f.svc->wait_idle(60s));
  CHECK(f.svc->conversation().size() == 5);
  CHECK(f.events->all<father::FinishedEvent>().size() == 2);
  REQUIRE(f.svc->release().is_ok());
}

TEST_CASE("diagnostics are redacted by default and carry no prompt or response text") {
  Fixture f;
  f.prepare("fast");
  father::ChatRequest req;
  req.user_message = "TOP-SECRET-PROMPT";
  req.max_new_tokens = 8;
  REQUIRE(f.svc->chat(req).is_ok());
  REQUIRE(f.svc->wait_idle(60s));
  auto d = f.svc->diagnostics();
  CHECK(d.conversation.empty());
  CHECK(d.conversation_messages == 2);
  CHECK(d.requests_completed == 1);
  CHECK(d.tokens_emitted == 8);
  CHECK(d.active_tier == "fast");
  CHECK(d.active_ready);
  CHECK(d.to_string().find("TOP-SECRET") == std::string::npos);
  CHECK(d.to_string().find("redacted") != std::string::npos);
  REQUIRE(d.tiers.size() == 3);
  auto full = f.svc->diagnostics(true);
  REQUIRE(full.conversation.size() == 2);
  CHECK(full.conversation[0].content == "TOP-SECRET-PROMPT");
}

TEST_CASE("cancel stops generation promptly at the next window and keeps the partial answer") {
  Fixture f;
  f.prepare("strong");
  std::atomic<father::RequestId> rid{0};
  std::atomic<bool> cancelled{false};
  f.svc->subscribe([&](const father::Event& e) {
    if (std::holds_alternative<father::TokensEvent>(e) && !cancelled.exchange(true)) {
      while (rid.load() == 0) std::this_thread::yield();  // chat() has not returned its id yet
      CHECK(f.svc->cancel(rid.load()).is_ok());
    }
  });
  father::ChatRequest req;
  req.user_message = "cancel me";
  req.max_new_tokens = 200;
  auto r = f.svc->chat(req);
  REQUIRE(r.is_ok());
  rid = r.value();
  REQUIRE(f.svc->wait_idle(60s));
  auto fin = f.events->all<father::FinishedEvent>();
  REQUIRE(fin.size() == 1);
  CHECK(fin[0].reason == father::FinishReason::kCancelled);
  CHECK(fin[0].stats.tokens < 200);
  CHECK(fin[0].stats.tokens >= 1);
  CHECK(f.svc->cancel(r.value()).code() == ErrorCode::kNotFound);  // already finished
  CHECK(f.svc->conversation().size() == 2);  // user + partial assistant
  REQUIRE(f.svc->release().is_ok());
}

TEST_CASE("a second request while one runs is refused") {
  Fixture f;
  f.prepare("fast");
  father::ChatRequest req;
  req.user_message = "one";
  req.max_new_tokens = 64;
  REQUIRE(f.svc->chat(req).is_ok());
  auto second = f.svc->chat(req);
  CHECK(second.status().code() == ErrorCode::kFailedPrecondition);
  CHECK(f.svc->reset_conversation().code() == ErrorCode::kFailedPrecondition);
  REQUIRE(f.svc->wait_idle(60s));
  CHECK(f.svc->conversation().size() == 2);  // the refused request left no trace
  CHECK(f.svc->reset_conversation().is_ok());
  CHECK(f.svc->conversation().empty());
}

TEST_CASE("invalid chat requests are rejected without touching the conversation") {
  Fixture f;
  father::ChatRequest req;
  req.user_message = "x";
  CHECK(f.svc->chat(req).status().code() == ErrorCode::kFailedPrecondition);  // no tier selected
  REQUIRE(f.svc->select_tier("fast").is_ok());
  CHECK(f.svc->select_tier("turbo").code() == ErrorCode::kNotFound);
  req.context_tokens = 128 * 1024;  // not offered for Fast
  CHECK(f.svc->chat(req).status().code() == ErrorCode::kInvalidArgument);
  req.context_tokens = 4096;
  req.max_new_tokens = 5000;  // prompt + output exceed the context
  CHECK(f.svc->chat(req).status().code() == ErrorCode::kOutOfRange);
  req.max_new_tokens = 0;
  CHECK(f.svc->chat(req).status().code() == ErrorCode::kInvalidArgument);
  CHECK(f.svc->conversation().empty());
  CHECK(f.svc->cancel(99).code() == ErrorCode::kNotFound);
}

TEST_CASE("node lost mid-generation: Ultra -> Fast with a readable event, the answer continues and names its model") {
  Fixture f;
  f.prepare("ultra");
  std::atomic<bool> poked{false};
  f.svc->subscribe([&](const father::Event& e) {
    if (std::holds_alternative<father::TokensEvent>(e) && !poked.exchange(true)) f.g14->on_local_activity();
  });
  father::ChatRequest req;
  req.user_message = "survive a node loss";
  req.max_new_tokens = 24;
  auto rid = f.svc->chat(req);
  REQUIRE(rid.is_ok());
  REQUIRE(f.svc->wait_idle(120s));

  const auto prompt = f.tokenizer->encode_chat(std::vector<father::ChatMessage>{{father::ChatRole::kUser, "survive a node loss"}});
  const auto ref = f.reference(prompt, 24);

  auto fb = f.events->all<father::FallbackEvent>();
  REQUIRE(fb.size() == 1);
  CHECK(fb[0].kind == father::FallbackEvent::Kind::kDowngrade);
  CHECK(fb[0].from_tier == "ultra");
  CHECK(fb[0].to_tier == "fast");  // Strong also needs G14, which is in use
  CHECK(fb[0].from_model == "Qwen3.8-Flash-Next GSQ-RCO IQ3_S");
  CHECK(fb[0].to_model == "Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S");
  CHECK(fb[0].message.find("G14 is in use") != std::string::npos);
  CHECK(fb[0].message.find(fb[0].from_model) != std::string::npos);
  CHECK(fb[0].message.find(fb[0].to_model) != std::string::npos);
  CHECK(fb[0].message.find("conversation is kept") != std::string::npos);

  // The whole answer is still the correct continuation, and each chunk names the model that produced it.
  auto toks = f.events->all<father::TokensEvent>();
  CHECK(Fixture::flatten(toks) == ref);
  REQUIRE(toks.size() >= 2);
  CHECK(toks.front().tier_id == "ultra");
  CHECK(toks.front().model_name == "Qwen3.8-Flash-Next GSQ-RCO IQ3_S");
  CHECK(toks.back().tier_id == "fast");
  CHECK(toks.back().model_name == "Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S");

  auto fin = f.events->all<father::FinishedEvent>();
  REQUIRE(fin.size() == 1);
  CHECK(fin[0].reason == father::FinishReason::kCompleted);
  CHECK(fin[0].stats.fallbacks == 1);
  CHECK(fin[0].stats.tokens == 24);
  CHECK(fin[0].stats.final_tier == "fast");
  CHECK(fin[0].stats.final_model == "Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S");
  REQUIRE(fin[0].stats.answered_by.size() == 2);
  CHECK(fin[0].stats.answered_by[0].tier_id == "ultra");
  CHECK(fin[0].stats.answered_by[1].tier_id == "fast");
  CHECK(fin[0].stats.answered_by[0].tokens + fin[0].stats.answered_by[1].tokens == 24);
  CHECK(f.events->all<father::ErrorEvent>().empty());

  // Conversation survived; readiness now tells the truth.
  const auto conv = f.svc->conversation();
  REQUIRE(conv.size() == 2);
  CHECK(conv[1].content == f.tokenizer->decode(ref));
  auto tiers = f.svc->list_tiers();
  CHECK(pick(tiers, "fast").state == TierState::kReady);
  CHECK(pick(tiers, "ultra").state == TierState::kUnavailable);
  CHECK(pick(tiers, "ultra").headline() == "G14 is in use");
  CHECK(pick(tiers, "ultra").suggested_fallback == std::vector<std::string>{"fast"});
  CHECK(f.svc->diagnostics().fallbacks == 1);

  // The surviving node was released by the downgrade and holds nothing.
  CHECK(f.n3060->status().state == node::NodeState::kAvailable);
  CHECK(f.n3060->status().staging_census_bytes == 0);
  CHECK(f.g14->status().staging_census_bytes == 0);
  REQUIRE(f.svc->release().is_ok());
}

TEST_CASE("the catalog policy retries the same tier when it is still viable, then continues on it") {
  Fixture f;
  f.prepare("strong");
  std::atomic<bool> poked{false};
  f.svc->subscribe([&](const father::Event& e) {
    if (std::holds_alternative<father::TokensEvent>(e) && !poked.exchange(true)) {
      f.g14->on_local_activity();  // lease revoked: the distributed session is lost ...
      f.g14->on_local_idle();      // ... but the machine is immediately available again
    }
  });
  father::ChatRequest req;
  req.user_message = "retry please";
  req.max_new_tokens = 16;
  REQUIRE(f.svc->chat(req).is_ok());
  REQUIRE(f.svc->wait_idle(120s));

  auto fb = f.events->all<father::FallbackEvent>();
  REQUIRE(fb.size() == 1);
  CHECK(fb[0].kind == father::FallbackEvent::Kind::kRetry);
  CHECK(fb[0].from_tier == "strong");
  CHECK(fb[0].to_tier == "strong");
  CHECK(fb[0].to_model == "Swift-1.5-Qwen3.8-Flash-Next GSQ-RCO IQ2_XS");
  CHECK(fb[0].message.find("retrying") != std::string::npos);

  const auto prompt = f.tokenizer->encode_chat(std::vector<father::ChatMessage>{{father::ChatRole::kUser, "retry please"}});
  CHECK(Fixture::flatten(f.events->all<father::TokensEvent>()) == f.reference(prompt, 16));
  auto fin = f.events->all<father::FinishedEvent>();
  REQUIRE(fin.size() == 1);
  CHECK(fin[0].reason == father::FinishReason::kCompleted);
  CHECK(fin[0].stats.final_tier == "strong");
  CHECK(fin[0].stats.answered_by.size() == 1);
  REQUIRE(f.svc->release().is_ok());
}

TEST_CASE("when no lower tier can continue, the request ends with an explicit error and no substitute model") {
  Fixture f;
  f.prepare("strong");
  // Strong is lost and the unpinned-model confirmation is withdrawn, so Fast cannot be prepared either.
  std::atomic<bool> poked{false};
  f.svc->subscribe([&](const father::Event& e) {
    if (std::holds_alternative<father::TokensEvent>(e) && !poked.exchange(true)) {
      f.g14->on_local_activity();
      f.source->confirm_ = false;  // every tier's unpinned model becomes unconfirmed: nothing can continue
    }
  });
  father::ChatRequest req;
  req.user_message = "no way out";
  req.max_new_tokens = 16;
  REQUIRE(f.svc->chat(req).is_ok());
  REQUIRE(f.svc->wait_idle(120s));
  CHECK(f.events->all<father::FallbackEvent>().empty());
  auto errs = f.events->all<father::ErrorEvent>();
  REQUIRE(errs.size() == 1);
  CHECK(errs[0].message.find("no lower tier") != std::string::npos);
  auto fin = f.events->all<father::FinishedEvent>();
  REQUIRE(fin.size() == 1);
  CHECK(fin[0].reason == father::FinishReason::kFailed);
  CHECK(fin[0].stats.tokens >= 1);  // the partial answer is reported, not discarded
  CHECK(f.svc->conversation().size() == 2);
  CHECK(f.svc->diagnostics().requests_failed == 1);
  CHECK_FALSE(f.svc->diagnostics().last_error.empty());
}

TEST_CASE("prepare progress is polled from the readiness source while the plan is provisioned") {
  // A source that reports provisioning progress while Coordinator::prepare runs.
  struct Progressing final : father::ReadinessSource {
    WorkerSource* inner;
    std::atomic<int> calls{0};
    catalog::ReadinessInputs observe(const catalog::TierEntry& t, const catalog::TierAssignment& a, std::uint32_t c) override {
      auto in = inner->observe(t, a, c);
      const int n = ++calls;
      if (n > 1) {  // after the admission check: provisioning in flight
        catalog::ProvisioningProgress p;
        p.bytes_total = 100;
        p.bytes_done = static_cast<std::uint64_t>(std::min(n * 10, 90));
        p.rate_bytes_per_s = 10;
        in.provisioning = p;
      }
      return in;
    }
  };
  Fixture f;
  auto prog = std::make_shared<Progressing>();
  prog->inner = f.source.get();
  father::ServiceDeps deps;
  deps.catalog = f.cat;
  deps.assignment.bind("father", "Father");
  deps.assignment.bind("node:laptop-class", "G14");
  deps.assignment.bind("node:designated-3060", "3060");
  deps.tokenizer = f.tokenizer;
  deps.readiness = prog;
  deps.deployments = f.provider;
  deps.options.progress_poll = 1ms;
  auto s = father::make_father_service(std::move(deps));
  REQUIRE(s.is_ok());
  auto coll = std::make_shared<Collector>();
  s.value()->subscribe([coll](const father::Event& e) { (*coll)(e); });
  REQUIRE(s.value()->prepare_tier("ultra", 4096).is_ok());
  REQUIRE(s.value()->wait_idle(30s));
  auto pe = coll->all<father::PrepareProgressEvent>();
  REQUIRE_FALSE(pe.empty());
  CHECK(pe.front().message.find("Preparing Ultra") != std::string::npos);
  CHECK(coll->all<father::TierReadyEvent>().size() == 1);
  REQUIRE(s.value()->release().is_ok());
}

TEST_CASE("the Coordinator's progress reaches events, the observer and the readiness board") {
  // Readiness that reports what the ProvisioningBoard knows, exactly like production (ProductionOptions::provisioning).
  struct BoardSource final : father::ReadinessSource {
    WorkerSource* inner;
    std::shared_ptr<father::ProvisioningBoard> board;
    std::mutex m;
    std::vector<catalog::ProvisioningProgress> observed;
    catalog::ReadinessInputs observe(const catalog::TierEntry& t, const catalog::TierAssignment& a, std::uint32_t c) override {
      auto in = inner->observe(t, a, c);
      in.provisioning = board->get(t.id);
      if (in.provisioning) {
        std::lock_guard lk(m);
        observed.push_back(*in.provisioning);
      }
      return in;
    }
  };
  Fixture f;
  auto board = std::make_shared<father::ProvisioningBoard>();
  auto src = std::make_shared<BoardSource>();
  src->inner = f.source.get();
  src->board = board;
  std::atomic<int> finished{0};
  std::atomic<int> progress_calls{0};
  father::ServiceDeps deps;
  deps.catalog = f.cat;
  deps.assignment.bind("father", "Father");
  deps.assignment.bind("node:laptop-class", "G14");
  deps.assignment.bind("node:designated-3060", "3060");
  deps.tokenizer = f.tokenizer;
  deps.readiness = src;
  deps.deployments = f.provider;
  deps.options.progress_poll = 5ms;
  deps.prepare_observer = board->observer();
  auto inner_progress = deps.prepare_observer.on_progress;
  deps.prepare_observer.on_progress = [&](const std::string& tier, const coordinator::PrepareProgress& p) {
    ++progress_calls;
    inner_progress(tier, p);
  };
  auto inner_finished = deps.prepare_observer.on_finished;
  deps.prepare_observer.on_finished = [&](const std::string& tier) {
    ++finished;
    inner_finished(tier);
  };
  auto s = father::make_father_service(std::move(deps));
  REQUIRE(s.is_ok());
  auto coll = std::make_shared<Collector>();
  s.value()->subscribe([coll](const father::Event& e) { (*coll)(e); });
  REQUIRE(s.value()->prepare_tier("ultra", 4096).is_ok());
  REQUIRE(s.value()->wait_idle(30s));

  CHECK(progress_calls.load() >= 4);
  CHECK(finished.load() == 1);
  CHECK_FALSE(board->get("ultra").has_value());  // cleared when the prepare ended: nothing can be stuck "Preparing"

  // Events: some carry per-Node detail; the last one has both Nodes, fully sent and sealed.
  std::vector<father::PrepareDetail> details;
  for (const auto& e : coll->all<father::PrepareProgressEvent>())
    if (e.detail) details.push_back(*e.detail);
  REQUIRE_FALSE(details.empty());
  const auto& last = details.back();
  REQUIRE(last.nodes.size() == 2);
  CHECK(last.nodes[0].name == "g14");
  CHECK(last.nodes[1].name == "n3060");
  for (const auto& n : last.nodes) {
    CHECK(n.bytes_total > 0);
    CHECK(n.bytes_sent == n.bytes_total);
    CHECK(n.objects_sealed == n.objects_total);
  }
  for (const auto& d : details)
    for (const auto& n : d.nodes) CHECK(n.bytes_sent <= n.bytes_total);
  CHECK(coll->all<father::TierReadyEvent>().size() == 1);

  // The readiness source saw board progress while preparing (the percentage the UI draws comes from it).
  {
    std::lock_guard lk(src->m);
    for (const auto& o : src->observed) CHECK(o.bytes_done <= o.bytes_total);
  }
  REQUIRE(s.value()->release().is_ok());
}

TEST_CASE("ProvisioningBoard tracks the latest progress, measures a rate only from real data, and forgets on clear") {
  father::ProvisioningBoard board;
  CHECK_FALSE(board.get("strong").has_value());
  coordinator::PrepareProgress p;
  p.bytes_total = 1000;
  p.bytes_sent = 0;
  board.update("strong", p);
  auto g = board.get("strong");
  REQUIRE(g.has_value());
  CHECK(g->bytes_done == 0);
  CHECK(g->bytes_total == 1000);
  CHECK_FALSE(g->rate_bytes_per_s.has_value());  // no data yet: no rate, so no ETA
  p.bytes_sent = 100;
  board.update("strong", p);
  CHECK_FALSE(board.get("strong")->rate_bytes_per_s.has_value());  // one sample is not a rate
  std::this_thread::sleep_for(600ms);
  p.bytes_sent = 400;
  board.update("strong", p);
  g = board.get("strong");
  REQUIRE(g->rate_bytes_per_s.has_value());
  CHECK(g->rate_is_measured);  // observed on this run
  CHECK(*g->rate_bytes_per_s > 100.0);  // 300 bytes in about 0.6 s
  CHECK(*g->rate_bytes_per_s < 3000.0);
  CHECK(g->bytes_done == 400);
  board.update("fast", p);
  board.clear("strong");
  CHECK_FALSE(board.get("strong").has_value());
  CHECK(board.get("fast").has_value());
  // The wiring helpers share the same state.
  auto provider = board.provider();
  CHECK(provider("fast").has_value());
  board.observer().on_finished("fast");
  CHECK_FALSE(provider("fast").has_value());
}
