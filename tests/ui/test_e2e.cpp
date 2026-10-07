// End to end: the Father view-model driving a real FatherService (fixture model, in-process NodeWorkers over real
// TCP + protocol) through InProcessFatherClient. Mirrors tests/father_service; nothing here is mocked except the
// hardware (machine states come from the real workers).
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/father/service.hpp"
#include "clusterlm/node/node_worker.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/ui/clients.hpp"
#include "clusterlm/ui/father_viewmodel.hpp"

using namespace clusterlm;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using catalog::MachineState;
using catalog::TierState;

namespace {

fs::path temp_dir(const char* name) {
  auto p = fs::temp_directory_path() / (std::string("clm-father-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  { std::error_code ec_rm; fs::remove_all(p, ec_rm); }
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
    // Destroy the workers (open journal files) before deleting their directories: Windows cannot delete open files.
    g14.reset();
    n3060.reset();
    { std::error_code ec_rm; fs::remove_all(dir, ec_rm); }
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


}  // namespace

namespace {

struct UiFixture : Fixture {
  std::unique_ptr<ui::InProcessFatherClient> client;
  std::unique_ptr<ui::FatherViewModel> vm;

  UiFixture() {
    ui::InProcessFatherClient::Options o;
    o.catalog = cat;
    o.assignment.bind("father", "Father PC");
    o.assignment.bind("node:laptop-class", "G14");
    o.assignment.bind("node:designated-3060", "3060");
    o.tokenizer = tokenizer;
    o.paired = {{"G14", "Node", "g14fp"}, {"3060", "Node", "n3060fp"}};
    client = std::make_unique<ui::InProcessFatherClient>(*svc, std::move(o));
    vm = std::make_unique<ui::FatherViewModel>(*client);
    ui::FatherSettings s;
    s.max_new_tokens = 24;
    REQUIRE(vm->apply_settings(s).is_ok());
    vm->tick(true);
  }
  ~UiFixture() {
    vm.reset();  // detach before the service goes away
    client.reset();
  }

  // Polls the view-model the way the draw loop does until `pred` holds.
  template <typename Pred>
  bool wait_for(Pred pred, std::chrono::milliseconds timeout = 60s) {
    const auto end = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < end) {
      vm->tick(true);
      if (pred(vm->snapshot())) return true;
      std::this_thread::sleep_for(20ms);
    }
    return false;
  }
};

const ui::TierRow& row(const ui::FatherViewState& s, const char* id) {
  for (const auto& r : s.tiers)
    if (r.id == id) return r;
  FAIL("tier row missing");
  return s.tiers.front();
}

}  // namespace

TEST_CASE("e2e: tiers, prepare Fast, chat through the view-model, exact context and redacted diagnostics") {
  UiFixture f;
  auto s = f.vm->snapshot();
  REQUIRE(s.tiers.size() == 3);
  for (const auto& r : s.tiers) CHECK(r.state_label == "Available");  // honest before any prepare
  CHECK(row(s, "ultra").participants == std::vector<std::string>{"Father PC", "G14", "3060"});

  REQUIRE(f.vm->select_tier("fast").is_ok());
  REQUIRE(f.vm->prepare_tier("fast").is_ok());
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return row(v, "fast").state_label == "Ready" && !v.prepare.active; }));

  REQUIRE(f.vm->send("hello cluster").is_ok());
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return !v.chat_busy && v.last_answer.valid; }));
  s = f.vm->snapshot();
  REQUIRE(s.entries.size() == 2);
  CHECK(s.entries[0].text == "hello cluster");
  const auto conv = f.svc->conversation();
  REQUIRE(conv.size() == 2);
  CHECK(s.entries[1].full_text() == conv[1].content);  // what streamed is what the Father kept
  CHECK_FALSE(s.entries[1].full_text().empty());
  CHECK(s.entries[1].attribution == "Answered by Fast (Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S)");
  CHECK(s.last_answer.tokens == 24);
  CHECK(s.last_answer.provenance == "Measured");
  CHECK(s.last_answer.tok_s != "--");
  CHECK(s.last_answer.ttft != "--");
  CHECK(s.context.exact);
  CHECK(s.context.tokens_used > 24);
  CHECK(s.context.label.find("/ 4096 tokens") != std::string::npos);

  // Diagnostics: redacted unless the user opts in, and the preview never carries text.
  f.vm->set_exporter(ui::make_file_exporter((f.dir / "export").string()));
  f.vm->set_diagnostics_open(true);
  s = f.vm->snapshot();
  CHECK(s.diagnostics.snapshot_text.find("hello cluster") == std::string::npos);
  CHECK(s.diagnostics.snapshot_text.find("[text redacted]") != std::string::npos);
  CHECK(s.diagnostics.plan_summary.empty());  // the service does not report a plan: the UI says so
  REQUIRE(f.vm->export_diagnostics().is_ok());
  CHECK(f.vm->snapshot().diagnostics.export_message.find("no prompts or answers") != std::string::npos);

  REQUIRE(f.vm->release().is_ok());
  CHECK(f.wait_for([](const ui::FatherViewState& v) { return row(v, "fast").state_label == "Available"; }));
}

TEST_CASE("e2e: Ultra prepares, then a node goes busy mid-answer and the transcript explains the switch") {
  UiFixture f;
  REQUIRE(f.vm->select_tier("ultra").is_ok());
  REQUIRE(f.vm->prepare_tier("ultra").is_ok());
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return row(v, "ultra").state_label == "Ready" && !v.prepare.active; }));

  std::atomic<bool> poked{false};
  f.client->subscribe([&](const father::Event& e) {
    if (std::holds_alternative<father::TokensEvent>(e) && !poked.exchange(true)) f.g14->on_local_activity();
  });
  REQUIRE(f.vm->send("survive a node loss").is_ok());
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return !v.chat_busy && v.last_answer.valid; }, 120s));

  auto s = f.vm->snapshot();
  const ui::ChatEntry* notice = nullptr;
  const ui::ChatEntry* last_assistant = nullptr;
  for (const auto& e : s.entries) {
    if (e.kind == ui::EntryKind::kNotice) notice = &e;
    if (e.kind == ui::EntryKind::kAssistant) last_assistant = &e;
  }
  REQUIRE(notice != nullptr);
  REQUIRE(last_assistant != nullptr);
  CHECK(notice->text.find("Qwen3.8-Flash-Next GSQ-RCO IQ3_S") != std::string::npos);
  CHECK(notice->text.find("Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S") != std::string::npos);
  CHECK(notice->text.find("conversation is kept") != std::string::npos);
  CHECK(s.last_answer.fallbacks == 1);
  CHECK(s.last_answer.final_model == "Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S");
  CHECK(s.active_model == "Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S");
  // The continuation is attributed to the model that actually produced it.
  CHECK(last_assistant->attribution.rfind("Continued by Fast", 0) == 0);
  // The conversation survived on Father: user + one assistant message.
  CHECK(f.svc->conversation().size() == 2);
  REQUIRE(f.vm->release().is_ok());
}

TEST_CASE("e2e: cancel from the UI stops the answer and keeps the partial text") {
  UiFixture f;
  ui::FatherSettings fs;
  fs.max_new_tokens = 400;
  REQUIRE(f.vm->apply_settings(fs).is_ok());
  REQUIRE(f.vm->select_tier("strong").is_ok());
  REQUIRE(f.vm->prepare_tier("strong").is_ok());
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return row(v, "strong").state_label == "Ready" && !v.prepare.active; }));
  REQUIRE(f.vm->send("cancel me").is_ok());
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return v.entries.size() >= 2 && !v.entries[1].full_text().empty(); }));
  (void)f.vm->cancel();  // may race with natural completion; both outcomes must leave a consistent UI
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return !v.chat_busy && v.last_answer.valid; }));
  auto s = f.vm->snapshot();
  CHECK((s.last_answer.reason == "cancelled" || s.last_answer.reason == "completed"));
  CHECK_FALSE(s.entries[1].full_text().empty());
  CHECK_FALSE(s.entries[1].streaming);
  if (s.last_answer.reason == "cancelled") CHECK(s.entries.back().text.find("stopped") != std::string::npos);
  REQUIRE(f.vm->release().is_ok());
}

TEST_CASE("e2e: an unconfirmed model is reported in the banner and the tier stays Unavailable") {
  UiFixture f;
  f.source->confirm_ = false;
  f.vm->tick(true);
  CHECK(row(f.vm->snapshot(), "ultra").state_label == "Unavailable");
  CHECK_FALSE(row(f.vm->snapshot(), "ultra").headline.empty());
  REQUIRE(f.vm->select_tier("ultra").is_ok());
  f.vm->tick(true);
  CHECK_FALSE(f.vm->snapshot().can_send);
  (void)f.vm->prepare_tier("ultra");  // the service accepts the job and then fails it
  REQUIRE(f.wait_for([](const ui::FatherViewState& v) { return !v.prepare.active && v.banner.kind == ui::Banner::Kind::kError; }));
  CHECK(f.vm->snapshot().banner.text.find("unpinned") != std::string::npos);
}
