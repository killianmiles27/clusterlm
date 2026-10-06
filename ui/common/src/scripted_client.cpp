// ScriptedFatherClient and the `--demo` script. Everything produced by the demo is labelled Synthetic.
#include <sstream>

#include "clusterlm/ui/clients.hpp"

namespace clusterlm::ui {

ScriptedFatherClient::~ScriptedFatherClient() {
  stop_ = true;
  for (auto& t : threads_)
    if (t.joinable()) t.join();
}

void ScriptedFatherClient::record(std::string call) { calls_.push_back(std::move(call)); }

Status ScriptedFatherClient::take_failure(const std::string& method) {
  auto it = failures_.find(method);
  if (it == failures_.end()) return Status::ok();
  Status s = it->second;
  failures_.erase(it);
  return s;
}

void ScriptedFatherClient::set_tiers(std::vector<catalog::TierReadiness> tiers) {
  std::lock_guard lk(mu_);
  tiers_ = std::move(tiers);
}
void ScriptedFatherClient::set_participants(std::string tier_id, std::vector<TierParticipant> p) {
  std::lock_guard lk(mu_);
  participants_[std::move(tier_id)] = std::move(p);
}
void ScriptedFatherClient::set_context_use(std::optional<ContextUse> use) {
  std::lock_guard lk(mu_);
  context_use_ = use;
}
void ScriptedFatherClient::set_paired(std::vector<PairedMachine> paired) {
  std::lock_guard lk(mu_);
  paired_ = std::move(paired);
}
void ScriptedFatherClient::set_diagnostics(DiagnosticsExport d) {
  std::lock_guard lk(mu_);
  diag_ = std::move(d);
}
void ScriptedFatherClient::fail_next(const std::string& method, Status status) {
  std::lock_guard lk(mu_);
  failures_[method] = std::move(status);
}
std::vector<std::string> ScriptedFatherClient::calls() const {
  std::lock_guard lk(mu_);
  return calls_;
}
std::vector<father::ChatRequest> ScriptedFatherClient::chats() const {
  std::lock_guard lk(mu_);
  return chats_;
}
std::optional<bool> ScriptedFatherClient::last_export_include_text() const {
  std::lock_guard lk(mu_);
  return last_export_include_text_;
}

void ScriptedFatherClient::emit(const father::Event& event) {
  std::vector<father::EventSink> sinks;
  {
    std::lock_guard lk(mu_);
    for (auto& [id, s] : sinks_) sinks.push_back(s);
  }
  for (auto& s : sinks) s(event);
}

void ScriptedFatherClient::spawn(std::function<void(const std::atomic<bool>& stop)> body) {
  std::lock_guard lk(mu_);
  if (stop_) return;
  threads_.emplace_back([this, b = std::move(body)] { b(stop_); });
}

void ScriptedFatherClient::patch_tier(const std::string& tier_id, const std::function<void(catalog::TierReadiness&)>& fn) {
  std::lock_guard lk(mu_);
  for (auto& t : tiers_)
    if (t.tier_id == tier_id) fn(t);
}

Result<std::vector<catalog::TierReadiness>> ScriptedFatherClient::list_tiers(std::uint32_t) {
  std::lock_guard lk(mu_);
  record("list_tiers");
  if (auto f = take_failure("list_tiers"); !f.is_ok()) return f;
  return tiers_;
}

Result<std::vector<TierParticipant>> ScriptedFatherClient::participants(std::string_view tier_id) {
  std::lock_guard lk(mu_);
  auto it = participants_.find(std::string(tier_id));
  if (it == participants_.end()) return std::vector<TierParticipant>{};
  return it->second;
}

std::string ScriptedFatherClient::selected_tier() {
  std::lock_guard lk(mu_);
  return selected_;
}

Status ScriptedFatherClient::select_tier(std::string_view tier_id) {
  std::lock_guard lk(mu_);
  record("select_tier:" + std::string(tier_id));
  if (auto f = take_failure("select_tier"); !f.is_ok()) return f;
  selected_ = std::string(tier_id);
  return Status::ok();
}

Status ScriptedFatherClient::prepare_tier(std::string_view tier_id, std::uint32_t ctx) {
  std::function<void(ScriptedFatherClient&, const std::string&, std::uint32_t)> hook;
  {
    std::lock_guard lk(mu_);
    record("prepare_tier:" + std::string(tier_id));
    if (auto f = take_failure("prepare_tier"); !f.is_ok()) return f;
    hook = on_prepare;
  }
  if (hook) hook(*this, std::string(tier_id), ctx);
  return Status::ok();
}

Result<father::RequestId> ScriptedFatherClient::send_chat(father::ChatRequest request) {
  father::RequestId id = 0;
  decltype(on_chat) hook;
  {
    std::lock_guard lk(mu_);
    record("send_chat");
    if (auto f = take_failure("send_chat"); !f.is_ok()) return f;
    id = next_request_++;
    chats_.push_back(request);
    hook = on_chat;
  }
  if (hook) hook(*this, id, request);
  return id;
}

Status ScriptedFatherClient::cancel(father::RequestId request) {
  decltype(on_cancel) hook;
  {
    std::lock_guard lk(mu_);
    record("cancel:" + std::to_string(request));
    if (auto f = take_failure("cancel"); !f.is_ok()) return f;
    hook = on_cancel;
  }
  if (hook) hook(*this, request);
  return Status::ok();
}

Status ScriptedFatherClient::release() {
  std::lock_guard lk(mu_);
  record("release");
  return take_failure("release");
}

Status ScriptedFatherClient::reset_conversation() {
  std::lock_guard lk(mu_);
  record("reset_conversation");
  return take_failure("reset_conversation");
}

Result<ContextUse> ScriptedFatherClient::context_use() {
  std::lock_guard lk(mu_);
  if (!context_use_) return make_error(ErrorCode::kUnavailable, "context use unknown");
  return *context_use_;
}

Result<DiagnosticsExport> ScriptedFatherClient::export_diagnostics(bool include_text) {
  std::lock_guard lk(mu_);
  record("export_diagnostics");
  if (auto f = take_failure("export_diagnostics"); !f.is_ok()) return f;
  last_export_include_text_ = include_text;
  DiagnosticsExport d = diag_;
  d.includes_conversation = include_text;
  return d;
}

Result<FatherSettings> ScriptedFatherClient::get_settings() {
  std::lock_guard lk(mu_);
  return settings_;
}
Status ScriptedFatherClient::set_settings(const FatherSettings& s) {
  std::lock_guard lk(mu_);
  record("set_settings");
  if (auto f = take_failure("set_settings"); !f.is_ok()) return f;
  if (auto v = validate(s); !v.is_ok()) return v;
  settings_ = s;
  return Status::ok();
}

Result<std::vector<PairedMachine>> ScriptedFatherClient::paired_machines() {
  std::lock_guard lk(mu_);
  if (auto f = take_failure("paired_machines"); !f.is_ok()) return f;
  return paired_;
}
Status ScriptedFatherClient::start_pairing() {
  std::lock_guard lk(mu_);
  record("start_pairing");
  return take_failure("start_pairing");
}
Status ScriptedFatherClient::unpair(std::string_view machine_id) {
  std::lock_guard lk(mu_);
  record("unpair:" + std::string(machine_id));
  return take_failure("unpair");
}

father::SubscriptionId ScriptedFatherClient::subscribe(father::EventSink sink) {
  std::lock_guard lk(mu_);
  const auto id = next_sub_++;
  sinks_[id] = std::move(sink);
  return id;
}
void ScriptedFatherClient::unsubscribe(father::SubscriptionId id) {
  std::lock_guard lk(mu_);
  sinks_.erase(id);
}

// ---- demo ---------------------------------------------------------------------------------------------------

namespace {

catalog::TierReadiness demo_tier(const char* id, const char* model, catalog::TierState st, const char* headline) {
  catalog::TierReadiness t;
  t.tier_id = id;
  t.model_name = model;
  t.state = st;
  t.reasons.push_back(headline);
  return t;
}

constexpr const char* kDemoFast = "Demo model (Fast)";
constexpr const char* kDemoStrong = "Demo model (Strong)";

void sleep_unless_stopped(const std::atomic<bool>& stop, int ms) {
  for (int waited = 0; waited < ms && !stop.load(); waited += 20) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

}  // namespace

std::unique_ptr<ScriptedFatherClient> make_demo_father_client() {
  auto c = std::make_unique<ScriptedFatherClient>("Synthetic");
  using catalog::TierState;
  c->set_tiers({demo_tier("fast", kDemoFast, TierState::kAvailable, "Fast is available (demo)"),
                demo_tier("strong", kDemoStrong, TierState::kAvailable, "Strong is available (demo)"),
                demo_tier("ultra", "Demo model (Ultra)", TierState::kUnavailable,
                          "Ultra needs the 3060 PC, which is offline (demo)")});
  c->set_participants("fast", {{"father", "This PC"}});
  c->set_participants("strong", {{"father", "This PC"}, {"node:laptop-class", "G14"}});
  c->set_participants("ultra", {{"father", "This PC"}, {"node:laptop-class", "G14"}, {"node:designated-3060", "3060"}});
  c->set_paired({{"G14", "Node", "demo-g14"}, {"3060", "Node", "demo-3060"}});
  DiagnosticsExport d;
  d.text = "DEMO MODE: scripted events, no model is running. Every number is Synthetic.\n";
  d.plan_summary = "demo plan: father 0-4, G14 4-10, father 10-16 (Synthetic)";
  d.stage_timings = {{"Father layers 0-3 (Synthetic)", 3.0}, {"G14 layers 4-9 (Synthetic)", 5.0}, {"Father layers 10-15 (Synthetic)", 4.0}};
  c->set_diagnostics(d);
  c->set_context_use(ContextUse{0, false});

  c->on_prepare = [](ScriptedFatherClient& self, const std::string& tier, std::uint32_t) {
    self.spawn([&self, tier](const std::atomic<bool>& stop) {
      const std::string model = tier == "fast" ? kDemoFast : tier == "strong" ? kDemoStrong : "Demo model (Ultra)";
      for (int pct = 0; pct <= 100 && !stop.load(); pct += 10) {
        self.patch_tier(tier, [&](catalog::TierReadiness& t) {
          t.state = TierState::kPreparing;
          t.progress = catalog::PrepareProgress{static_cast<double>(pct), (100 - pct) * 0.3, true};
          t.reasons = {"Preparing " + tier + " (demo)"};
        });
        self.emit(father::PrepareProgressEvent{tier, model, static_cast<double>(pct), (100 - pct) * 0.3,
                                               "Copying scripted demo data"});
        sleep_unless_stopped(stop, 250);
      }
      if (stop.load()) return;
      self.patch_tier(tier, [&](catalog::TierReadiness& t) {
        t.state = TierState::kReady;
        t.progress.reset();
        t.reasons = {tier + " is ready (demo)"};
      });
      self.emit(father::TierReadyEvent{tier, model});
    });
  };

  auto cancel_id = std::make_shared<std::atomic<std::uint64_t>>(0);
  c->on_cancel = [cancel_id](ScriptedFatherClient&, father::RequestId id) { cancel_id->store(id); };
  c->on_chat = [cancel_id](ScriptedFatherClient& self, father::RequestId id, const father::ChatRequest& req) {
    const bool fallback = req.user_message.find("fallback") != std::string::npos;
    const std::string tier = self.selected_tier().empty() ? "fast" : self.selected_tier();
    self.spawn([&self, id, fallback, tier, cancel_id](const std::atomic<bool>& stop) {
      std::vector<std::string> words = {"This ", "is ", "a ", "scripted ", "demo ", "answer. ", "No ", "model ",
                                        "is ", "running; ", "all ", "numbers ", "are ", "Synthetic."};
      std::string cur_tier = tier, cur_model = tier == "strong" ? kDemoStrong : kDemoFast;
      father::GenerationStats stats;
      stats.decode_tok_s = 12.5;
      stats.ttft_ms = 420;
      stats.accepted_per_round = 1.0;
      father::AnswerSegment seg{cur_tier, cur_model, 0};
      for (std::size_t i = 0; i < words.size(); ++i) {
        if (stop.load()) return;
        if (cancel_id->load() == id) {
          stats.answered_by.push_back(seg);
          stats.final_tier = cur_tier;
          stats.final_model = cur_model;
          self.emit(father::FinishedEvent{id, father::FinishReason::kCancelled, stats});
          return;
        }
        if (fallback && tier != "fast" && i == words.size() / 2) {
          father::FallbackEvent fe;
          fe.request = id;
          fe.kind = father::FallbackEvent::Kind::kDowngrade;
          fe.from_tier = tier;
          fe.to_tier = "fast";
          fe.from_model = cur_model;
          fe.to_model = kDemoFast;
          fe.message = "G14 became busy - switching from Strong (" + cur_model + ") to Fast (" + std::string(kDemoFast) +
                       "); your conversation is kept.";
          stats.answered_by.push_back(seg);
          cur_tier = "fast";
          cur_model = kDemoFast;
          seg = {cur_tier, cur_model, 0};
          ++stats.fallbacks;
          self.emit(fe);
        }
        father::TokensEvent te;
        te.request = id;
        te.tier_id = cur_tier;
        te.model_name = cur_model;
        te.tokens = {static_cast<std::int32_t>(i)};
        te.text = words[i];
        self.emit(te);
        ++stats.tokens;
        ++seg.tokens;
        sleep_unless_stopped(stop, 120);
      }
      stats.answered_by.push_back(seg);
      stats.rounds = stats.tokens;
      stats.final_tier = cur_tier;
      stats.final_model = cur_model;
      self.emit(father::FinishedEvent{id, father::FinishReason::kCompleted, stats});
    });
  };
  return c;
}

}  // namespace clusterlm::ui
