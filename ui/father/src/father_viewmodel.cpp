#include "clusterlm/ui/father_viewmodel.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <filesystem>

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

namespace {

std::string capitalise(std::string s) {
  if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
  return s;
}

std::string trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

bool contains_ci(const std::string& hay, const char* needle) {
  std::string h = hay, n = needle;
  for (auto& c : h) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  for (auto& c : n) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return h.find(n) != std::string::npos;
}

std::string state_label(catalog::TierState s, std::optional<double> percent) {
  switch (s) {
    case catalog::TierState::kUnavailable: return "Unavailable";
    case catalog::TierState::kAvailable: return "Available";
    case catalog::TierState::kPreparing:
      if (percent) return "Preparing " + std::to_string(static_cast<int>(std::clamp(*percent, 0.0, 100.0))) + "%";
      return "Preparing";
    case catalog::TierState::kReady: return "Ready";
  }
  return "Unavailable";
}

}  // namespace

std::string ChatEntry::full_text() const {
  std::string out;
  for (const auto& s : spans) out += s.text;
  return out;
}

DiagnosticsExporter make_file_exporter(std::string dir) {
  return [dir = std::move(dir)](const std::string& text) -> Result<std::string> {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return make_error(ErrorCode::kUnavailable, "Could not create the folder " + dir);
    const auto stamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const fs::path path = fs::path(dir) / ("clusterlm-diagnostics-" + std::to_string(stamp) + ".txt");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return make_error(ErrorCode::kUnavailable, "Could not write " + path.string());
    f << text;
    f.flush();
    if (!f) return make_error(ErrorCode::kUnavailable, "Could not write " + path.string());
    return path.string();
  };
}

// The event sink outlives neither client nor view-model use: the gate lets the destructor detach safely.
struct FatherViewModel::Gate {
  std::mutex m;
  FatherViewModel* vm = nullptr;
};

FatherViewModel::FatherViewModel(FatherClient& client, bool demo)
    : client_(client), gate_(std::make_shared<Gate>()) {
  gate_->vm = this;
  st_.demo = demo;
  if (auto s = client_.get_settings(); s.is_ok()) {
    st_.settings = s.value();
    st_.context.context_tokens = s->context_tokens;
  }
  if (auto p = client_.paired_machines(); p.is_ok()) {
    st_.pairing.machines = p.value();
  } else {
    st_.pairing.message = ascii_display(p.status().message());
  }
  std::weak_ptr<Gate> w = gate_;
  sub_ = client_.subscribe([w](const father::Event& e) {
    if (auto g = w.lock()) {
      std::lock_guard lk(g->m);
      if (g->vm) g->vm->on_event(e);
    }
  });
  recompute_send_state();
}

FatherViewModel::~FatherViewModel() {
  {
    std::lock_guard lk(gate_->m);
    gate_->vm = nullptr;
  }
  client_.unsubscribe(sub_);
}

void FatherViewModel::set_exporter(DiagnosticsExporter e) {
  std::lock_guard lk(mu_);
  exporter_ = std::move(e);
}

FatherViewState FatherViewModel::snapshot() const {
  std::lock_guard lk(mu_);
  return st_;
}

// ---- helpers (under mu_) -------------------------------------------------------------------------------------

void FatherViewModel::set_banner(Banner::Kind k, std::string text) {
  st_.banner.kind = k;
  st_.banner.text = std::move(text);
}

std::string FatherViewModel::tier_label(const std::string& tier_id, const std::string& model) const {
  std::string t = tier_id.empty() ? std::string() : capitalise(tier_id);
  if (t.empty()) return model;
  return model.empty() ? t : t + " (" + model + ")";
}

void FatherViewModel::rebuild_attribution(ChatEntry& e) const {
  if (e.spans.empty()) {
    e.attribution.clear();
    return;
  }
  std::string s = e.continued ? "Continued by " : "Answered by ";
  for (std::size_t i = 0; i < e.spans.size(); ++i) {
    if (i) s += ", then ";
    s += tier_label(e.spans[i].tier_id, e.spans[i].model);
  }
  e.attribution = ascii_display(s);
}

ChatEntry& FatherViewModel::assistant_entry(father::RequestId id) {
  if (!st_.entries.empty()) {
    auto& last = st_.entries.back();
    if (last.kind == EntryKind::kAssistant && last.request == id && last.streaming) return last;
  }
  ChatEntry e;
  e.kind = EntryKind::kAssistant;
  e.request = id;
  e.streaming = true;
  // A continuation: this request already produced an answer entry (a notice interrupted it).
  for (const auto& old : st_.entries)
    if (old.kind == EntryKind::kAssistant && old.request == id) e.continued = true;
  st_.entries.push_back(std::move(e));
  return st_.entries.back();
}

void FatherViewModel::recompute_send_state() {
  auto& s = st_;
  s.can_send = false;
  s.send_hint.clear();
  if (!s.connection_message.empty()) {
    s.send_hint = s.connection_message;
    return;
  }
  if (s.chat_busy) {
    s.send_hint = "An answer is being written. You can stop it with Cancel.";
    return;
  }
  if (s.prepare.active) {
    s.send_hint = "A model is being prepared. You can chat when it is ready.";
    return;
  }
  const TierRow* sel = nullptr;
  for (const auto& r : s.tiers)
    if (r.selected) sel = &r;
  if (!sel) {
    s.send_hint = "Choose Fast, Strong or Ultra first.";
    return;
  }
  switch (sel->state) {
    case catalog::TierState::kUnavailable:
      s.send_hint = sel->name + " is not available: " + sel->headline;
      return;
    case catalog::TierState::kPreparing:
      s.send_hint = sel->name + " is still preparing.";
      return;
    case catalog::TierState::kAvailable:
      s.can_send = true;
      s.send_hint = sel->name + " is not prepared yet; it will be prepared first, which can take a while.";
      return;
    case catalog::TierState::kReady:
      s.can_send = true;
      return;
  }
}

// ---- reloading -----------------------------------------------------------------------------------------------

void FatherViewModel::reload_tiers() {
  std::uint32_t ctx;
  {
    std::lock_guard lk(mu_);
    ctx = st_.context.context_tokens;
  }
  auto tiers = client_.list_tiers(ctx);
  std::string sel = client_.selected_tier();
  std::vector<std::vector<TierParticipant>> parts;
  if (tiers.is_ok())
    for (const auto& t : tiers.value()) {
      auto p = client_.participants(t.tier_id);
      parts.push_back(p.is_ok() ? p.value() : std::vector<TierParticipant>{});
    }

  std::lock_guard lk(mu_);
  if (!tiers.is_ok()) {
    st_.connection_message = ascii_display(tiers.status().message());
    st_.tiers.clear();
    recompute_send_state();
    return;
  }
  st_.connection_message.clear();
  st_.tiers.clear();
  st_.participants.clear();
  const bool other_running = st_.chat_busy;
  for (std::size_t i = 0; i < tiers->size(); ++i) {
    const auto& t = (*tiers)[i];
    TierRow r;
    r.id = t.tier_id;
    r.name = capitalise(t.tier_id);
    r.model = ascii_display(t.model_name);
    r.state = t.state;
    r.headline = ascii_display(t.headline());
    for (const auto& n : t.notes) r.notes.push_back(ascii_display(n));
    r.selected = (t.tier_id == sel);
    if (t.state == catalog::TierState::kPreparing) {
      if (st_.prepare.active && st_.prepare.tier_id == t.tier_id && st_.prepare.percent) {
        r.progress_percent = st_.prepare.percent;
        r.eta = st_.prepare.eta;
      } else if (t.progress) {
        r.progress_percent = t.progress->percent;
        r.eta = format_eta(t.progress->eta_seconds);
      }
    }
    r.state_label = state_label(t.state, r.progress_percent);
    for (const auto& p : parts[i]) r.participants.push_back(ascii_display(p.machine_id));
    r.can_prepare = t.state == catalog::TierState::kAvailable && !other_running && !st_.prepare.active;
    if (r.selected) {
      st_.participants = r.participants;
      if (!st_.chat_busy && !active_from_answer_) {
        st_.active_tier = r.id;
        st_.active_model = r.model;
      }
    }
    st_.tiers.push_back(std::move(r));
  }
  st_.selected_tier = sel;
  tiers_loaded_ = true;
  recompute_send_state();
}

void FatherViewModel::reload_context() {
  auto use = client_.context_use();
  std::lock_guard lk(mu_);
  auto& c = st_.context;
  if (use.is_ok()) {
    c.tokens_used = use->tokens_used;
    c.exact = use->exact;
  } else if (st_.entries.empty()) {
    c.tokens_used = 0;
    c.exact = true;
  } else {
    c.tokens_used = generated_tokens_;
    c.exact = false;
  }
  c.label = (c.exact ? "" : "at least ") + std::to_string(c.tokens_used) + " / " + std::to_string(c.context_tokens) + " tokens";
  context_pending_ = false;
}

void FatherViewModel::tick(bool force) {
  bool do_tiers, do_context;
  {
    std::lock_guard lk(mu_);
    const auto now = std::chrono::steady_clock::now();
    const auto interval = (st_.prepare.active || st_.chat_busy) ? std::chrono::seconds(1) : std::chrono::seconds(3);
    do_tiers = force || refresh_pending_ || !tiers_loaded_ || now - last_poll_ >= interval;
    do_context = force || context_pending_;
    if (do_tiers) {
      last_poll_ = now;
      refresh_pending_ = false;
    }
  }
  if (do_tiers) reload_tiers();
  if (do_context) reload_context();
}

// ---- events --------------------------------------------------------------------------------------------------

void FatherViewModel::on_event(const father::Event& ev) {
  std::lock_guard lk(mu_);
  struct V {
    FatherViewModel& vm;
    FatherViewState& st;
    void operator()(const father::TierSelectedEvent& e) {
      st.selected_tier = e.tier_id;
      st.active_tier = e.tier_id;
      st.active_model = ascii_display(e.model_name);
      vm.active_from_answer_ = false;
      vm.refresh_pending_ = true;
    }
    void operator()(const father::PrepareProgressEvent& e) {
      auto& p = st.prepare;
      p.active = true;
      p.tier_id = e.tier_id;
      p.model = ascii_display(e.model_name);
      p.percent = e.percent;
      p.eta = format_eta(e.eta_seconds);
      p.message = ascii_display(e.message);
      for (auto& r : st.tiers)
        if (r.id == e.tier_id && r.state != catalog::TierState::kReady) {
          r.state = catalog::TierState::kPreparing;
          r.progress_percent = e.percent;
          r.eta = p.eta;
          r.state_label = state_label(r.state, r.progress_percent);
          r.can_prepare = false;
        }
      vm.recompute_send_state();
    }
    void operator()(const father::TierReadyEvent& e) {
      st.prepare = PrepareView{};
      vm.set_banner(Banner::Kind::kInfo, capitalise(e.tier_id) + " finished preparing (" + ascii_display(e.model_name) + ").");
      vm.refresh_pending_ = true;
      vm.recompute_send_state();
    }
    void operator()(const father::TokensEvent& e) {
      auto& entry = vm.assistant_entry(e.request);
      const std::string model = ascii_display(e.model_name);
      if (entry.spans.empty() || entry.spans.back().tier_id != e.tier_id || entry.spans.back().model != model)
        entry.spans.push_back({e.tier_id, model, {}});
      entry.spans.back().text += e.text;
      vm.generated_tokens_ += static_cast<std::uint32_t>(e.tokens.size());
      vm.rebuild_attribution(entry);
      st.active_tier = e.tier_id;
      st.active_model = model;
      vm.active_from_answer_ = true;
    }
    void operator()(const father::FallbackEvent& e) {
      for (auto& en : st.entries)
        if (en.request == e.request && en.kind == EntryKind::kAssistant) en.streaming = false;
      ChatEntry n;
      n.kind = EntryKind::kNotice;
      n.request = e.request;
      n.text = ascii_display(e.message);
      if (!contains_ci(n.text, "conversation is kept")) n.text += " Your conversation is kept.";
      st.entries.push_back(std::move(n));
      if (!e.to_tier.empty()) st.active_tier = e.to_tier;
      if (!e.to_model.empty()) st.active_model = ascii_display(e.to_model);
      vm.active_from_answer_ = true;
      vm.refresh_pending_ = true;
    }
    void operator()(const father::FinishedEvent& e) {
      vm.finished_.insert(e.request);
      ChatEntry* last = nullptr;
      for (auto& en : st.entries)
        if (en.request == e.request && en.kind == EntryKind::kAssistant) {
          en.streaming = false;
          en.cancelled = e.reason == father::FinishReason::kCancelled;
          last = &en;
        }
      if (last && e.stats.answered_by.size() > 1) {
        std::string whole = " (whole answer: ";
        for (std::size_t i = 0; i < e.stats.answered_by.size(); ++i) {
          if (i) whole += ", then ";
          whole += vm.tier_label(e.stats.answered_by[i].tier_id, e.stats.answered_by[i].model_name);
        }
        last->attribution += ascii_display(whole + ")");
      }
      if (e.reason == father::FinishReason::kCancelled) {
        ChatEntry n;
        n.kind = EntryKind::kNotice;
        n.request = e.request;
        n.text = "Answer stopped. What was written so far is kept in your conversation.";
        st.entries.push_back(std::move(n));
      } else if (e.reason == father::FinishReason::kFailed) {
        bool has_error = false;
        for (const auto& en : st.entries)
          if (en.request == e.request && en.kind == EntryKind::kError) has_error = true;
        if (!has_error) {
          ChatEntry n;
          n.kind = EntryKind::kError;
          n.request = e.request;
          n.text = "The answer could not be completed. Your conversation is kept.";
          st.entries.push_back(std::move(n));
        }
      }
      auto& l = st.last_answer;
      l.valid = true;
      l.ttft = format_millis(e.stats.ttft_ms);
      l.tok_s = format_rate(e.stats.decode_tok_s);
      char buf[32];
      std::snprintf(buf, sizeof buf, "%.2f", e.stats.accepted_per_round);
      l.accepted_per_round = e.stats.accepted_per_round > 0 ? buf : "--";
      l.tokens = e.stats.tokens;
      l.rounds = e.stats.rounds;
      l.fallbacks = e.stats.fallbacks;
      l.final_model = ascii_display(e.stats.final_model);
      l.reason = std::string(father::to_string(e.reason));
      l.provenance = std::string(vm.client_.stats_provenance());
      if (!e.stats.final_tier.empty()) st.active_tier = e.stats.final_tier;
      if (!e.stats.final_model.empty()) {
        st.active_model = ascii_display(e.stats.final_model);
        vm.active_from_answer_ = true;
      }
      if (vm.current_request_ == 0 || vm.current_request_ == e.request) st.chat_busy = false;
      vm.context_pending_ = true;
      vm.refresh_pending_ = true;
      vm.recompute_send_state();
    }
    void operator()(const father::ErrorEvent& e) {
      ChatEntry n;
      n.kind = EntryKind::kError;
      n.request = e.request;
      n.text = describe_error(e.code, e.message);
      for (auto& en : st.entries)
        if (en.request == e.request && en.kind == EntryKind::kAssistant) en.streaming = false;
      vm.set_banner(Banner::Kind::kError, n.text);
      st.entries.push_back(std::move(n));
      if (e.request == 0) {
        st.prepare = PrepareView{};
      } else {
        vm.finished_.insert(e.request);
        if (vm.current_request_ == 0 || vm.current_request_ == e.request) st.chat_busy = false;
      }
      vm.refresh_pending_ = true;
      vm.recompute_send_state();
    }
    void operator()(const father::ReleasedEvent& e) {
      st.prepare = PrepareView{};
      vm.set_banner(Banner::Kind::kInfo, e.message.empty() ? "Released. Helper machines are cleaning up." : ascii_display(e.message));
      vm.refresh_pending_ = true;
      vm.recompute_send_state();
    }
  };
  std::visit(V{*this, st_}, ev);
}

// ---- commands ------------------------------------------------------------------------------------------------

Status FatherViewModel::select_tier(const std::string& tier_id) {
  auto s = client_.select_tier(tier_id);
  std::lock_guard lk(mu_);
  if (!s.is_ok()) {
    set_banner(Banner::Kind::kError, describe_error(s.code(), s.message()));
    return s;
  }
  st_.selected_tier = tier_id;
  active_from_answer_ = false;
  for (auto& r : st_.tiers) r.selected = (r.id == tier_id);
  refresh_pending_ = true;
  recompute_send_state();
  return s;
}

Status FatherViewModel::prepare_tier(const std::string& tier_id) {
  std::uint32_t ctx;
  {
    std::lock_guard lk(mu_);
    if (st_.chat_busy || st_.prepare.active)
      return make_error(ErrorCode::kFailedPrecondition, "Another job is already running.");
    ctx = st_.context.context_tokens;
    st_.prepare = PrepareView{};
    st_.prepare.active = true;  // set before the call: events may arrive before it returns
    st_.prepare.tier_id = tier_id;
    st_.prepare.message = "Starting";
    for (const auto& r : st_.tiers)
      if (r.id == tier_id) st_.prepare.model = r.model;
    set_banner(Banner::Kind::kNone, {});
    recompute_send_state();
  }
  auto s = client_.prepare_tier(tier_id, ctx);
  std::lock_guard lk(mu_);
  if (!s.is_ok()) {
    st_.prepare = PrepareView{};
    set_banner(Banner::Kind::kError, describe_error(s.code(), s.message()));
  }
  refresh_pending_ = true;
  recompute_send_state();
  return s;
}

Status FatherViewModel::send(const std::string& text_in) {
  const std::string text = trim(text_in);
  if (text.empty()) return make_error(ErrorCode::kInvalidArgument, "Type a message first.");
  father::ChatRequest req;
  std::size_t entry_index;
  {
    std::lock_guard lk(mu_);
    if (!st_.can_send) return make_error(ErrorCode::kFailedPrecondition, st_.send_hint);
    req.user_message = text;
    if (!st_.settings.system_prompt.empty()) req.system_prompt = st_.settings.system_prompt;
    req.max_new_tokens = st_.settings.max_new_tokens;
    req.context_tokens = st_.context.context_tokens;
    req.q = 1;
    ChatEntry u;
    u.kind = EntryKind::kUser;
    u.text = text;
    entry_index = st_.entries.size();
    st_.entries.push_back(std::move(u));
    st_.chat_busy = true;  // before the call: events may arrive before it returns
    current_request_ = 0;
    if (finished_.size() > 64) finished_.clear();
    set_banner(Banner::Kind::kNone, {});
    recompute_send_state();
  }
  auto r = client_.send_chat(std::move(req));
  std::lock_guard lk(mu_);
  if (!r.is_ok()) {
    st_.chat_busy = false;
    if (st_.entries.size() == entry_index + 1) st_.entries.pop_back();
    set_banner(Banner::Kind::kError, "Your message was not sent: " + describe_error(r.status().code(), r.status().message()));
    recompute_send_state();
    return r.status();
  }
  current_request_ = r.value();
  if (finished_.count(r.value())) st_.chat_busy = false;
  recompute_send_state();
  return Status::ok();
}

Status FatherViewModel::cancel() {
  father::RequestId id;
  {
    std::lock_guard lk(mu_);
    if (!st_.chat_busy) return make_error(ErrorCode::kFailedPrecondition, "There is nothing to cancel.");
    id = current_request_;
    if (id == 0) return make_error(ErrorCode::kFailedPrecondition, "The message is still being sent; try again in a moment.");
  }
  auto s = client_.cancel(id);
  if (!s.is_ok()) {
    std::lock_guard lk(mu_);
    set_banner(Banner::Kind::kError, "Could not cancel: " + describe_error(s.code(), s.message()));
  }
  return s;
}

Status FatherViewModel::release() {
  auto s = client_.release();
  std::lock_guard lk(mu_);
  if (!s.is_ok()) {
    set_banner(Banner::Kind::kError, "Could not release: " + describe_error(s.code(), s.message()));
    return s;
  }
  st_.prepare = PrepareView{};
  set_banner(Banner::Kind::kInfo, "Released. Helper machines remove their temporary files.");
  refresh_pending_ = true;
  recompute_send_state();
  return s;
}

Status FatherViewModel::new_conversation() {
  auto s = client_.reset_conversation();
  std::lock_guard lk(mu_);
  if (!s.is_ok()) {
    set_banner(Banner::Kind::kError, "Could not start a new conversation: " + describe_error(s.code(), s.message()));
    return s;
  }
  st_.entries.clear();
  st_.last_answer = StatsView{};
  generated_tokens_ = 0;
  finished_.clear();
  context_pending_ = true;
  set_banner(Banner::Kind::kNone, {});
  return s;
}

Status FatherViewModel::apply_settings(const FatherSettings& s) {
  auto r = client_.set_settings(s);
  std::lock_guard lk(mu_);
  if (!r.is_ok()) {
    set_banner(Banner::Kind::kError, "Settings not saved: " + describe_error(r.code(), r.message()));
    return r;
  }
  st_.settings = s;
  st_.context.context_tokens = s.context_tokens;
  refresh_pending_ = true;
  context_pending_ = true;
  set_banner(Banner::Kind::kNone, {});
  return r;
}

void FatherViewModel::set_diagnostics_open(bool open) {
  {
    std::lock_guard lk(mu_);
    st_.diagnostics.open = open;
  }
  if (open) refresh_diagnostics();
}

void FatherViewModel::set_include_conversation(bool include) {
  std::lock_guard lk(mu_);
  st_.diagnostics.include_conversation = include;
}

void FatherViewModel::refresh_diagnostics() {
  // The on-screen preview is always the redacted snapshot; only an explicit export may include text.
  auto d = client_.export_diagnostics(false);
  std::lock_guard lk(mu_);
  auto& v = st_.diagnostics;
  v.rows.clear();
  const auto& l = st_.last_answer;
  if (l.valid) {
    v.rows.push_back({"Time to first token", l.ttft, l.provenance});
    v.rows.push_back({"Decode speed", l.tok_s, l.provenance});
    v.rows.push_back({"Accepted per round", l.accepted_per_round, l.provenance});
    v.rows.push_back({"Tokens / rounds", std::to_string(l.tokens) + " / " + std::to_string(l.rounds), l.provenance});
    v.rows.push_back({"Fallbacks during answer", std::to_string(l.fallbacks), l.provenance});
    v.rows.push_back({"Finished", l.reason, l.provenance});
  }
  if (!d.is_ok()) {
    v.snapshot_text = "Diagnostics are not available: " + ascii_display(d.status().message());
    v.plan_summary.clear();
    v.stage_timings.clear();
    return;
  }
  v.snapshot_text = ascii_display(d->text);
  v.plan_summary = ascii_display(d->plan_summary);
  v.stage_timings = d->stage_timings;
}

Status FatherViewModel::export_diagnostics() {
  bool include;
  DiagnosticsExporter exporter;
  {
    std::lock_guard lk(mu_);
    include = st_.diagnostics.include_conversation;
    exporter = exporter_;
  }
  auto d = client_.export_diagnostics(include);
  std::lock_guard lk(mu_);
  auto& v = st_.diagnostics;
  if (!d.is_ok()) {
    v.export_message = "Export failed: " + describe_error(d.status().code(), d.status().message());
    return d.status();
  }
  std::string text = d->text;
  if (d->includes_conversation) text = "WARNING: this file contains your conversation text. Do not share it.\n" + text;
  if (!exporter) {
    v.export_message = "Saving is not available here.";
    return make_error(ErrorCode::kFailedPrecondition, v.export_message);
  }
  auto path = exporter(text);
  if (!path.is_ok()) {
    v.export_message = "Export failed: " + ascii_display(path.status().message());
    return path.status();
  }
  v.last_export_path = path.value();
  v.export_message = d->includes_conversation ? "Saved (includes your conversation text): " + path.value()
                                              : "Saved (no prompts or answers included): " + path.value();
  return Status::ok();
}

Status FatherViewModel::start_pairing(const PairingRequest& request) {
  auto s = client_.start_pairing(request);
  Result<std::vector<PairedMachine>> list = s.is_ok() ? client_.paired_machines() : Result<std::vector<PairedMachine>>(s);
  std::lock_guard lk(mu_);
  st_.pairing.message = s.is_ok() ? "Paired. Compare the short code on both screens." : ascii_display(s.message());
  if (s.is_ok() && list.is_ok()) st_.pairing.machines = list.value();
  refresh_pending_ = true;
  return s;
}

Status FatherViewModel::unpair(const std::string& machine_id) {
  auto s = client_.unpair(machine_id);
  std::lock_guard lk(mu_);
  if (s.is_ok()) {
    st_.pairing.message.clear();
    st_.pairing.machines.erase(std::remove_if(st_.pairing.machines.begin(), st_.pairing.machines.end(),
                                              [&](const PairedMachine& m) { return m.machine_id == machine_id; }),
                               st_.pairing.machines.end());
  } else {
    st_.pairing.message = ascii_display(s.message());
  }
  return s;
}

void FatherViewModel::dismiss_banner() {
  std::lock_guard lk(mu_);
  set_banner(Banner::Kind::kNone, {});
}

}  // namespace clusterlm::ui
