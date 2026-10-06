#include "clusterlm/father/service.hpp"

#include <algorithm>
#include <condition_variable>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"

namespace clusterlm::father {

std::string_view to_string(FinishReason r) noexcept {
  switch (r) {
    case FinishReason::kCompleted: return "completed";
    case FinishReason::kCancelled: return "cancelled";
    case FinishReason::kFailed: return "failed";
  }
  return "failed";
}

std::string DiagnosticsSnapshot::to_string() const {
  std::ostringstream o;
  o << "selected=" << selected_tier << " active=" << active_tier << (active_ready ? " (ready)" : "")
    << " model=\"" << active_model << "\" job=" << (job_running ? "running" : "idle") << "\n";
  o << "requests: started=" << requests_started << " completed=" << requests_completed
    << " cancelled=" << requests_cancelled << " failed=" << requests_failed << " fallbacks=" << fallbacks
    << " tokens=" << tokens_emitted << "\n";
  o << "conversation: " << conversation_messages << " message(s)"
    << (conversation.empty() ? " [text redacted]" : "") << "\n";
  for (const auto& t : tiers) o << "tier " << t.tier_id << ": " << t.state << " - " << t.headline << "\n";
  if (!last_error.empty()) o << "last_error: " << last_error << "\n";
  return o.str();
}

namespace {

using catalog::TierEntry;
using catalog::TierState;

bool is_viable(TierState s) { return s == TierState::kReady || s == TierState::kAvailable; }

class ServiceImpl final : public FatherService {
 public:
  explicit ServiceImpl(ServiceDeps d) : deps_(std::move(d)) {}

  ~ServiceImpl() override {
    cancel_.store(true);
    join_job();
    (void)teardown_active();
  }

  // ---- queries -------------------------------------------------------------------------------------------

  std::vector<catalog::TierReadiness> list_tiers(std::uint32_t ctx) override {
    if (ctx == 0) ctx = deps_.options.default_context_tokens;
    std::vector<std::pair<std::string, catalog::ReadinessInputs>> all;
    for (const auto& t : deps_.catalog.tiers()) all.emplace_back(t.id, observe(t, ctx));
    std::vector<catalog::TierReadiness> out;
    for (const auto& t : deps_.catalog.tiers()) out.push_back(catalog::evaluate_with_fallback(deps_.catalog, t, all));
    return out;
  }

  std::string selected_tier() const override {
    std::lock_guard lk(m_);
    return selected_;
  }

  std::vector<ChatMessage> conversation() const override {
    std::lock_guard lk(m_);
    return conversation_;
  }

  DiagnosticsSnapshot diagnostics(bool include_text) const override {
    DiagnosticsSnapshot d;
    {
      std::lock_guard lk(m_);
      d.selected_tier = selected_;
      d.active_tier = active_tier_;
      d.active_ready = active_ready_;
      d.job_running = job_running_;
      d.requests_started = started_;
      d.requests_completed = completed_;
      d.requests_cancelled = cancelled_count_;
      d.requests_failed = failed_;
      d.fallbacks = fallbacks_;
      d.tokens_emitted = tokens_emitted_;
      d.conversation_messages = static_cast<std::uint32_t>(conversation_.size());
      d.last_error = last_error_;
      if (include_text) d.conversation = conversation_;
      if (const auto* t = deps_.catalog.find(active_tier_)) d.active_model = t->model.display_name;
    }
    // list_tiers is a const-safe observation; the source is thread-safe by contract.
    for (const auto& r : const_cast<ServiceImpl*>(this)->list_tiers(0))
      d.tiers.push_back({r.tier_id, r.model_name, std::string(catalog::to_string(r.state)), r.headline()});
    return d;
  }

  // ---- control -------------------------------------------------------------------------------------------

  Status select_tier(std::string_view tier_id) override {
    const auto* t = deps_.catalog.find(tier_id);
    if (!t) return make_error(ErrorCode::kNotFound, "unknown tier \"" + std::string(tier_id) + "\"");
    {
      std::lock_guard lk(m_);
      selected_ = t->id;
    }
    emit(TierSelectedEvent{t->id, t->model.display_name});
    return Status::ok();
  }

  Status prepare_tier(std::string_view tier_id, std::uint32_t ctx) override {
    const auto* t = deps_.catalog.find(tier_id);
    if (!t) return make_error(ErrorCode::kNotFound, "unknown tier \"" + std::string(tier_id) + "\"");
    if (ctx == 0) ctx = deps_.options.default_context_tokens;
    const auto* cp = t->find_context(ctx);
    if (!cp || !cp->offered)
      return make_error(ErrorCode::kInvalidArgument, std::to_string(ctx) + "-token context is not offered for " + t->display_name);
    const std::string id = t->id;
    return launch_job([this] { cancel_.store(false); return Status::ok(); }, [this, id, ctx] {
      Status st = ensure_tier(id, ctx);
      if (!st.is_ok()) {
        note_error(st);
        emit(ErrorEvent{0, st.code(), st.message()});
      }
    });
  }

  Result<RequestId> chat(ChatRequest req) override {
    if (req.user_message.empty()) return make_error(ErrorCode::kInvalidArgument, "empty message");
    if (req.max_new_tokens == 0 || req.q == 0) return make_error(ErrorCode::kInvalidArgument, "max_new_tokens and q must be > 0");
    std::string tier_id;
    {
      std::lock_guard lk(m_);
      tier_id = selected_;
    }
    if (tier_id.empty()) return make_error(ErrorCode::kFailedPrecondition, "no tier selected");
    const auto* tier = deps_.catalog.find(tier_id);
    const auto* cp = tier->find_context(req.context_tokens);
    if (!cp || !cp->offered)
      return make_error(ErrorCode::kInvalidArgument, std::to_string(req.context_tokens) + "-token context is not offered for " + tier->display_name);

    auto prompt = std::make_shared<std::vector<std::int32_t>>();
    const RequestId rid = next_request_.fetch_add(1) + 1;
    auto commit = [&, prompt]() -> Status {
      // Runs under m_ after the busy check: reject before touching the conversation, then append the user turn.
      std::vector<ChatMessage> trial = conversation_;
      if (trial.empty() && req.system_prompt) trial.push_back({ChatRole::kSystem, *req.system_prompt});
      trial.push_back({ChatRole::kUser, req.user_message});
      *prompt = deps_.tokenizer->encode_chat(trial);
      if (prompt->size() + req.max_new_tokens > req.context_tokens)
        return make_error(ErrorCode::kOutOfRange, "conversation plus requested tokens exceed the " + std::to_string(req.context_tokens) + "-token context");
      conversation_ = std::move(trial);
      ++started_;
      current_request_ = rid;
      cancel_.store(false);
      return Status::ok();
    };
    Status st = launch_job(commit, [this, rid, req, tier_id, prompt] { run_chat(rid, req, *prompt, tier_id); });
    if (!st.is_ok()) return st;
    return rid;
  }

  Status cancel(RequestId request) override {
    std::lock_guard lk(m_);
    if (!job_running_ || current_request_ != request || request == 0)
      return make_error(ErrorCode::kNotFound, "no such running request");
    cancel_.store(true);
    return Status::ok();
  }

  Status release() override {
    cancel_.store(true);
    join_job();
    Status st = teardown_active();
    cancel_.store(false);
    emit(ReleasedEvent{st.is_ok() ? "Released all leases; Nodes hold no model data." : "Release finished with problems: " + st.message()});
    return st;
  }

  Status reset_conversation() override {
    std::lock_guard lk(m_);
    if (job_running_) return make_error(ErrorCode::kFailedPrecondition, "a request is running");
    conversation_.clear();
    return Status::ok();
  }

  SubscriptionId subscribe(EventSink sink) override {
    std::lock_guard lk(m_);
    const auto id = ++next_sub_;
    sinks_[id] = std::move(sink);
    return id;
  }
  void unsubscribe(SubscriptionId id) override {
    std::lock_guard lk(m_);
    sinks_.erase(id);
  }

  bool wait_idle(std::chrono::milliseconds timeout) override {
    std::unique_lock lk(m_);
    return cv_.wait_for(lk, timeout, [&] { return !job_running_; });
  }

 private:
  struct Active {
    std::string tier_id;
    std::uint32_t max_context = 0;
    std::unique_ptr<coordinator::Coordinator> coord;
    Deployment dep;
    // The distributed session holding this tier's conversation state between turns. Reset on any failure or
    // tier change; a history that is not an extension of what it holds also starts a new one.
    std::shared_ptr<coordinator::Conversation> conversation;
  };

  // ---- plumbing ------------------------------------------------------------------------------------------

  void emit(const Event& e) {
    std::vector<EventSink> sinks;
    {
      std::lock_guard lk(m_);
      for (auto& [id, s] : sinks_) sinks.push_back(s);
    }
    std::lock_guard g(emit_m_);
    for (auto& s : sinks) s(e);
  }

  void note_error(const Status& st) {
    std::lock_guard lk(m_);
    last_error_ = st.to_string();
  }

  void join_job() {
    std::thread t;
    {
      std::unique_lock lk(m_);
      cv_.wait(lk, [&] { return !job_running_; });
      t = std::move(job_);
    }
    if (t.joinable()) t.join();
  }

  // Reserves the single job slot. `commit` runs under the lock (it may refuse); `body` runs on the job thread.
  Status launch_job(const std::function<Status()>& commit, std::function<void()> body) {
    std::thread old;
    {
      std::lock_guard lk(m_);
      if (job_running_) return make_error(ErrorCode::kFailedPrecondition, "another request or preparation is already running");
      CLM_RETURN_IF_ERROR(commit());
      job_running_ = true;
      old = std::move(job_);
    }
    if (old.joinable()) old.join();
    std::lock_guard lk(m_);
    job_ = std::thread([this, body = std::move(body)] {
      body();
      {
        std::lock_guard l(m_);
        job_running_ = false;
        current_request_ = 0;
      }
      cv_.notify_all();
    });
    return Status::ok();
  }

  catalog::ReadinessInputs observe(const TierEntry& tier, std::uint32_t ctx) {
    auto in = deps_.readiness->observe(tier, deps_.assignment, ctx);
    in.context_tokens = ctx;
    bool ours = false;
    {
      std::lock_guard lk(m_);
      ours = active_ready_ && active_tier_ == tier.id && active_ctx_ >= ctx;
    }
    // Only the service knows whether its own plan is prepared and current; Father's domains are Ready iff it is.
    in.plan.plan_ready = ours;
    if (ours)
      for (auto& m : in.machines)
        if (m.role == catalog::kRoleFather) m.state = catalog::MachineState::kReady;
    return in;
  }

  catalog::TierReadiness readiness_of(const TierEntry& tier, std::uint32_t ctx, catalog::ReadinessInputs* out = nullptr) {
    auto in = observe(tier, ctx);
    auto r = catalog::evaluate(tier, in);
    if (out) *out = std::move(in);
    return r;
  }

  Status teardown_active() {
    if (!active_.coord) return Status::ok();
    if (active_.conversation) (void)active_.coord->close_conversation(*active_.conversation);
    active_.conversation.reset();
    {
      std::lock_guard lk(m_);
      active_ready_ = false;  // not ready from this moment
    }
    std::string problems;
    auto rel = active_.coord->release();
    if (!rel.is_ok()) {
      problems = rel.status().to_string();
    } else {
      for (const auto& n : rel->nodes)
        if (!n.storage_cleaned || !n.errors.empty()) problems += n.node + ": " + (n.errors.empty() ? "storage not cleaned" : n.errors) + "; ";
    }
    log::info("father.tier_released", {{"tier", active_.tier_id}, {"clean", problems.empty() ? "true" : "false"}});
    active_ = Active{};
    {
      std::lock_guard lk(m_);
      active_tier_.clear();
      active_ctx_ = 0;
    }
    if (!problems.empty()) return make_error(ErrorCode::kInternal, problems);
    return Status::ok();
  }

  // Makes `tier_id` the prepared tier for `ctx`. Fails (without substituting anything) if it cannot.
  Status ensure_tier(const std::string& tier_id, std::uint32_t ctx) {
    const auto* tier = deps_.catalog.find(tier_id);
    if (!tier) return make_error(ErrorCode::kNotFound, "unknown tier");
    {
      std::lock_guard lk(m_);
      if (active_ready_ && active_tier_ == tier_id && active_ctx_ >= ctx) return Status::ok();
    }
    (void)teardown_active();  // one prepared tier at a time; a failed one is cleaned here

    CLM_RETURN_IF_ERROR(deps_.assignment.validate_for(*tier));
    catalog::ReadinessInputs in;
    const auto r = readiness_of(*tier, ctx, &in);
    if (r.state == TierState::kUnavailable) {
      std::string why;
      for (const auto& s : r.reasons) why += (why.empty() ? "" : "; ") + s;
      return make_error(ErrorCode::kFailedPrecondition, tier->display_name + " is unavailable: " + why);
    }

    auto dep = deps_.deployments->resolve(*tier, ctx);
    if (!dep.is_ok()) return dep.status();
    auto coord = coordinator::Coordinator::create(dep->config);
    if (!coord.is_ok()) return coord.status();

    // Model identity: pinned hash must match; an unpinned model must be exactly the manifest the user confirmed.
    const std::string root = coord.value()->manifest().root_hash().hex();
    if (tier->model.expected_root_hash) {
      if (root != *tier->model.expected_root_hash)
        return make_error(ErrorCode::kDataLoss, "manifest root does not match the pinned catalog hash for " + tier->display_name);
    } else if (in.model.user_confirmed_root_hex != root) {
      return make_error(ErrorCode::kPermissionDenied, "the " + tier->display_name + " manifest has not been confirmed by the user");
    }

    emit(PrepareProgressEvent{tier->id, tier->model.display_name, std::nullopt, std::nullopt,
                              "Preparing " + tier->display_name + " (" + tier->model.display_name + ")"});

    std::mutex pm;
    std::condition_variable pcv;
    bool stop_poll = false;
    std::thread poller([&] {
      double last = -1;
      std::unique_lock lk(pm);
      while (!pcv.wait_for(lk, deps_.options.progress_poll, [&] { return stop_poll; })) {
        lk.unlock();
        const auto cur = readiness_of(*tier, ctx);
        if (cur.state == TierState::kPreparing && cur.progress && cur.progress->percent != last) {
          last = cur.progress->percent;
          emit(PrepareProgressEvent{tier->id, tier->model.display_name, cur.progress->percent, cur.progress->eta_seconds, cur.headline()});
        }
        lk.lock();
      }
    });
    auto stop = [&] {
      { std::lock_guard lk(pm); stop_poll = true; }
      pcv.notify_all();
      poller.join();
    };

    Status st = coord.value()->connect();
    if (st.is_ok()) {
      auto rep = coord.value()->prepare(dep->plan);
      if (!rep.is_ok()) st = rep.status();
    }
    stop();
    if (!st.is_ok()) {
      (void)coord.value()->release();
      return st;
    }
    active_.tier_id = tier->id;
    active_.max_context = dep->plan.max_context;
    active_.dep = std::move(dep).value();
    active_.coord = std::move(coord).value();
    {
      std::lock_guard lk(m_);
      active_tier_ = tier->id;
      active_ctx_ = ctx;
      active_ready_ = true;
    }
    log::info("father.tier_ready", {{"tier", tier->id}});
    emit(TierReadyEvent{tier->id, tier->model.display_name});
    return Status::ok();
  }

  // ---- chat ----------------------------------------------------------------------------------------------

  struct Run {
    RequestId id = 0;
    Stopwatch since_start;
    bool got_first = false;
    double ttft_ms = 0;
    double gen_ms = 0;
    std::uint64_t accepted_sum = 0, decode_rounds = 0;
    std::vector<std::int32_t> out;
    std::vector<AnswerSegment> segments;
    std::uint32_t fallbacks = 0;
  };

  struct LoopResult {
    Status status;
    bool cancelled = false;
    bool fatal = false;  // the request itself is invalid for this plan; no tier change can help
  };

  LoopResult generate_loop(Run& run, const TierEntry& tier, const std::vector<std::int32_t>& prompt, const ChatRequest& req) {
    LoopResult lr;
    if (prompt.size() + req.max_new_tokens > active_.max_context) {
      lr.fatal = true;
      lr.status = make_error(ErrorCode::kOutOfRange, "the request does not fit the " + tier.display_name + " plan's " + std::to_string(active_.max_context) + "-token window");
      return lr;
    }
    std::shared_ptr<domain::Drafter> drafter;
    if (req.q > 1) {
      if (!active_.dep.make_drafter) {
        lr.fatal = true;
        lr.status = make_error(ErrorCode::kInvalidArgument, "q > 1 requires a drafter from the deployment");
        return lr;
      }
      drafter = active_.dep.make_drafter();
    }
    // Full token history for this answer: the rendered conversation plus what this answer has emitted so far
    // (non-empty after a fallback: the new tier continues the same answer).
    std::vector<std::int32_t> history = prompt;
    history.insert(history.end(), run.out.begin(), run.out.end());
    auto& conv = active_.conversation;
    auto held = [&] {
      std::vector<std::int32_t> h = conv->committed_tokens();
      h.insert(h.end(), conv->pending_tokens().begin(), conv->pending_tokens().end());
      return h;
    };
    if (conv && conv->valid()) {
      const auto h = held();
      if (h.size() > history.size() || !std::equal(h.begin(), h.end(), history.begin())) {
        (void)active_.coord->close_conversation(*conv);
        conv.reset();
      }
    }
    if (!conv || !conv->valid()) {
      auto opened = active_.coord->open_conversation();
      if (!opened.is_ok()) {
        lr.status = opened.status();
        return lr;
      }
      conv = std::move(opened).value();
    }
    const auto already = held().size();
    coordinator::GenerationRequest g;
    g.conversation = conv;
    g.prompt.assign(history.begin() + static_cast<std::ptrdiff_t>(already), history.end());
    g.max_new_tokens = static_cast<std::uint32_t>(req.max_new_tokens - run.out.size());
    g.q = req.q;
    g.drafter = drafter;
    g.cancel = run_cancel_;
    g.on_tokens = [&](std::span<const std::int32_t> toks) {
      std::vector<std::int32_t> v(toks.begin(), toks.end());
      run.out.insert(run.out.end(), v.begin(), v.end());
      if (!run.segments.empty() && run.segments.back().tier_id == tier.id)
        run.segments.back().tokens += static_cast<std::uint32_t>(v.size());
      else
        run.segments.push_back({tier.id, tier.model.display_name, static_cast<std::uint32_t>(v.size())});
      if (!run.got_first) {
        run.got_first = true;
        run.ttft_ms = run.since_start.elapsed_ms();
      }
      {
        std::lock_guard lk(m_);
        tokens_emitted_ += v.size();
      }
      TokensEvent ev;
      ev.request = run.id;
      ev.tier_id = tier.id;
      ev.model_name = tier.model.display_name;
      ev.text = deps_.tokenizer->decode(v);
      ev.tokens = std::move(v);
      emit(ev);
    };
    Stopwatch sw;
    auto res = active_.coord->generate(g);
    run.gen_ms += sw.elapsed_ms();
    if (!res.is_ok()) {
      {
        std::lock_guard lk(m_);
        active_ready_ = false;  // the distributed session is invalid; the prepared plan is no longer current
      }
      conv.reset();
      lr.status = res.status();
      if (res.status().code() == ErrorCode::kInvalidArgument) lr.fatal = true;
      return lr;
    }
    for (const auto& r : res->rounds)
      if (!r.prefill) {
        ++run.decode_rounds;
        run.accepted_sum += r.accepted;
      }
    if (res->cancelled) {
      lr.cancelled = true;
      return lr;
    }
    if (res->tokens.empty()) lr.status = make_error(ErrorCode::kInternal, "backend returned no tokens");
    return lr;
  }

  void run_chat(RequestId rid, const ChatRequest& req, const std::vector<std::int32_t>& prompt, std::string tier_id) {
    Run run;
    run.id = rid;
    std::uint32_t retries_used = 0;
    FinishReason reason = FinishReason::kCompleted;
    Status final_error;

    while (true) {
      const auto* tier = deps_.catalog.find(tier_id);
      Status st = ensure_tier(tier_id, req.context_tokens);
      LoopResult lr;
      if (st.is_ok()) {
        lr = generate_loop(run, *tier, prompt, req);
        st = lr.status;
        if (lr.cancelled) { reason = FinishReason::kCancelled; break; }
        if (st.is_ok()) break;  // finished
        if (lr.fatal) { final_error = st; reason = FinishReason::kFailed; break; }
      }
      if (cancel_.load()) { reason = FinishReason::kCancelled; break; }
      note_error(st);
      log::warn("father.session_lost", {{"tier", tier_id}, {"code", std::string(to_string(st.code()))}});

      // The session (or preparation) of `tier_id` failed. Explicit catalog policy decides what happens next.
      const auto cur = readiness_of(*tier, req.context_tokens);
      const std::string why = cur.state == TierState::kUnavailable
                                  ? cur.headline()
                                  : "The " + tier->display_name + " session was interrupted (" + st.message() + ")";
      if (retries_used < tier->on_session_loss.max_retries && cur.state != TierState::kUnavailable) {
        ++retries_used;
        ++run.fallbacks;
        FallbackEvent ev;
        ev.request = rid;
        ev.kind = FallbackEvent::Kind::kRetry;
        ev.from_tier = ev.to_tier = tier->id;
        ev.from_model = ev.to_model = tier->model.display_name;
        ev.message = why + " \xE2\x80\x94 retrying " + tier->display_name + " (" + tier->model.display_name + "); your conversation is kept.";
        bump_fallbacks();
        emit(ev);
        continue;
      }
      const TierEntry* next = nullptr;
      if (tier->on_session_loss.then == catalog::LossAction::kDowngrade) {
        for (const auto* lower : deps_.catalog.fallbacks_after(tier->id)) {
          const auto* cp = lower->find_context(req.context_tokens);
          if (!cp || !cp->offered) continue;
          if (is_viable(readiness_of(*lower, req.context_tokens).state)) { next = lower; break; }
        }
      }
      if (!next) {
        final_error = make_error(ErrorCode::kUnavailable, why + "; no lower tier can continue this answer");
        reason = FinishReason::kFailed;
        break;
      }
      ++run.fallbacks;
      FallbackEvent ev;
      ev.request = rid;
      ev.kind = FallbackEvent::Kind::kDowngrade;
      ev.from_tier = tier->id;
      ev.to_tier = next->id;
      ev.from_model = tier->model.display_name;
      ev.to_model = next->model.display_name;
      ev.message = why + " \xE2\x80\x94 switching from " + tier->display_name + " (" + tier->model.display_name + ") to " +
                   next->display_name + " (" + next->model.display_name + "); your conversation is kept" +
                   (run.out.empty() ? "." : " and the answer continues from where it stopped.");
      bump_fallbacks();
      emit(ev);
      tier_id = next->id;
      retries_used = 0;
    }

    if (reason == FinishReason::kFailed) {
      note_error(final_error);
      emit(ErrorEvent{rid, final_error.code(), final_error.message()});
    }
    // Keep the (possibly partial) answer in the Father-held conversation.
    if (!run.out.empty()) {
      auto text = deps_.tokenizer->decode(run.out);
      std::lock_guard lk(m_);
      conversation_.push_back({ChatRole::kAssistant, std::move(text)});
    }
    FinishedEvent fin;
    fin.request = rid;
    fin.reason = reason;
    auto& s = fin.stats;
    s.ttft_ms = run.ttft_ms;
    s.tokens = static_cast<std::uint32_t>(run.out.size());
    s.rounds = static_cast<std::uint32_t>(run.decode_rounds);
    s.decode_tok_s = run.gen_ms > 0 ? 1000.0 * static_cast<double>(run.out.size()) / run.gen_ms : 0;
    s.accepted_per_round = run.decode_rounds ? static_cast<double>(run.accepted_sum) / static_cast<double>(run.decode_rounds) : 0;
    s.fallbacks = run.fallbacks;
    s.answered_by = run.segments;
    if (!run.segments.empty()) {
      s.final_tier = run.segments.back().tier_id;
      s.final_model = run.segments.back().model_name;
    }
    {
      std::lock_guard lk(m_);
      if (reason == FinishReason::kCompleted) ++completed_;
      else if (reason == FinishReason::kCancelled) ++cancelled_count_;
      else ++failed_;
    }
    log::info("father.request_finished", {{"request", std::to_string(rid)}, {"reason", std::string(to_string(reason))},
                                          {"tokens", std::to_string(run.out.size())}});
    emit(fin);
  }

  void bump_fallbacks() {
    std::lock_guard lk(m_);
    ++fallbacks_;
  }

  ServiceDeps deps_;
  mutable std::mutex m_;
  std::mutex emit_m_;
  std::condition_variable cv_;
  std::thread job_;
  bool job_running_ = false;
  // Shared with the Coordinator's GenerationRequest so cancel() reaches the in-flight window directly.
  std::shared_ptr<std::atomic<bool>> run_cancel_ = std::make_shared<std::atomic<bool>>(false);
  std::atomic<bool>& cancel_ = *run_cancel_;
  std::atomic<RequestId> next_request_{0};
  RequestId current_request_ = 0;
  std::string selected_;
  std::vector<ChatMessage> conversation_;
  std::map<SubscriptionId, EventSink> sinks_;
  SubscriptionId next_sub_ = 0;
  Active active_;  // job thread / release only
  std::string active_tier_;
  std::uint32_t active_ctx_ = 0;
  bool active_ready_ = false;
  std::uint64_t started_ = 0, completed_ = 0, cancelled_count_ = 0, failed_ = 0, fallbacks_ = 0, tokens_emitted_ = 0;
  std::string last_error_;
};

}  // namespace

Result<std::unique_ptr<FatherService>> make_father_service(ServiceDeps deps) {
  if (!deps.tokenizer || !deps.readiness || !deps.deployments)
    return make_error(ErrorCode::kInvalidArgument, "tokenizer, readiness source and deployment provider are required");
  return std::unique_ptr<FatherService>(new ServiceImpl(std::move(deps)));
}

}  // namespace clusterlm::father
