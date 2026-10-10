#include "clusterlm/scheduler/scheduler.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <tuple>

namespace clusterlm::scheduler {

std::string_view to_string(InvalidateReason r) noexcept {
  switch (r) {
    case InvalidateReason::kAbortNotConfirmed: return "abort_not_confirmed";
    case InvalidateReason::kWorkerLost: return "worker_lost";
    case InvalidateReason::kLocalUserReturned: return "local_user_returned";
    case InvalidateReason::kPolicy: return "policy";
    case InvalidateReason::kSessionInvalidated: return "session_invalidated";
    case InvalidateReason::kShutdown: return "shutdown";
  }
  return "unknown";
}

namespace {

constexpr std::size_t kDoneHistory = 512;
constexpr auto kPoll = std::chrono::milliseconds(20);

std::uint64_t mix(std::uint64_t x) {  // splitmix64 finalizer
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

struct HintKey { std::uint64_t k0, k1; };
HintKey process_hint_key() {
  static const HintKey k = [] {
    std::random_device rd;
    return HintKey{(std::uint64_t{rd()} << 32) | rd(), (std::uint64_t{rd()} << 32) | rd()};
  }();
  return k;
}

// Keyed 128-bit digest of the conversation hint. Not a cryptographic MAC: its job is to avoid retaining the raw hint
// (which may identify a person) and to compare equal hints; isolation between clients rests on the client id, not on it.
void digest_hint(const std::string& hint, std::uint64_t& lo, std::uint64_t& hi) {
  lo = hi = 0;
  if (hint.empty()) return;
  const HintKey k = process_hint_key();
  std::uint64_t a = k.k0, b = k.k1;
  for (unsigned char c : hint) {
    a = mix(a ^ c);
    b = mix(b + a + c);
  }
  lo = a | 1ULL;  // never both zero: zero means "no hint"
  hi = b;
}

}  // namespace

// ============================================================================================================
struct Scheduler::Impl final : BackendEvents {
  // ---- records -------------------------------------------------------------------------------------------
  enum class StartStep : std::uint8_t { kNone, kDroppingRetained, kStartIssued };

  struct Ticket {
    TicketId id = 0;
    std::string request_id, client_id;
    Priority prio = Priority::kApi;
    std::string profile_id, api_model_id, routed_via, model_identity, backend_id, fingerprint;
    std::uint32_t revision = 0;
    bool alias = false, alias_allow_prepare = false, may_prepare = false;
    std::uint32_t prepare_wait_ms = 0;
    bool interactive = false;
    std::uint32_t context_class = 0, prompt_tokens = 0, completion_budget = 0;
    TimePoint enqueued;
    std::optional<TimePoint> queue_deadline, prepare_deadline, cleanup_deadline;
    TicketState state = TicketState::kQueued;
    StartStep step = StartStep::kNone;
    EndCause cause = EndCause::kNone;
    bool terminal_sent = false, started_notified = false;
    PlanId plan = 0;
    std::size_t slot = 0;
    std::optional<JobSeq> waiting_job;
    RetentionKey key;
    bool retain = false;
    std::shared_ptr<GenerationInput> input;
    std::shared_ptr<GatedSink> sink;
    std::shared_ptr<TicketObserver> observer;
    std::optional<TimePoint> slot_since;
    std::uint32_t completion_tokens = 0;
  };

  struct Slot {
    enum class St : std::uint8_t { kFree, kReserved, kRetained, kDropping, kQuarantined } st = St::kFree;
    TicketId ticket = 0;
    RetentionKey key;
    RetainedSessionRef ref;
    std::uint32_t retained_tokens = 0;
    TicketId for_ticket = 0;
  };

  struct Plan {
    enum class St : std::uint8_t { kReady, kReleasing, kInvalidating } state = St::kReady;
    PlanId id = 0;
    std::string profile_id, fingerprint;
    std::uint32_t revision = 0, context = 0;
    PlanInfo info;
    std::vector<Slot> slots;
    std::optional<TimePoint> idle_deadline, invalidate_deadline;
    bool stuck_reported = false;
    std::uint64_t session_seq = 0;
    TimePoint last_used;
  };

  struct Job {
    JobSeq seq = 0;
    std::string id;
    JobKind kind = JobKind::kPrepare;
    JobState state = JobState::kQueued;
    std::string profile_id, fingerprint;
    std::uint32_t revision = 0, context = 0;
    ProfileSnapshot profile;
    PlanId plan = 0;
    std::optional<JobSeq> after;
    Code fail_code = Code::kModelNotReady;   // what waiters of a failed job are told
    bool cancel_requested = false;
    PrepareProgressView progress;
    std::string last_error;
    TimePoint created, finished;
  };

  enum class Phase : std::uint8_t { kRunning, kDraining, kReleasing, kDone };

  // ---- members -------------------------------------------------------------------------------------------
  std::shared_ptr<ExecutionBackend> backend;
  SchedulerOptions opt;
  std::shared_ptr<ClockSource> clock;
  mutable std::mutex mu;
  std::condition_variable cv_work, cv_delivery, cv_done;
  std::deque<std::function<void()>> actions, deliveries;
  bool stop_threads = false;
  std::thread dispatcher, deliverer;
  std::thread::id dispatcher_id, deliverer_id;
  std::size_t inflight_actions = 0, inflight_deliveries = 0;

  std::shared_ptr<const WorldSnapshot> world;
  std::map<std::string, std::string> model_index;     // api_model_id -> "p:<id>" | "a:<id>"
  std::set<std::string> revoked;

  TimePoint last_now;
  std::uint64_t next_ticket = 1, next_plan = 1, next_job = 1;
  std::map<TicketId, Ticket> tickets;
  std::deque<TicketId> done_order;
  std::map<std::string, TicketId> by_request;
  FairQueue queue;
  std::map<PlanId, Plan> plans;
  std::map<JobSeq, Job> jobs_;
  std::map<std::string, std::string> last_error_by_profile;
  struct Interval { TimePoint start; std::optional<TimePoint> end; };
  std::map<std::string, std::deque<Interval>> usage;
  std::map<std::string, std::deque<TimePoint>> recent_requests;
  SchedulerCounters ctr;
  Phase phase = Phase::kRunning;
  TimePoint grace_deadline, release_deadline;
  bool grace_expired = false, release_extended = false, detach_issued = false;
  TimePoint hard_deadline;
  ShutdownReport report;
  std::uint64_t req_salt = 0;

  Impl(std::shared_ptr<ExecutionBackend> b, SchedulerOptions o) : backend(std::move(b)), opt(std::move(o)) {
    clock = opt.clock ? opt.clock : std::make_shared<SteadyClockSource>();
    opt.max_queue_depth = std::min<std::uint32_t>(opt.max_queue_depth, 256);
    opt.max_queue_per_client = std::min<std::uint32_t>(opt.max_queue_per_client, 64);
    opt.queue_timeout = std::min<Duration>(opt.queue_timeout, std::chrono::seconds(600));
    last_now = clock->now();
    std::random_device rd;
    req_salt = (std::uint64_t{rd()} << 32) | rd();
    world = std::make_shared<WorldSnapshot>();
    backend->attach(this);
  }

  // ---- time ----------------------------------------------------------------------------------------------
  TimePoint now() {  // never goes backwards
    last_now = std::max(last_now, clock->now());
    return last_now;
  }

  // ---- outgoing work -------------------------------------------------------------------------------------
  void act(std::function<void()> f) {
    actions.push_back(std::move(f));
    cv_work.notify_all();
  }
  void deliver(std::function<void()> f) {
    deliveries.push_back(std::move(f));
    cv_delivery.notify_all();
  }

  // ---- helpers -------------------------------------------------------------------------------------------
  const ProfileSnapshot* profile_of(const std::string& id) const {
    auto it = world->profiles.find(id);
    return it == world->profiles.end() ? nullptr : &it->second;
  }

  Plan* plan_for(const std::string& fingerprint, std::uint32_t ctx) {
    Plan* best = nullptr;
    for (auto& [id, p] : plans)
      if (p.state == Plan::St::kReady && p.fingerprint == fingerprint && p.context >= ctx)
        if (!best || p.context < best->context) best = &p;
    return best;
  }

  static bool slot_active(const Slot& s) {
    return s.st == Slot::St::kReserved || s.st == Slot::St::kDropping || s.st == Slot::St::kQuarantined;
  }
  static std::uint32_t active_slots(const Plan& p) {
    std::uint32_t n = 0;
    for (const Slot& s : p.slots) n += slot_active(s) ? 1U : 0U;
    return n;
  }

  ReadinessView base_readiness(const std::string& profile_id, std::uint32_t ctx) const {
    auto it = world->readiness.find(WorldSnapshot::key(profile_id, ctx));
    if (it != world->readiness.end()) return it->second;
    ReadinessView v;
    v.state = ReadyState::kUnavailable;
    v.reasons = {"no readiness information"};
    v.blockers = {"no_readiness"};
    return v;
  }

  // `ready` is never inferred from the evaluator: it requires the scheduler's own prepared plan for this exact
  // fingerprint and context, confirmed by the backend. Blockers from the evaluator always win.
  ReadinessView effective(const ProfileSnapshot& p, std::uint32_t ctx, bool for_dispatch = false) const {
    ReadinessView v = base_readiness(p.id, ctx);
    if (v.state == ReadyState::kUnavailable || v.state == ReadyState::kInstalled) return v;
    // A plan that is going away is a transient reason to show (compatible), but a queued ticket waits for it (the next
    // prepare is ordered behind the release) instead of being rejected.
    for (const auto& [pid, pl] : plans) {
      if (!for_dispatch && pl.profile_id == p.id && pl.state != Plan::St::kReady) {
        ReadinessView r;
        r.state = ReadyState::kCompatible;
        r.reasons = {pl.state == Plan::St::kInvalidating ? "the previous session has not stopped yet"
                                                         : "the previous plan is being released"};
        return r;
      }
    }
    for (const auto& [pid, pl] : plans) {
      if (pl.profile_id == p.id && pl.state == Plan::St::kReady && pl.fingerprint == p.plan_fingerprint &&
          pl.context >= ctx) {
        ReadinessView r;
        r.state = active_slots(pl) >= pl.slots.size() ? ReadyState::kBusy : ReadyState::kReady;
        return r;
      }
    }
    for (const auto& [seq, j] : jobs_) {
      if (j.kind == JobKind::kPrepare && (j.state == JobState::kQueued || j.state == JobState::kRunning) &&
          j.fingerprint == p.plan_fingerprint && j.context >= ctx) {
        ReadinessView r;
        r.state = ReadyState::kPreparing;
        r.job_id = j.id;
        r.percent = j.progress.percent;
        r.eta_seconds = j.progress.eta_seconds;
        return r;
      }
    }
    if (v.state == ReadyState::kCompatible) return v;
    v.state = ReadyState::kLoadable;   // the evaluator may say ready/busy/preparing; only our own records can
    v.job_id.reset();
    v.percent.reset();
    v.eta_seconds.reset();
    if (auto e = last_error_by_profile.find(p.id); e != last_error_by_profile.end())
      v.reasons.push_back("last preparation failed: " + e->second);
    return v;
  }

  ResourceView resources_of(const std::string& profile_id) const {
    auto it = world->resources.find(profile_id);
    return it == world->resources.end() ? ResourceView{} : it->second;
  }

  std::string new_request_id() {
    char buf[40];
    std::snprintf(buf, sizeof buf, "req_%016llx", static_cast<unsigned long long>(mix(req_salt ^ next_ticket)));
    return buf;
  }

  // ---- usage / fairness ----------------------------------------------------------------------------------
  bool demoted(const std::string& client, TimePoint t) {
    auto it = usage.find(client);
    if (it == usage.end()) return false;
    const TimePoint horizon = t - std::chrono::minutes(10);
    auto& dq = it->second;
    while (!dq.empty() && dq.front().end && *dq.front().end < horizon) dq.pop_front();
    Duration sum{};
    for (const auto& iv : dq) {
      const TimePoint s = std::max(iv.start, horizon);
      const TimePoint e = std::min(iv.end ? *iv.end : t, t);
      if (e > s) sum += e - s;
    }
    return sum > opt.max_slot_seconds;
  }
  FairQueue::DemotedFn demoted_fn(TimePoint t) {
    return [this, t](const std::string& c) { return demoted(c, t); };
  }
  void usage_begin(Ticket& t, TimePoint at) {
    t.slot_since = at;
    usage[t.client_id].push_back({at, std::nullopt});
  }
  void usage_end(Ticket& t, TimePoint at) {
    if (!t.slot_since) return;
    auto& dq = usage[t.client_id];
    for (auto it = dq.rbegin(); it != dq.rend(); ++it)
      if (!it->end && it->start == *t.slot_since) { it->end = at; break; }
    t.slot_since.reset();
  }

  // ---- terminals -----------------------------------------------------------------------------------------
  StartedInfo info_of(const Ticket& t) const {
    return {t.request_id, t.profile_id, t.api_model_id, t.routed_via, t.model_identity, t.backend_id};
  }

  Terminal terminal_for(const Ticket& t, EndCause cause, const std::string& detail = {}) {
    Terminal term;
    term.request_id = t.request_id;
    term.cause = cause;
    term.info = info_of(t);
    term.kind = Terminal::Kind::kError;
    switch (cause) {
      case EndCause::kClientGone:
      case EndCause::kClientCancel: term.kind = Terminal::Kind::kSilent; break;
      case EndCause::kQueueTimeout:
        term.code = Code::kQueueTimeout; term.message = "the request waited too long in the queue"; break;
      case EndCause::kPrepareWaitExpired:
        term.code = Code::kModelNotReady; term.message = "the model is still being prepared";
        term.retry_after_s = 5; break;
      case EndCause::kPrepareFailed:
        term.code = Code::kModelNotReady;
        term.message = detail.empty() ? "the model could not be prepared" : detail; break;
      case EndCause::kKeyRevoked: term.code = Code::kInvalidApiKey; term.message = "invalid API key"; break;
      case EndCause::kShutdown: term.code = Code::kShuttingDown; term.message = "the server is shutting down"; break;
      case EndCause::kManagement: term.code = Code::kRequestCancelled; term.message = "the request was cancelled"; break;
      case EndCause::kWorkerLost:
        term.code = Code::kWorkerLost; term.message = "a Worker was lost while the request was running"; break;
      case EndCause::kBackendError: term.code = Code::kInternalError; term.message = "internal error"; break;
      case EndCause::kRejectedAtDequeue:
      case EndCause::kCompleted:
      case EndCause::kNone: term.code = Code::kInternalError; term.message = "internal error"; break;
    }
    if (t.waiting_job) {
      auto j = jobs_.find(*t.waiting_job);
      if (j != jobs_.end()) term.job_id = j->second.id;
    }
    return term;
  }

  void send_terminal(Ticket& t, Terminal term) {
    if (t.terminal_sent) return;
    t.terminal_sent = true;
    auto sink = t.sink;
    auto obs = t.observer;
    deliver([sink, obs, term = std::move(term)] {
      if (sink) sink->close();      // no token may follow the final response
      if (obs) obs->on_terminal(term);
    });
  }

  void finish(Ticket& t, TimePoint at) {  // -> done
    t.state = TicketState::kDone;
    t.cleanup_deadline.reset();
    t.queue_deadline.reset();
    t.prepare_deadline.reset();
    t.waiting_job.reset();
    usage_end(t, at);
    t.input.reset();
    done_order.push_back(t.id);
    while (done_order.size() > kDoneHistory) {
      auto old = tickets.find(done_order.front());
      if (old != tickets.end()) { by_request.erase(old->second.request_id); tickets.erase(old); }
      done_order.pop_front();
    }
  }

  // ---- slots / plans -------------------------------------------------------------------------------------
  void slot_released(Plan& p, Slot& s) {   // the session state of this slot is confirmed gone
    s = Slot{};
    if (p.state == Plan::St::kInvalidating) s.st = Slot::St::kQuarantined;
  }

  void invalidate_plan(Plan& p, InvalidateReason reason) {
    if (p.state == Plan::St::kInvalidating) return;
    const TimePoint at = now();
    p.state = Plan::St::kInvalidating;
    p.invalidate_deadline = at + opt.invalidate_timeout;
    p.idle_deadline.reset();
    ++ctr.plan_invalidations;
    for (Slot& s : p.slots) {
      const TicketId owner = s.ticket;
      s.st = Slot::St::kQuarantined;
      s.ticket = owner;
    }
    EndCause c = EndCause::kWorkerLost;
    if (reason == InvalidateReason::kShutdown) c = EndCause::kShutdown;
    if (reason == InvalidateReason::kAbortNotConfirmed) c = EndCause::kBackendError;
    for (auto& [id, t] : tickets) {
      if (t.plan != p.id) continue;
      if (t.state == TicketState::kStarting && t.step == StartStep::kStartIssued) {
        if (!t.terminal_sent) send_terminal(t, terminal_for(t, c));
        t.state = TicketState::kCancelling;
        if (t.cause == EndCause::kNone) t.cause = c;
        t.cleanup_deadline.reset();
      } else if (t.state == TicketState::kRunning) {
        if (!t.terminal_sent) send_terminal(t, terminal_for(t, c));
        t.state = TicketState::kCancelling;
        if (t.cause == EndCause::kNone) t.cause = c;
        t.cleanup_deadline.reset();
      }
    }
    auto backend_ptr = backend;
    const PlanId pid = p.id;
    act([backend_ptr, pid, reason] { backend_ptr->invalidate_plan(pid, reason); });
  }

  void cancel_running_prepare(const std::string& why) {
    for (auto& [seq, j] : jobs_) {
      if (j.kind == JobKind::kPrepare && j.state == JobState::kRunning && !j.cancel_requested) {
        j.cancel_requested = true;
        j.last_error = why;
        auto b = backend;
        const JobSeq s = seq;
        act([b, s] { b->cancel_prepare(s); });
      }
    }
  }

  // ---- jobs ----------------------------------------------------------------------------------------------
  Job* find_prepare(const std::string& fingerprint, std::uint32_t ctx) {
    Job* best = nullptr;
    for (auto& [seq, j] : jobs_) {
      if (j.kind != JobKind::kPrepare || j.fingerprint != fingerprint) continue;
      if (j.state != JobState::kQueued && j.state != JobState::kRunning) continue;
      if (j.cancel_requested) continue;
      if (j.context >= ctx && (!best || j.seq < best->seq)) best = &j;
    }
    return best;
  }

  std::string ensure_prepare(const ProfileSnapshot& p, std::uint32_t ctx) {
    if (Job* j = find_prepare(p.plan_fingerprint, ctx)) return j->id;
    // A queued smaller job for the same plan is raised to the larger context (D7).
    for (auto& [seq, j] : jobs_) {
      if (j.kind == JobKind::kPrepare && j.state == JobState::kQueued && j.fingerprint == p.plan_fingerprint &&
          !j.cancel_requested && j.context < ctx) {
        j.context = ctx;
        return j.id;
      }
    }
    Job j;
    j.seq = next_job++;
    j.id = "job_" + std::to_string(j.seq);
    j.kind = JobKind::kPrepare;
    j.profile_id = p.id;
    j.fingerprint = p.plan_fingerprint;
    j.revision = p.revision;
    j.context = ctx;
    j.profile = p;
    j.plan = next_plan++;
    j.created = now();
    jobs_[j.seq] = j;
    return j.id;
  }

  Job* ensure_release(Plan& pl) {
    for (auto& [seq, j] : jobs_)
      if (j.kind == JobKind::kRelease && j.plan == pl.id &&
          (j.state == JobState::kQueued || j.state == JobState::kRunning))
        return &j;
    Job j;
    j.seq = next_job++;
    j.id = "job_" + std::to_string(j.seq);
    j.kind = JobKind::kRelease;
    j.profile_id = pl.profile_id;
    j.fingerprint = pl.fingerprint;
    j.context = pl.context;
    j.plan = pl.id;
    j.created = now();
    return &(jobs_[j.seq] = j);
  }

  bool any_job_running() const {
    for (const auto& [s, j] : jobs_)
      if (j.state == JobState::kRunning) return true;
    return false;
  }

  void end_job(Job& j, JobState st, const std::string& err) {
    j.state = st;
    j.finished = now();
    if (!err.empty()) j.last_error = err;
    if (j.kind == JobKind::kPrepare) {
      if (st == JobState::kSucceeded) last_error_by_profile.erase(j.profile_id);
      else if (!j.last_error.empty()) last_error_by_profile[j.profile_id] = j.last_error;
      for (auto& [id, t] : tickets) {
        if (t.waiting_job != j.seq) continue;
        t.waiting_job.reset();
        t.prepare_deadline.reset();
        if (st != JobState::kSucceeded && t.state == TicketState::kQueued) {
          Terminal term = terminal_for(t, EndCause::kPrepareFailed,
                                       st == JobState::kCancelled && j.last_error.empty() ? "preparation was cancelled"
                                                                                           : j.last_error);
          term.job_id = j.id;
          if (st == JobState::kFailed) term.code = j.fail_code;
          cancel_queued(t, EndCause::kPrepareFailed, std::move(term));
        }
      }
    }
    gc_jobs();
  }

  void gc_jobs() {
    const TimePoint at = now();
    std::vector<JobSeq> finished;
    for (auto& [s, j] : jobs_) {
      if (j.state == JobState::kQueued || j.state == JobState::kRunning) continue;
      if (at - j.finished > opt.job_history_age) { finished.push_back(s); continue; }
    }
    for (JobSeq s : finished) jobs_.erase(s);
    std::vector<JobSeq> done;
    for (auto& [s, j] : jobs_)
      if (j.state != JobState::kQueued && j.state != JobState::kRunning) done.push_back(s);
    while (done.size() > opt.max_job_history) { jobs_.erase(done.front()); done.erase(done.begin()); }
  }

  Plan* lru_ready_plan() {
    Plan* best = nullptr;
    for (auto& [id, p] : plans)
      if (p.state == Plan::St::kReady && (!best || p.last_used < best->last_used)) best = &p;
    return best;
  }

  void run_jobs() {
    // A queued prepare whose profile is no longer loadable (a Worker went away, a blocker appeared) fails now, with the
    // evaluator's reason, instead of waiting for its turn: its waiters are told at once.
    for (auto& [seq, j] : jobs_) {
      if (j.kind != JobKind::kPrepare || j.state != JobState::kQueued) continue;
      const ReadinessView b = base_readiness(j.profile_id, j.context);
      if (b.state == ReadyState::kUnavailable || b.state == ReadyState::kInstalled || b.state == ReadyState::kCompatible) {
        j.fail_code = b.state == ReadyState::kCompatible ? Code::kWorkersUnavailable : Code::kModelNotReady;
        end_job(j, JobState::kFailed, b.reasons.empty() ? "the model is not available" : b.reasons.front());
      }
    }
    if (any_job_running()) return;
    for (auto& [seq, j] : jobs_) {
      if (j.state != JobState::kQueued) continue;
      if (j.after) {
        auto a = jobs_.find(*j.after);
        if (a != jobs_.end() && (a->second.state == JobState::kQueued || a->second.state == JobState::kRunning)) continue;
      }
      if (phase != Phase::kRunning && j.kind == JobKind::kPrepare) {
        end_job(j, JobState::kCancelled, "the server is shutting down");
        continue;
      }
      if (j.kind == JobKind::kRelease) {
        auto it = plans.find(j.plan);
        if (it == plans.end()) { end_job(j, JobState::kSucceeded, {}); continue; }
        Plan& pl = it->second;
        if (pl.state != Plan::St::kReady) {   // already releasing/invalidating: the job completes at plan_gone
          j.state = JobState::kRunning;
          return;
        }
        for (const Slot& s : pl.slots)
          if (slot_active(s)) goto next_job;   // swap never kills an active request (contract §4)
        j.state = JobState::kRunning;
        pl.state = Plan::St::kReleasing;
        pl.idle_deadline.reset();
        {
          auto b = backend;
          const PlanId pid = pl.id;
          act([b, pid] { b->release(pid); });
        }
        return;
      }
      // prepare
      {
        bool blocked = false;
        for (auto& [pid, pl] : plans)
          if (pl.state != Plan::St::kReady) blocked = true;   // allocations of two plans never overlap
        if (blocked) continue;
        // One plan per profile, and at most max_prepared_profiles: otherwise queue a swap release (lazily).
        Plan* victim = nullptr;
        for (auto& [pid, pl] : plans)
          if (pl.profile_id == j.profile_id) victim = &pl;
        if (!victim && plans.size() >= opt.max_prepared_profiles) victim = lru_ready_plan();
        if (victim) {
          Job* r = ensure_release(*victim);
          j.after = r->seq;
          ++ctr.swaps;
          continue;
        }
        if (j.after) {
          auto a = jobs_.find(*j.after);
          if (a != jobs_.end() && a->second.state != JobState::kSucceeded) {
            end_job(j, JobState::kFailed, "the previous model could not be released");
            continue;
          }
        }
        j.state = JobState::kRunning;
        ++ctr.prepares_started;
        PrepareCommand cmd{j.seq, j.plan, j.profile, j.context};
        auto b = backend;
        act([b, cmd] { b->prepare(cmd); });
        return;
      }
    next_job:;
    }
  }

  // ---- ticket transitions ----------------------------------------------------------------------------------
  void cancel_queued(Ticket& t, EndCause cause, Terminal term) {
    queue.erase(t.id);
    t.cause = cause;
    send_terminal(t, std::move(term));
    finish(t, now());
    if (cause != EndCause::kClientGone && cause != EndCause::kClientCancel) ++ctr.failed;
    else ++ctr.cancelled;
  }

  void cancel_locked(Ticket& t, EndCause cause) {
    const TimePoint at = now();
    switch (t.state) {
      case TicketState::kQueued:
        if (cause == EndCause::kQueueTimeout) ++ctr.timed_out;
        cancel_queued(t, cause, terminal_for(t, cause));
        return;
      case TicketState::kStarting:
      case TicketState::kRunning: {
        const bool issued = t.state == TicketState::kRunning || t.step == StartStep::kStartIssued;
        t.state = TicketState::kCancelling;
        t.cause = cause;
        send_terminal(t, terminal_for(t, cause));
        if (cause == EndCause::kClientGone || cause == EndCause::kClientCancel) ++ctr.cancelled;
        if (issued) {
          t.cleanup_deadline = at + opt.cancel_timeout;
          auto b = backend;
          const TicketId id = t.id;
          act([b, id] { b->abort_request(id); });
        } else {
          t.cleanup_deadline = at + opt.cancel_timeout;   // the drop in flight is bounded too
        }
        return;
      }
      case TicketState::kFinishing:
      case TicketState::kCancelling:
      case TicketState::kDone: return;
    }
    return;
  }

  // Starts `t` on slot `si` of plan `p` (T4/T6).
  void issue_start(Plan& p, std::size_t si, Ticket& t, bool reuse) {
    Slot& s = p.slots[si];
    StartCommand cmd;
    cmd.ticket = t.id;
    cmd.plan = p.id;
    cmd.input = t.input;
    cmd.sink = t.sink;
    cmd.key = t.key;
    cmd.retain_on_finish = t.retain;
    cmd.completion_budget = t.completion_budget;
    cmd.session_seq = ++p.session_seq;
    if (reuse) cmd.reuse = s.ref;
    s.st = Slot::St::kReserved;
    s.ticket = t.id;
    s.for_ticket = 0;
    t.state = TicketState::kStarting;
    t.step = StartStep::kStartIssued;
    t.plan = p.id;
    t.slot = si;
    t.cleanup_deadline = now() + opt.start_timeout;
    auto b = backend;
    act([b, cmd] { b->start_request(cmd); });
  }

  std::optional<Code> recheck(const Ticket& t) {
    auto cit = world->clients.find(t.client_id);
    if (revoked.count(t.client_id) || (cit != world->clients.end() && cit->second.revoked)) return Code::kInvalidApiKey;
    const ProfileSnapshot* p = profile_of(t.profile_id);
    if (!p || !p->exposed_api) return Code::kModelNotFound;
    if (cit != world->clients.end()) {
      const auto& c = cit->second;
      const std::string& asked = t.routed_via.empty() ? t.api_model_id : t.routed_via;
      const bool ok = std::any_of(c.allowed_models.begin(), c.allowed_models.end(),
                                  [&](const std::string& m) { return m == "*" || m == asked; });
      if (!ok || !c.scope_inference) return Code::kModelNotFound;
    }
    if (p->plan_fingerprint != t.fingerprint) return Code::kModelNotReady;   // "the model behind this name was changed"
    ReadinessView r = effective(*p, t.context_class, true);
    if (t.alias) {
      const bool viable = is_prepared(r.state) || (t.alias_allow_prepare && (r.state == ReadyState::kLoadable || r.state == ReadyState::kPreparing));
      if (!viable) return readiness_reject_code(r, resources_of(p->id));
    }
    if (!is_loadable(r.state)) return readiness_reject_code(r, resources_of(p->id));
    if (!resources_of(p->id).workers_allowed) return Code::kWorkersUnavailable;
    return std::nullopt;
  }

  void reject_at_dequeue(Ticket& t, Code code) {
    Terminal term;
    term.request_id = t.request_id;
    term.cause = EndCause::kRejectedAtDequeue;
    term.code = code;
    term.info = info_of(t);
    const ProfileSnapshot* p = profile_of(t.profile_id);
    if (code == Code::kModelNotReady && p && p->plan_fingerprint != t.fingerprint)
      term.message = "the model behind this name was changed";
    else if (code == Code::kModelNotFound) term.message = "the model does not exist or is not available to this key";
    else if (code == Code::kInvalidApiKey) term.message = "invalid API key";
    else term.message = code == Code::kWorkersUnavailable ? "a required Worker is not available" : "the model is not ready";
    if (code == Code::kWorkersUnavailable || code == Code::kModelNotReady) term.reasons = {term.message};
    FairQueue::Entry e{t.id, t.client_id, t.prio};
    queue.served(t.client_id, t.prio);
    cancel_queued(t, EndCause::kRejectedAtDequeue, std::move(term));
  }

  void attach_prepare(Ticket& t, Job& j) {
    t.waiting_job = j.seq;
    if (!t.prepare_deadline && !t.interactive) {
      Duration w = std::chrono::milliseconds(t.prepare_wait_ms);
      w = std::min<Duration>(w, opt.prepare_wait_cap);
      t.prepare_deadline = now() + w;
    }
  }

  void dispatch() {
    const TimePoint at = now();
    while (true) {
      auto head = queue.head(demoted_fn(at));
      if (!head) break;
      Ticket& t = tickets.at(*head);
      if (phase != Phase::kRunning) {
        queue.served(t.client_id, t.prio);
        cancel_queued(t, EndCause::kShutdown, terminal_for(t, EndCause::kShutdown));
        continue;
      }
      if (auto code = recheck(t)) { reject_at_dequeue(t, *code); continue; }
      Plan* p = plan_for(t.fingerprint, t.context_class);
      if (p) {
        t.waiting_job.reset();
        t.prepare_deadline.reset();
        int free_i = -1, same_i = -1, other_i = -1;
        for (std::size_t i = 0; i < p->slots.size(); ++i) {
          const Slot& s = p->slots[i];
          if (s.st == Slot::St::kFree && free_i < 0) free_i = static_cast<int>(i);
          if (s.st == Slot::St::kRetained) {
            if (s.key == t.key && same_i < 0) same_i = static_cast<int>(i);
            else if (!(s.key == t.key) && other_i < 0) other_i = static_cast<int>(i);
          }
        }
        p->idle_deadline.reset();
        p->last_used = at;
        if (same_i >= 0 || free_i >= 0) {
          const bool reuse = same_i >= 0;
          const std::size_t si = static_cast<std::size_t>(reuse ? same_i : free_i);
          queue.served(t.client_id, t.prio);
          queue.erase(t.id);
          usage_begin(t, at);
          issue_start(*p, si, t, reuse);
          continue;
        }
        if (other_i >= 0) {   // T5: abort the other client's retained state before this session opens
          Slot& s = p->slots[static_cast<std::size_t>(other_i)];
          s.st = Slot::St::kDropping;
          s.for_ticket = t.id;
          t.state = TicketState::kStarting;
          t.step = StartStep::kDroppingRetained;
          t.plan = p->id;
          t.slot = static_cast<std::size_t>(other_i);
          t.cleanup_deadline = at + opt.cancel_timeout;
          queue.served(t.client_id, t.prio);
          queue.erase(t.id);
          usage_begin(t, at);
          auto b = backend;
          const PlanId pid = p->id;
          const RetainedSessionRef ref = s.ref;
          act([b, pid, ref] { b->drop_session(pid, ref); });
          break;
        }
        break;   // every slot busy: strict head-of-line
      }
      // Not prepared: join or start a prepare job, per lifecycle and wait budget.
      const ProfileSnapshot* prof = profile_of(t.profile_id);
      if (t.waiting_job) {   // the job this ticket waits on is still alive (even if being cancelled): wait for its outcome
        auto wj = jobs_.find(*t.waiting_job);
        if (wj != jobs_.end() && (wj->second.state == JobState::kQueued || wj->second.state == JobState::kRunning)) break;
        t.waiting_job.reset();
      }
      if (Job* j = find_prepare(t.fingerprint, t.context_class)) {
        if (!t.waiting_job) attach_prepare(t, *j);
        break;
      }
      if (!t.may_prepare) {
        queue.served(t.client_id, t.prio);
        Terminal term = terminal_for(t, EndCause::kPrepareFailed, "the model is not prepared and preparation is manual");
        cancel_queued(t, EndCause::kPrepareFailed, std::move(term));
        continue;
      }
      std::string jid = ensure_prepare(*prof, t.context_class);
      if (t.prepare_wait_ms == 0 && !t.interactive) {   // fail fast, but the job is started (it benefits the next request)
        queue.served(t.client_id, t.prio);
        Terminal term = terminal_for(t, EndCause::kPrepareFailed, "the model is being prepared");
        term.job_id = jid;
        term.retry_after_s = 5;
        cancel_queued(t, EndCause::kPrepareFailed, std::move(term));
        continue;
      }
      if (Job* j = find_prepare(t.fingerprint, t.context_class)) attach_prepare(t, *j);
      break;
    }
  }

  // ---- idle retention / timers -------------------------------------------------------------------------
  void idle_bookkeeping() {
    const TimePoint at = now();
    for (auto& [id, p] : plans) {
      if (p.state != Plan::St::kReady) continue;
      bool idle = true;
      for (const Slot& s : p.slots) idle = idle && !slot_active(s);
      bool wanted = false;
      for (auto& [tid, t] : tickets)
        if (t.state == TicketState::kQueued && t.fingerprint == p.fingerprint) wanted = true;
      if (!idle || wanted) { p.idle_deadline.reset(); continue; }
      if (!p.idle_deadline) {
        const ProfileSnapshot* prof = profile_of(p.profile_id);
        if (!prof) p.idle_deadline = at;
        else if (prof->release_after_idle_seconds > 0)
          p.idle_deadline = at + std::chrono::seconds(prof->release_after_idle_seconds);
      }
    }
  }

  struct Due { TimePoint when; int kind; std::uint64_t id; };

  void timers() {
    const TimePoint at = now();
    std::vector<Due> due;
    for (auto& [id, t] : tickets) {
      if (t.state == TicketState::kQueued) {
        if (t.waiting_job) {
          if (t.prepare_deadline && *t.prepare_deadline <= at) due.push_back({*t.prepare_deadline, 1, id});
        } else if (t.queue_deadline && *t.queue_deadline <= at) {
          due.push_back({*t.queue_deadline, 2, id});
        }
      }
      if (t.cleanup_deadline && *t.cleanup_deadline <= at) due.push_back({*t.cleanup_deadline, 0, id});
    }
    for (auto& [id, p] : plans) {
      if (p.idle_deadline && *p.idle_deadline <= at && p.state == Plan::St::kReady) due.push_back({*p.idle_deadline, 3, id});
      if (p.invalidate_deadline && *p.invalidate_deadline <= at && !p.stuck_reported) due.push_back({*p.invalidate_deadline, 4, id});
    }
    std::sort(due.begin(), due.end(), [](const Due& a, const Due& b) {
      return std::tie(a.when, a.kind, a.id) < std::tie(b.when, b.kind, b.id);
    });
    for (const Due& d : due) {
      switch (d.kind) {
        case 0: {  // cleanup/start timeout: a timer only escalates, it never frees a slot
          auto it = tickets.find(d.id);
          if (it == tickets.end() || !it->second.cleanup_deadline) break;
          Ticket& t = it->second;
          t.cleanup_deadline.reset();
          auto pit = plans.find(t.plan);
          if (pit == plans.end()) break;
          if (t.state == TicketState::kStarting && t.step == StartStep::kStartIssued && !t.terminal_sent) {
            t.state = TicketState::kCancelling;
            t.cause = EndCause::kBackendError;
            send_terminal(t, terminal_for(t, EndCause::kBackendError));
          }
          invalidate_plan(pit->second, InvalidateReason::kAbortNotConfirmed);
          break;
        }
        case 1:
        case 2: {
          auto it = tickets.find(d.id);
          if (it == tickets.end() || it->second.state != TicketState::kQueued) break;
          if (d.kind == 1) {
            Terminal term = terminal_for(it->second, EndCause::kPrepareWaitExpired);
            cancel_queued(it->second, EndCause::kPrepareWaitExpired, std::move(term));
          } else {
            ++ctr.timed_out;
            cancel_queued(it->second, EndCause::kQueueTimeout, terminal_for(it->second, EndCause::kQueueTimeout));
          }
          break;
        }
        case 3: {
          auto it = plans.find(d.id);
          if (it == plans.end() || it->second.state != Plan::St::kReady) break;
          it->second.idle_deadline.reset();
          ensure_release(it->second);
          break;
        }
        case 4: {
          auto it = plans.find(d.id);
          if (it == plans.end()) break;
          it->second.stuck_reported = true;
          ++ctr.stuck_invalidations;
          break;
        }
        default: break;
      }
    }
    gc_jobs();
  }

  void shutdown_progress() {
    if (phase == Phase::kRunning || phase == Phase::kDone) return;
    const TimePoint at = now();
    if (at >= hard_deadline) {   // the documented upper bound: report what is stuck and finish
      report.clean = false;
      report.problems.push_back("shutdown reached its time bound with work still outstanding");
      for (auto& [id, p] : plans) invalidate_plan(p, InvalidateReason::kShutdown);
      phase = Phase::kDone;
      if (!detach_issued) {
        detach_issued = true;
        auto b = backend;
        act([b] { b->detach(); });
      }
      cv_done.notify_all();
      return;
    }
    bool active = false;
    for (auto& [id, t] : tickets)
      if (t.state == TicketState::kStarting || t.state == TicketState::kRunning || t.state == TicketState::kFinishing ||
          t.state == TicketState::kCancelling)
        active = true;
    if (phase == Phase::kDraining) {
      if (!grace_expired && at >= grace_deadline) {
        grace_expired = true;
        for (auto& [id, t] : tickets)
          if (t.state == TicketState::kStarting || t.state == TicketState::kRunning) cancel_locked(t, EndCause::kShutdown);
      }
      if (!active && !any_job_running()) {
        phase = Phase::kReleasing;
        release_deadline = at + opt.release_timeout;
        for (auto& [id, p] : plans) {
          if (p.state != Plan::St::kReady) continue;
          p.state = Plan::St::kReleasing;
          auto b = backend;
          const PlanId pid = id;
          act([b, pid] { b->release(pid); });
        }
      }
      return;
    }
    // kReleasing
    if (plans.empty() && !active) {
      phase = Phase::kDone;
      if (!detach_issued) {
        detach_issued = true;
        auto b = backend;
        act([b] { b->detach(); });
      }
      cv_done.notify_all();
      return;
    }
    if (at >= release_deadline) {
      for (auto& [id, p] : plans) {
        report.clean = false;
        report.problems.push_back("plan " + std::to_string(id) + " was not released in time");
        invalidate_plan(p, InvalidateReason::kShutdown);
      }
      if (release_extended) {
        phase = Phase::kDone;   // give up: bounded
        if (!detach_issued) {
          detach_issued = true;
          auto b = backend;
          act([b] { b->detach(); });
        }
        cv_done.notify_all();
      } else {
        release_extended = true;
        release_deadline = at + opt.release_timeout;
      }
    }
  }

  void pump() {
    timers();
    run_jobs();
    dispatch();
    run_jobs();
    idle_bookkeeping();
    shutdown_progress();
  }

  // ---- BackendEvents -------------------------------------------------------------------------------------
  void on_prepare_progress(JobSeq seq, const PrepareProgressView& pv) override {
    std::lock_guard<std::mutex> lock(mu);
    auto it = jobs_.find(seq);
    if (it == jobs_.end() || it->second.state != JobState::kRunning) { ++ctr.stale_events; return; }
    it->second.progress = pv;
  }

  void on_prepare_done(JobSeq seq, PlanId pid, Result<PlanInfo> res) override {
    std::lock_guard<std::mutex> lock(mu);
    auto it = jobs_.find(seq);
    if (it == jobs_.end() || it->second.state != JobState::kRunning || it->second.kind != JobKind::kPrepare) {
      ++ctr.stale_events;
      return;
    }
    Job& j = it->second;
    if (!res.is_ok()) {
      end_job(j, j.cancel_requested ? JobState::kCancelled : JobState::kFailed,
              j.cancel_requested && !j.last_error.empty() ? j.last_error : res.status().message());
    } else {
      Plan pl;
      pl.id = pid;
      pl.profile_id = j.profile_id;
      pl.fingerprint = j.fingerprint;
      pl.revision = j.revision;
      pl.context = j.context;
      pl.info = res.value();
      pl.last_used = now();
      const std::uint32_t n = std::max<std::uint32_t>(1, std::min(res.value().concurrency, j.profile.max_sessions));
      pl.slots.assign(n, Slot{});
      plans[pid] = std::move(pl);
      if (j.cancel_requested) {   // a cancelled job that still produced a plan: release it right away
        plans[pid].state = Plan::St::kReleasing;
        auto b = backend;
        act([b, pid] { b->release(pid); });
        end_job(j, JobState::kCancelled, j.last_error);
      } else {
        end_job(j, JobState::kSucceeded, {});
      }
    }
    pump();
  }

  void on_request_running(TicketId id) override {
    std::lock_guard<std::mutex> lock(mu);
    auto it = tickets.find(id);
    if (it == tickets.end() || it->second.state != TicketState::kStarting || it->second.step != StartStep::kStartIssued) {
      ++ctr.stale_events;
      return;
    }
    Ticket& t = it->second;
    t.state = TicketState::kRunning;
    t.cleanup_deadline.reset();
    notify_started(t);
    pump();
  }

  void notify_started(Ticket& t) {
    if (t.started_notified) return;
    t.started_notified = true;
    auto obs = t.observer;
    const StartedInfo info = info_of(t);
    deliver([obs, info] { if (obs) obs->on_started(info); });
  }

  void on_request_output_done(TicketId id, const OutputSummary& s) override {
    std::lock_guard<std::mutex> lock(mu);
    auto it = tickets.find(id);
    if (it == tickets.end() ||
        !(it->second.state == TicketState::kRunning ||
          (it->second.state == TicketState::kStarting && it->second.step == StartStep::kStartIssued))) {
      ++ctr.stale_events;
      return;
    }
    Ticket& t = it->second;
    if (t.state == TicketState::kStarting) { t.state = TicketState::kRunning; notify_started(t); }
    t.state = TicketState::kFinishing;
    t.completion_tokens = s.completion_tokens;
    t.cleanup_deadline = now() + opt.cancel_timeout;
    Terminal term;
    term.kind = Terminal::Kind::kSuccess;
    term.request_id = t.request_id;
    term.summary = s;
    term.cause = EndCause::kCompleted;
    term.info = info_of(t);
    send_terminal(t, std::move(term));
    pump();
  }

  void on_request_ended(TicketId id, RequestEnd end) override {
    std::lock_guard<std::mutex> lock(mu);
    auto it = tickets.find(id);
    if (it == tickets.end()) { ++ctr.stale_events; return; }
    Ticket& t = it->second;
    const TimePoint at = now();
    auto pit = plans.find(t.plan);
    Plan* p = pit == plans.end() ? nullptr : &pit->second;
    const bool owns_slot = p && t.slot < p->slots.size() && p->slots[t.slot].ticket == id &&
                           (p->slots[t.slot].st == Slot::St::kReserved || p->slots[t.slot].st == Slot::St::kQuarantined);
    auto settle_slot = [&](bool retain) {
      if (!p || !owns_slot) return;
      Slot& s = p->slots[t.slot];
      if (p->state == Plan::St::kInvalidating) { s = Slot{}; s.st = Slot::St::kQuarantined; return; }
      if (retain && end.retained) {
        s = Slot{};
        s.st = Slot::St::kRetained;
        s.key = t.key;
        s.ref = *end.retained;
        s.retained_tokens = end.retained_tokens;
      } else {
        s = Slot{};
      }
    };
    switch (t.state) {
      case TicketState::kFinishing:   // T9
        settle_slot(end.disposition == Disposition::kRetained && t.retain);
        ++ctr.completed;
        finish(t, at);
        break;
      case TicketState::kCancelling:  // T12
        settle_slot(end.disposition == Disposition::kRetained);
        finish(t, at);
        break;
      case TicketState::kRunning:
      case TicketState::kStarting: {
        if (t.state == TicketState::kStarting && t.step != StartStep::kStartIssued) { ++ctr.stale_events; return; }
        // T15/T16: ended without a prior output_done or cancel: a failure (or a contract violation) invalidates the plan.
        const bool lost = end.fail == FailKind::kWorkerLost || end.fail == FailKind::kSessionInvalidated;
        const EndCause c = lost ? EndCause::kWorkerLost : EndCause::kBackendError;
        if (lost) ++ctr.worker_lost;
        t.cause = c;
        send_terminal(t, terminal_for(t, c));
        if (p && owns_slot) { p->slots[t.slot].st = Slot::St::kQuarantined; }
        finish(t, at);
        if (p) invalidate_plan(*p, lost ? InvalidateReason::kWorkerLost : InvalidateReason::kSessionInvalidated);
        break;
      }
      default: ++ctr.stale_events; return;
    }
    pump();
  }

  void on_session_dropped(PlanId pid, RetainedSessionRef ref, Status st) override {
    std::lock_guard<std::mutex> lock(mu);
    auto pit = plans.find(pid);
    if (pit == plans.end()) { ++ctr.stale_events; return; }
    Plan& p = pit->second;
    for (std::size_t i = 0; i < p.slots.size(); ++i) {
      Slot& s = p.slots[i];
      if (s.st != Slot::St::kDropping || !(s.ref == ref)) continue;
      const TicketId waiter = s.for_ticket;
      if (!st.is_ok()) { invalidate_plan(p, InvalidateReason::kAbortNotConfirmed); pump(); return; }
      slot_released(p, s);
      auto tit = tickets.find(waiter);
      if (waiter && tit != tickets.end()) {
        Ticket& t = tit->second;
        if (t.state == TicketState::kStarting && t.step == StartStep::kDroppingRetained && p.state == Plan::St::kReady) {
          issue_start(p, i, t, false);   // T6
        } else if (t.state == TicketState::kCancelling) {
          finish(t, now());              // T11
        }
      }
      pump();
      return;
    }
    ++ctr.stale_events;
  }

  void on_plan_gone(PlanId pid, PlanGone gone) override {
    std::lock_guard<std::mutex> lock(mu);
    auto pit = plans.find(pid);
    if (pit == plans.end()) { ++ctr.stale_events; return; }
    const TimePoint at = now();
    const bool was_ready = pit->second.state == Plan::St::kReady;
    if (!gone.clean) report.clean = false;
    if (was_ready) ++ctr.plan_invalidations;
    for (auto& [id, t] : tickets) {
      if (t.plan != pid) continue;
      switch (t.state) {
        case TicketState::kStarting:
          if (t.step == StartStep::kDroppingRetained) {   // T17: nothing of this ticket was ever opened
            t.state = TicketState::kQueued;
            t.step = StartStep::kNone;
            t.plan = 0;
            t.cleanup_deadline.reset();
            usage_end(t, at);
            queue.push({t.id, t.client_id, t.prio});
            break;
          }
          [[fallthrough]];
        case TicketState::kRunning:
          if (!t.terminal_sent) send_terminal(t, terminal_for(t, EndCause::kWorkerLost));
          t.cause = t.cause == EndCause::kNone ? EndCause::kWorkerLost : t.cause;
          finish(t, at);
          break;
        case TicketState::kCancelling:
        case TicketState::kFinishing: finish(t, at); break;
        default: break;
      }
    }
    plans.erase(pit);
    for (auto& [seq, j] : jobs_)
      if (j.kind == JobKind::kRelease && j.plan == pid && (j.state == JobState::kRunning || j.state == JobState::kQueued))
        end_job(j, JobState::kSucceeded, gone.problems);
    pump();
  }

  void on_plan_lost(PlanId pid, InvalidateReason reason) override {
    std::lock_guard<std::mutex> lock(mu);
    auto pit = plans.find(pid);
    if (pit == plans.end()) { ++ctr.stale_events; return; }
    invalidate_plan(pit->second, reason);
    pump();
  }
};

// ============================================================================================================
Scheduler::Scheduler(std::shared_ptr<ExecutionBackend> backend, SchedulerOptions options)
    : impl_(std::make_unique<Impl>(std::move(backend), std::move(options))) {
  Impl& s = *impl_;
  if (s.opt.threading == SchedulerOptions::Threading::kThreaded) {
    s.dispatcher = std::thread([&s] {
      std::unique_lock<std::mutex> lock(s.mu);
      while (true) {
        if (s.actions.empty()) {
          if (s.stop_threads) break;
          s.cv_work.wait_for(lock, kPoll);
          if (s.actions.empty()) { s.pump(); continue; }
        }
        auto f = std::move(s.actions.front());
        s.actions.pop_front();
        ++s.inflight_actions;
        lock.unlock();
        f();
        lock.lock();
        --s.inflight_actions;
        s.cv_done.notify_all();
      }
    });
    s.deliverer = std::thread([&s] {
      std::unique_lock<std::mutex> lock(s.mu);
      while (true) {
        if (s.deliveries.empty()) {
          if (s.stop_threads) break;
          s.cv_delivery.wait_for(lock, kPoll);
          continue;
        }
        auto f = std::move(s.deliveries.front());
        s.deliveries.pop_front();
        ++s.inflight_deliveries;
        lock.unlock();
        f();
        lock.lock();
        --s.inflight_deliveries;
        s.cv_done.notify_all();
      }
    });
    s.dispatcher_id = s.dispatcher.get_id();
    s.deliverer_id = s.deliverer.get_id();
  }
}

Scheduler::~Scheduler() {
  Impl& s = *impl_;
  if (s.opt.threading == SchedulerOptions::Threading::kThreaded) {
    if (s.phase != Impl::Phase::kDone) (void)shutdown();
    {
      std::lock_guard<std::mutex> lock(s.mu);
      s.stop_threads = true;
    }
    s.cv_work.notify_all();
    s.cv_delivery.notify_all();
    if (s.dispatcher.joinable()) s.dispatcher.join();
    if (s.deliverer.joinable()) s.deliverer.join();
  } else if (s.phase != Impl::Phase::kDone) {
    (void)shutdown();
    for (int i = 0; i < 8 && s.phase != Impl::Phase::kDone; ++i) run_pending();
  }
  if (!s.detach_issued) s.backend->detach();
}

std::size_t Scheduler::run_pending() {
  Impl& s = *impl_;
  std::size_t n = 0;
  while (true) {
    std::function<void()> f;
    bool is_action = false;
    {
      std::lock_guard<std::mutex> lock(s.mu);
      s.pump();
      if (!s.actions.empty()) { f = std::move(s.actions.front()); s.actions.pop_front(); is_action = true; }
      else if (!s.deliveries.empty()) { f = std::move(s.deliveries.front()); s.deliveries.pop_front(); }
    }
    if (!f) break;
    (void)is_action;
    f();
    ++n;
  }
  return n;
}

EnqueueResult Scheduler::enqueue(EnqueueRequest req) {
  Impl& s = *impl_;
  EnqueueResult out;
  std::lock_guard<std::mutex> lock(s.mu);
  const TimePoint at = s.now();
  auto rej = [&](Code c, std::string msg) {
    out.decision.kind = Decision::Kind::kReject;
    out.decision.code = c;
    out.decision.message = std::move(msg);
    ++s.ctr.rejected;
    return out;
  };
  if (s.phase != Impl::Phase::kRunning) return rej(Code::kShuttingDown, "the server is shutting down");
  if (req.client.revoked || s.revoked.count(req.client.id)) return rej(Code::kInvalidApiKey, "invalid API key");
  if (!req.client.scope_inference) return rej(Code::kInsufficientScope, "this key may not run inference");

  const WorldSnapshot& w = *s.world;
  auto mi = s.model_index.find(req.api_model_id);
  const auto allowed = [&](const std::string& id) {
    return std::any_of(req.client.allowed_models.begin(), req.client.allowed_models.end(),
                       [&](const std::string& m) { return m == "*" || m == id; });
  };
  if (mi == s.model_index.end() || !allowed(req.api_model_id))
    return rej(Code::kModelNotFound, "the model '" + req.api_model_id + "' does not exist or is not available to this key");

  ResolvedTarget target;
  target.api_model_id = req.api_model_id;
  const ProfileSnapshot* chosen = nullptr;
  const AliasSnapshot* alias = nullptr;
  ReadinessView chosen_ready;
  const auto ready_for = [&](const ProfileSnapshot& p) {
    const std::uint32_t budget = completion_budget(p, req.shape, s.opt.admission);
    const std::uint64_t need = std::uint64_t{req.shape.prompt_tokens} + budget;
    const auto cls = need <= p.context_max_tokens ? context_class_for(p, static_cast<std::uint32_t>(need)) : std::nullopt;
    return s.effective(p, cls ? *cls : p.context_max_tokens);
  };
  if (mi->second[0] == 'p') {
    chosen = &w.profiles.at(mi->second.substr(2));
    target.profile = chosen;
    chosen_ready = ready_for(*chosen);
  } else {
    alias = &w.aliases.at(mi->second.substr(2));
    if (req.client.limits.lan && !alias->exposed_lan)
      return rej(Code::kModelNotFound, "the model '" + req.api_model_id + "' does not exist or is not available to this key");
    target.alias = alias;
    std::vector<CandidateView> cands;
    for (const std::string& pid : alias->candidates) {
      CandidateView c;
      auto pit = w.profiles.find(pid);
      if (pit != w.profiles.end()) {
        c.profile = &pit->second;
        c.visible = pit->second.exposed_api && (!req.client.limits.lan || pit->second.exposed_lan) &&
                    allowed(pit->second.api_model_id);   // an alias never widens access
        c.readiness = ready_for(pit->second);
      }
      cands.push_back(std::move(c));
    }
    std::vector<std::string> why;
    auto idx = select_candidate(*alias, cands, &why);
    if (!idx) {
      out = EnqueueResult{};
      out.decision.kind = Decision::Kind::kReject;
      out.decision.code = Code::kModelNotReady;
      out.decision.message = "no candidate of '" + req.api_model_id + "' can serve the request";
      if (req.client.limits.can_see_machine_names) out.decision.reasons = why;
      else out.decision.reasons = {"the model is not ready"};
      ++s.ctr.rejected;
      return out;
    }
    chosen = cands[*idx].profile;
    chosen_ready = cands[*idx].readiness;
  }

  // Queue view.
  QueueView q;
  q.depth = s.queue.depth();
  q.max_queue_depth = s.opt.max_queue_depth;
  q.max_queue_per_client = s.opt.max_queue_per_client;
  q.shutting_down = false;
  for (auto& [id, t] : s.tickets) {
    if (t.client_id != req.client.id || t.state == TicketState::kDone) continue;
    ++q.client_in_flight;
    if (t.state == TicketState::kQueued) ++q.client_queued;
  }
  auto& recent = s.recent_requests[req.client.id];
  while (!recent.empty() && at - recent.front() > std::chrono::minutes(1)) recent.pop_front();
  q.client_requests_last_minute = recent.size();
  {
    const std::uint32_t budget = completion_budget(*chosen, req.shape, s.opt.admission);
    const std::uint64_t need = std::uint64_t{req.shape.prompt_tokens} + budget;
    if (need <= chosen->context_max_tokens) {
      const auto cls = context_class_for(*chosen, static_cast<std::uint32_t>(need));
      Impl::Plan* p = cls ? s.plan_for(chosen->plan_fingerprint, *cls) : nullptr;
      if (p)
        for (const auto& sl : p->slots)
          if (sl.st == Impl::Slot::St::kFree || sl.st == Impl::Slot::St::kRetained) q.slot_free = true;
    }
  }
  Decision d = admit(req.client, target, *chosen, req.shape, chosen_ready, s.resources_of(chosen->id), q, s.opt.admission,
                     alias && alias->allow_prepare);
  if (d.kind == Decision::Kind::kReject) {
    if (d.start_prepare) d.job_id = s.ensure_prepare(*chosen, d.context_class);
    if (d.job_id) d.retry_after_s = 5;
    out.decision = std::move(d);
    ++s.ctr.rejected;
    s.pump();
    return out;
  }

  Impl::Ticket t;
  t.id = s.next_ticket++;
  t.request_id = s.new_request_id();
  t.client_id = req.client.id;
  t.prio = s.opt.priorities_enabled ? req.client.priority : Priority::kApi;
  t.interactive = req.client.priority == Priority::kInteractive;
  t.profile_id = chosen->id;
  t.api_model_id = chosen->api_model_id;
  t.routed_via = alias ? alias->api_model_id : std::string();
  t.model_identity = chosen->model_identity;
  t.backend_id = chosen->backend_id;
  t.fingerprint = chosen->plan_fingerprint;
  t.revision = chosen->revision;
  t.alias = alias != nullptr;
  t.alias_allow_prepare = alias && alias->allow_prepare;
  t.may_prepare = chosen->preparation != Preparation::kManual || t.alias_allow_prepare;
  t.prepare_wait_ms = req.client.limits.prepare_wait_ms;
  t.context_class = d.context_class;
  t.prompt_tokens = req.shape.prompt_tokens;
  t.completion_budget = d.completion_budget;
  t.enqueued = at;
  t.queue_deadline = at + s.opt.queue_timeout;
  digest_hint(req.conversation_hint, t.key.hint_lo, t.key.hint_hi);
  t.key.client_id = req.client.id;
  t.retain = !t.key.empty();
  t.input = std::move(req.input);
  t.sink = std::make_shared<GatedSink>(std::move(req.sink));
  t.observer = std::move(req.observer);
  if (d.start_prepare) s.ensure_prepare(*chosen, d.context_class);
  out.accepted = true;
  out.ticket = t.id;
  out.request_id = t.request_id;
  out.decision = d;
  recent.push_back(at);
  ++s.ctr.admitted;
  s.queue.push({t.id, t.client_id, t.prio});
  s.by_request[t.request_id] = t.id;
  s.tickets[t.id] = std::move(t);
  s.pump();
  return out;
}

Status Scheduler::cancel(TicketId ticket, EndCause cause) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  auto it = s.tickets.find(ticket);
  if (it == s.tickets.end()) return Status(ErrorCode::kNotFound, "unknown ticket");
  s.cancel_locked(it->second, cause);
  s.pump();
  return Status::ok();
}

Status Scheduler::cancel_request(const std::string& request_id, EndCause cause) {
  TicketId id = 0;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->by_request.find(request_id);
    if (it == impl_->by_request.end()) return Status(ErrorCode::kNotFound, "unknown request");
    id = it->second;
  }
  return cancel(id, cause);
}

Result<std::string> Scheduler::prepare(const std::string& profile_id, std::uint32_t ctx) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  if (s.phase != Impl::Phase::kRunning) return Status(ErrorCode::kFailedPrecondition, "shutting_down");
  const ProfileSnapshot* p = s.profile_of(profile_id);
  if (!p) return Status(ErrorCode::kNotFound, "unknown profile");
  if (ctx == 0) ctx = p->context_default_tokens ? p->context_default_tokens : p->context_max_tokens;
  const auto cls = context_class_for(*p, ctx);
  if (!cls) return Status(ErrorCode::kInvalidArgument, "context exceeds the profile maximum");
  ReadinessView r = s.effective(*p, *cls);
  if (is_prepared(r.state)) return std::string();   // already prepared: nothing to start
  if (!is_loadable(r.state)) return Status(ErrorCode::kFailedPrecondition, r.reasons.empty() ? "profile is not loadable" : r.reasons.front());
  if (!s.resources_of(profile_id).workers_allowed) return Status(ErrorCode::kFailedPrecondition, "a required Worker is not available");
  std::string id = s.ensure_prepare(*p, *cls);
  s.pump();
  return id;
}

Status Scheduler::cancel_job(const std::string& job_id) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  for (auto& [seq, j] : s.jobs_) {
    if (j.id != job_id) continue;
    if (j.state == JobState::kQueued) {
      s.end_job(j, JobState::kCancelled, "preparation was cancelled");
      for (auto& [sq, d] : s.jobs_)   // a swap prepare that depends on a cancelled release is cancelled with it
        if (d.after == seq && d.state == JobState::kQueued) s.end_job(d, JobState::kCancelled, "preparation was cancelled");
    } else if (j.state == JobState::kRunning) {
      if (j.kind == JobKind::kRelease) return Status(ErrorCode::kFailedPrecondition, "a release cannot be cancelled");
      if (!j.cancel_requested) {
        j.cancel_requested = true;
        j.last_error = "preparation was cancelled";
        auto b = s.backend;
        const JobSeq sq = seq;
        s.act([b, sq] { b->cancel_prepare(sq); });
      }
    }
    s.pump();
    return Status::ok();
  }
  return Status(ErrorCode::kNotFound, "unknown job");
}

Status Scheduler::release(const std::string& profile_id, bool force) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  std::vector<Impl::Plan*> targets;
  for (auto& [id, p] : s.plans)
    if (p.profile_id == profile_id && p.state == Impl::Plan::St::kReady) targets.push_back(&p);
  if (targets.empty()) return Status(ErrorCode::kNotFound, "no prepared plan for this profile");
  for (Impl::Plan* p : targets) {
    bool active = false;
    for (const auto& sl : p->slots) active = active || Impl::slot_active(sl);
    if (active && !force) return Status(ErrorCode::kFailedPrecondition, "conflict: a request is active");
  }
  for (Impl::Plan* p : targets) {
    for (auto& [id, t] : s.tickets)
      if (t.plan == p->id && (t.state == TicketState::kStarting || t.state == TicketState::kRunning))
        s.cancel_locked(t, EndCause::kManagement);
    s.ensure_release(*p);
  }
  s.pump();
  return Status::ok();
}

void Scheduler::update_world(std::shared_ptr<const WorldSnapshot> w) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  s.world = w ? std::move(w) : std::make_shared<WorldSnapshot>();
  s.model_index.clear();
  for (const auto& [id, p] : s.world->profiles) s.model_index[p.api_model_id] = "p:" + id;
  for (const auto& [id, a] : s.world->aliases) s.model_index[a.api_model_id] = "a:" + id;
  // A blocker on a profile whose plan is ready invalidates the plan; one on a running prepare cancels it.
  for (auto& [pid, pl] : s.plans) {
    if (pl.state != Impl::Plan::St::kReady) continue;
    const ProfileSnapshot* prof = s.profile_of(pl.profile_id);
    if (!prof) { s.invalidate_plan(pl, InvalidateReason::kPolicy); continue; }
    const ReadinessView b = s.base_readiness(pl.profile_id, pl.context);
    if (b.state == ReadyState::kUnavailable || b.state == ReadyState::kInstalled)
      s.invalidate_plan(pl, InvalidateReason::kPolicy);
  }
  for (auto& [seq, j] : s.jobs_) {
    if (j.kind != JobKind::kPrepare || j.state != JobState::kRunning || j.cancel_requested) continue;
    const ReadinessView b = s.base_readiness(j.profile_id, j.context);
    if (b.state == ReadyState::kUnavailable || b.state == ReadyState::kInstalled || b.state == ReadyState::kCompatible)
      s.cancel_running_prepare(b.reasons.empty() ? "a required Worker is no longer available" : b.reasons.front());
  }
  s.pump();
}

void Scheduler::worker_event(const WorkerEvent& ev) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  if (ev.kind == WorkerEventKind::kBack) {
    for (const auto& [id, p] : s.world->profiles) {
      if (!p.prepare_when_available || p.preparation == Preparation::kManual) continue;
      const std::uint32_t ctx = p.context_default_tokens ? p.context_default_tokens : p.context_max_tokens;
      const auto cls = context_class_for(p, ctx);
      if (cls && s.effective(p, *cls).state == ReadyState::kLoadable && s.resources_of(id).workers_allowed)
        s.ensure_prepare(p, *cls);
    }
    s.pump();
    return;
  }
  InvalidateReason r = InvalidateReason::kWorkerLost;
  if (ev.kind == WorkerEventKind::kLocalUserActive) r = InvalidateReason::kLocalUserReturned;
  if (ev.kind == WorkerEventKind::kPolicyForbids) r = InvalidateReason::kPolicy;
  for (auto& [pid, pl] : s.plans) {
    if (pl.state != Impl::Plan::St::kReady) continue;
    if (std::find(pl.info.worker_ids.begin(), pl.info.worker_ids.end(), ev.worker_id) != pl.info.worker_ids.end())
      s.invalidate_plan(pl, r);
  }
  // A prepare that depended on this Worker is cancelled by the next world update, which reports the profile below
  // `loadable` (the scheduler does not know which Workers a prepare will use).
  s.pump();
}

void Scheduler::key_revoked(const std::string& client_id) {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  s.revoked.insert(client_id);
  for (auto& [id, t] : s.tickets)
    if (t.client_id == client_id) s.cancel_locked(t, EndCause::kKeyRevoked);
  for (auto& [pid, p] : s.plans) {
    for (Impl::Slot& sl : p.slots) {
      if (sl.st == Impl::Slot::St::kRetained && sl.key.client_id == client_id && p.state == Impl::Plan::St::kReady) {
        sl.st = Impl::Slot::St::kDropping;
        sl.for_ticket = 0;
        auto b = s.backend;
        const PlanId plan = pid;
        const RetainedSessionRef ref = sl.ref;
        s.act([b, plan, ref] { b->drop_session(plan, ref); });
      }
    }
  }
  s.pump();
}

ReadinessView Scheduler::readiness(const std::string& profile_id, std::uint32_t ctx) const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  const ProfileSnapshot* p = impl_->profile_of(profile_id);
  if (!p) {
    ReadinessView v;
    v.reasons = {"unknown profile"};
    return v;
  }
  if (ctx == 0) ctx = p->context_default_tokens ? p->context_default_tokens : p->context_max_tokens;
  const auto cls = context_class_for(*p, ctx);
  return impl_->effective(*p, cls ? *cls : p->context_max_tokens);
}

PlanOverlay Scheduler::overlay(const std::string& profile_id, std::uint32_t ctx) const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  PlanOverlay o;
  const ProfileSnapshot* prof = impl_->profile_of(profile_id);
  if (!prof) return o;
  o.profile_revision = prof->revision;
  o.plan_fingerprint = prof->plan_fingerprint;
  o.concurrency_limit = prof->max_sessions;
  for (const auto& [id, p] : impl_->plans) {
    if (p.profile_id == profile_id && p.state == Impl::Plan::St::kReady && p.fingerprint == prof->plan_fingerprint &&
        p.context >= ctx) {
      o.plan_ready = true;
      o.plan_context = p.context;
      o.active_requests = Impl::active_slots(p);
      o.concurrency_limit = static_cast<std::uint32_t>(p.slots.size());
    }
  }
  for (const auto& [seq, j] : impl_->jobs_) {
    if (j.profile_id != profile_id || j.kind != JobKind::kPrepare) continue;
    o.job_id = j.id;
    o.job_state = std::string(to_string(j.state));
    o.job_phase = j.progress.phase;
    o.last_error = j.last_error;
  }
  return o;
}

bool Scheduler::shutdown_done() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->phase == Impl::Phase::kDone;
}

ShutdownReport Scheduler::shutdown() {
  Impl& s = *impl_;
  if (std::this_thread::get_id() == s.deliverer_id || std::this_thread::get_id() == s.dispatcher_id) {
    ShutdownReport r;
    r.clean = false;
    r.problems.push_back("shutdown() called from a scheduler thread");
    return r;
  }
  std::unique_lock<std::mutex> lock(s.mu);
  if (s.phase == Impl::Phase::kRunning) {
    const TimePoint at = s.now();
    s.phase = Impl::Phase::kDraining;
    s.grace_deadline = at + s.opt.shutdown_grace;
    s.hard_deadline = s.grace_deadline + s.opt.cancel_timeout + s.opt.invalidate_timeout + 2 * s.opt.release_timeout;
    for (auto& [id, t] : s.tickets) {
      if (t.state == TicketState::kQueued) {
        s.queue.served(t.client_id, t.prio);
        s.cancel_queued(t, EndCause::kShutdown, s.terminal_for(t, EndCause::kShutdown));
      }
    }
    for (auto& [seq, j] : s.jobs_) {
      if (j.state == JobState::kQueued && j.kind == JobKind::kPrepare) s.end_job(j, JobState::kCancelled, "the server is shutting down");
      else if (j.state == JobState::kRunning && j.kind == JobKind::kPrepare) s.cancel_running_prepare("the server is shutting down");
    }
    s.pump();
  }
  if (s.opt.threading == SchedulerOptions::Threading::kManual) return s.report;
  // Threaded: wait for completion (real time, bounded by the documented maximum).
  const auto bound = s.opt.shutdown_grace + s.opt.cancel_timeout + s.opt.invalidate_timeout + 2 * s.opt.release_timeout;
  const auto limit = std::chrono::steady_clock::now() + bound + std::chrono::seconds(5);
  while (s.phase != Impl::Phase::kDone && std::chrono::steady_clock::now() < limit)
    s.cv_done.wait_for(lock, kPoll);
  // Let queued actions and deliveries finish so every terminal is delivered.
  while ((!s.actions.empty() || !s.deliveries.empty() || s.inflight_actions || s.inflight_deliveries) &&
         std::chrono::steady_clock::now() < limit)
    s.cv_done.wait_for(lock, kPoll);
  if (s.phase != Impl::Phase::kDone) { s.report.clean = false; s.report.problems.push_back("shutdown did not complete in time"); }
  return s.report;
}

// ---- StatusProvider ----------------------------------------------------------------------------------------

std::vector<MachineStatus> Scheduler::machines() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::vector<MachineStatus> out = impl_->world->machines;
  for (MachineStatus& m : out) {
    m.leases_held = 0;
    for (const auto& [id, p] : impl_->plans)
      if (std::find(p.info.worker_ids.begin(), p.info.worker_ids.end(), m.display_name) != p.info.worker_ids.end())
        ++m.leases_held;
  }
  return out;
}

QueueStatus Scheduler::queue(const std::string& client_id) const {
  Impl& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mu);
  QueueStatus q;
  const TimePoint at = s.last_now;
  q.depth = static_cast<std::uint32_t>(s.queue.depth());
  q.capacity = s.opt.max_queue_depth;
  q.per_priority[0] = static_cast<std::uint32_t>(s.queue.depth(Priority::kInteractive));
  q.per_priority[1] = static_cast<std::uint32_t>(s.queue.depth(Priority::kApi));
  const auto order = s.queue.order(s.demoted_fn(at));
  std::map<TicketId, std::uint32_t> pos;
  for (std::size_t i = 0; i < order.size(); ++i) pos[order[i]] = static_cast<std::uint32_t>(i + 1);
  for (const auto& [id, t] : s.tickets) {
    if (t.state == TicketState::kDone) continue;
    if (t.state != TicketState::kQueued) ++q.active;
    if (t.client_id == client_id) {
      TicketStatus ts;
      ts.request_id = t.request_id;
      ts.state = t.state;
      ts.profile_id = t.profile_id;
      ts.api_model_id = t.api_model_id;
      ts.position = pos.count(id) ? pos[id] : 0;
      ts.age_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(at - t.enqueued).count());
      ts.waiting_for_prepare = t.waiting_job.has_value();
      if (t.waiting_job) { auto j = s.jobs_.find(*t.waiting_job); if (j != s.jobs_.end()) ts.job_id = j->second.id; }
      ts.prompt_tokens = t.prompt_tokens;
      ts.completion_budget = t.completion_budget;
      q.own.push_back(std::move(ts));
    } else if (t.state == TicketState::kQueued && pos.count(id)) {
      q.others_positions.push_back(pos[id]);
    }
  }
  std::sort(q.others_positions.begin(), q.others_positions.end());
  return q;
}

std::vector<JobStatus> Scheduler::jobs() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::vector<JobStatus> out;
  for (const auto& [seq, j] : impl_->jobs_) {
    JobStatus st;
    st.job_id = j.id;
    st.kind = j.kind;
    st.state = j.state;
    st.profile_id = j.profile_id;
    st.context_tokens = j.context;
    if (j.state == JobState::kRunning) st.phase = j.progress.phase;
    st.percent = j.progress.percent;
    st.eta_seconds = j.progress.eta_seconds;
    st.last_error = j.last_error;
    for (const auto& [id, t] : impl_->tickets)
      if (t.waiting_job == seq && t.state == TicketState::kQueued) ++st.waiting_requests;
    st.age_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(impl_->last_now - j.created).count());
    out.push_back(std::move(st));
  }
  return out;
}

std::optional<JobStatus> Scheduler::job(const std::string& job_id) const {
  for (auto& j : jobs())
    if (j.job_id == job_id) return j;
  return std::nullopt;
}

std::vector<AvailabilityEntry> Scheduler::availability() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::vector<AvailabilityEntry> out;
  for (const auto& [id, p] : impl_->world->profiles) {
    AvailabilityEntry e;
    e.profile_id = id;
    e.api_model_id = p.api_model_id;
    const std::uint32_t ctx = p.context_default_tokens ? p.context_default_tokens : p.context_max_tokens;
    const auto cls = context_class_for(p, ctx);
    ReadinessView r = impl_->effective(p, cls ? *cls : p.context_max_tokens);
    e.state = r.state;
    e.prepared = is_prepared(r.state);
    const ResourceView res = impl_->resources_of(id);
    e.can_prepare_now = r.state == ReadyState::kLoadable && p.preparation != Preparation::kManual && res.workers_allowed &&
                        impl_->phase == Impl::Phase::kRunning;
    if (!e.can_prepare_now && !e.prepared) {
      e.why_not = r.reasons;
      if (!res.workers_allowed) e.why_not.push_back(res.reason.empty() ? "a required Worker is not available" : res.reason);
      if (r.state == ReadyState::kLoadable && p.preparation == Preparation::kManual)
        e.why_not.push_back("preparation is manual");
      if (e.why_not.empty()) e.why_not.push_back(std::string(to_string(r.state)));
    }
    out.push_back(std::move(e));
  }
  return out;
}

SchedulerCounters Scheduler::counters() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->ctr;
}

DebugSnapshot Scheduler::debug_snapshot() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  DebugSnapshot d;
  using St = Impl::Slot::St;
  for (const auto& [id, p] : impl_->plans) {
    DebugSnapshot::PlanView pv;
    pv.id = id;
    pv.profile_id = p.profile_id;
    pv.context = p.context;
    pv.state = p.state == Impl::Plan::St::kReady ? "ready" : p.state == Impl::Plan::St::kReleasing ? "releasing" : "invalidating";
    for (const auto& sl : p.slots) {
      const char* n = sl.st == St::kFree ? "free" : sl.st == St::kReserved ? "reserved" : sl.st == St::kRetained ? "retained"
                      : sl.st == St::kDropping ? "dropping" : "quarantined";
      pv.slots.push_back({n, sl.ticket});
    }
    d.plans.push_back(std::move(pv));
  }
  for (const auto& [id, t] : impl_->tickets) {
    d.ticket_states[id] = std::string(to_string(t.state));
    if (t.state == TicketState::kQueued) ++d.queued;
    else if (t.state != TicketState::kDone) ++d.active;
  }
  for (const auto& [s, j] : impl_->jobs_) {
    if (j.state == JobState::kRunning) ++d.jobs_running;
    if (j.state == JobState::kQueued) ++d.jobs_queued;
  }
  return d;
}

}  // namespace clusterlm::scheduler
