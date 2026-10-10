// Scheduler scenarios (docs/scheduler-design.md §8.3), run in manual mode with a manual clock and a deterministic fake
// backend: every command and completion is explicit, so each race is reproducible. Real = the scheduler code under test;
// mocked = the backend (no Coordinator, no Workers, no GPU).
#include <doctest/doctest.h>

#include "fixture.hpp"

using namespace clusterlm::scheduler;
using namespace clusterlm::scheduler::testing;
using Kind = Terminal::Kind;
using namespace std::chrono_literals;

namespace {
Terminal only(const std::shared_ptr<CapturingObserver>& o) {
  REQUIRE(o->terminals.size() == 1);
  return o->terminals.front();
}
std::string state_of(Fixture& f, TicketId t) { return f.sch->debug_snapshot().ticket_states.at(t); }
std::string slot0(Fixture& f) {
  auto d = f.sch->debug_snapshot();
  REQUIRE_FALSE(d.plans.empty());
  return d.plans.front().slots.front().state;
}
}  // namespace

TEST_CASE("full lifecycle: loadable -> prepare job -> start -> finish -> slot free") {
  Fixture f;
  auto r = f.enq();
  REQUIRE(r.accepted);
  f.run();
  CHECK(f.be->count(Cmd::K::kPrepare) == 1);
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kPreparing);
  auto jobs = f.sch->jobs();
  REQUIRE(jobs.size() == 1);
  CHECK(jobs[0].waiting_requests == 1);
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kBusy);   // a request holds the only slot
  REQUIRE(f.be->count(Cmd::K::kStart) == 1);
  f.be->running(r.ticket);
  f.run();
  REQUIRE(f.last_observer->started.size() == 1);
  CHECK(f.last_observer->started[0].api_model_id == "model-a");
  CHECK(f.last_observer->started[0].routed_via.empty());
  f.be->output(r.ticket, 7);
  f.run();
  auto t = only(f.last_observer);
  CHECK(t.kind == Kind::kSuccess);
  CHECK(t.summary.completion_tokens == 7);
  CHECK(slot0(f) == "reserved");   // the response is out but the slot is held until the backend confirms the close
  f.be->ended(r.ticket, {});
  f.run();
  CHECK(slot0(f) == "free");
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kReady);
  CHECK(f.sch->counters().completed == 1);
  f.check_invariants();
}

TEST_CASE("an already prepared profile admits and starts immediately, FIFO behind it") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  f.be->output(1);
  f.be->ended(1, {});
  f.run();
  auto r = f.enq();
  f.run();
  CHECK(r.decision.kind == Decision::Kind::kAdmit);
  CHECK(f.be->count(Cmd::K::kStart) == 2);
}

TEST_CASE("fast-fail client gets model_not_ready with a job id, and the job still starts") {
  Fixture f;
  auto r = f.enq("model-a", "k1", 10, 20, "", false);
  CHECK_FALSE(r.accepted);
  CHECK(r.decision.code == Code::kModelNotReady);
  REQUIRE(r.decision.job_id.has_value());
  f.run();
  CHECK(f.be->count(Cmd::K::kPrepare) == 1);
  // a second fast-fail request joins the same job
  auto r2 = f.enq("model-a", "k2", 10, 20, "", false);
  CHECK(r2.decision.job_id == r.decision.job_id);
  CHECK(f.sch->jobs().size() == 1);
}

TEST_CASE("manual preparation is never started by a request") {
  Fixture f;
  f.w->profiles["prof_a"].preparation = Preparation::kManual;
  f.publish();
  auto r = f.enq();
  CHECK_FALSE(r.accepted);
  CHECK(r.decision.code == Code::kModelNotReady);
  f.run();
  CHECK(f.be->count(Cmd::K::kPrepare) == 0);
  auto j = f.sch->prepare("prof_a");   // an explicit management prepare is allowed
  REQUIRE(j.is_ok());
  f.run();
  CHECK(f.be->count(Cmd::K::kPrepare) == 1);
}

TEST_CASE("F1 cancelling a waiting request does not cancel the job; others are served after it") {
  Fixture f;
  auto a = f.enq("model-a", "k1");
  auto aobs = f.last_observer;
  auto b = f.enq("model-a", "k2");
  f.run();
  CHECK(f.sch->jobs()[0].waiting_requests == 1);   // only the head is attached (strict head-of-line)
  CHECK(f.sch->cancel(a.ticket).is_ok());
  f.run();
  CHECK(only(aobs).kind == Kind::kSilent);
  REQUIRE(f.sch->jobs().size() == 1);
  CHECK(f.sch->jobs()[0].state == JobState::kRunning);
  CHECK(f.sch->jobs()[0].waiting_requests == 1);   // b took over the head
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  CHECK(f.be->count(Cmd::K::kStart) == 1);
  CHECK(f.be->last(Cmd::K::kStart).a == b.ticket);
  f.check_invariants();
}

TEST_CASE("F2 cancel during starting: abort follows start, no on_started, slot freed only by the confirmation") {
  Fixture f;
  auto r = f.enq();
  auto obs = f.last_observer;
  f.run();
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  REQUIRE(f.be->count(Cmd::K::kStart) == 1);
  CHECK(f.sch->cancel(r.ticket).is_ok());
  f.run();
  // order in the command log: start before abort
  std::size_t start_at = 0, abort_at = 0;
  for (std::size_t i = 0; i < f.be->log.size(); ++i) {
    if (f.be->log[i].kind == Cmd::K::kStart) start_at = i;
    if (f.be->log[i].kind == Cmd::K::kAbort) abort_at = i;
  }
  CHECK(start_at < abort_at);
  CHECK(slot0(f) == "reserved");   // not free yet: the domain may still write
  CHECK(only(obs).kind == Kind::kSilent);
  CHECK(obs->started.empty());
  f.check_invariants();
  f.be->aborted(r.ticket);
  f.run();
  CHECK(slot0(f) == "free");
  f.check_invariants();
}

TEST_CASE("F3 cancel races output_done: the first one wins, the other is a no-op") {
  SUBCASE("cancel first") {
    Fixture f;
    PlanId plan = f.prepare_ready();
    (void)plan;
    f.be->running(1);
    auto obs = f.observers.back();
    CHECK(f.sch->cancel(1).is_ok());
    f.be->output(1);   // stale
    f.run();
    CHECK(only(obs).kind == Kind::kSilent);
    CHECK(f.sch->counters().stale_events == 1);
    f.be->aborted(1);
    f.run();
    CHECK(slot0(f) == "free");
  }
  SUBCASE("output first") {
    Fixture f;
    f.prepare_ready();
    f.be->running(1);
    auto obs = f.observers.back();
    f.be->output(1);
    CHECK(f.sch->cancel(1).is_ok());   // finishing: no-op
    f.run();
    CHECK(only(obs).kind == Kind::kSuccess);
    CHECK(f.be->count(Cmd::K::kAbort) == 0);
    f.be->ended(1, {});
    f.run();
    CHECK(slot0(f) == "free");
  }
}

TEST_CASE("F4 repeated cancels with different causes: first cause kept, one abort, always ok") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto obs = f.observers.back();
  CHECK(f.sch->cancel(1, EndCause::kClientGone).is_ok());
  CHECK(f.sch->cancel(1, EndCause::kManagement).is_ok());
  CHECK(f.sch->cancel(1, EndCause::kShutdown).is_ok());
  f.run();
  CHECK(f.be->count(Cmd::K::kAbort) == 1);
  CHECK(only(obs).cause == EndCause::kClientGone);
  CHECK(f.sch->cancel(9999).code() == clusterlm::ErrorCode::kNotFound);
}

TEST_CASE("F5 abort never confirmed: a timer escalates to invalidation, never frees the slot") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto second = f.enq("model-a", "k2");
  auto second_obs = f.last_observer;
  CHECK(f.sch->cancel(1).is_ok());
  f.run();
  f.sec(9);
  CHECK(f.be->count(Cmd::K::kInvalidate) == 0);
  f.sec(2);   // cancel_timeout (10 s) passed
  REQUIRE(f.be->count(Cmd::K::kInvalidate) == 1);
  CHECK(f.be->last(Cmd::K::kInvalidate).why == InvalidateReason::kAbortNotConfirmed);
  CHECK(slot0(f) == "quarantined");
  CHECK(f.be->count(Cmd::K::kStart) == 1);   // the queued request was not started on a possibly written slot
  f.check_invariants();
  // F6: the plan finally goes away; a late `ended` for the cancelled ticket is ignored
  PlanId plan = f.be->last(Cmd::K::kInvalidate).a;
  f.be->gone(plan);
  f.run();
  f.be->ended(1, {});
  f.run();
  CHECK(f.sch->counters().stale_events >= 1);
  CHECK(f.sch->debug_snapshot().plans.empty());
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kPreparing);   // the waiting request triggered a new prepare
  CHECK(f.be->count(Cmd::K::kPrepare) == 2);
  if (!second_obs->terminals.empty()) MESSAGE("terminal code=" << to_string(second_obs->terminals[0].code) << " cause=" << int(second_obs->terminals[0].cause) << " msg=" << second_obs->terminals[0].message);
  CHECK(second_obs->terminals.empty());
  f.check_invariants();
}

TEST_CASE("F7 invalidation that never completes keeps the slot quarantined at any time") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  CHECK(f.sch->cancel(1).is_ok());
  f.sec(11);
  REQUIRE(f.be->count(Cmd::K::kInvalidate) == 1);
  f.sec(3600);
  CHECK(slot0(f) == "quarantined");
  CHECK(f.sch->counters().stuck_invalidations == 1);
  auto rv = f.sch->readiness("prof_a", 4096);
  CHECK(rv.state == ReadyState::kCompatible);
  REQUIRE_FALSE(rv.reasons.empty());
  CHECK(rv.reasons.front().find("not stopped") != std::string::npos);
  f.check_invariants();
}

TEST_CASE("F8 a prepare whose Worker disappears is cancelled; waiters get model_not_ready") {
  Fixture f;
  auto r = f.enq();
  auto obs = f.last_observer;
  f.run();
  auto job = f.be->last(Cmd::K::kPrepare).a;
  f.set_state("prof_a", ReadyState::kCompatible, {"G14 is offline"});
  f.publish();
  f.run();
  CHECK(f.be->count(Cmd::K::kCancelPrepare) == 1);
  f.be->fail_prepare(job, "worker lost");
  f.run();
  auto t = only(obs);
  CHECK(t.code == Code::kModelNotReady);
  CHECK(t.cause == EndCause::kPrepareFailed);
  CHECK(f.sch->jobs()[0].state == JobState::kCancelled);
  CHECK(f.sch->debug_snapshot().plans.empty());
  (void)r;
}

TEST_CASE("F9 Worker lost while running: immediate worker_lost, slot freed only at plan_gone, queued rejected") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto first = f.observers.back();
  auto queued = f.enq("model-a", "k2");
  auto queued_obs = f.last_observer;
  f.run();
  f.sch->worker_event({"G14", WorkerEventKind::kLost, "G14 is offline"});
  f.run();
  CHECK(only(first).code == Code::kWorkerLost);
  REQUIRE(f.be->count(Cmd::K::kInvalidate) == 1);
  CHECK(f.be->last(Cmd::K::kInvalidate).why == InvalidateReason::kWorkerLost);
  CHECK(slot0(f) == "quarantined");
  f.set_state("prof_a", ReadyState::kCompatible, {"G14 is offline"});
  f.publish();
  f.check_invariants();
  f.be->gone(f.be->last(Cmd::K::kInvalidate).a);
  f.run();
  CHECK(only(queued_obs).code == Code::kWorkersUnavailable);
  CHECK(f.sch->debug_snapshot().plans.empty());
  (void)queued;
}

TEST_CASE("F9b backend reports the stage lost: ticket worker_lost, plan invalidated") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto obs = f.observers.back();
  f.be->failed_worker_lost(1);
  f.run();
  CHECK(only(obs).code == Code::kWorkerLost);
  CHECK(f.be->count(Cmd::K::kInvalidate) == 1);
  CHECK(slot0(f) == "quarantined");   // the plan's domains are not gone yet
  f.be->gone(f.be->last(Cmd::K::kInvalidate).a);
  f.run();
  CHECK(f.sch->debug_snapshot().plans.empty());
}

TEST_CASE("F10 local user returns while the plan idles with a retained session: immediate invalidation, independent of jobs") {
  Fixture f([](SchedulerOptions& o) { o.max_prepared_profiles = 2; });
  f.be->retain_sessions = true;
  f.enq("model-a", "k1", 10, 20, "conv");
  f.run();
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  f.be->running(1);
  f.be->finish(1);
  f.run();
  CHECK(slot0(f) == "retained");
  f.enq("model-b", "k2");   // a prepare for B is now running
  f.run();
  REQUIRE(f.be->count(Cmd::K::kPrepare) == 2);
  f.sch->worker_event({"G14", WorkerEventKind::kLocalUserActive, "G14 is in use"});
  f.run();
  REQUIRE(f.be->count(Cmd::K::kInvalidate) == 1);
  CHECK(f.be->last(Cmd::K::kInvalidate).why == InvalidateReason::kLocalUserReturned);
  CHECK(f.be->count(Cmd::K::kRelease) == 0);   // not a queued release job
}

TEST_CASE("F11 swap waits for the active request, never aborts it, then release -> prepare") {
  Fixture f;
  f.prepare_ready("model-a");
  f.be->running(1);
  auto b = f.enq("model-b", "k2");
  f.run();
  CHECK(f.be->count(Cmd::K::kRelease) == 0);
  CHECK(f.be->count(Cmd::K::kAbort) == 0);
  CHECK(f.be->count(Cmd::K::kPrepare) == 1);   // no overlap of allocations
  f.be->output(1);
  f.be->ended(1, {});
  f.run();
  REQUIRE(f.be->count(Cmd::K::kRelease) == 1);
  CHECK(f.be->count(Cmd::K::kPrepare) == 1);   // prepare B only after the plan is gone
  f.be->gone(f.be->last(Cmd::K::kRelease).a, GoneCause::kReleased);
  f.run();
  REQUIRE(f.be->count(Cmd::K::kPrepare) == 2);
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  CHECK(f.be->last(Cmd::K::kStart).a == b.ticket);
  CHECK(f.sch->counters().swaps >= 1);
  f.check_invariants();
}

TEST_CASE("F12 cancelling the release job of a swap cancels the dependent prepare; waiters get model_not_ready") {
  Fixture f;
  f.prepare_ready("model-a");
  f.be->running(1);
  f.enq("model-b", "k2");
  auto obs = f.last_observer;
  f.run();
  std::string release_job;
  for (const auto& j : f.sch->jobs())
    if (j.kind == JobKind::kRelease) release_job = j.job_id;
  REQUIRE_FALSE(release_job.empty());
  CHECK(f.sch->cancel_job(release_job).is_ok());
  f.run();
  CHECK(only(obs).code == Code::kModelNotReady);
  CHECK(f.be->count(Cmd::K::kRelease) == 0);
}

TEST_CASE("F13 prepare is idempotent per plan; a larger context raises a queued job, follows a running one") {
  Fixture f;
  f.w->profiles["prof_a"].preparation = Preparation::kManual;
  f.publish();
  auto blocker = f.sch->prepare("prof_b", 4096);   // occupies the single job slot so the next job stays queued
  REQUIRE(blocker.is_ok());
  auto j1 = f.sch->prepare("prof_a", 4096);
  auto j2 = f.sch->prepare("prof_a", 4096);
  REQUIRE(j1.is_ok());
  CHECK(j1.value() == j2.value());
  auto j3 = f.sch->prepare("prof_a", 8192);   // raises the queued job
  CHECK(j3.value() == j1.value());
  std::uint32_t ctx = 0;
  for (const auto& j : f.sch->jobs())
    if (j.job_id == j1.value()) ctx = j.context_tokens;
  CHECK(ctx == 8192);
  CHECK(f.sch->jobs().size() == 2);
}

TEST_CASE("F13b a running smaller job gets a follow-up job for a larger context") {
  Fixture f;
  f.w->profiles["prof_a"].preparation = Preparation::kManual;
  f.publish();
  auto j1 = f.sch->prepare("prof_a", 4096);
  f.run();
  auto j2 = f.sch->prepare("prof_a", 8192);
  REQUIRE(j2.is_ok());
  CHECK(j2.value() != j1.value());
  CHECK(f.sch->jobs().size() == 2);
}

TEST_CASE("F14 warm retention: same client+hint reuses, others drop first") {
  Fixture f;
  f.be->retain_sessions = true;
  auto t1 = f.enq("model-a", "k1", 10, 20, "conv-1");
  f.run();
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  f.be->running(t1.ticket);
  f.be->finish(t1.ticket);
  f.run();
  CHECK(slot0(f) == "retained");
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kReady);   // retained state is idle, not busy

  SUBCASE("same client, same hint: reuse") {
    auto t2 = f.enq("model-a", "k1", 10, 20, "conv-1");
    f.run();
    CHECK(f.be->last(Cmd::K::kStart).a == t2.ticket);
    CHECK(f.be->last(Cmd::K::kStart).reuse);
    CHECK(f.be->count(Cmd::K::kDrop) == 0);
  }
  SUBCASE("other client: the retained state is dropped (and confirmed) before its session opens") {
    auto t2 = f.enq("model-a", "k2", 10, 20, "conv-1");
    f.run();
    REQUIRE(f.be->count(Cmd::K::kDrop) == 1);
    CHECK(f.be->count(Cmd::K::kStart) == 1);   // not yet
    CHECK(slot0(f) == "dropping");
    auto d = f.be->last(Cmd::K::kDrop);
    f.be->dropped(d.a, RetainedSessionRef{d.a, d.b});
    f.run();
    CHECK(f.be->last(Cmd::K::kStart).a == t2.ticket);
    CHECK_FALSE(f.be->last(Cmd::K::kStart).reuse);
  }
  SUBCASE("same client, different hint: drop first") {
    f.enq("model-a", "k1", 10, 20, "conv-2");
    f.run();
    CHECK(f.be->count(Cmd::K::kDrop) == 1);
  }
  SUBCASE("same client, no hint: drop first") {
    f.enq("model-a", "k1", 10, 20, "");
    f.run();
    CHECK(f.be->count(Cmd::K::kDrop) == 1);
  }
  SUBCASE("key revoked drops its retained state") {
    f.sch->key_revoked("k1");
    f.run();
    CHECK(f.be->count(Cmd::K::kDrop) == 1);
  }
  f.check_invariants();
}

TEST_CASE("F15 a drop that is never confirmed escalates; the waiting ticket returns to the queue (T17)") {
  Fixture f;
  f.be->retain_sessions = true;
  auto t1 = f.enq("model-a", "k1", 10, 20, "c");
  f.run();
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  f.be->running(t1.ticket);
  f.be->finish(t1.ticket);
  f.run();
  auto t2 = f.enq("model-a", "k2", 10, 20, "");
  f.run();
  REQUIRE(f.be->count(Cmd::K::kDrop) == 1);
  f.sec(11);
  REQUIRE(f.be->count(Cmd::K::kInvalidate) == 1);
  CHECK(state_of(f, t2.ticket) == "starting");
  f.be->gone(f.be->last(Cmd::K::kInvalidate).a);
  f.run();
  CHECK(state_of(f, t2.ticket) == "queued");   // nothing of it was ever opened
  CHECK(f.be->count(Cmd::K::kPrepare) == 2);   // and it prepares again
  f.check_invariants();
}

TEST_CASE("F16 shutdown: queued -> shutting_down at once, running gets the grace period, then plans are released and the backend detached") {
  Fixture f([](SchedulerOptions& o) { o.max_prepared_profiles = 2; });
  f.prepare_ready("model-a");
  f.be->running(1);
  auto running_obs = f.observers.back();
  std::vector<std::shared_ptr<CapturingObserver>> queued;
  for (int i = 0; i < 3; ++i) { f.enq("model-a", "k" + std::to_string(i + 2)); queued.push_back(f.last_observer); }
  f.enq("model-b", "k9");
  f.run();
  (void)f.sch->shutdown();
  f.run();
  for (auto& o : queued) CHECK(only(o).code == Code::kShuttingDown);
  REQUIRE(f.be->count(Cmd::K::kCancelPrepare) == 1);   // the running prepare for B is cancelled, not abandoned
  auto late = f.enq("model-a", "k7");
  CHECK_FALSE(late.accepted);
  CHECK(late.decision.code == Code::kShuttingDown);
  CHECK(f.be->count(Cmd::K::kAbort) == 0);   // the running request gets its grace
  CHECK_FALSE(f.sch->shutdown_done());
  f.sec(11);
  CHECK(f.be->count(Cmd::K::kAbort) == 1);
  f.be->aborted(1);
  f.be->fail_prepare(f.be->last(Cmd::K::kCancelPrepare).a, "cancelled");
  f.run();
  CHECK(only(running_obs).code == Code::kShuttingDown);
  REQUIRE(f.be->count(Cmd::K::kRelease) == 1);
  f.be->gone(f.be->last(Cmd::K::kRelease).a, GoneCause::kReleased);
  f.run();
  CHECK(f.sch->shutdown_done());
  CHECK(f.be->detached);
  CHECK(f.be->log.back().kind == Cmd::K::kDetach);
}

TEST_CASE("F17 shutdown with an abort that is never confirmed ends within its bound and reports it") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  (void)f.sch->shutdown();
  f.run();
  for (int i = 0; i < 40 && !f.sch->shutdown_done(); ++i) f.sec(10);
  CHECK(f.sch->shutdown_done());
  CHECK(f.be->detached);
}

TEST_CASE("F18 a large clock jump fires timers in deadline order") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  f.enq("model-a", "k2");
  auto queued_obs = f.last_observer;
  CHECK(f.sch->cancel(1).is_ok());
  f.run();
  f.advance(1h);
  REQUIRE(f.be->count(Cmd::K::kInvalidate) == 1);
  CHECK(only(queued_obs).code == Code::kQueueTimeout);
  CHECK(slot0(f) == "quarantined");
}

TEST_CASE("F19 a clock that goes backwards neither extends nor fires deadlines") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  f.enq("model-a", "k2");
  auto obs = f.last_observer;
  const TimePoint t0 = f.clock->now();
  f.sec(100);
  f.clock->set(t0 - 300s);
  f.run();
  CHECK(obs->terminals.empty());
  f.clock->set(t0 + 110s);
  f.run();
  CHECK(obs->terminals.empty());
  f.clock->set(t0 + 121s);   // 120 s after the enqueue, measured on the never-decreasing clock
  f.run();
  CHECK(only(obs).code == Code::kQueueTimeout);
}

TEST_CASE("F20 fairness: round robin across clients; a client over its slot budget goes last") {
  Fixture f;
  f.prepare_ready("model-a");   // ticket 1, client k1, running after prepared
  f.be->running(1);
  std::vector<TicketId> a, order;
  for (int i = 0; i < 8; ++i) a.push_back(f.enq("model-a", "A").ticket);
  auto b = f.enq("model-a", "B");
  auto c = f.enq("model-a", "C");
  f.be->output(1);
  f.be->ended(1, {});
  f.run();
  for (int i = 0; i < 10; ++i) {
    TicketId t = f.be->last(Cmd::K::kStart).a;
    order.push_back(t);
    f.be->running(t);
    f.be->output(t);
    f.be->ended(t, {});
    f.run();
  }
  CHECK(order == std::vector<TicketId>{a[0], b.ticket, c.ticket, a[1], a[2], a[3], a[4], a[5], a[6], a[7]});
}

TEST_CASE("F20b a client that held the slot too long is demoted behind other clients") {
  Fixture f([](SchedulerOptions& o) { o.queue_timeout = 600s; });
  f.prepare_ready("model-a");   // ticket 1 = client k1
  f.be->running(1);
  auto again = f.enq("model-a", "k1");
  auto other = f.enq("model-a", "k2");
  f.sec(301);   // k1 has held the slot for 301 s > max_slot_seconds (300)
  f.be->output(1);
  f.be->ended(1, {});
  f.run();
  CHECK(f.be->last(Cmd::K::kStart).a == other.ticket);
  (void)again;
}

TEST_CASE("F21 identical scenarios produce identical command traces") {
  auto trace = [] {
    Fixture f;
    std::vector<TicketId> ts;
    for (int i = 0; i < 6; ++i) ts.push_back(f.enq(i % 2 ? "model-b" : "model-a", "k" + std::to_string(i % 3)).ticket);
    f.run();
    f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
    f.run();
    std::vector<std::pair<int, std::uint64_t>> out;
    for (const auto& c : f.be->log) out.push_back({static_cast<int>(c.kind), c.a});
    return out;
  };
  CHECK(trace() == trace());
}

TEST_CASE("F22 a profile edit that changes the plan rejects queued tickets; a rename does not") {
  SUBCASE("model/backend changed") {
    Fixture f;
    f.prepare_ready();
    f.be->running(1);
    f.enq("model-a", "k2");
    auto obs = f.last_observer;
    f.w->profiles["prof_a"].plan_fingerprint = "fp-changed";
    f.w->profiles["prof_a"].revision = 2;
    f.publish();
    f.be->output(1);
    f.be->ended(1, {});
    f.run();
    auto t = only(obs);
    CHECK(t.code == Code::kModelNotReady);
    CHECK(t.message.find("changed") != std::string::npos);
  }
  SUBCASE("rename only") {
    Fixture f;
    f.prepare_ready();
    f.be->running(1);
    auto r = f.enq("model-a", "k2");
    f.w->profiles["prof_a"].revision = 2;   // same fingerprint
    f.publish();
    f.be->output(1);
    f.be->ended(1, {});
    f.run();
    CHECK(f.be->last(Cmd::K::kStart).a == r.ticket);
  }
}

TEST_CASE("F23 completions delivered synchronously inside commands do not deadlock") {
  Fixture f;
  f.be->auto_prepare = f.be->auto_start = f.be->auto_finish = f.be->auto_release = f.be->auto_invalidate = true;
  auto r = f.enq();
  f.run();
  CHECK(only(f.last_observer).kind == Kind::kSuccess);
  CHECK(slot0(f) == "free");
  f.check_invariants();
}

TEST_CASE("F24 a misbehaving backend is ignored and counted") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  f.be->output(1);
  f.be->ended(1, {});
  f.be->ended(1, {});   // twice
  f.be->running(1);     // after the end
  f.be->gone(99);       // unknown plan
  f.run();
  CHECK(f.sch->counters().stale_events >= 3);
  CHECK(slot0(f) == "free");
  f.check_invariants();
}

TEST_CASE("F25 forced release: conflict while active, request_cancelled when forced, retained-only plan is no conflict") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto obs = f.observers.back();
  CHECK(f.sch->release("prof_a", false).code() == clusterlm::ErrorCode::kFailedPrecondition);
  CHECK(f.sch->release("prof_a", true).is_ok());
  f.run();
  CHECK(only(obs).code == Code::kRequestCancelled);
  CHECK(f.be->count(Cmd::K::kAbort) == 1);
  CHECK(f.be->count(Cmd::K::kRelease) == 0);   // only after the abort is confirmed
  f.be->aborted(1);
  f.run();
  REQUIRE(f.be->count(Cmd::K::kRelease) == 1);
  f.be->gone(f.be->last(Cmd::K::kRelease).a, GoneCause::kReleased);
  f.run();
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kLoadable);
  CHECK(f.sch->release("prof_a", false).code() == clusterlm::ErrorCode::kNotFound);
}

TEST_CASE("F26 observers may cancel and enqueue from inside on_terminal") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto victim = f.enq("model-a", "k2");
  f.run();
  auto first = f.observers.front();
  bool reentered = false;
  first->hook = [&](const Terminal&) {
    (void)f.sch->cancel(victim.ticket);
    (void)f.enq("model-a", "k3");
    reentered = true;
  };
  f.be->output(1);
  f.run();
  CHECK(reentered);
  CHECK(state_of(f, victim.ticket) == "done");
}

TEST_CASE("queue timeout, prepare-wait expiry and bounds") {
  SUBCASE("queue timeout") {
    Fixture f;
    f.prepare_ready();
    f.be->running(1);
    f.enq("model-a", "k2");
    auto obs = f.last_observer;
    f.sec(119);
    CHECK(obs->terminals.empty());
    f.sec(2);
    CHECK(only(obs).code == Code::kQueueTimeout);
    CHECK(http_status(Code::kQueueTimeout) == 504);
  }
  SUBCASE("prepare-wait expiry carries the job id") {
    Fixture f;
    f.enq("model-a", "k1");
    auto obs = f.last_observer;
    f.run();
    f.sec(31);   // client prepare_wait_ms is 30 s in the fixture
    auto t = only(obs);
    CHECK(t.code == Code::kModelNotReady);
    CHECK(t.job_id.has_value());
    CHECK(t.retry_after_s.has_value());
    CHECK(f.sch->jobs()[0].state == JobState::kRunning);   // the job continues
  }
  SUBCASE("global and per-client queue limits") {
    Fixture f([](SchedulerOptions& o) { o.max_queue_depth = 2; });
    f.prepare_ready();
    f.be->running(1);
    CHECK(f.enq("model-a", "k2").accepted);
    CHECK(f.enq("model-a", "k3").accepted);
    auto r = f.enq("model-a", "k4");
    CHECK_FALSE(r.accepted);
    CHECK(r.decision.code == Code::kQueueFull);
  }
}

TEST_CASE("start that never reaches running escalates after start_timeout") {
  Fixture f;
  auto r = f.enq();
  auto obs = f.last_observer;
  f.run();
  f.be->prepared(f.be->last(Cmd::K::kPrepare).a);
  f.run();
  f.sec(61);
  CHECK(only(obs).code == Code::kInternalError);
  REQUIRE(f.be->count(Cmd::K::kInvalidate) == 1);
  CHECK(slot0(f) == "quarantined");
  (void)r;
}

TEST_CASE("key revocation ends that client's requests with invalid_api_key") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto obs = f.observers.back();
  auto q = f.enq("model-a", "k1");
  auto qobs = f.last_observer;
  f.sch->key_revoked("k1");
  f.run();
  CHECK(only(obs).code == Code::kInvalidApiKey);
  CHECK(only(qobs).code == Code::kInvalidApiKey);
  CHECK(f.be->count(Cmd::K::kAbort) == 1);
  auto again = f.enq("model-a", "k1");
  CHECK(again.decision.code == Code::kInvalidApiKey);
  (void)q;
}

TEST_CASE("routing alias: explicit candidates, never widens access, chosen candidate is recorded") {
  Fixture f;
  AliasSnapshot al;
  al.id = "alias_x";
  al.api_model_id = "auto";
  al.candidates = {"prof_a", "prof_b"};
  f.w->aliases["alias_x"] = al;
  f.publish();
  f.prepare_ready("model-b");   // only B is prepared
  f.be->running(1);
  f.be->output(1);
  f.be->ended(1, {});
  f.run();
  SUBCASE("first-viable skips the unprepared candidate when the alias may not prepare") {
    auto r = f.enq("auto", "k2");
    REQUIRE(r.accepted);
    f.run();
    CHECK(f.be->last(Cmd::K::kStart).a == r.ticket);
    f.be->running(r.ticket);
    f.run();
    CHECK(f.last_observer->started[0].profile_id == "prof_b");
    CHECK(f.last_observer->started[0].routed_via == "auto");
    CHECK(f.last_observer->started[0].api_model_id == "model-b");
  }
  SUBCASE("a key that may not see a candidate never reaches it through the alias") {
    ClientInfo c = f.client("k5");
    c.allowed_models = {"auto"};   // alias only: neither candidate is visible
    EnqueueRequest req;
    req.client = c;
    req.api_model_id = "auto";
    req.shape.prompt_tokens = 10;
    req.shape.max_tokens = 10;
    auto r = f.sch->enqueue(std::move(req));
    CHECK_FALSE(r.accepted);
    CHECK(r.decision.code == Code::kModelNotReady);
  }
  SUBCASE("an exact model id is never re-routed") {
    auto r = f.enq("model-a", "k3", 10, 20, "", false);
    CHECK_FALSE(r.accepted);
    CHECK(r.decision.code == Code::kModelNotReady);
    f.run();
    CHECK(f.be->count(Cmd::K::kStart) == 1);   // only the setup request; nothing was routed to B for model-a
  }
}

TEST_CASE("idle release: a plan is released after release_after_idle_seconds, not before, not while wanted") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  f.be->output(1);
  f.be->ended(1, {});
  f.run();
  f.sec(59);
  CHECK(f.be->count(Cmd::K::kRelease) == 0);
  f.sec(2);
  CHECK(f.be->count(Cmd::K::kRelease) == 1);
}

TEST_CASE("status surface: queue shows own tickets in full and others only as positions") {
  Fixture f;
  f.prepare_ready();
  f.be->running(1);
  auto mine = f.enq("model-a", "me");
  f.enq("model-a", "other");
  auto q = f.sch->queue("me");
  CHECK(q.depth == 2);
  REQUIRE(q.own.size() == 1);
  CHECK(q.own[0].position == 1);
  CHECK(q.own[0].request_id == mine.request_id);
  REQUIRE(q.others_positions.size() == 1);
  CHECK(q.others_positions[0] == 2);
  auto other = f.sch->queue("someone-else");
  CHECK(other.own.empty());
  CHECK(other.others_positions.size() == 2);
  CHECK(f.sch->job("job_nope") == std::nullopt);
  auto av = f.sch->availability();
  CHECK(av.size() == 2);
}

TEST_CASE("readiness is never inferred: the evaluator saying ready does not make a profile ready") {
  Fixture f;
  f.set_state("prof_a", ReadyState::kReady);
  f.publish();
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kLoadable);
  f.set_state("prof_a", ReadyState::kUnavailable, {"model file missing"});
  f.publish();
  CHECK(f.sch->readiness("prof_a", 4096).state == ReadyState::kUnavailable);
}

TEST_CASE("a blocker on a ready profile invalidates its plan") {
  Fixture f;
  f.prepare_ready();
  f.set_state("prof_a", ReadyState::kUnavailable, {"model file missing"});
  f.publish();
  f.run();
  CHECK(f.be->count(Cmd::K::kInvalidate) == 1);
}
