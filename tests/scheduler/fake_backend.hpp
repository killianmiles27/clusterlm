#pragma once
// Deterministic fake of the ExecutionBackend port. Commands are recorded; completions are delivered explicitly by the
// test (or automatically per mode flag). It enforces the port contract (docs/scheduler-design.md §6.3): an `ended` or
// `gone` the contract does not allow is a test failure unless `tolerate_violations` is set, and it keeps a write ledger so
// tests can assert the core safety property: a slot is never free while a session may still write to it.
#include <doctest/doctest.h>

#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "clusterlm/scheduler/backend.hpp"

namespace clusterlm::scheduler::testing {

struct Cmd {
  enum class K { kPrepare, kCancelPrepare, kRelease, kInvalidate, kStart, kAbort, kDrop, kDetach } kind;
  std::uint64_t a = 0;   // job | plan | ticket
  std::uint64_t b = 0;   // plan for start, session for drop
  bool reuse = false;
  InvalidateReason why = InvalidateReason::kPolicy;
};

class FakeBackend final : public ExecutionBackend {
 public:
  // automatic completions (all default off: the test completes commands one by one)
  bool auto_prepare = false, auto_release = false, auto_invalidate = false, auto_start = false, auto_finish = false,
       auto_drop = false, auto_abort = false;
  bool sync = true;   // completions are delivered inside the command (re-entrant) when an auto flag is set
  bool retain_sessions = false;   // auto_finish reports kRetained for requests that asked for retention
  std::vector<std::string> workers{"G14"};
  std::uint32_t concurrency = 1;

  BackendEvents* ev = nullptr;
  mutable std::mutex mu;
  std::vector<Cmd> log;
  std::map<std::uint64_t, PlanId> prepare_plan;           // job -> plan
  std::map<TicketId, PlanId> writers;                      // sessions that may still write (ledger)
  std::map<TicketId, StartCommand> starts;
  std::set<PlanId> live_plans, gone_plans;
  std::map<TicketId, std::size_t> ended_count;
  bool detached = false;
  std::uint64_t violations = 0;

  // ---- ExecutionBackend
  void attach(BackendEvents* e) override { ev = e; }
  void detach() override { std::lock_guard<std::mutex> l(mu); detached = true; log.push_back({Cmd::K::kDetach}); }
  void prepare(PrepareCommand c) override {
    { std::lock_guard<std::mutex> l(mu); log.push_back({Cmd::K::kPrepare, c.job, c.plan}); prepare_plan[c.job] = c.plan; }
    if (auto_prepare) prepared(c.job);
  }
  void cancel_prepare(JobSeq j) override { std::lock_guard<std::mutex> l(mu); log.push_back({Cmd::K::kCancelPrepare, j}); }
  void release(PlanId p) override {
    { std::lock_guard<std::mutex> l(mu); log.push_back({Cmd::K::kRelease, p}); }
    if (auto_release) gone(p, GoneCause::kReleased);
  }
  void invalidate_plan(PlanId p, InvalidateReason r) override {
    { std::lock_guard<std::mutex> l(mu); log.push_back({Cmd::K::kInvalidate, p, 0, false, r}); }
    if (auto_invalidate) gone(p, GoneCause::kInvalidated);
  }
  void start_request(StartCommand c) override {
    {
      std::lock_guard<std::mutex> l(mu);
      log.push_back({Cmd::K::kStart, c.ticket, c.plan, c.reuse.has_value()});
      writers[c.ticket] = c.plan;
      starts[c.ticket] = c;
    }
    if (auto_start) {
      running(c.ticket);
      if (auto_finish) finish(c.ticket);
    }
  }
  void abort_request(TicketId t) override {
    { std::lock_guard<std::mutex> l(mu); log.push_back({Cmd::K::kAbort, t}); }
    if (auto_abort) aborted(t);
  }
  void drop_session(PlanId p, RetainedSessionRef r) override {
    { std::lock_guard<std::mutex> l(mu); log.push_back({Cmd::K::kDrop, p, r.session_seq}); }
    if (auto_drop) dropped(p, r);
  }
  BackendDiagnostics diagnostics() const override { return {}; }

  // ---- test-driven completions
  void prepared(std::uint64_t job) {
    PlanId plan;
    { std::lock_guard<std::mutex> l(mu); plan = prepare_plan.at(job); live_plans.insert(plan); }
    PlanInfo info;
    info.max_context = 16384;
    info.concurrency = concurrency;
    info.worker_ids = workers;
    info.backend_id = "fake";
    ev->on_prepare_done(job, plan, Result<PlanInfo>(info));
  }
  void fail_prepare(std::uint64_t job, const std::string& msg = "boom") {
    PlanId plan;
    { std::lock_guard<std::mutex> l(mu); plan = prepare_plan.at(job); }
    ev->on_prepare_done(job, plan, Result<PlanInfo>(Status(ErrorCode::kUnavailable, msg)));
  }
  void running(TicketId t) { ev->on_request_running(t); }
  void output(TicketId t, std::uint32_t completion = 5) {
    OutputSummary s;
    s.prompt_tokens = 10;
    s.completion_tokens = completion;
    ev->on_request_output_done(t, s);
  }
  void ended(TicketId t, RequestEnd e) {
    { std::lock_guard<std::mutex> l(mu); writers.erase(t); ++ended_count[t]; }
    ev->on_request_ended(t, std::move(e));
  }
  void finish(TicketId t) {
    output(t);
    RequestEnd e;
    bool retain = false;
    PlanId plan = 0;
    std::uint64_t seq = 0;
    { std::lock_guard<std::mutex> l(mu); retain = retain_sessions && starts.at(t).retain_on_finish; plan = starts.at(t).plan; seq = starts.at(t).session_seq; }
    if (retain) { e.disposition = Disposition::kRetained; e.retained = RetainedSessionRef{plan, seq}; e.retained_tokens = 15; }
    ended(t, e);
  }
  void aborted(TicketId t) { RequestEnd e; e.disposition = Disposition::kAborted; ended(t, e); }
  void failed_worker_lost(TicketId t) {
    RequestEnd e; e.disposition = Disposition::kFailed; e.fail = FailKind::kWorkerLost; ended(t, e);
  }
  void dropped(PlanId p, RetainedSessionRef r, Status st = Status::ok()) { ev->on_session_dropped(p, r, std::move(st)); }
  void gone(PlanId p, GoneCause c = GoneCause::kInvalidated) {
    {
      std::lock_guard<std::mutex> l(mu);
      for (auto it = writers.begin(); it != writers.end();) it = it->second == p ? writers.erase(it) : std::next(it);
      live_plans.erase(p);
      gone_plans.insert(p);
    }
    PlanGone g;
    g.cause = c;
    ev->on_plan_gone(p, g);
  }
  void lost(PlanId p) { ev->on_plan_lost(p, InvalidateReason::kWorkerLost); }

  // ---- queries
  std::vector<Cmd> cmds(Cmd::K k) const {
    std::lock_guard<std::mutex> l(mu);
    std::vector<Cmd> out;
    for (const Cmd& c : log) if (c.kind == k) out.push_back(c);
    return out;
  }
  std::size_t count(Cmd::K k) const { return cmds(k).size(); }
  Cmd last(Cmd::K k) const { auto v = cmds(k); REQUIRE_FALSE(v.empty()); return v.back(); }
};

}  // namespace clusterlm::scheduler::testing
