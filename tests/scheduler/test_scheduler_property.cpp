// Model-based property test: random sequences of API calls, backend completions, Worker events, world edits and clock
// movements against the scheduler (manual mode, fake backend). After every step the safety invariants of
// docs/scheduler-design.md §8.2 must hold; after a drain every ticket must have exactly one terminal (liveness).
#include <doctest/doctest.h>

#include <cstdlib>
#include <random>

#include "fixture.hpp"

using namespace clusterlm::scheduler;
using namespace clusterlm::scheduler::testing;
using namespace std::chrono_literals;

namespace {

struct Driver {
  Fixture f;
  std::mt19937_64 rng;
  std::size_t seen = 0;
  std::set<std::uint64_t> pending_prepare, cancelled_prepare;
  struct Run { PlanId plan; int stage = 0; bool aborted = false; };   // stage: 0 started, 1 running, 2 output done
  std::map<TicketId, Run> runs;
  std::set<PlanId> pending_release, pending_invalidate;
  std::map<PlanId, RetainedSessionRef> pending_drop;
  std::vector<TicketId> accepted;
  std::map<TicketId, std::shared_ptr<CapturingObserver>> obs;
  std::uint64_t fp_counter = 0;

  explicit Driver(std::uint64_t seed) : f([](SchedulerOptions& o) { o.max_prepared_profiles = 1; }), rng(seed) {
    f.be->retain_sessions = true;
  }

  std::uint64_t pick(std::uint64_t n) { return n ? rng() % n : 0; }

  void scan() {
    std::lock_guard<std::mutex> l(f.be->mu);
    for (; seen < f.be->log.size(); ++seen) {
      const Cmd& c = f.be->log[seen];
      switch (c.kind) {
        case Cmd::K::kPrepare: pending_prepare.insert(c.a); break;
        case Cmd::K::kCancelPrepare: cancelled_prepare.insert(c.a); break;
        case Cmd::K::kRelease: pending_release.insert(c.a); break;
        case Cmd::K::kInvalidate: pending_invalidate.insert(c.a); break;
        case Cmd::K::kStart: runs[c.a] = Run{c.b, 0, false}; break;
        case Cmd::K::kAbort: if (runs.count(c.a)) runs[c.a].aborted = true; break;
        case Cmd::K::kDrop: pending_drop[c.a] = RetainedSessionRef{c.a, c.b}; break;
        case Cmd::K::kDetach: break;
      }
    }
  }

  bool has_pending() const {
    return !pending_prepare.empty() || !runs.empty() || !pending_release.empty() || !pending_invalidate.empty() ||
           !pending_drop.empty();
  }

  void complete_one() {
    scan();
    std::vector<int> choices;
    if (!pending_prepare.empty()) choices.push_back(0);
    if (!runs.empty()) choices.push_back(1);
    if (!pending_release.empty()) choices.push_back(2);
    if (!pending_invalidate.empty()) choices.push_back(3);
    if (!pending_drop.empty()) choices.push_back(4);
    if (choices.empty()) return;
    switch (choices[pick(choices.size())]) {
      case 0: {
        auto it = pending_prepare.begin();
        std::advance(it, static_cast<long>(pick(pending_prepare.size())));
        const std::uint64_t job = *it;
        pending_prepare.erase(it);
        if (cancelled_prepare.count(job) || pick(5) == 0) f.be->fail_prepare(job);
        else f.be->prepared(job);
        break;
      }
      case 1: {
        auto it = runs.begin();
        std::advance(it, static_cast<long>(pick(runs.size())));
        const TicketId t = it->first;
        Run& r = it->second;
        if (r.aborted) { runs.erase(it); f.be->aborted(t); break; }
        if (pick(25) == 0) { runs.erase(it); f.be->failed_worker_lost(t); break; }
        if (r.stage == 0) { r.stage = 1; f.be->running(t); }
        else if (r.stage == 1) { r.stage = 2; f.be->output(t); }
        else { runs.erase(it); f.be->finish(t); }
        break;
      }
      case 2: {
        auto it = pending_release.begin();
        std::advance(it, static_cast<long>(pick(pending_release.size())));
        PlanId p = *it;
        pending_release.erase(it);
        f.be->gone(p, GoneCause::kReleased);
        for (auto r = runs.begin(); r != runs.end();) r = r->second.plan == p ? runs.erase(r) : std::next(r);
        break;
      }
      case 3: {
        auto it = pending_invalidate.begin();
        std::advance(it, static_cast<long>(pick(pending_invalidate.size())));
        PlanId p = *it;
        pending_invalidate.erase(it);
        for (auto r = runs.begin(); r != runs.end();) r = r->second.plan == p ? runs.erase(r) : std::next(r);
        f.be->gone(p, GoneCause::kInvalidated);
        break;
      }
      case 4: {
        auto it = pending_drop.begin();
        std::advance(it, static_cast<long>(pick(pending_drop.size())));
        auto d = *it;
        pending_drop.erase(it);
        f.be->dropped(d.first, d.second);
        break;
      }
    }
  }

  void step() {
    switch (pick(12)) {
      case 0: case 1: case 2: {
        const std::string client = "c" + std::to_string(pick(3));
        const std::string model = pick(2) ? "model-a" : "model-b";
        const std::string hint = pick(2) ? "h" + std::to_string(pick(2)) : "";
        auto r = f.enq(model, client, static_cast<std::uint32_t>(10 + pick(5000)), static_cast<std::uint32_t>(1 + pick(3000)), hint,
                       pick(4) != 0);
        if (r.accepted) { accepted.push_back(r.ticket); obs[r.ticket] = f.last_observer; }
        break;
      }
      case 3:
        if (!accepted.empty()) (void)f.sch->cancel(accepted[pick(accepted.size())], static_cast<EndCause>(1 + pick(2)));
        else (void)f.sch->cancel(999);
        break;
      case 4: case 5: case 6: case 7: complete_one(); break;
      case 8: {
        WorkerEvent ev{"G14", pick(3) == 0 ? WorkerEventKind::kLocalUserActive : WorkerEventKind::kLost, "x"};
        if (pick(4) == 0) ev.kind = WorkerEventKind::kBack;
        f.sch->worker_event(ev);
        break;
      }
      case 9: {
        const std::string prof = pick(2) ? "prof_a" : "prof_b";
        const std::uint64_t what = pick(8);
        if (what < 2) f.set_state(prof, ReadyState::kCompatible, {"G14 is in use"});
        else if (what < 6) f.set_state(prof, ReadyState::kLoadable);
        else if (what == 6) f.w->profiles[prof].plan_fingerprint = "fp-" + prof + "-" + std::to_string(++fp_counter);
        else f.set_state(prof, ReadyState::kUnavailable, {"model missing"});
        f.publish();
        break;
      }
      case 10: {
        const std::uint64_t which = pick(5);
        if (which == 0) (void)f.sch->prepare(pick(2) ? "prof_a" : "prof_b", pick(2) ? 4096 : 8192);
        else if (which == 1 && !f.sch->jobs().empty()) { auto j = f.sch->jobs(); (void)f.sch->cancel_job(j[pick(j.size())].job_id); }
        else if (which == 2) (void)f.sch->release(pick(2) ? "prof_a" : "prof_b", pick(2) == 0);
        else if (which == 3) f.sch->key_revoked("c" + std::to_string(pick(3)));
        else (void)f.sch->release("prof_a", false);
        break;
      }
      case 11:
        if (pick(20) == 0) f.clock->set(f.clock->now() - std::chrono::seconds(pick(300)));
        else f.clock->advance(std::chrono::milliseconds(pick(3) == 0 ? pick(700'000) : pick(3000)));
        break;
    }
    f.run();
    f.check_invariants();
  }

  void drain() {
    f.set_state("prof_a", ReadyState::kLoadable);
    f.set_state("prof_b", ReadyState::kLoadable);
    f.publish();
    for (int i = 0; i < 3000; ++i) {
      f.run();
      scan();
      if (has_pending()) { complete_one(); f.run(); continue; }
      auto d = f.sch->debug_snapshot();
      if (d.queued == 0 && d.active == 0 && d.jobs_running == 0 && d.jobs_queued == 0) return;
      f.clock->advance(30s);
      f.run();
    }
  }
};

}  // namespace

TEST_CASE("property: safety invariants hold and every accepted ticket ends exactly once") {
  const char* env = std::getenv("CLUSTERLM_PROPERTY_SEEDS");
  const std::uint64_t seeds = env ? std::strtoull(env, nullptr, 10) : 200;
  std::map<std::string, std::uint64_t> stats;
  for (std::uint64_t seed = 1; seed <= seeds; ++seed) {
    INFO("seed " << seed);
    Driver d(seed);
    for (int i = 0; i < 300; ++i) {
      d.step();
    }
    d.drain();
    d.f.check_invariants();
    for (const Cmd& c : d.f.be->log) ++stats["cmd" + std::to_string(static_cast<int>(c.kind))];
    for (TicketId t : d.accepted)
      for (const auto& term : d.obs[t]->terminals) ++stats[term.kind == Terminal::Kind::kSuccess ? std::string("ok") : term.kind == Terminal::Kind::kSilent ? std::string("silent") : std::string(to_string(term.code))];
    const auto sc = d.f.sch->counters();
    stats["stale"] += sc.stale_events;
    stats["stuck"] += sc.stuck_invalidations;
    stats["swaps"] += sc.swaps;
    auto snap = d.f.sch->debug_snapshot();
    CHECK_MESSAGE(snap.queued == 0, "seed " << seed << ": queued tickets remain after drain");
    CHECK_MESSAGE(snap.active == 0, "seed " << seed << ": active tickets remain after drain");
    for (TicketId t : d.accepted) {
      const std::size_t n = d.obs[t]->terminals.size();
      CHECK_MESSAGE(n == 1, "seed " << seed << ": ticket " << t << " has " << n << " terminals");
    }
    for (const auto& p : snap.plans)
      for (const auto& s : p.slots)
        CHECK_MESSAGE((s.state == "free" || s.state == "retained"), "seed " << seed << ": slot " << s.state << " after drain");
  }
  if (std::getenv("CLUSTERLM_PROPERTY_VERBOSE"))
    for (const auto& [k, v] : stats) std::printf("  %-24s %llu\n", k.c_str(), static_cast<unsigned long long>(v));
}
