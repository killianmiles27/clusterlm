#pragma once
#include <doctest/doctest.h>

#include <set>

#include "clusterlm/scheduler/scheduler.hpp"
#include "fake_backend.hpp"

namespace clusterlm::scheduler::testing {

struct CapturingObserver final : TicketObserver {
  std::vector<Terminal> terminals;
  std::vector<StartedInfo> started;
  std::function<void(const Terminal&)> hook;
  void on_started(const StartedInfo& i) override { started.push_back(i); }
  void on_terminal(const Terminal& t) override {
    terminals.push_back(t);
    if (hook) hook(t);
  }
};

struct CountingSink final : TokenSink {
  std::atomic<int> tokens_after_terminal{0};
  std::atomic<int> total{0};
  std::atomic<bool> terminal_seen{false};
  void tokens(std::span<const std::int32_t> ids, std::string_view) override {
    total += static_cast<int>(ids.size());
    if (terminal_seen) tokens_after_terminal += static_cast<int>(ids.size());
  }
};

inline ProfileSnapshot make_profile(const std::string& id, const std::string& api) {
  ProfileSnapshot p;
  p.id = id;
  p.revision = 1;
  p.api_model_id = api;
  p.exposed_api = true;
  p.exposed_lan = true;
  p.plan_fingerprint = "fp-" + id;
  p.model_identity = "fam/q4/" + id;
  p.backend_id = "fake";
  p.context_max_tokens = 8192;
  p.context_default_tokens = 4096;
  p.offered_contexts = {4096, 8192};
  p.release_after_idle_seconds = 60;
  p.max_sessions = 1;
  return p;
}

struct Fixture {
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::shared_ptr<FakeBackend> be = std::make_shared<FakeBackend>();
  std::unique_ptr<Scheduler> sch;
  std::shared_ptr<WorldSnapshot> w = std::make_shared<WorldSnapshot>();
  std::vector<std::shared_ptr<CapturingObserver>> observers;

  explicit Fixture(std::function<void(SchedulerOptions&)> tweak = {}, bool two_profiles = true) {
    SchedulerOptions o;
    o.threading = SchedulerOptions::Threading::kManual;
    o.clock = clock;
    if (tweak) tweak(o);
    w->profiles["prof_a"] = make_profile("prof_a", "model-a");
    if (two_profiles) w->profiles["prof_b"] = make_profile("prof_b", "model-b");
    publish();
    sch = std::make_unique<Scheduler>(be, o);
    sch->update_world(w);
  }

  void set_state(const std::string& profile, ReadyState st, std::vector<std::string> reasons = {}) {
    for (std::uint32_t ctx : {4096U, 8192U}) {
      ReadinessView v;
      v.state = st;
      v.reasons = reasons;
      w->readiness[WorldSnapshot::key(profile, ctx)] = v;
    }
  }
  void publish() {
    for (auto& [id, p] : w->profiles) {
      (void)p;
      if (!w->readiness.count(WorldSnapshot::key(id, 4096))) set_state(id, ReadyState::kLoadable);
    }
    ++w->version;
    if (sch) sch->update_world(std::make_shared<WorldSnapshot>(*w));
  }

  ClientInfo client(const std::string& id = "k1") {
    ClientInfo c;
    c.id = id;
    c.allowed_models = {"*"};
    c.limits.prepare_wait_ms = 30'000;   // tests of the fast-fail path set this to 0
    c.limits.can_see_machine_names = true;
    c.limits.max_queued = 64;
    c.limits.max_concurrent = 64;
    return c;
  }

  EnqueueResult enq(const std::string& model = "model-a", const std::string& cid = "k1", std::uint32_t prompt = 10,
                    std::uint32_t max_tokens = 20, const std::string& hint = "", bool wait = true) {
    EnqueueRequest r;
    r.client = client(cid);
    if (!wait) r.client.limits.prepare_wait_ms = 0;
    r.api_model_id = model;
    r.shape.prompt_tokens = prompt;
    r.shape.max_tokens = max_tokens;
    r.conversation_hint = hint;
    auto obs = std::make_shared<CapturingObserver>();
    observers.push_back(obs);
    r.observer = obs;
    last_observer = obs;
    return sch->enqueue(std::move(r));
  }
  std::shared_ptr<CapturingObserver> last_observer;

  void run() { sch->run_pending(); }
  void advance(std::chrono::milliseconds d) { clock->advance(d); run(); }
  void sec(int n) { advance(std::chrono::seconds(n)); }

  // Drives a profile from loadable to a ready plan via the fake; returns the plan id.
  PlanId prepare_ready(const std::string& model = "model-a") {
    auto r = enq(model);
    REQUIRE(r.accepted);
    run();
    auto prep = be->cmds(Cmd::K::kPrepare);
    REQUIRE_FALSE(prep.empty());
    be->prepared(prep.back().a);
    run();
    return prep.back().b;
  }

  // Safety invariants P1/P2 of the design: a slot is never free while a session may still write to it.
  void check_invariants() {
    auto d = sch->debug_snapshot();
    std::lock_guard<std::mutex> l(be->mu);
    for (const auto& [ticket, plan] : be->writers) {
      bool found = false;
      for (const auto& p : d.plans) {
        if (p.id != plan) continue;
        for (const auto& s : p.slots)
          if (s.ticket == ticket) { found = true; CHECK_MESSAGE(s.state != "free", "slot freed while ticket may write"); }
      }
      // A writer whose plan is gone from the scheduler is a violation too, unless the fake already saw `gone`.
      const bool plan_gone = be->gone_plans.count(plan) != 0;
      CHECK_MESSAGE((found || plan_gone), "writer without a slot");
    }
    for (const auto& p : d.plans) {
      std::size_t active = 0;
      for (const auto& s : p.slots) active += (s.state == "reserved" || s.state == "dropping" || s.state == "quarantined");
      CHECK(active <= p.slots.size());
    }
    CHECK(d.jobs_running <= 1);
  }
};

}  // namespace clusterlm::scheduler::testing
