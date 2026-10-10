// Threaded stress: the real dispatcher and delivery threads, many caller threads, and a backend whose completions arrive
// on their own threads after random delays. Intended to be run under ThreadSanitizer (see docs/scheduler-design.md §8.4:
// cmake -DCMAKE_CXX_FLAGS=-fsanitize=thread). Without TSAN it still checks the functional invariants: exactly one terminal
// per accepted ticket, no token after a terminal, a clean shutdown.
#include <doctest/doctest.h>

#include <atomic>
#include <random>
#include <thread>

#include "fixture.hpp"

using namespace clusterlm;
using namespace clusterlm::scheduler;
using namespace clusterlm::scheduler::testing;
using namespace std::chrono_literals;

namespace {

class ThreadedBackend final : public ExecutionBackend {
 public:
  void attach(BackendEvents* e) override { ev_ = e; }
  void detach() override {
    std::vector<std::thread> ts;
    { std::lock_guard<std::mutex> l(mu_); ts.swap(threads_); }
    for (auto& t : ts) t.join();
    detached = true;
  }
  void prepare(PrepareCommand c) override {
    spawn([this, c] {
      nap();
      if (cancelled(c.job)) { ev_->on_prepare_done(c.job, c.plan, Result<PlanInfo>(Status(ErrorCode::kCancelled, "cancelled"))); return; }
      { std::lock_guard<std::mutex> l(mu_); plan_tickets_[c.plan] = 0; }
      PlanInfo info;
      info.concurrency = 1;
      info.worker_ids = {"G14"};
      ev_->on_prepare_done(c.job, c.plan, Result<PlanInfo>(info));
    });
  }
  void cancel_prepare(JobSeq j) override { std::lock_guard<std::mutex> l(mu_); cancelled_jobs_.insert(j); }
  void release(PlanId p) override { spawn([this, p] { retire(p, GoneCause::kReleased); }); }
  void invalidate_plan(PlanId p, InvalidateReason) override {
    spawn([this, p] { retire(p, GoneCause::kInvalidated); });
  }
  void start_request(StartCommand c) override {
    { std::lock_guard<std::mutex> l(mu_); ++plan_tickets_[c.plan]; }
    spawn([this, c] {
      nap();
      auto aborted_now = [&] { std::lock_guard<std::mutex> l(mu_); return aborted_.count(c.ticket) != 0; };
      auto end = [&](RequestEnd e) {
        ev_->on_request_ended(c.ticket, std::move(e));
        std::lock_guard<std::mutex> l(mu_);
        --plan_tickets_[c.plan];
      };
      if (aborted_now()) { RequestEnd e; e.disposition = Disposition::kAborted; end(e); return; }
      ev_->on_request_running(c.ticket);
      const std::int32_t ids[3] = {1, 2, 3};
      for (int i = 0; i < 4; ++i) {
        nap();
        if (aborted_now()) { RequestEnd e; e.disposition = Disposition::kAborted; end(e); return; }
        c.sink->tokens(ids, "abc");
      }
      OutputSummary s;
      s.completion_tokens = 12;
      ev_->on_request_output_done(c.ticket, s);
      nap();
      end(RequestEnd{});
    });
  }
  void abort_request(TicketId t) override { std::lock_guard<std::mutex> l(mu_); aborted_.insert(t); }
  void drop_session(PlanId p, RetainedSessionRef r) override {
    spawn([this, p, r] { nap(); ev_->on_session_dropped(p, r, Status::ok()); });
  }
  BackendDiagnostics diagnostics() const override { return {}; }
  std::atomic<bool> detached{false};

 private:
  void spawn(std::function<void()> f) {
    std::lock_guard<std::mutex> l(mu_);
    threads_.emplace_back(std::move(f));
  }
  void nap() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::this_thread::sleep_for(std::chrono::microseconds(rng() % 800));
  }
  bool cancelled(JobSeq j) { std::lock_guard<std::mutex> l(mu_); return cancelled_jobs_.count(j) != 0; }
  void retire(PlanId p, GoneCause c) {
    // E3: every request of the plan has ended before the plan is gone.
    for (int i = 0; i < 20000; ++i) {
      { std::lock_guard<std::mutex> l(mu_); if (plan_tickets_[p] == 0) break; }
      std::this_thread::sleep_for(100us);
    }
    nap();
    PlanGone g;
    g.cause = c;
    ev_->on_plan_gone(p, g);
  }
  BackendEvents* ev_ = nullptr;
  std::mutex mu_;
  std::vector<std::thread> threads_;
  std::set<JobSeq> cancelled_jobs_;
  std::set<TicketId> aborted_;
  std::map<PlanId, int> plan_tickets_;
};

struct Rec final : TicketObserver {
  std::atomic<int> terminals{0};
  std::shared_ptr<CountingSink> sink;
  void on_terminal(const Terminal&) override {
    ++terminals;
    sink->terminal_seen = true;
  }
};

}  // namespace

TEST_CASE("threaded stress: callers, dispatcher, delivery and backend threads race without breaking invariants") {
  auto be = std::make_shared<ThreadedBackend>();
  SchedulerOptions o;
  o.threading = SchedulerOptions::Threading::kThreaded;
  o.queue_timeout = 3s;
  o.cancel_timeout = 300ms;
  o.invalidate_timeout = 500ms;
  o.release_timeout = 2s;
  o.shutdown_grace = 1s;
  o.start_timeout = 2s;
  o.max_queue_depth = 64;
  o.max_queue_per_client = 64;
  auto sch = std::make_unique<Scheduler>(be, o);
  auto w = std::make_shared<WorldSnapshot>();
  for (const char* id : {"prof_a", "prof_b"}) {
    w->profiles[id] = make_profile(id, std::string("model-") + (id[5] == 'a' ? "a" : "b"));
    for (std::uint32_t ctx : {4096U, 8192U}) {
      ReadinessView v;
      v.state = ReadyState::kLoadable;
      w->readiness[WorldSnapshot::key(id, ctx)] = v;
    }
  }
  sch->update_world(w);

  std::mutex rm;
  std::vector<std::shared_ptr<Rec>> recs;
  std::vector<TicketId> tickets;
  std::atomic<int> accepted{0};
  std::vector<std::thread> callers;
  for (int tno = 0; tno < 6; ++tno) {
    callers.emplace_back([&, tno] {
      std::mt19937 rng(static_cast<unsigned>(tno) * 7919U + 1U);
      for (int i = 0; i < 150; ++i) {
        const auto op = rng() % 10;
        if (op < 6) {
          EnqueueRequest r;
          r.client.id = "c" + std::to_string(rng() % 4);
          r.client.allowed_models = {"*"};
          r.client.limits.prepare_wait_ms = 30'000;
          r.client.limits.max_queued = 64;
          r.client.limits.max_concurrent = 64;
          r.api_model_id = rng() % 2 ? "model-a" : "model-b";
          r.shape.prompt_tokens = 20;
          r.shape.max_tokens = 20;
          r.conversation_hint = rng() % 2 ? "conv" : "";
          auto rec = std::make_shared<Rec>();
          rec->sink = std::make_shared<CountingSink>();
          r.observer = rec;
          r.sink = rec->sink;
          auto res = sch->enqueue(std::move(r));
          if (res.accepted) {
            ++accepted;
            std::lock_guard<std::mutex> l(rm);
            recs.push_back(rec);
            tickets.push_back(res.ticket);
          }
        } else if (op < 8) {
          TicketId t = 0;
          { std::lock_guard<std::mutex> l(rm); if (!tickets.empty()) t = tickets[rng() % tickets.size()]; }
          if (t) (void)sch->cancel(t, rng() % 2 ? EndCause::kClientGone : EndCause::kClientCancel);
        } else if (op == 8) {
          (void)sch->readiness("prof_a", 4096);
          (void)sch->queue("c1");
          (void)sch->jobs();
          (void)sch->availability();
          (void)sch->debug_snapshot();
        } else if (rng() % 8 == 0) {
          sch->worker_event({"G14", WorkerEventKind::kLocalUserActive, "x"});
        } else {
          (void)sch->release(rng() % 2 ? "prof_a" : "prof_b", rng() % 2 == 0);
        }
        if (rng() % 16 == 0) std::this_thread::sleep_for(1ms);
      }
    });
  }
  for (auto& t : callers) t.join();
  // Let the system settle, then shut down.
  auto report = sch->shutdown();
  CHECK(report.clean);
  CHECK(be->detached);
  CHECK(sch->shutdown_done());
  std::lock_guard<std::mutex> l(rm);
  CHECK(static_cast<int>(recs.size()) == accepted.load());
  for (auto& r : recs) {
    CHECK(r->terminals.load() == 1);
    CHECK(r->sink->tokens_after_terminal.load() == 0);
  }
  CHECK(accepted.load() > 20);
}
