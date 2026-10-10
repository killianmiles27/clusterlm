#pragma once
// The scheduler: admission, a bounded fair queue, async prepare/release jobs, per-plan generation slots, warm retention,
// cancellation and Worker-loss handling. Design: docs/scheduler-design.md. Contract: docs/interfaces/scheduler-admission-v1.md.
//
// Threading: one mutex guards all state. Decisions are made under it and turned into *actions* that a single dispatcher
// executes in order with no lock held (backend commands), and *deliveries* that a delivery thread executes with no lock
// held (observer callbacks). In SchedulerOptions::Threading::kManual no threads exist and tests drive run_pending().
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "clusterlm/scheduler/admission.hpp"
#include "clusterlm/scheduler/backend.hpp"
#include "clusterlm/scheduler/clock.hpp"
#include "clusterlm/scheduler/fair_queue.hpp"
#include "clusterlm/scheduler/status.hpp"

namespace clusterlm::scheduler {

// ---- inputs pushed by the host (never pulled under the lock) ---------------------------------------------------------

struct WorldSnapshot {
  std::uint64_t version = 0;
  std::map<std::string, ProfileSnapshot> profiles;            // by profile id
  std::map<std::string, AliasSnapshot> aliases;               // by alias id
  std::map<std::string, ReadinessView> readiness;             // "prof_x@4096": evaluator output per offered context
  std::map<std::string, ResourceView> resources;              // by profile id: may the required Workers be used now?
  std::map<std::string, ClientInfo> clients;                  // latest key state (revocation, models) by client id
  std::vector<MachineStatus> machines;                        // for GET /machines
  static std::string key(const std::string& profile_id, std::uint32_t ctx) { return profile_id + "@" + std::to_string(ctx); }
};

enum class WorkerEventKind : std::uint8_t { kLost, kLocalUserActive, kPolicyForbids, kLeaseRevoked, kBack };
struct WorkerEvent {
  std::string worker_id;                                      // matches PlanInfo::worker_ids
  WorkerEventKind kind = WorkerEventKind::kLost;
  std::string reason;
};

// ---- ticket observer -------------------------------------------------------------------------------------------------

enum class EndCause : std::uint8_t {
  kNone, kClientGone, kClientCancel, kQueueTimeout, kPrepareWaitExpired, kPrepareFailed, kKeyRevoked, kShutdown,
  kManagement, kWorkerLost, kBackendError, kRejectedAtDequeue, kCompleted,
};

struct StartedInfo {
  std::string request_id, profile_id, api_model_id, routed_via, model_identity, backend_id;
};

struct Terminal {
  enum class Kind : std::uint8_t { kSuccess, kError, kSilent } kind = Kind::kError;
  std::string request_id;
  Code code = Code::kInternalError;
  std::string message;
  std::vector<std::string> reasons;
  std::optional<std::string> job_id;
  std::optional<std::uint32_t> retry_after_s;
  EndCause cause = EndCause::kNone;
  OutputSummary summary;                                       // success only
  StartedInfo info;                                            // x_clusterlm carriage
};

// Called from the delivery thread (or run_pending) with no scheduler lock held. May call back into the scheduler.
class TicketObserver {
 public:
  virtual ~TicketObserver() = default;
  virtual void on_started(const StartedInfo&) {}
  virtual void on_terminal(const Terminal&) = 0;               // exactly once per accepted ticket
};

// ---- API structs -----------------------------------------------------------------------------------------------------

struct EnqueueRequest {
  ClientInfo client;
  std::string api_model_id;
  RequestShape shape;
  std::string conversation_hint;                              // raw session_id/user; hashed immediately, never stored
  std::shared_ptr<GenerationInput> input;
  std::shared_ptr<TokenSink> sink;
  std::shared_ptr<TicketObserver> observer;
};

struct EnqueueResult {
  bool accepted = false;
  TicketId ticket = 0;
  std::string request_id;
  Decision decision;                                          // rejection details (accepted: kind admit|queue)
};

struct SchedulerOptions {
  enum class Threading : std::uint8_t { kThreaded, kManual } threading = Threading::kThreaded;
  std::shared_ptr<ClockSource> clock;                          // default: steady clock
  std::uint32_t max_queue_depth = 32, max_queue_per_client = 8;       // hard caps 256 / 64
  Duration queue_timeout = std::chrono::seconds(120);                 // hard cap 600 s
  Duration prepare_wait_cap = std::chrono::seconds(60);               // API clients; interactive waits unbounded
  Duration cancel_timeout = std::chrono::seconds(10);
  Duration start_timeout = std::chrono::seconds(60);
  Duration invalidate_timeout = std::chrono::seconds(30);
  Duration release_timeout = std::chrono::seconds(30);
  Duration shutdown_grace = std::chrono::seconds(10);
  Duration max_slot_seconds = std::chrono::seconds(300);
  std::uint32_t max_prepared_profiles = 1;
  std::uint32_t max_job_history = 64;
  Duration job_history_age = std::chrono::hours(1);
  bool priorities_enabled = false;
  AdmissionPolicy admission;
};

struct ShutdownReport {
  bool clean = true;
  std::vector<std::string> problems;
};

// Slot/plan view for invariant checks and diagnostics. Counts and ids only.
struct DebugSnapshot {
  struct SlotView { std::string state; TicketId ticket = 0; };
  struct PlanView { PlanId id = 0; std::string profile_id, state; std::uint32_t context = 0; std::vector<SlotView> slots; };
  std::vector<PlanView> plans;
  std::size_t queued = 0, active = 0, jobs_running = 0, jobs_queued = 0;
  std::map<TicketId, std::string> ticket_states;
};

// What the readiness evaluator needs from the scheduler for one (profile, context class).
struct PlanOverlay {
  bool plan_ready = false;
  std::uint32_t plan_context = 0, profile_revision = 0;
  std::string plan_fingerprint;
  std::uint32_t active_requests = 0, concurrency_limit = 0;
  std::optional<std::string> job_id;
  std::string job_state, job_phase, last_error;
};

class Scheduler : public StatusProvider {
 public:
  Scheduler(std::shared_ptr<ExecutionBackend> backend, SchedulerOptions options);
  ~Scheduler() override;
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  // ---- requests
  EnqueueResult enqueue(EnqueueRequest request);
  Status cancel(TicketId ticket, EndCause cause = EndCause::kClientCancel);
  Status cancel_request(const std::string& request_id, EndCause cause = EndCause::kClientCancel);

  // ---- management (scope checks are the serving layer's)
  Result<std::string> prepare(const std::string& profile_id, std::uint32_t context_tokens = 0);  // idempotent per plan
  Status cancel_job(const std::string& job_id);
  Status release(const std::string& profile_id, bool force);

  // ---- world
  void update_world(std::shared_ptr<const WorldSnapshot> world);
  void worker_event(const WorkerEvent& event);
  void key_revoked(const std::string& client_id);

  // ---- readiness
  ReadinessView readiness(const std::string& profile_id, std::uint32_t context_tokens) const;
  PlanOverlay overlay(const std::string& profile_id, std::uint32_t context_tokens) const;

  // ---- lifecycle
  // Manual mode only: runs queued backend actions and deliveries until none remain; returns how many ran.
  std::size_t run_pending();
  // Begins shutdown (idempotent). Threaded mode: waits for completion up to the documented bound.
  // Manual mode: returns immediately; drive with run_pending() + clock advance; poll shutdown_done().
  ShutdownReport shutdown();
  bool shutdown_done() const;

  // ---- StatusProvider
  std::vector<MachineStatus> machines() const override;
  QueueStatus queue(const std::string& client_id) const override;
  std::vector<JobStatus> jobs() const override;
  std::optional<JobStatus> job(const std::string& job_id) const override;
  std::vector<AvailabilityEntry> availability() const override;
  SchedulerCounters counters() const override;

  DebugSnapshot debug_snapshot() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace clusterlm::scheduler
