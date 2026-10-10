#pragma once
// The port between the scheduler and whatever executes plans (the Coordinator adapter in production, FakeBackend in
// tests). See docs/scheduler-design.md §6. Contract clauses E1-E7 are normative and are enforced by the test fake.
//
//   E1 non-blocking commands, invoked from one thread (the scheduler's dispatcher)
//   E2 exactly-once completions; a failed/cancelled prepare has released every partial lease before on_prepare_done
//   E3 per-entity ordering: running < output_done < ended per ticket; ended/dropped < plan_gone per plan
//   E4 abort valid in any phase; kAborted is reported only after the domains confirmed (no further writes)
//   E5 invalidate_plan is idempotent; on_plan_gone only once Host domains are destroyed (late, never early)
//   E6 no exceptions across the port; lost stage/Worker => kFailed/kWorkerLost, other => kBackendError
//   E7 events carry counts, ids and status text only: no prompt, token ids, paths or addresses
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/scheduler/types.hpp"

namespace clusterlm::scheduler {

using TicketId = std::uint64_t;
using PlanId = std::uint64_t;
using JobSeq = std::uint64_t;

// Opaque to the scheduler: prompt tokens, sampling parameters, stop tokens, tool schema live behind this type in the
// serving/adapter library. The scheduler can move the pointer; it cannot read, log or serialize what it points to.
struct GenerationInput;

// Receives generated tokens. Implemented by the serving layer; called with the ticket's gate lock held, so it must not
// block for long, and it MAY call Scheduler::cancel (client gone on write failure).
class TokenSink {
 public:
  virtual ~TokenSink() = default;
  virtual void tokens(std::span<const std::int32_t> ids, std::string_view text) = 0;
};

// Gate between backend token threads and the serving sink: once closed (which happens before the terminal response is
// delivered) nothing more reaches the sink, so no token can follow a final response.
class GatedSink {
 public:
  explicit GatedSink(std::shared_ptr<TokenSink> sink) : sink_(std::move(sink)) {}
  void tokens(std::span<const std::int32_t> ids, std::string_view text) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_ || !sink_) return;
    sink_->tokens(ids, text);
  }
  void close() {
    std::lock_guard<std::mutex> lock(mu_);  // waits for an in-progress delivery
    closed_ = true;
  }
  bool closed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
  }

 private:
  mutable std::mutex mu_;
  bool closed_ = false;
  std::shared_ptr<TokenSink> sink_;
};

// Isolation key of retained session state: same client AND same conversation hint (compared as a keyed digest; the raw
// hint, which may identify a person, is never stored by the scheduler).
struct RetentionKey {
  std::string client_id;
  std::uint64_t hint_lo = 0, hint_hi = 0;
  bool empty() const { return hint_lo == 0 && hint_hi == 0; }
  friend bool operator==(const RetentionKey& a, const RetentionKey& b) {
    return a.client_id == b.client_id && a.hint_lo == b.hint_lo && a.hint_hi == b.hint_hi;
  }
};

struct RetainedSessionRef {
  PlanId plan = 0;
  std::uint64_t session_seq = 0;
  friend bool operator==(const RetainedSessionRef& a, const RetainedSessionRef& b) {
    return a.plan == b.plan && a.session_seq == b.session_seq;
  }
};

struct PlanInfo {
  std::uint32_t max_context = 0;
  std::uint32_t concurrency = 1;
  bool per_client_isolation = false;
  std::vector<std::string> worker_ids;   // Worker display names or ids the plan holds leases on
  std::string backend_id, model_identity;
};

struct PrepareProgressView {
  std::string phase;                     // father-domains | provisioning | node-preparing | node-ready
  std::optional<double> percent;
  std::optional<double> eta_seconds;     // estimate
  std::uint64_t bytes_sent = 0, bytes_total = 0;
  std::uint32_t objects_sealed = 0, objects_total = 0;
};

struct PrepareCommand {
  JobSeq job = 0;
  PlanId plan = 0;                       // pre-assigned id the plan will have on success
  ProfileSnapshot profile;
  std::uint32_t context_tokens = 0;
};

struct StartCommand {
  TicketId ticket = 0;
  PlanId plan = 0;
  std::shared_ptr<GenerationInput> input;
  std::shared_ptr<GatedSink> sink;
  std::optional<RetainedSessionRef> reuse;   // the backend still checks exact token extension itself
  RetentionKey key;
  bool retain_on_finish = false;
  std::uint32_t completion_budget = 0;
  std::uint64_t session_seq = 0;             // identifies the session this start opens (for RetainedSessionRef)
};

enum class Disposition : std::uint8_t { kClosed, kRetained, kAborted, kFailed };
enum class FailKind : std::uint8_t { kNone, kWorkerLost, kSessionInvalidated, kBackendError };
struct OutputSummary {
  enum class Finish : std::uint8_t { kStop, kLength } finish = Finish::kStop;
  std::uint32_t prompt_tokens = 0, completion_tokens = 0, reused_prefix_tokens = 0;
};
struct RequestEnd {
  Disposition disposition = Disposition::kClosed;
  FailKind fail = FailKind::kNone;
  Status status;
  std::optional<RetainedSessionRef> retained;
  std::uint32_t retained_tokens = 0;
};
enum class GoneCause : std::uint8_t { kReleased, kInvalidated, kLost };
struct PlanGone {
  GoneCause cause = GoneCause::kReleased;
  bool clean = true;
  std::string problems;                  // status text, no paths
};
enum class InvalidateReason : std::uint8_t {
  kAbortNotConfirmed, kWorkerLost, kLocalUserReturned, kPolicy, kSessionInvalidated, kShutdown,
};
std::string_view to_string(InvalidateReason r) noexcept;

struct BackendDiagnostics {
  std::uint32_t plans = 0, sessions = 0;
  std::uint64_t commands_outstanding = 0;
};

// Completions, implemented by the scheduler. Thread-safe, non-blocking, never call back into the backend.
class BackendEvents {
 public:
  virtual void on_prepare_progress(JobSeq, const PrepareProgressView&) = 0;
  virtual void on_prepare_done(JobSeq, PlanId, Result<PlanInfo>) = 0;
  virtual void on_request_running(TicketId) = 0;
  virtual void on_request_output_done(TicketId, const OutputSummary&) = 0;
  virtual void on_request_ended(TicketId, RequestEnd) = 0;
  virtual void on_session_dropped(PlanId, RetainedSessionRef, Status) = 0;
  virtual void on_plan_gone(PlanId, PlanGone) = 0;
  virtual void on_plan_lost(PlanId, InvalidateReason) = 0;   // spontaneous: lease lost while idle
 protected:
  ~BackendEvents() = default;
};

class ExecutionBackend {
 public:
  virtual ~ExecutionBackend() = default;
  virtual void attach(BackendEvents* events) = 0;
  virtual void detach() = 0;                                       // blocks until no event call is running; none after
  virtual void prepare(PrepareCommand) = 0;                        // progress*, then exactly one on_prepare_done
  virtual void cancel_prepare(JobSeq) = 0;
  virtual void release(PlanId) = 0;                                // exactly one on_plan_gone
  virtual void invalidate_plan(PlanId, InvalidateReason) = 0;      // idempotent; one on_plan_gone per plan in total
  virtual void start_request(StartCommand) = 0;                    // [running] [output_done] then exactly one ended
  virtual void abort_request(TicketId) = 0;
  virtual void drop_session(PlanId, RetainedSessionRef) = 0;       // exactly one on_session_dropped
  virtual BackendDiagnostics diagnostics() const = 0;
};

}  // namespace clusterlm::scheduler
