#pragma once
// Status surface of the scheduler: read-only snapshots for the management API (`GET /machines`, `/queue`, `/availability`,
// `/jobs/{id}`, `/profiles`), local IPC and MCP (workstream D). Stable from v1: fields are only ever added.
//
// Privacy / disclosure rules (scheduler-admission-v1 §8, auth-scopes-v1 §5):
//   * never a prompt, token id, key id of another client, device fingerprint, network address or file path;
//   * `machines` carries the user's Worker display names; callers without `status:read` must not be given this surface at
//     all (the serving layer enforces the scope; the scheduler additionally returns generic reasons to such callers in
//     admission errors);
//   * a client sees its own tickets in full and other clients' tickets only as anonymous queue positions.
// Every value is an observation at `as_of_ms`; none is a prediction unless flagged `estimate`.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/scheduler/resource_policy.hpp"
#include "clusterlm/scheduler/types.hpp"

namespace clusterlm::scheduler {

struct MachineStatus {
  std::string display_name;
  std::string state;                  // catalog::MachineState text: busy | available | preparing | ready | inferencing | ...
  bool usable = false;                // would satisfy a selector right now (policy allows and state usable)
  std::string block;                  // PolicyBlock code when !usable by policy, else ""
  std::string reason;                 // user-readable
  std::optional<std::uint32_t> schedule_opens_in_minutes;
  std::uint64_t ram_free_bytes = 0, vram_free_bytes = 0;
  std::optional<std::uint64_t> observed_age_ms;   // age of the last report; absent = never reported
  std::uint32_t leases_held = 0;                  // plans of ours that currently hold this Worker
};

enum class TicketState : std::uint8_t { kQueued, kStarting, kRunning, kFinishing, kCancelling, kDone };
std::string_view to_string(TicketState s) noexcept;

struct TicketStatus {
  std::string request_id;             // req_*; only for the caller's own tickets
  TicketState state = TicketState::kQueued;
  std::string profile_id, api_model_id;
  std::uint32_t position = 0;         // 1-based queue position; 0 when not queued
  std::uint64_t age_ms = 0;
  bool waiting_for_prepare = false;
  std::optional<std::string> job_id;
  std::uint32_t prompt_tokens = 0, completion_budget = 0;   // counts only
};

struct QueueStatus {
  std::uint32_t depth = 0, capacity = 0;
  std::uint32_t per_priority[kPriorityCount] = {};
  std::uint32_t active = 0;                       // starting + running + finishing
  std::vector<TicketStatus> own;                  // the caller's tickets (queued or active)
  std::vector<std::uint32_t> others_positions;    // anonymous 1-based positions of other clients' queued tickets
};

enum class JobKind : std::uint8_t { kPrepare, kRelease };
enum class JobState : std::uint8_t { kQueued, kRunning, kSucceeded, kFailed, kCancelled };
std::string_view to_string(JobKind k) noexcept;
std::string_view to_string(JobState s) noexcept;

struct JobStatus {
  std::string job_id;                 // job_*
  JobKind kind = JobKind::kPrepare;
  JobState state = JobState::kQueued;
  std::string profile_id;
  std::uint32_t context_tokens = 0;
  std::string phase;                  // father-domains | provisioning | node-preparing | node-ready (running only)
  std::optional<double> percent;
  std::optional<double> eta_seconds;  // estimate; absent without a measured rate
  std::string last_error;              // status text; empty when none
  std::uint32_t waiting_requests = 0;
  std::uint64_t age_ms = 0;
};

struct AvailabilityEntry {
  std::string profile_id, api_model_id;
  ReadyState state = ReadyState::kUnavailable;
  bool can_prepare_now = false;       // a prepare job would be admitted and could complete
  bool prepared = false;
  std::vector<std::string> why_not;   // empty when can_prepare_now or prepared
};

struct SchedulerCounters {
  std::uint64_t admitted = 0, rejected = 0, completed = 0, cancelled = 0, failed = 0, worker_lost = 0, timed_out = 0;
  std::uint64_t plan_invalidations = 0, swaps = 0, prepares_started = 0;
};

// Read-only view the serving layer, IPC and MCP depend on. Thread-safe; every call returns a consistent snapshot.
class StatusProvider {
 public:
  virtual ~StatusProvider() = default;
  virtual std::vector<MachineStatus> machines() const = 0;
  virtual QueueStatus queue(const std::string& client_id) const = 0;
  virtual std::vector<JobStatus> jobs() const = 0;
  virtual std::optional<JobStatus> job(const std::string& job_id) const = 0;
  virtual std::vector<AvailabilityEntry> availability() const = 0;
  virtual SchedulerCounters counters() const = 0;
};

}  // namespace clusterlm::scheduler
