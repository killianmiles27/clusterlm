#pragma once
// NodeSupervisor: the ClusterLM Node service's policy loop around one worker process.
//
// The service never runs inference itself. It launches `clusterlm-node` as a worker (inside a Job Object on
// Windows), decides eligibility from local policy — user activity, session lock, power — and tells the
// worker to offer or revoke. Local policy wins without consulting Father (spec addendum §11):
//   1. on revocation the worker is told immediately and must reach Busy with zero staged bytes;
//   2. if it does not within the cooperative deadline (default 2 s), the supervisor terminates the worker's
//      job and relaunches it — the relaunched worker's orphan recovery deletes leftover staging before it can
//      accept another lease;
//   3. a worker that dies on its own is relaunched the same way;
//   4. suspend revokes immediately like (1); resume restarts from Busy (on_suspend / on_resume).
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/platform/adapters.hpp"
#include "clusterlm/platform/process.hpp"

namespace clusterlm::node {

struct IdlePolicy {
  std::uint32_t idle_seconds_required = 300;  // conservative default; user-configurable
  bool participate_when_locked = true;        // a locked session counts as idle
  bool require_ac_power = true;
  bool allow_battery_saver = false;
};

struct SupervisorConfig {
  std::filesystem::path worker_binary;
  std::vector<std::string> worker_args;  // listen/staging/identity/... (the supervisor adds --start-busy)
  // Device ids (certificate fingerprints) of paired Fathers; each is passed to the worker as `--trust <id>`.
  std::vector<std::string> trusted_peers;
  IdlePolicy policy;
  std::chrono::milliseconds cooperative_deadline{2000};
  std::chrono::milliseconds status_poll{20};
  std::uint64_t worker_memory_limit_bytes = 0;  // Job Object per-process limit (0 = none)
  // How often tick() asks the worker for its lease state (cached for status replies). 0 = every tick.
  std::chrono::milliseconds lease_poll_interval{250};
};

// Resource caps that are worker launch arguments (--ram-gib, --vram-gib, --disk-gib, --threads).
struct WorkerCaps {
  std::uint32_t ram_gib = 0;
  std::uint32_t vram_gib = 0;
  std::uint32_t disk_gib = 0;  // 0 = flag removed (no explicit cap)
  std::uint32_t threads = 0;   // 0 = flag removed (automatic)
  friend bool operator==(const WorkerCaps&, const WorkerCaps&) = default;
};
// Returns `args` with the cap flags replaced (or appended, or removed for 0 disk/threads). Other arguments keep
// their order. Values are plain decimal numbers; nothing from the caller is passed through as text.
std::vector<std::string> with_worker_caps(std::vector<std::string> args, const WorkerCaps& caps);

// What the worker last reported about its lease (states and counts only; never model, prompt or object names).
struct WorkerLeaseView {
  bool known = false;               // false until the first successful poll, or when the worker is unresponsive
  std::string state;                // the worker's NodeState name: Busy, Available, Preparing, Ready, ...
  std::uint32_t sealed_objects = 0;
  std::uint32_t planned_objects = 0;
};

enum class SupervisorEventKind {
  kWorkerStarted,
  kOffered,          // policy became eligible; worker told to become Available
  kRevoked,          // policy became ineligible; worker reached Busy with zero staged bytes in time
  kForcedTermination,// worker missed the cooperative deadline and was terminated
  kWorkerRestarted,  // relaunched after a crash or forced termination
};
std::string_view to_string(SupervisorEventKind k);

struct SupervisorEvent {
  SupervisorEventKind kind;
  double latency_ms = 0;               // revocation: activity observed -> worker Busy & clean
  std::uint64_t residual_bytes = 0;    // staged bytes reported by the worker at that point
  std::string detail;
};

class NodeSupervisor {
 public:
  NodeSupervisor(SupervisorConfig config, platform::ActivityMonitor& activity, platform::PowerMonitor& power,
                 std::unique_ptr<platform::ProcessJob> job = nullptr);
  ~NodeSupervisor();

  Status start();
  // One policy evaluation. The service calls this periodically (e.g. every 100 ms); tests call it directly.
  Result<std::vector<SupervisorEvent>> tick();
  void stop();

  // The machine is going to sleep. Local policy wins without consulting Father: the worker is revoked right now
  // (cooperative deadline, then forced termination) and the supervisor stays ineligible until on_resume() and a
  // fresh policy evaluation say otherwise. Idempotent.
  Result<std::vector<SupervisorEvent>> on_suspend();
  // The machine woke up. The worker is guaranteed Busy: nothing is offered until policy is satisfied again
  // (the next tick() must observe an eligible machine; callers should invalidate stale activity data first).
  Result<std::vector<SupervisorEvent>> on_resume();

  // Pairing state change: replaces the trusted Father list. The running worker keeps the old list in memory, so
  // it is revoked (cooperatively, then forcibly) and restarted with the new one: an unpaired Father loses every
  // connection and lease immediately. The worker comes back Busy; policy re-offers it on the next tick.
  Result<std::vector<SupervisorEvent>> set_trusted_peers(std::vector<std::string> fingerprints);
  const std::vector<std::string>& trusted_peers() const { return cfg_.trusted_peers; }

  // Applies a changed participation policy (immediately; the next tick re-evaluates) and/or worker arguments. A
  // changed argument list restarts the worker like set_trusted_peers does: revoked cooperatively first, so a lease
  // never survives a cap change. Unchanged inputs do nothing.
  Result<std::vector<SupervisorEvent>> reconfigure(std::optional<std::vector<std::string>> worker_args,
                                                   std::optional<IdlePolicy> policy);
  const std::vector<std::string>& worker_args() const { return cfg_.worker_args; }
  const IdlePolicy& policy() const { return cfg_.policy; }

  // The lease state the worker reported at the last poll (see SupervisorConfig::lease_poll_interval).
  WorkerLeaseView worker_lease() const { return lease_view_; }
  // Set when the worker reported an accepted UnpairNotice from its paired Father; returns the notifying device id
  // (possibly empty in insecure loopback mode) once, then clears it.
  std::optional<std::string> take_unpair_notice();

  bool eligible() const { return eligible_; }
  bool suspended() const { return suspended_; }
  platform::ChildProcess* worker() { return worker_.get(); }
  std::string worker_endpoint() const { return endpoint_; }
  std::string worker_device_id() const { return device_id_; }

 private:
  Result<bool> policy_eligible();
  Status launch();
  struct WorkerStatus {
    std::string state;
    std::uint64_t census_bytes = 0;
    std::uint32_t sealed = 0, planned = 0;
  };
  // read_until on the worker's stdout that also notices unsolicited UnpairNotice lines.
  Result<std::string> read_worker(const std::string& prefix, std::chrono::milliseconds timeout);
  void poll_lease();
  Status restart_worker();
  Result<WorkerStatus> worker_status(std::chrono::milliseconds timeout);
  Result<SupervisorEvent> revoke();
  Result<SupervisorEvent> force_and_relaunch(const std::string& why);

  SupervisorConfig cfg_;
  platform::ActivityMonitor& activity_;
  platform::PowerMonitor& power_;
  std::unique_ptr<platform::ProcessJob> job_;
  std::unique_ptr<platform::ChildProcess> worker_;
  std::string endpoint_, device_id_;
  bool eligible_ = false;
  bool suspended_ = false;
  WorkerLeaseView lease_view_;
  std::chrono::steady_clock::time_point last_lease_poll_{};
  std::optional<std::string> unpair_notice_;
};

}  // namespace clusterlm::node
