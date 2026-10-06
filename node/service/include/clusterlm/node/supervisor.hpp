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
  };
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
};

}  // namespace clusterlm::node
