#pragma once
// NodeWorker: the ClusterLM Node execution service.
//
// Serves one Father over the three protocol channels, owns at most one lease at a time, provisions only the
// plan-assigned objects into its ephemeral LeaseStore, hosts the execution domain(s) for its assigned stages,
// optionally forwards stage results directly to an authorized downstream peer, and releases everything when
// the lease ends — on Father request, local user activity, Father loss or fault.
//
// Availability states follow spec addendum §11:
//   Busy -> Available -> Preparing -> Ready <-> Inferencing -> Releasing -> Busy/Available (or CleanupPending)
// Local policy wins: on_local_activity() revokes the lease without consulting Father.
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "clusterlm/common/status.hpp"
#include "clusterlm/transport/impairment.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::node {

enum class NodeState : std::uint8_t {
  kBusy,            // local use or paused: holds no model storage
  kAvailable,       // idle and willing; not provisioned
  kPreparing,       // plan accepted; objects transferring/loading
  kReady,           // every assigned object validated, domains prepared
  kInferencing,     // executing a window
  kReleasing,
  kCleanupPending,  // storage cleanup blocked; retried before any new lease
};
std::string_view to_string(NodeState s);

struct NodeConfig {
  std::string name = "node";
  transport::Endpoint listen{"127.0.0.1", 0};
  transport::SecurityConfig security;
  std::filesystem::path staging_root;
  std::uint64_t ram_allowance = 0;      // safe RAM this Node offers (from live policy / ClusterLM Bench)
  std::uint64_t vram_allowance = 0;
  std::uint64_t disk_allowance = 0;     // 0 = disk staging not permitted
  std::string backend = "reference";
  // Upper bound on compute threads any domain of this Node may use (0 = automatic). Set from the user's
  // resource cap (config::ResourceCaps::threads); reported in the offer, passed to every DomainSpec.
  std::uint32_t cpu_threads = 0;
  bool start_busy = false;
  // Called once, from the control-channel thread and after the reply was sent, when the paired Father sent an
  // UnpairNotice: the lease is already released and `peer_device_id` is no longer trusted. The supervising service
  // clears its paired-Father setting. Must not call back into the worker.
  std::function<void(const std::string& peer_device_id)> on_unpair_notice;
  // Network emulation applied to every connection this Node opens or accepts (simulation only).
  std::optional<transport::NetworkConditions> impairment;
  std::shared_ptr<transport::FaultInjector> faults;
  // Test hook: invoked at named lifecycle phases ("transfer", "ready", "inference", ...).
  std::function<void(std::string_view)> phase_hook;
};

struct NodeStatus {
  NodeState state = NodeState::kBusy;
  std::uint64_t lease_generation = 0;
  std::uint64_t leases_granted = 0;
  std::uint64_t leases_released = 0;
  std::uint64_t windows_executed = 0;
  std::uint64_t windows_forwarded = 0;
  std::uint64_t stale_rejections = 0;
  std::uint64_t provisioned_bytes = 0;
  std::uint64_t last_release_ns = 0;
  bool last_storage_cleaned = true;
  std::uint64_t staging_census_bytes = 0;
  // Lease progress, counts only: objects of the current plan that are sealed, and objects the plan assigns.
  std::uint32_t sealed_objects = 0;
  std::uint32_t planned_objects = 0;
};

class NodeWorker {
 public:
  static Result<std::unique_ptr<NodeWorker>> start(NodeConfig config);
  ~NodeWorker();
  NodeWorker(const NodeWorker&) = delete;
  NodeWorker& operator=(const NodeWorker&) = delete;

  transport::Endpoint endpoint() const;
  std::string device_id() const;
  NodeStatus status() const;

  // Local activity policy: revoke eligibility and release immediately.
  void on_local_activity();
  // Local idle policy satisfied again: become Available and offer resources to a connected Father.
  void on_local_idle();
  void stop();

  struct Impl;

 private:
  explicit NodeWorker(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace clusterlm::node
