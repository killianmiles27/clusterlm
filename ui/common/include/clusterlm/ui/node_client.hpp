#pragma once
// NodeClient: what the Node UI may ask of the local Node service. Intentionally small: no chat, no model, no
// GGUF concepts. Implementations: IpcNodeClient (helper pipe) and test fakes.
#include <cstdint>
#include <string>
#include <string_view>

#include "clusterlm/catalog/readiness.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/platform/ipc_messages.hpp"

namespace clusterlm::ui {

// What the user sees. The helper pipe reports only a subset (Starting/Busy/Offering/Paused/...); Preparing, Ready,
// In use and Cleanup needed come from the lease layer and are mapped from catalog::MachineState once the service
// reports them (WP14 seam).
enum class NodeUiState : std::uint8_t {
  kUnreachable,  // the Node service is not running / not answering
  kStarting,
  kAvailable,
  kPreparing,
  kReady,
  kInUse,
  kBusy,  // someone is using this PC: the Node stays out of the way
  kPaused,
  kCleanupNeeded,
  kStopping,
};
std::string_view to_string(NodeUiState s) noexcept;
// One plain sentence per state, shown under the state label.
std::string_view describe(NodeUiState s) noexcept;
NodeUiState from_ipc(ipc::NodeState s) noexcept;
NodeUiState from_machine_state(catalog::MachineState s) noexcept;

struct NodeStatus {
  NodeUiState state = NodeUiState::kUnreachable;
  std::string paired_father;        // fingerprint prefix; empty = not paired
  std::uint64_t storage_bytes = 0;  // temporary storage currently in use
  bool fresh = false;               // the session helper is reporting
  std::string detail;               // why unreachable, etc.
};

struct NodeSettings {
  bool allow_when_idle = true;
  bool ac_power_only = true;
  std::uint32_t temp_storage_limit_gb = 64;  // 1..4096
  bool start_with_windows = true;
  // Advanced resource caps. 100 / 0 mean "no cap".
  std::uint32_t cpu_cap_percent = 100;         // 10..100
  std::uint32_t gpu_memory_cap_percent = 100;  // 10..100
  std::uint32_t ram_cap_gb = 0;                // 0 = none, else 1..1024
};
// Field-level validation: the message names the field in words a user can act on.
Status validate(const NodeSettings& s);

class NodeClient {
 public:
  virtual ~NodeClient() = default;
  virtual Result<NodeStatus> status() = 0;
  virtual Status pause() = 0;  // until resume
  virtual Status resume() = 0;
  virtual Result<NodeSettings> get_settings() = 0;
  // kUnimplemented until the Node config store exists; the UI says so rather than pretending to save.
  virtual Status set_settings(const NodeSettings& settings) = 0;
};

}  // namespace clusterlm::ui
