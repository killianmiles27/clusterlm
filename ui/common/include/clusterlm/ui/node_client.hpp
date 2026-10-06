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

// What the user sees. The helper pipe reports the Node's own state (Starting/Busy/Offering/Paused/...) plus, since
// protocol version 2, the worker's lease state (ipc::LeaseState). Preparing, Ready, In use and Cleanup needed come from
// that lease state while the Node is offering; a Node that is paused, suspended or stopping shows that instead.
// from_machine_state() maps the same states from the Father-side catalog vocabulary.
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
// With the worker's lease state: only an offering Node takes its state from the lease (a busy, paused, suspended or
// stopping Node stays out of the way whatever its worker still holds).
NodeUiState from_ipc(ipc::NodeState s, ipc::LeaseState lease) noexcept;
NodeUiState from_machine_state(catalog::MachineState s) noexcept;

struct NodeStatus {
  NodeUiState state = NodeUiState::kUnreachable;
  std::string paired_father;        // fingerprint prefix; empty = not paired
  std::uint64_t storage_bytes = 0;  // temporary storage currently in use
  bool fresh = false;               // the session helper is reporting
  std::string detail;               // why unreachable, etc.
  // The worker's lease, counts only: parts received and verified out of the parts assigned. 0/0 when no lease.
  ipc::LeaseState lease = ipc::LeaseState::kNone;
  std::uint32_t lease_parts_done = 0;
  std::uint32_t lease_parts_total = 0;
};

// The user-editable part of the Node settings (the service owns the rest: name, paired Father, trust, paths).
struct NodeSettings {
  bool allow_when_idle = true;
  bool ac_power_only = true;
  std::uint32_t temp_storage_limit_gb = 64;  // 0 = no limit, else 1..4096
  bool start_with_windows = true;
  // Advanced resource limits: what this PC may give. They apply to the next help session (the Node restarts its
  // worker to apply a change, after releasing anything it holds).
  std::uint32_t cpu_cap_percent = 100;  // 10..100 (of this PC's logical processors; 100 = no limit)
  std::uint32_t gpu_memory_gb = 0;      // graphics memory it may use; 0 = do not use the graphics card
  std::uint32_t ram_gb = 4;             // memory it may use, 1..1024
};

// Conversions to and from the service's wire form (ipc::NodeSettingsView). `logical_processors` is this PC's count
// (std::thread::hardware_concurrency()); the CPU limit travels as a thread count. `base` supplies the service
// fields the UI does not edit (idle time), so a save never changes them.
NodeSettings from_view(const ipc::NodeSettingsView& v, std::uint32_t logical_processors);
ipc::NodeSettingsView to_view(const NodeSettings& s, const ipc::NodeSettingsView& base, std::uint32_t logical_processors);

// The line the Node shows while pairing mode is on: what to type on the Father.
struct NodePairingInfo {
  std::string code;         // "ABCD-EFGH"
  std::string endpoint;     // "HOST:PORT"
  std::string fingerprint;  // short form, to compare on the Father afterwards
  std::uint32_t window_seconds = 0;
};
// Field-level validation: the message names the field in words a user can act on.
Status validate(const NodeSettings& s);

class NodeClient {
 public:
  virtual ~NodeClient() = default;
  virtual Result<NodeStatus> status() = 0;
  virtual Status pause() = 0;  // until resume
  virtual Status resume() = 0;
  // Read and saved through the Node service (which validates, persists and applies them). A service that cannot
  // (older build, no settings store) answers an error; the UI says so rather than pretending to save.
  virtual Result<NodeSettings> get_settings() = 0;
  virtual Status set_settings(const NodeSettings& settings) = 0;
  // Asks the service to enter pairing mode (rate limited, signed-in user only) and returns the line to show.
  virtual Result<NodePairingInfo> enter_pairing_mode() = 0;
};

}  // namespace clusterlm::ui
