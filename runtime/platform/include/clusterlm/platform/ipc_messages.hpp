#pragma once
// Message schemas carried in ipc::Envelope: helper <-> Node service, and the Father UI <-> agent transport.
//
// Privacy: nothing here can carry token IDs, text, logits, prompts or activations. Helper messages describe
// only local input activity (idle seconds, lock state) and service status.
#include <cstdint>
#include <string>
#include <string_view>

#include "clusterlm/common/status.hpp"
#include "clusterlm/platform/ipc.hpp"

namespace clusterlm::ipc {

// Version 2: StatusReply carries the worker lease state and counts; settings and pairing-mode messages added.
constexpr std::uint16_t kHelperProtocolVersion = 2;
constexpr std::uint16_t kFatherUiProtocolVersion = 1;

// Logical pipe names. Windows: \\.\pipe\<name>. The Father UI pipe is per user: <prefix>.<user tag>.
constexpr const char* kNodeHelperPipeName = "ClusterLM.Node.Helper";
constexpr const char* kFatherUiPipePrefix = "ClusterLM.Father.UI";

enum class MessageKind : std::uint16_t {
  kActivityReport = 1,  // helper -> service
  kPauseRequest = 2,    // helper -> service
  kResumeRequest = 3,   // helper -> service
  kStatusRequest = 4,   // helper -> service
  kStatusReply = 5,     // service -> helper
  kAck = 6,             // service -> helper (reply to everything but StatusRequest, SettingsRequest, PairingModeRequest)
  kSettingsRequest = 7,     // helper/UI -> service: read the participation policy and resource caps
  kSettingsReply = 8,       // service -> helper/UI
  kSettingsUpdate = 9,      // helper/UI -> service: replace them (validated, persisted and applied by the service)
  kPairingModeRequest = 10, // helper/UI -> service: enter pairing mode (rate limited)
  kPairingModeReply = 11,   // service -> helper/UI: the code line to show the user
  // Father UI <-> agent: opaque payloads, the API itself belongs to the Father service workstream.
  kFatherRequest = 0x100,
  kFatherReply = 0x101,
  kFatherEvent = 0x102,
};

struct ActivityReport {
  std::uint32_t idle_seconds = 0;
  bool session_locked = false;
  std::uint32_t session_id = 0;  // must equal the pipe peer's session; the service checks
};
struct PauseRequest {
  std::uint32_t duration_seconds = 0;  // 0 = until ResumeRequest
};
struct ResumeRequest {};
struct StatusRequest {};

enum class NodeState : std::uint8_t { kStarting = 0, kBusy, kOffering, kPaused, kSuspended, kStopping };
std::string_view to_string(NodeState s);

// What the worker's lease is doing, as the supervisor last observed it. Counts and states only: never a model
// name, object name, prompt or token.
enum class LeaseState : std::uint8_t { kNone = 0, kPreparing, kReady, kInferencing, kReleasing, kCleanupPending };
std::string_view to_string(LeaseState s);

struct StatusReply {
  NodeState state = NodeState::kStarting;
  std::string paired_father;        // device-id fingerprint prefix or empty; never an address list
  std::uint64_t storage_bytes = 0;  // bytes currently staged under the Node staging root
  bool helper_reports_fresh = false;
  LeaseState lease_state = LeaseState::kNone;   // kNone also when the worker has not reported yet
  std::uint32_t lease_objects_sealed = 0;       // objects of the current plan received and verified
  std::uint32_t lease_objects_total = 0;        // objects the current plan assigns to this Node
};

// The part of the Node settings an interactive user may read and change through the service. Deliberately NOT here:
// the Node name, the paired Father, trust, any path, any command line. Ranges are enforced by the service
// (config::validate), not by this wire type.
struct NodeSettingsView {
  bool allow_when_idle = true;
  std::uint32_t idle_seconds = 300;
  bool ac_only = true;
  std::uint32_t temp_storage_limit_gib = 0;  // 0 = no explicit cap
  bool start_with_system = true;
  std::uint32_t ram_gib = 4;
  std::uint32_t vram_gib = 0;
  std::uint32_t threads = 0;                 // 0 = automatic
  friend bool operator==(const NodeSettingsView&, const NodeSettingsView&) = default;
};
struct SettingsRequest {};
struct SettingsReply {
  NodeSettingsView settings;
};
struct SettingsUpdate {
  NodeSettingsView settings;
};

struct PairingModeRequest {};
// The single line the user needs, as the service prints it for `--pair`: code, endpoint, short fingerprint.
struct PairingModeReply {
  std::string code;         // "ABCD-EFGH"
  std::string endpoint;     // "HOST:PORT" the Father user types in
  std::string fingerprint;  // short form for the visual comparison step
  std::uint32_t window_seconds = 0;
};
struct Ack {
  ErrorCode code = ErrorCode::kOk;
  std::string message;  // short, free of user data
};

Envelope encode(const ActivityReport& m);
Envelope encode(const PauseRequest& m);
Envelope encode(const ResumeRequest& m);
Envelope encode(const StatusRequest& m);
Envelope encode(const StatusReply& m);
Envelope encode(const Ack& m);
Envelope encode(const SettingsRequest& m);
Envelope encode(const SettingsReply& m);
Envelope encode(const SettingsUpdate& m);
Envelope encode(const PairingModeRequest& m);
Envelope encode(const PairingModeReply& m);

// Each decode checks kind, version (kVersionMismatch) and that the payload is consumed exactly.
Result<ActivityReport> decode_activity_report(const Envelope& e);
Result<PauseRequest> decode_pause_request(const Envelope& e);
Status decode_resume_request(const Envelope& e);
Status decode_status_request(const Envelope& e);
Result<StatusReply> decode_status_reply(const Envelope& e);
Result<Ack> decode_ack(const Envelope& e);
Status decode_settings_request(const Envelope& e);
Result<SettingsReply> decode_settings_reply(const Envelope& e);
Result<SettingsUpdate> decode_settings_update(const Envelope& e);
Status decode_pairing_mode_request(const Envelope& e);
Result<PairingModeReply> decode_pairing_mode_reply(const Envelope& e);

// Father UI envelopes: version-stamped wrappers around an opaque payload.
Envelope father_envelope(MessageKind kind, Bytes payload);

// Per-user pipe name for the Father agent, e.g. "ClusterLM.Father.UI.<tag>". `user_tag` is sanitised to the
// pipe-name alphabet and truncated (it is typically a SID or user name).
std::string father_ui_pipe_name(const std::string& user_tag);

}  // namespace clusterlm::ipc
