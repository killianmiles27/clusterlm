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

constexpr std::uint16_t kHelperProtocolVersion = 1;
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
  kAck = 6,             // service -> helper (reply to everything but StatusRequest)
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

struct StatusReply {
  NodeState state = NodeState::kStarting;
  std::string paired_father;        // device-id fingerprint prefix or empty; never an address list
  std::uint64_t storage_bytes = 0;  // bytes currently staged under the Node staging root
  bool helper_reports_fresh = false;
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

// Each decode checks kind, version (kVersionMismatch) and that the payload is consumed exactly.
Result<ActivityReport> decode_activity_report(const Envelope& e);
Result<PauseRequest> decode_pause_request(const Envelope& e);
Status decode_resume_request(const Envelope& e);
Status decode_status_request(const Envelope& e);
Result<StatusReply> decode_status_reply(const Envelope& e);
Result<Ack> decode_ack(const Envelope& e);

// Father UI envelopes: version-stamped wrappers around an opaque payload.
Envelope father_envelope(MessageKind kind, Bytes payload);

// Per-user pipe name for the Father agent, e.g. "ClusterLM.Father.UI.<tag>". `user_tag` is sanitised to the
// pipe-name alphabet and truncated (it is typically a SID or user name).
std::string father_ui_pipe_name(const std::string& user_tag);

}  // namespace clusterlm::ipc
