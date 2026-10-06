#include "clusterlm/platform/ipc_messages.hpp"

namespace clusterlm::ipc {

namespace {
constexpr std::size_t kMaxStringBytes = 256;

Envelope make(MessageKind kind, ByteWriter&& w, std::uint16_t version = kHelperProtocolVersion) {
  Envelope e;
  e.kind = static_cast<std::uint16_t>(kind);
  e.version = version;
  e.payload = std::move(w).take();
  return e;
}

Status check_header(const Envelope& e, MessageKind kind) {
  if (e.kind != static_cast<std::uint16_t>(kind))
    return make_error(ErrorCode::kProtocolError, "unexpected ipc message kind");
  if (e.version != kHelperProtocolVersion)
    return make_error(ErrorCode::kVersionMismatch, "unsupported helper protocol version");
  return Status::ok();
}
}  // namespace

std::string_view to_string(NodeState s) {
  switch (s) {
    case NodeState::kStarting: return "Starting";
    case NodeState::kBusy: return "Busy";
    case NodeState::kOffering: return "Offering";
    case NodeState::kPaused: return "Paused";
    case NodeState::kSuspended: return "Suspended";
    case NodeState::kStopping: return "Stopping";
  }
  return "?";
}

std::string_view to_string(LeaseState s) {
  switch (s) {
    case LeaseState::kNone: return "None";
    case LeaseState::kPreparing: return "Preparing";
    case LeaseState::kReady: return "Ready";
    case LeaseState::kInferencing: return "Inferencing";
    case LeaseState::kReleasing: return "Releasing";
    case LeaseState::kCleanupPending: return "CleanupPending";
  }
  return "?";
}

namespace {
void put_settings(ByteWriter& w, const NodeSettingsView& v) {
  w.boolean(v.allow_when_idle);
  w.u32(v.idle_seconds);
  w.boolean(v.ac_only);
  w.u32(v.temp_storage_limit_gib);
  w.boolean(v.start_with_system);
  w.u32(v.ram_gib);
  w.u32(v.vram_gib);
  w.u32(v.threads);
}
void get_settings(ByteReader& r, NodeSettingsView& v) {
  r.boolean(v.allow_when_idle);
  r.u32(v.idle_seconds);
  r.boolean(v.ac_only);
  r.u32(v.temp_storage_limit_gib);
  r.boolean(v.start_with_system);
  r.u32(v.ram_gib);
  r.u32(v.vram_gib);
  r.u32(v.threads);
}
}  // namespace

Envelope encode(const ActivityReport& m) {
  ByteWriter w;
  w.u32(m.idle_seconds);
  w.boolean(m.session_locked);
  w.u32(m.session_id);
  return make(MessageKind::kActivityReport, std::move(w));
}
Envelope encode(const PauseRequest& m) {
  ByteWriter w;
  w.u32(m.duration_seconds);
  return make(MessageKind::kPauseRequest, std::move(w));
}
Envelope encode(const ResumeRequest&) { return make(MessageKind::kResumeRequest, ByteWriter{}); }
Envelope encode(const StatusRequest&) { return make(MessageKind::kStatusRequest, ByteWriter{}); }
Envelope encode(const StatusReply& m) {
  ByteWriter w;
  w.u8(static_cast<std::uint8_t>(m.state));
  w.str(m.paired_father);
  w.u64(m.storage_bytes);
  w.boolean(m.helper_reports_fresh);
  w.u8(static_cast<std::uint8_t>(m.lease_state));
  w.u32(m.lease_objects_sealed);
  w.u32(m.lease_objects_total);
  return make(MessageKind::kStatusReply, std::move(w));
}
Envelope encode(const Ack& m) {
  ByteWriter w;
  w.u16(static_cast<std::uint16_t>(m.code));
  w.str(m.message);
  return make(MessageKind::kAck, std::move(w));
}

Envelope encode(const SettingsRequest&) { return make(MessageKind::kSettingsRequest, ByteWriter{}); }
Envelope encode(const SettingsReply& m) {
  ByteWriter w;
  put_settings(w, m.settings);
  return make(MessageKind::kSettingsReply, std::move(w));
}
Envelope encode(const SettingsUpdate& m) {
  ByteWriter w;
  put_settings(w, m.settings);
  return make(MessageKind::kSettingsUpdate, std::move(w));
}
Envelope encode(const PairingModeRequest&) { return make(MessageKind::kPairingModeRequest, ByteWriter{}); }
Envelope encode(const PairingModeReply& m) {
  ByteWriter w;
  w.str(m.code);
  w.str(m.endpoint);
  w.str(m.fingerprint);
  w.u32(m.window_seconds);
  return make(MessageKind::kPairingModeReply, std::move(w));
}

Result<ActivityReport> decode_activity_report(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kActivityReport));
  ByteReader r(e.payload);
  ActivityReport m;
  r.u32(m.idle_seconds);
  r.boolean(m.session_locked);
  r.u32(m.session_id);
  CLM_RETURN_IF_ERROR(r.finish("ActivityReport"));
  return m;
}
Result<PauseRequest> decode_pause_request(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kPauseRequest));
  ByteReader r(e.payload);
  PauseRequest m;
  r.u32(m.duration_seconds);
  CLM_RETURN_IF_ERROR(r.finish("PauseRequest"));
  return m;
}
Status decode_resume_request(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kResumeRequest));
  return ByteReader(e.payload).finish("ResumeRequest");
}
Status decode_status_request(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kStatusRequest));
  return ByteReader(e.payload).finish("StatusRequest");
}
Result<StatusReply> decode_status_reply(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kStatusReply));
  ByteReader r(e.payload);
  StatusReply m;
  std::uint8_t state = 0;
  r.u8(state);
  r.str(m.paired_father, kMaxStringBytes);
  r.u64(m.storage_bytes);
  r.boolean(m.helper_reports_fresh);
  std::uint8_t lease = 0;
  r.u8(lease);
  r.u32(m.lease_objects_sealed);
  r.u32(m.lease_objects_total);
  CLM_RETURN_IF_ERROR(r.finish("StatusReply"));
  if (state > static_cast<std::uint8_t>(NodeState::kStopping))
    return make_error(ErrorCode::kProtocolError, "StatusReply: unknown node state");
  if (lease > static_cast<std::uint8_t>(LeaseState::kCleanupPending))
    return make_error(ErrorCode::kProtocolError, "StatusReply: unknown lease state");
  m.state = static_cast<NodeState>(state);
  m.lease_state = static_cast<LeaseState>(lease);
  return m;
}
Result<Ack> decode_ack(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kAck));
  ByteReader r(e.payload);
  Ack m;
  std::uint16_t code = 0;
  r.u16(code);
  r.str(m.message, kMaxStringBytes);
  CLM_RETURN_IF_ERROR(r.finish("Ack"));
  if (code > static_cast<std::uint16_t>(ErrorCode::kHardwareUnavailable))
    return make_error(ErrorCode::kProtocolError, "Ack: unknown error code");
  m.code = static_cast<ErrorCode>(code);
  return m;
}

Status decode_settings_request(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kSettingsRequest));
  return ByteReader(e.payload).finish("SettingsRequest");
}
Result<SettingsReply> decode_settings_reply(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kSettingsReply));
  ByteReader r(e.payload);
  SettingsReply m;
  get_settings(r, m.settings);
  CLM_RETURN_IF_ERROR(r.finish("SettingsReply"));
  return m;
}
Result<SettingsUpdate> decode_settings_update(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kSettingsUpdate));
  ByteReader r(e.payload);
  SettingsUpdate m;
  get_settings(r, m.settings);
  CLM_RETURN_IF_ERROR(r.finish("SettingsUpdate"));
  return m;
}
Status decode_pairing_mode_request(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kPairingModeRequest));
  return ByteReader(e.payload).finish("PairingModeRequest");
}
Result<PairingModeReply> decode_pairing_mode_reply(const Envelope& e) {
  CLM_RETURN_IF_ERROR(check_header(e, MessageKind::kPairingModeReply));
  ByteReader r(e.payload);
  PairingModeReply m;
  r.str(m.code, kMaxStringBytes);
  r.str(m.endpoint, kMaxStringBytes);
  r.str(m.fingerprint, kMaxStringBytes);
  r.u32(m.window_seconds);
  CLM_RETURN_IF_ERROR(r.finish("PairingModeReply"));
  return m;
}

Envelope father_envelope(MessageKind kind, Bytes payload) {
  Envelope e;
  e.kind = static_cast<std::uint16_t>(kind);
  e.version = kFatherUiProtocolVersion;
  e.payload = std::move(payload);
  return e;
}

std::string father_ui_pipe_name(const std::string& user_tag) {
  std::string name = kFatherUiPipePrefix;
  name.push_back('.');
  std::size_t n = 0;
  for (char c : user_tag) {
    if (n >= 40) break;
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
    name.push_back(ok ? c : '_');
    ++n;
  }
  if (n == 0) name += "default";
  return name;
}

}  // namespace clusterlm::ipc
