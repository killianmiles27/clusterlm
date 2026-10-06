#include "clusterlm/protocol/messages.hpp"

#include <type_traits>

namespace clusterlm::protocol {
namespace {

// ---- field helpers ---------------------------------------------------------------------------------

void put(ByteWriter& w, const Digest256& d) { w.raw(d.bytes); }
bool get(ByteReader& r, Digest256& d) {
  ByteSpan s;
  if (!r.raw(d.bytes.size(), s)) return false;
  std::copy(s.begin(), s.end(), d.bytes.begin());
  return true;
}

template <typename Tag, typename Rep>
void put(ByteWriter& w, StrongId<Tag, Rep> id) {
  if constexpr (sizeof(Rep) == 8) {
    w.u64(id.value);
  } else {
    w.u32(id.value);
  }
}
template <typename Tag, typename Rep>
bool get(ByteReader& r, StrongId<Tag, Rep>& id) {
  if constexpr (sizeof(Rep) == 8) {
    return r.u64(id.value);
  } else {
    return r.u32(id.value);
  }
}

template <typename E>
bool get_enum(ByteReader& r, E& out, std::uint8_t max_value) {
  std::uint8_t v;
  if (!r.u8(v) || v > max_value) return false;
  out = static_cast<E>(v);
  return true;
}

void put(ByteWriter& w, const objects::LayerRange& range) {
  w.u32(range.begin);
  w.u32(range.end);
}
bool get(ByteReader& r, objects::LayerRange& range) { return r.u32(range.begin) && r.u32(range.end); }

void put(ByteWriter& w, const domain::WindowRequest& q) {
  put(w, q.epoch);
  put(w, q.session);
  put(w, q.window);
  w.u64(q.base_position);
  put(w, q.expected_state);
  w.u32(q.positions);
}
bool get(ByteReader& r, domain::WindowRequest& q) {
  return get(r, q.epoch) && get(r, q.session) && get(r, q.window) && r.u64(q.base_position) &&
         get(r, q.expected_state) && r.u32(q.positions);
}

void put(ByteWriter& w, const domain::CommitRequest& c) {
  put(w, c.epoch);
  put(w, c.session);
  put(w, c.window);
  w.u32(c.accepted);
  put(w, c.expected_state);
}
bool get(ByteReader& r, domain::CommitRequest& c) {
  return get(r, c.epoch) && get(r, c.session) && get(r, c.window) && r.u32(c.accepted) && get(r, c.expected_state);
}

void put(ByteWriter& w, const domain::CommitAck& a) {
  put(w, a.session);
  put(w, a.window);
  w.u64(a.committed_position);
  put(w, a.state);
}
bool get(ByteReader& r, domain::CommitAck& a) {
  return get(r, a.session) && get(r, a.window) && r.u64(a.committed_position) && get(r, a.state);
}

void put(ByteWriter& w, const domain::StageTiming& t) {
  w.u64(t.compute_ns);
  w.u64(t.cpu_expert_ns);
  w.u64(t.gpu_ns);
  w.u32(t.experts_selected);
  w.u32(t.experts_cpu);
  w.u32(t.experts_gpu);
}
bool get(ByteReader& r, domain::StageTiming& t) {
  return r.u64(t.compute_ns) && r.u64(t.cpu_expert_ns) && r.u64(t.gpu_ns) && r.u32(t.experts_selected) &&
         r.u32(t.experts_cpu) && r.u32(t.experts_gpu);
}

constexpr std::uint32_t kMaxStages = 64;
constexpr std::uint32_t kMaxTimings = 64;

// ---- per-message codecs ----------------------------------------------------------------------------
// encode_body(w, m) and decode_body(r, m, limits) per message type.

void encode_body(ByteWriter& w, const Hello& m) {
  w.u16(m.protocol_version);
  w.u8(static_cast<std::uint8_t>(m.role));
  w.u8(static_cast<std::uint8_t>(m.channel));
  w.str(m.device_id);
  w.str(m.backend_build);
  put(w, m.lease);
}
bool decode_body(ByteReader& r, Hello& m, const DecodeLimits& l) {
  return r.u16(m.protocol_version) && get_enum(r, m.role, 2) && get_enum(r, m.channel, 2) &&
         r.str(m.device_id, l.max_string) && r.str(m.backend_build, l.max_string) && get(r, m.lease);
}

void encode_body(ByteWriter& w, const HelloAck& m) {
  w.u16(m.protocol_version);
  w.str(m.device_id);
  put(w, m.lease);
}
bool decode_body(ByteReader& r, HelloAck& m, const DecodeLimits& l) {
  return r.u16(m.protocol_version) && r.str(m.device_id, l.max_string) && get(r, m.lease);
}

void encode_body(ByteWriter& w, const OfferResources& m) {
  put(w, m.lease);
  w.u64(m.safe_ram_bytes);
  w.u64(m.safe_vram_bytes);
  w.u64(m.staging_disk_bytes);
  w.str(m.cpu_summary);
  w.str(m.gpu_summary);
  w.u8(static_cast<std::uint8_t>(m.power));
}
bool decode_body(ByteReader& r, OfferResources& m, const DecodeLimits& l) {
  return get(r, m.lease) && r.u64(m.safe_ram_bytes) && r.u64(m.safe_vram_bytes) && r.u64(m.staging_disk_bytes) &&
         r.str(m.cpu_summary, l.max_string) && r.str(m.gpu_summary, l.max_string) && get_enum(r, m.power, 2);
}

void encode_body(ByteWriter& w, const PreparePlan& m) {
  put(w, m.lease);
  put(w, m.model_root);
  w.str(m.backend_build);
  put(w, m.plan_hash);
  w.u32(static_cast<std::uint32_t>(m.stages.size()));
  for (const auto& s : m.stages) {
    put(w, s.stage);
    w.u8(static_cast<std::uint8_t>(s.role));
    put(w, s.layers);
    w.u32(s.max_context);
    w.u32(s.max_window);
    w.u32(s.max_local_batch);
  }
  w.u64(m.ram_cap_bytes);
  w.u64(m.vram_cap_bytes);
  w.u64(m.staging_cap_bytes);
  m.manifest.encode(w);
  w.u32(static_cast<std::uint32_t>(m.assignments.size()));
  for (const auto& a : m.assignments) {
    w.u32(a.object_index);
    w.u8(static_cast<std::uint8_t>(a.target));
  }
}
bool decode_body(ByteReader& r, PreparePlan& m, const DecodeLimits& l) {
  std::uint32_t n;
  if (!(get(r, m.lease) && get(r, m.model_root) && r.str(m.backend_build, l.max_string) && get(r, m.plan_hash) &&
        r.u32(n) && n <= kMaxStages))
    return false;
  m.stages.resize(n);
  for (auto& s : m.stages) {
    if (!(get(r, s.stage) && get_enum(r, s.role, 2) && get(r, s.layers) && r.u32(s.max_context) &&
          r.u32(s.max_window) && r.u32(s.max_local_batch)))
      return false;
  }
  if (!(r.u64(m.ram_cap_bytes) && r.u64(m.vram_cap_bytes) && r.u64(m.staging_cap_bytes))) return false;
  auto manifest = objects::ModelManifest::decode(r);
  if (!manifest.is_ok()) return false;
  m.manifest = std::move(manifest).value();
  if (!r.u32(n) || n > l.max_manifest_objects) return false;
  m.assignments.resize(n);
  for (auto& a : m.assignments) {
    if (!(r.u32(a.object_index) && get_enum(r, a.target, 3))) return false;
  }
  return true;
}

void encode_body(ByteWriter& w, const PlanAccepted& m) {
  put(w, m.lease);
  put(w, m.plan_hash);
  w.u64(m.reserved_ram_bytes);
  w.u64(m.reserved_vram_bytes);
  w.u64(m.reserved_staging_bytes);
}
bool decode_body(ByteReader& r, PlanAccepted& m, const DecodeLimits&) {
  return get(r, m.lease) && get(r, m.plan_hash) && r.u64(m.reserved_ram_bytes) && r.u64(m.reserved_vram_bytes) &&
         r.u64(m.reserved_staging_bytes);
}

void encode_body(ByteWriter& w, const ProvisionChunk& m) {
  put(w, m.lease);
  w.u32(m.object_index);
  w.u64(m.offset);
  put(w, m.chunk_digest);
  w.blob(m.data);
}
bool decode_body(ByteReader& r, ProvisionChunk& m, const DecodeLimits& l) {
  return get(r, m.lease) && r.u32(m.object_index) && r.u64(m.offset) && get(r, m.chunk_digest) &&
         r.blob(m.data, l.max_provision_chunk);
}

void encode_body(ByteWriter& w, const SealObject& m) {
  put(w, m.lease);
  w.u32(m.object_index);
  w.u64(m.total_length);
  put(w, m.object_digest);
}
bool decode_body(ByteReader& r, SealObject& m, const DecodeLimits&) {
  return get(r, m.lease) && r.u32(m.object_index) && r.u64(m.total_length) && get(r, m.object_digest);
}

void encode_body(ByteWriter& w, const ObjectSealed& m) {
  put(w, m.lease);
  w.u32(m.object_index);
}
bool decode_body(ByteReader& r, ObjectSealed& m, const DecodeLimits&) {
  return get(r, m.lease) && r.u32(m.object_index);
}

void encode_body(ByteWriter& w, const PlanReady& m) {
  put(w, m.lease);
  put(w, m.plan_hash);
  w.u64(m.resident_bytes);
  w.u64(m.prepare_ns);
}
bool decode_body(ByteReader& r, PlanReady& m, const DecodeLimits&) {
  return get(r, m.lease) && get(r, m.plan_hash) && r.u64(m.resident_bytes) && r.u64(m.prepare_ns);
}

void encode_body(ByteWriter& w, const AuthorizePeer& m) {
  put(w, m.lease);
  put(w, m.plan_hash);
  put(w, m.from_stage);
  put(w, m.to_stage);
  w.str(m.peer_device_id);
  w.str(m.peer_endpoint);
  put(w, m.peer_lease);
}
bool decode_body(ByteReader& r, AuthorizePeer& m, const DecodeLimits& l) {
  return get(r, m.lease) && get(r, m.plan_hash) && get(r, m.from_stage) && get(r, m.to_stage) &&
         r.str(m.peer_device_id, l.max_string) && r.str(m.peer_endpoint, l.max_string) && get(r, m.peer_lease);
}

void encode_body(ByteWriter& w, const OpenSession& m) {
  put(w, m.epoch);
  put(w, m.session);
}
bool decode_body(ByteReader& r, OpenSession& m, const DecodeLimits&) { return get(r, m.epoch) && get(r, m.session); }

void encode_body(ByteWriter& w, const SessionOpened& m) {
  put(w, m.epoch);
  put(w, m.session);
}
bool decode_body(ByteReader& r, SessionOpened& m, const DecodeLimits&) {
  return get(r, m.epoch) && get(r, m.session);
}

void encode_body(ByteWriter& w, const RunWindow& m) {
  put(w, m.lease);
  put(w, m.request);
  put(w, m.stage);
  w.boolean(m.forward_to_peer);
  w.boolean(m.auto_commit);
  w.u32(static_cast<std::uint32_t>(m.upstream_timings.size()));
  for (const auto& t : m.upstream_timings) put(w, t);
  m.activations.encode(w);
}
bool decode_body(ByteReader& r, RunWindow& m, const DecodeLimits& l) {
  std::uint32_t n;
  if (!(get(r, m.lease) && get(r, m.request) && get(r, m.stage) && r.boolean(m.forward_to_peer) &&
        r.boolean(m.auto_commit) && r.u32(n) &&
        n <= kMaxTimings))
    return false;
  m.upstream_timings.resize(n);
  for (auto& t : m.upstream_timings)
    if (!get(r, t)) return false;
  auto acts = domain::StageActivations::decode(r, l.max_window_positions);
  if (!acts.is_ok()) return false;
  m.activations = std::move(acts).value();
  return true;
}

void encode_body(ByteWriter& w, const StageResult& m) {
  put(w, m.epoch);
  put(w, m.session);
  put(w, m.window);
  put(w, m.stage);
  w.u16(static_cast<std::uint16_t>(m.status));
  w.str(m.error_message);
  w.boolean(m.status == ErrorCode::kOk);
  if (m.status == ErrorCode::kOk) m.activations.encode(w);
  w.u32(static_cast<std::uint32_t>(m.timings.size()));
  for (const auto& t : m.timings) put(w, t);
}
bool decode_body(ByteReader& r, StageResult& m, const DecodeLimits& l) {
  std::uint16_t status;
  bool has_acts;
  if (!(get(r, m.epoch) && get(r, m.session) && get(r, m.window) && get(r, m.stage) && r.u16(status) &&
        r.str(m.error_message, l.max_string) && r.boolean(has_acts)))
    return false;
  if (status > static_cast<std::uint16_t>(ErrorCode::kHardwareUnavailable)) return false;
  m.status = static_cast<ErrorCode>(status);
  if (has_acts != (m.status == ErrorCode::kOk)) return false;
  if (has_acts) {
    auto acts = domain::StageActivations::decode(r, l.max_window_positions);
    if (!acts.is_ok()) return false;
    m.activations = std::move(acts).value();
  }
  std::uint32_t n;
  if (!r.u32(n) || n > kMaxTimings) return false;
  m.timings.resize(n);
  for (auto& t : m.timings)
    if (!get(r, t)) return false;
  return true;
}

void encode_body(ByteWriter& w, const CommitWindow& m) {
  put(w, m.request);
  put(w, m.stage);
}
bool decode_body(ByteReader& r, CommitWindow& m, const DecodeLimits&) { return get(r, m.request) && get(r, m.stage); }

void encode_body(ByteWriter& w, const CommitAckMessage& m) {
  put(w, m.stage);
  put(w, m.ack);
}
bool decode_body(ByteReader& r, CommitAckMessage& m, const DecodeLimits&) { return get(r, m.stage) && get(r, m.ack); }

void encode_body(ByteWriter& w, const AbortWindow& m) {
  put(w, m.epoch);
  put(w, m.session);
  put(w, m.window);
  put(w, m.stage);
}
bool decode_body(ByteReader& r, AbortWindow& m, const DecodeLimits&) {
  return get(r, m.epoch) && get(r, m.session) && get(r, m.window) && get(r, m.stage);
}

void encode_body(ByteWriter& w, const WindowAborted& m) {
  put(w, m.stage);
  put(w, m.ack.session);
  put(w, m.ack.window);
  w.u64(m.ack.committed_position);
  put(w, m.ack.state);
}
bool decode_body(ByteReader& r, WindowAborted& m, const DecodeLimits&) {
  return get(r, m.stage) && get(r, m.ack.session) && get(r, m.ack.window) && r.u64(m.ack.committed_position) &&
         get(r, m.ack.state);
}

void encode_body(ByteWriter& w, const ProvisionStatus& m) {
  put(w, m.lease);
  w.u32(static_cast<std::uint32_t>(m.sealed_objects.size()));
  for (auto i : m.sealed_objects) w.u32(i);
}
bool decode_body(ByteReader& r, ProvisionStatus& m, const DecodeLimits& l) {
  std::uint32_t n;
  if (!(get(r, m.lease) && r.u32(n) && n <= l.max_manifest_objects)) return false;
  m.sealed_objects.resize(n);
  for (auto& i : m.sealed_objects)
    if (!r.u32(i)) return false;
  return true;
}

void encode_body(ByteWriter& w, const AbortSession& m) {
  put(w, m.epoch);
  put(w, m.session);
  w.str(m.reason);
}
bool decode_body(ByteReader& r, AbortSession& m, const DecodeLimits& l) {
  return get(r, m.epoch) && get(r, m.session) && r.str(m.reason, l.max_string);
}

void encode_body(ByteWriter& w, const ReleaseLease& m) {
  put(w, m.lease);
  w.u8(static_cast<std::uint8_t>(m.reason));
}
bool decode_body(ByteReader& r, ReleaseLease& m, const DecodeLimits&) {
  return get(r, m.lease) && get_enum(r, m.reason, 4);
}

void encode_body(ByteWriter& w, const ReleaseComplete& m) {
  put(w, m.lease);
  w.boolean(m.resources_released);
  w.boolean(m.storage_cleaned);
  w.u64(m.residual_bytes);
  w.u64(m.release_ns);
  w.str(m.errors);
}
bool decode_body(ByteReader& r, ReleaseComplete& m, const DecodeLimits& l) {
  return get(r, m.lease) && r.boolean(m.resources_released) && r.boolean(m.storage_cleaned) &&
         r.u64(m.residual_bytes) && r.u64(m.release_ns) && r.str(m.errors, l.max_string);
}

void encode_body(ByteWriter& w, const ErrorMessage& m) {
  w.u16(static_cast<std::uint16_t>(m.code));
  w.str(m.message);
  w.u16(static_cast<std::uint16_t>(m.in_reply_to));
}
bool decode_body(ByteReader& r, ErrorMessage& m, const DecodeLimits& l) {
  std::uint16_t code, reply;
  if (!(r.u16(code) && r.str(m.message, l.max_string) && r.u16(reply))) return false;
  if (code > static_cast<std::uint16_t>(ErrorCode::kHardwareUnavailable)) return false;
  m.code = static_cast<ErrorCode>(code);
  m.in_reply_to = static_cast<MessageType>(reply);
  return true;
}

void encode_body(ByteWriter& w, const Ping& m) { w.u64(m.nonce); }
bool decode_body(ByteReader& r, Ping& m, const DecodeLimits&) { return r.u64(m.nonce); }
void encode_body(ByteWriter& w, const Pong& m) { w.u64(m.nonce); }
bool decode_body(ByteReader& r, Pong& m, const DecodeLimits&) { return r.u64(m.nonce); }

template <typename T>
constexpr MessageType type_for();
#define CLM_MESSAGE_TYPE(T, E) \
  template <>                  \
  constexpr MessageType type_for<T>() { return MessageType::E; }
CLM_MESSAGE_TYPE(Hello, kHello)
CLM_MESSAGE_TYPE(HelloAck, kHelloAck)
CLM_MESSAGE_TYPE(OfferResources, kOfferResources)
CLM_MESSAGE_TYPE(PreparePlan, kPreparePlan)
CLM_MESSAGE_TYPE(PlanAccepted, kPlanAccepted)
CLM_MESSAGE_TYPE(ProvisionChunk, kProvisionChunk)
CLM_MESSAGE_TYPE(SealObject, kSealObject)
CLM_MESSAGE_TYPE(ObjectSealed, kObjectSealed)
CLM_MESSAGE_TYPE(PlanReady, kPlanReady)
CLM_MESSAGE_TYPE(AuthorizePeer, kAuthorizePeer)
CLM_MESSAGE_TYPE(OpenSession, kOpenSession)
CLM_MESSAGE_TYPE(SessionOpened, kSessionOpened)
CLM_MESSAGE_TYPE(RunWindow, kRunWindow)
CLM_MESSAGE_TYPE(StageResult, kStageResult)
CLM_MESSAGE_TYPE(CommitWindow, kCommitWindow)
CLM_MESSAGE_TYPE(CommitAckMessage, kCommitAck)
CLM_MESSAGE_TYPE(AbortSession, kAbortSession)
CLM_MESSAGE_TYPE(ReleaseLease, kReleaseLease)
CLM_MESSAGE_TYPE(ReleaseComplete, kReleaseComplete)
CLM_MESSAGE_TYPE(ErrorMessage, kError)
CLM_MESSAGE_TYPE(Ping, kPing)
CLM_MESSAGE_TYPE(Pong, kPong)
CLM_MESSAGE_TYPE(AbortWindow, kAbortWindow)
CLM_MESSAGE_TYPE(WindowAborted, kWindowAborted)
CLM_MESSAGE_TYPE(ProvisionStatus, kProvisionStatus)
#undef CLM_MESSAGE_TYPE

template <typename T>
Result<Message> decode_as(ByteSpan payload, const DecodeLimits& limits) {
  T m;
  ByteReader r(payload);
  if (!decode_body(r, m, limits)) {
    return make_error(ErrorCode::kProtocolError,
                      std::string("malformed ") + std::string(to_string(type_for<T>())) + " message");
  }
  CLM_RETURN_IF_ERROR(r.finish(to_string(type_for<T>())));
  return Message(std::move(m));
}

}  // namespace

std::string_view to_string(Channel c) {
  switch (c) {
    case Channel::kControl: return "control";
    case Channel::kActivation: return "activation";
    case Channel::kProvision: return "provision";
  }
  return "?";
}

std::string_view to_string(MessageType t) {
  switch (t) {
    case MessageType::kHello: return "Hello";
    case MessageType::kHelloAck: return "HelloAck";
    case MessageType::kOfferResources: return "OfferResources";
    case MessageType::kPreparePlan: return "PreparePlan";
    case MessageType::kPlanAccepted: return "PlanAccepted";
    case MessageType::kProvisionChunk: return "ProvisionChunk";
    case MessageType::kSealObject: return "SealObject";
    case MessageType::kObjectSealed: return "ObjectSealed";
    case MessageType::kPlanReady: return "PlanReady";
    case MessageType::kAuthorizePeer: return "AuthorizePeer";
    case MessageType::kOpenSession: return "OpenSession";
    case MessageType::kSessionOpened: return "SessionOpened";
    case MessageType::kRunWindow: return "RunWindow";
    case MessageType::kStageResult: return "StageResult";
    case MessageType::kCommitWindow: return "CommitWindow";
    case MessageType::kCommitAck: return "CommitAck";
    case MessageType::kAbortSession: return "AbortSession";
    case MessageType::kReleaseLease: return "ReleaseLease";
    case MessageType::kReleaseComplete: return "ReleaseComplete";
    case MessageType::kError: return "Error";
    case MessageType::kPing: return "Ping";
    case MessageType::kPong: return "Pong";
    case MessageType::kAbortWindow: return "AbortWindow";
    case MessageType::kWindowAborted: return "WindowAborted";
    case MessageType::kProvisionStatus: return "ProvisionStatus";
  }
  return "Unknown";
}

MessageType type_of(const Message& m) {
  return std::visit([](const auto& v) { return type_for<std::decay_t<decltype(v)>>(); }, m);
}

Channel channel_of(MessageType t) {
  switch (t) {
    case MessageType::kRunWindow:
    case MessageType::kStageResult:
      return Channel::kActivation;
    case MessageType::kProvisionChunk:
    case MessageType::kSealObject:
    case MessageType::kObjectSealed:
    case MessageType::kProvisionStatus:
      return Channel::kProvision;
    default:
      return Channel::kControl;
  }
}

Bytes encode(const Message& m) {
  ByteWriter w;
  std::visit([&w](const auto& v) { encode_body(w, v); }, m);
  return std::move(w).take();
}

Result<Message> decode(MessageType type, ByteSpan payload, const DecodeLimits& limits) {
  switch (type) {
    case MessageType::kHello: return decode_as<Hello>(payload, limits);
    case MessageType::kHelloAck: return decode_as<HelloAck>(payload, limits);
    case MessageType::kOfferResources: return decode_as<OfferResources>(payload, limits);
    case MessageType::kPreparePlan: return decode_as<PreparePlan>(payload, limits);
    case MessageType::kPlanAccepted: return decode_as<PlanAccepted>(payload, limits);
    case MessageType::kProvisionChunk: return decode_as<ProvisionChunk>(payload, limits);
    case MessageType::kSealObject: return decode_as<SealObject>(payload, limits);
    case MessageType::kObjectSealed: return decode_as<ObjectSealed>(payload, limits);
    case MessageType::kPlanReady: return decode_as<PlanReady>(payload, limits);
    case MessageType::kAuthorizePeer: return decode_as<AuthorizePeer>(payload, limits);
    case MessageType::kOpenSession: return decode_as<OpenSession>(payload, limits);
    case MessageType::kSessionOpened: return decode_as<SessionOpened>(payload, limits);
    case MessageType::kRunWindow: return decode_as<RunWindow>(payload, limits);
    case MessageType::kStageResult: return decode_as<StageResult>(payload, limits);
    case MessageType::kCommitWindow: return decode_as<CommitWindow>(payload, limits);
    case MessageType::kCommitAck: return decode_as<CommitAckMessage>(payload, limits);
    case MessageType::kAbortSession: return decode_as<AbortSession>(payload, limits);
    case MessageType::kReleaseLease: return decode_as<ReleaseLease>(payload, limits);
    case MessageType::kReleaseComplete: return decode_as<ReleaseComplete>(payload, limits);
    case MessageType::kError: return decode_as<ErrorMessage>(payload, limits);
    case MessageType::kPing: return decode_as<Ping>(payload, limits);
    case MessageType::kPong: return decode_as<Pong>(payload, limits);
    case MessageType::kAbortWindow: return decode_as<AbortWindow>(payload, limits);
    case MessageType::kWindowAborted: return decode_as<WindowAborted>(payload, limits);
    case MessageType::kProvisionStatus: return decode_as<ProvisionStatus>(payload, limits);
  }
  return make_error(ErrorCode::kProtocolError, "unknown message type " + std::to_string(static_cast<int>(type)));
}

}  // namespace clusterlm::protocol
