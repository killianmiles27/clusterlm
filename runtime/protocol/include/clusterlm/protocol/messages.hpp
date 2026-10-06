#pragma once
// ClusterLM Father/Node execution protocol (spec addendum §9).
//
// Messages are encoded with the explicit little-endian ByteWriter and carried in transport frames. Three
// logical channels exist per peer pair, each its own authenticated connection, so bulk provisioning can
// never block cancellation and large prefill activations never delay commits:
//   kControl    — lease, plan, commit/abort, release, errors (small, latency critical)
//   kActivation — RunWindow / StageResult (bounded activation payloads)
//   kProvision  — ProvisionChunk / SealObject (bulk, lowest priority)
//
// Invariants enforced by the codecs and by both endpoints:
//   * No message type carries vocabulary token IDs, logits, sampling state or candidate strings.
//   * Every execution message names its lease generation and/or epoch so stale traffic is rejected.
//   * Every variable-length field is bounded at decode time before allocation.
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/digest.hpp"
#include "clusterlm/common/ids.hpp"
#include "clusterlm/domain/boundary.hpp"
#include "clusterlm/domain/execution_domain.hpp"
#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::protocol {

inline constexpr std::uint16_t kProtocolVersion = 1;

enum class Channel : std::uint8_t { kControl = 0, kActivation = 1, kProvision = 2 };
std::string_view to_string(Channel c);

enum class MessageType : std::uint16_t {
  kHello = 1,
  kHelloAck = 2,
  kOfferResources = 10,
  kPreparePlan = 11,
  kPlanAccepted = 12,
  kProvisionChunk = 13,
  kSealObject = 14,
  kObjectSealed = 15,
  kPlanReady = 16,
  kAuthorizePeer = 17,
  kProvisionStatus = 18,
  kOpenSession = 20,
  kSessionOpened = 21,
  kRunWindow = 22,
  kStageResult = 23,
  kCommitWindow = 24,
  kCommitAck = 25,
  kAbortSession = 26,
  kAbortWindow = 27,
  kWindowAborted = 28,
  kReleaseLease = 30,
  kReleaseComplete = 31,
  kUnpairNotice = 32,
  kError = 40,
  kPing = 41,
  kPong = 42,
};
std::string_view to_string(MessageType t);

enum class NodeRole : std::uint8_t { kFather = 0, kNode = 1, kBench = 2 };

// ---- session establishment --------------------------------------------------------------------------

struct Hello {
  std::uint16_t protocol_version = kProtocolVersion;
  NodeRole role = NodeRole::kFather;
  Channel channel = Channel::kControl;
  std::string device_id;        // paired device identity (certificate fingerprint on TLS transports)
  std::string backend_build;    // backend build hash; mismatches are rejected at PreparePlan
  LeaseGeneration lease;        // 0 on the first control connection; data channels name the lease they join
};

struct HelloAck {
  std::uint16_t protocol_version = kProtocolVersion;
  std::string device_id;
  LeaseGeneration lease;        // current lease generation offered by the Node
};

// ---- resources, plan and provisioning ----------------------------------------------------------------

enum class PowerSource : std::uint8_t { kUnknown = 0, kAc = 1, kBattery = 2 };

struct OfferResources {
  LeaseGeneration lease;
  std::uint64_t safe_ram_bytes = 0;
  std::uint64_t safe_vram_bytes = 0;
  std::uint64_t staging_disk_bytes = 0;  // 0 = disk staging not permitted
  std::string cpu_summary;               // e.g. "x86-64 avx2 avx512f 16t"; full profile via ClusterLM Bench
  std::string gpu_summary;
  PowerSource power = PowerSource::kUnknown;
};

struct StageAssignment {
  StageId stage;
  domain::StageRole role = domain::StageRole::kMiddle;
  objects::LayerRange layers;
  std::uint32_t max_context = 0;
  std::uint32_t max_window = 0;
  std::uint32_t max_local_batch = 0;  // DomainSpec::max_local_batch (0 = max_window)
};

struct ObjectAssignment {
  std::uint32_t object_index = 0;  // index into PreparePlan::manifest.objects
  objects::AllocationTarget target = objects::AllocationTarget::kCpuResident;
};

struct PreparePlan {
  LeaseGeneration lease;
  Digest256 model_root;          // ModelManifest::root_hash() of Father's canonical manifest
  std::string backend_build;
  Digest256 plan_hash;
  std::vector<StageAssignment> stages;
  std::uint64_t ram_cap_bytes = 0;
  std::uint64_t vram_cap_bytes = 0;
  std::uint64_t staging_cap_bytes = 0;
  // Plan-scoped manifest: geometry + only the objects assigned to this Node (no Father-only objects).
  objects::ModelManifest manifest;
  std::vector<ObjectAssignment> assignments;
};

struct PlanAccepted {
  LeaseGeneration lease;
  Digest256 plan_hash;
  std::uint64_t reserved_ram_bytes = 0;
  std::uint64_t reserved_vram_bytes = 0;
  std::uint64_t reserved_staging_bytes = 0;
};

struct ProvisionChunk {
  LeaseGeneration lease;
  std::uint32_t object_index = 0;
  std::uint64_t offset = 0;
  Digest256 chunk_digest;
  Bytes data;  // bounded by kMaxProvisionChunk
};

struct SealObject {
  LeaseGeneration lease;
  std::uint32_t object_index = 0;
  std::uint64_t total_length = 0;
  Digest256 object_digest;
};

struct ObjectSealed {
  LeaseGeneration lease;
  std::uint32_t object_index = 0;
};

// Sent by a Node right after HelloAck on every provision channel of a Preparing lease: the objects already
// sealed under this lease. Father resends only the rest, so a broken bulk connection resumes within the same
// uninterrupted lease instead of restarting from zero (partially received objects are reset and resent whole).
struct ProvisionStatus {
  LeaseGeneration lease;
  std::vector<std::uint32_t> sealed_objects;
};

struct PlanReady {
  LeaseGeneration lease;
  Digest256 plan_hash;
  std::uint64_t resident_bytes = 0;
  std::uint64_t prepare_ns = 0;
};

// Father authorizes a direct peer channel for one plan/lease: the Node with stage N forwards its
// StageResult activations directly to the peer owning stage N+1 instead of relaying through Father.
struct AuthorizePeer {
  LeaseGeneration lease;
  Digest256 plan_hash;
  StageId from_stage;
  StageId to_stage;
  std::string peer_device_id;   // must match the peer's authenticated identity
  std::string peer_endpoint;    // host:port of the peer's listener on the selected interface
  LeaseGeneration peer_lease;   // the lease generation the peer granted for this plan
};

// ---- execution ---------------------------------------------------------------------------------------

struct OpenSession {
  Epoch epoch;
  SessionId session;
};

struct SessionOpened {
  Epoch epoch;
  SessionId session;
};

struct RunWindow {
  LeaseGeneration lease;
  domain::WindowRequest request;   // epoch, session, window, base position, expected state, positions
  StageId stage;
  domain::StageActivations activations;
  // Direct routing: if set, the receiver forwards its result to the next stage's authorized peer instead of
  // returning it. The last stage of a forwarded chain always returns its StageResult to Father.
  bool forward_to_peer = false;
  // Timings of stages already traversed by a forwarded chain, so Father receives the whole breakdown.
  std::vector<domain::StageTiming> upstream_timings;
  // Prefill: every stage commits all positions as soon as it has computed them. Prefill windows are never
  // speculative, so this lets successive chunks pipeline through the stages without a commit round trip; each
  // domain still processes its chunks strictly in window order.
  bool auto_commit = false;
};

struct StageResult {
  Epoch epoch;
  SessionId session;
  WindowId window;
  StageId stage;               // stage that produced `activations`
  ErrorCode status = ErrorCode::kOk;
  std::string error_message;
  domain::StageActivations activations;  // empty on error
  std::vector<domain::StageTiming> timings;  // one entry per stage this result traversed
};

struct CommitWindow {
  domain::CommitRequest request;
  StageId stage;
};

struct CommitAckMessage {
  StageId stage;
  domain::CommitAck ack;
};

struct AbortWindow {
  Epoch epoch;
  SessionId session;
  WindowId window;
  StageId stage;
};

struct WindowAborted {
  StageId stage;
  domain::WindowAbortAck ack;
};

struct AbortSession {
  Epoch epoch;
  SessionId session;
  std::string reason;
};

// ---- release -----------------------------------------------------------------------------------------

enum class ReleaseReason : std::uint8_t { kFatherRequest = 0, kLocalActivity = 1, kPause = 2, kFault = 3, kShutdown = 4 };

struct ReleaseLease {
  LeaseGeneration lease;
  ReleaseReason reason = ReleaseReason::kFatherRequest;
};

struct ReleaseComplete {
  LeaseGeneration lease;
  bool resources_released = false;
  bool storage_cleaned = false;           // reported separately: never claimed while files remain
  std::uint64_t residual_bytes = 0;       // application-owned model bytes still present
  std::uint64_t release_ns = 0;
  std::string errors;
};

// Father -> Node, control channel: "I am unpairing you; stop trusting me." Accepted by a Node only on a channel
// whose TLS identity is one of its pinned (paired) Fathers; the Node then releases its lease, drops that trust and
// the service clears its paired-Father setting. Carries no data, only a nonce echoed in the Pong reply.
struct UnpairNotice {
  std::uint64_t nonce = 0;
};

struct ErrorMessage {
  ErrorCode code = ErrorCode::kInternal;
  std::string message;
  MessageType in_reply_to = MessageType::kError;
};

struct Ping {
  std::uint64_t nonce = 0;
};
struct Pong {
  std::uint64_t nonce = 0;
};

using Message = std::variant<Hello, HelloAck, OfferResources, PreparePlan, PlanAccepted, ProvisionChunk, SealObject,
                             ObjectSealed, PlanReady, AuthorizePeer, OpenSession, SessionOpened, RunWindow, StageResult,
                             CommitWindow, CommitAckMessage, AbortSession, ReleaseLease, ReleaseComplete, ErrorMessage,
                             Ping, Pong, AbortWindow, WindowAborted, ProvisionStatus, UnpairNotice>;

MessageType type_of(const Message& m);
Channel channel_of(MessageType t);

// Decode limits — enforced before allocation.
struct DecodeLimits {
  std::uint32_t max_window_positions = 4096;          // activations per RunWindow/StageResult
  std::uint32_t max_provision_chunk = 8u << 20;       // bytes per ProvisionChunk
  std::uint32_t max_manifest_objects = 1u << 20;
  std::uint32_t max_string = 64u << 10;
};

Bytes encode(const Message& m);
Result<Message> decode(MessageType type, ByteSpan payload, const DecodeLimits& limits = {});

}  // namespace clusterlm::protocol
