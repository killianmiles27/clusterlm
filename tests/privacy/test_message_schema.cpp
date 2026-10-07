// Privacy schema test: the protocol can structurally not carry vocabulary token arrays, decoded text, logits, chat
// objects, system prompts or roles to a Node.
//
// Method. Every protocol::Message alternative has an entry in the registry below stating (a) the exact encoded size
// when all of its strings and collections are empty (`fixed`), (b) its allowlisted variable-length fields: strings
// (device ids, endpoints, error messages, reasons ...) and element-counted collections, and how many bytes each
// element adds. The test encodes a minimal instance and a maximal instance and requires the sizes to match the
// registry EXACTLY. Consequently:
//   * a new field (of any type) in an existing message changes the fixed size and fails here until this registry,
//     and with it the privacy review, is updated deliberately;
//   * a new message type fails the static_assert in message_samples.hpp and the coverage check below;
//   * RunWindow/StageResult are exactly "fixed header + positions x bytes_per_position (+ timings)": there is no
//     room for a token array or text next to the activations.
#include <doctest/doctest.h>

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>

#include "message_samples.hpp"

using namespace clusterlm;
using namespace clusterlm::protocol;
using clusterlm::testing::kMessageAlternatives;

namespace {

constexpr std::size_t kStageTimingBytes = 36;      // 3 x u64 + 3 x u32
constexpr std::size_t kStageAssignmentBytes = 25;  // u32 + u8 + 2 x u32 + 3 x u32

struct Shape {
  std::size_t fixed = 0;                    // bytes with every string/collection empty and zero positions
  std::vector<std::string> string_fields;   // allowlist (documented below in `kAllowedStringFields`)
  std::vector<std::string> counted_fields;  // allowlisted element-counted collections (indices, timings)
  std::function<Message()> minimal;
  // Fills every allowlisted string with `n` characters and every counted collection with `n` elements; returns the
  // number of encoded bytes that adds.
  std::function<std::size_t(Message&, std::size_t)> grow;
};

// The ONLY places where a message may carry variable-length text. Reviewed together with the threat model: none of
// them can hold a prompt, response, role or token, and none of them is ever interpreted as a path or command.
const std::set<std::string> kAllowedStringFields = {
    "Hello.device_id",          "Hello.backend_build",       "HelloAck.device_id",      "OfferResources.cpu_summary",
    "OfferResources.gpu_summary", "PreparePlan.backend_build", "AuthorizePeer.peer_device_id",
    "AuthorizePeer.peer_endpoint", "StageResult.error_message", "AbortSession.reason", "ReleaseComplete.errors",
    "ErrorMessage.message"};

// Field names that would indicate a way to make a Node execute or open something.
const std::set<std::string> kForbiddenFieldNames = {"command", "cmd",  "path",  "exe",   "executable", "argv",
                                                    "args",    "script", "shell", "file", "url",        "uri"};

std::size_t size_of(const Message& m) { return encode(m).size(); }

template <typename T>
Shape fixed_shape(std::size_t fixed, std::function<Message()> minimal = [] { return Message(T{}); }) {
  Shape s;
  s.fixed = fixed;
  s.minimal = std::move(minimal);
  s.grow = [](Message&, std::size_t) { return std::size_t{0}; };
  return s;
}

std::size_t empty_manifest_bytes() {
  ByteWriter w;
  objects::ModelManifest{}.encode(w);
  return w.size();
}

std::map<MessageType, Shape> build_registry() {
  std::map<MessageType, Shape> r;
  auto str = [](std::string& s, std::size_t n) {
    s.assign(n, 'x');
    return n;
  };

  {  // Hello: u16 + role u8 + channel u8 + 2 strings + lease u64
    Shape s = fixed_shape<Hello>(2 + 1 + 1 + 4 + 4 + 8);
    s.string_fields = {"Hello.device_id", "Hello.backend_build"};
    s.grow = [str](Message& m, std::size_t n) {
      auto& v = std::get<Hello>(m);
      return str(v.device_id, n) + str(v.backend_build, n);
    };
    r[MessageType::kHello] = s;
  }
  {
    Shape s = fixed_shape<HelloAck>(2 + 4 + 8);
    s.string_fields = {"HelloAck.device_id"};
    s.grow = [str](Message& m, std::size_t n) { return str(std::get<HelloAck>(m).device_id, n); };
    r[MessageType::kHelloAck] = s;
  }
  {
    Shape s = fixed_shape<OfferResources>(8 + 3 * 8 + 4 + 4 + 1);
    s.string_fields = {"OfferResources.cpu_summary", "OfferResources.gpu_summary"};
    s.grow = [str](Message& m, std::size_t n) {
      auto& v = std::get<OfferResources>(m);
      return str(v.cpu_summary, n) + str(v.gpu_summary, n);
    };
    r[MessageType::kOfferResources] = s;
  }
  {  // PreparePlan: lease, root, build str, plan hash, stage count, caps, manifest, assignment count
    Shape s = fixed_shape<PreparePlan>(8 + 32 + 4 + 32 + 4 + 3 * 8 + empty_manifest_bytes() + 4);
    s.string_fields = {"PreparePlan.backend_build"};
    s.counted_fields = {"PreparePlan.stages", "PreparePlan.assignments"};
    s.grow = [str](Message& m, std::size_t n) {
      auto& v = std::get<PreparePlan>(m);
      v.stages.assign(n, StageAssignment{});
      v.assignments.assign(n, ObjectAssignment{});
      return str(v.backend_build, n) + n * kStageAssignmentBytes + n * 5;
    };
    r[MessageType::kPreparePlan] = s;
  }
  r[MessageType::kPlanAccepted] = fixed_shape<PlanAccepted>(8 + 32 + 3 * 8);
  {  // ProvisionChunk: lease, index, offset, digest, blob. The blob is digest-checked object bytes.
    Shape s = fixed_shape<ProvisionChunk>(8 + 4 + 8 + 32 + 4);
    s.string_fields = {};
    s.counted_fields = {"ProvisionChunk.data"};
    s.grow = [](Message& m, std::size_t n) {
      std::get<ProvisionChunk>(m).data.assign(n, 0x5A);
      return n;
    };
    r[MessageType::kProvisionChunk] = s;
  }
  r[MessageType::kSealObject] = fixed_shape<SealObject>(8 + 4 + 8 + 32);
  r[MessageType::kObjectSealed] = fixed_shape<ObjectSealed>(8 + 4);
  r[MessageType::kPlanReady] = fixed_shape<PlanReady>(8 + 32 + 8 + 8 + 3 * 8);
  {
    Shape s = fixed_shape<AuthorizePeer>(8 + 32 + 4 + 4 + 4 + 4 + 8);
    s.string_fields = {"AuthorizePeer.peer_device_id", "AuthorizePeer.peer_endpoint"};
    s.grow = [str](Message& m, std::size_t n) {
      auto& v = std::get<AuthorizePeer>(m);
      return str(v.peer_device_id, n) + str(v.peer_endpoint, n);
    };
    r[MessageType::kAuthorizePeer] = s;
  }
  r[MessageType::kOpenSession] = fixed_shape<OpenSession>(8 + 8);
  r[MessageType::kSessionOpened] = fixed_shape<SessionOpened>(8 + 8);
  {  // RunWindow: lease, WindowRequest(44), stage, 2 bools, timing count, activations header(24) [+ payload]
    Shape s = fixed_shape<RunWindow>(8 + 44 + 4 + 2 + 4 + 24, [] {
      RunWindow m;
      m.activations.layout = clusterlm::testing::sample_layout();
      return Message(m);
    });
    s.counted_fields = {"RunWindow.upstream_timings"};
    s.grow = [](Message& m, std::size_t n) {
      std::get<RunWindow>(m).upstream_timings.assign(n, domain::StageTiming{});
      return n * kStageTimingBytes;
    };
    r[MessageType::kRunWindow] = s;
  }
  {  // StageResult (OK): epoch, session, window, stage, status u16, error str, has_acts, acts header, timing count
    Shape s = fixed_shape<StageResult>(8 + 8 + 8 + 4 + 2 + 4 + 1 + 24 + 4, [] {
      StageResult m;
      m.activations.layout = clusterlm::testing::sample_layout();
      return Message(m);
    });
    s.string_fields = {"StageResult.error_message"};
    s.counted_fields = {"StageResult.timings"};
    s.grow = [str](Message& m, std::size_t n) {
      auto& v = std::get<StageResult>(m);
      v.timings.assign(n, domain::StageTiming{});
      return str(v.error_message, n) + n * kStageTimingBytes;
    };
    r[MessageType::kStageResult] = s;
  }
  r[MessageType::kCommitWindow] = fixed_shape<CommitWindow>(36 + 4);
  r[MessageType::kCommitAck] = fixed_shape<CommitAckMessage>(4 + 8 + 8 + 8 + 8);
  {
    Shape s = fixed_shape<AbortSession>(8 + 8 + 4);
    s.string_fields = {"AbortSession.reason"};
    s.grow = [str](Message& m, std::size_t n) { return str(std::get<AbortSession>(m).reason, n); };
    r[MessageType::kAbortSession] = s;
  }
  r[MessageType::kReleaseLease] = fixed_shape<ReleaseLease>(8 + 1);
  {
    Shape s = fixed_shape<ReleaseComplete>(8 + 1 + 1 + 8 + 8 + 4);
    s.string_fields = {"ReleaseComplete.errors"};
    s.grow = [str](Message& m, std::size_t n) { return str(std::get<ReleaseComplete>(m).errors, n); };
    r[MessageType::kReleaseComplete] = s;
  }
  {
    Shape s = fixed_shape<ErrorMessage>(2 + 4 + 2);
    s.string_fields = {"ErrorMessage.message"};
    s.grow = [str](Message& m, std::size_t n) { return str(std::get<ErrorMessage>(m).message, n); };
    r[MessageType::kError] = s;
  }
  r[MessageType::kPing] = fixed_shape<Ping>(8);
  r[MessageType::kPong] = fixed_shape<Pong>(8);
  r[MessageType::kUnpairNotice] = fixed_shape<UnpairNotice>(8);
  r[MessageType::kAbortWindow] = fixed_shape<AbortWindow>(8 + 8 + 8 + 4);
  r[MessageType::kWindowAborted] = fixed_shape<WindowAborted>(4 + 8 + 8 + 8 + 8);
  {
    Shape s = fixed_shape<ProvisionStatus>(8 + 4);
    s.counted_fields = {"ProvisionStatus.sealed_objects"};
    s.grow = [](Message& m, std::size_t n) {
      std::get<ProvisionStatus>(m).sealed_objects.assign(n, 1);
      return n * 4;
    };
    r[MessageType::kProvisionStatus] = s;
  }
  return r;
}

}  // namespace

TEST_CASE("every protocol message has a reviewed shape: exact encoded size with and without variable fields") {
  const auto registry = build_registry();
  REQUIRE(registry.size() == kMessageAlternatives);
  std::set<MessageType> seen;
  for (const auto& [type, shape] : registry) {
    CAPTURE(std::string(to_string(type)));
    Message min = shape.minimal();
    CHECK(type_of(min) == type);
    CHECK(size_of(min) == shape.fixed);
    for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{200}}) {
      Message big = shape.minimal();
      const std::size_t added = shape.grow(big, n);
      CHECK(size_of(big) == shape.fixed + added);
    }
    seen.insert(type);
  }
  // Every alternative of the variant is covered (the sample set is the variant's order).
  std::set<MessageType> in_variant;
  for (const Message& m : clusterlm::testing::sample_messages()) in_variant.insert(type_of(m));
  CHECK(in_variant == seen);
}

TEST_CASE("string fields are exactly the reviewed allowlist, and none can name a command or path") {
  const auto registry = build_registry();
  std::set<std::string> declared;
  for (const auto& [type, shape] : registry) {
    for (const auto& f : shape.string_fields) {
      declared.insert(f);
      // The field name is the part after the dot: it must not suggest execution or file access.
      const std::string field = f.substr(f.find('.') + 1);
      for (const auto& bad : kForbiddenFieldNames) CHECK_MESSAGE(field.find(bad) == std::string::npos, f);
    }
  }
  CHECK(declared == kAllowedStringFields);
}

TEST_CASE("activation-carrying messages are exactly header + positions x bytes_per_position") {
  const objects::ModelManifest manifest = clusterlm::testing::sample_manifest();
  const domain::BoundaryLayout layout = domain::BoundaryLayout::for_geometry(manifest.geometry);
  REQUIRE(layout.bytes_per_position() == (std::uint64_t{4} * 32 + 32 + 4) * 4);  // (hc*H + H + hc) FP32 values
  for (std::uint32_t q : {0u, 1u, 2u, 3u, 17u, 64u}) {
    CAPTURE(q);
    RunWindow run;
    run.activations = clusterlm::testing::sample_activations(q, layout, 5);
    run.request.positions = q;
    CHECK(size_of(run) == 8 + 44 + 4 + 2 + 4 + 24 + q * layout.bytes_per_position());

    StageResult result;
    result.activations = clusterlm::testing::sample_activations(q, layout, 5);
    CHECK(size_of(result) == 8 + 8 + 8 + 4 + 2 + 4 + 1 + 24 + q * layout.bytes_per_position() + 4);

    // With timings the only additional bytes are the fixed-width timing records.
    run.upstream_timings.assign(3, domain::StageTiming{});
    CHECK(size_of(run) == 8 + 44 + 4 + 2 + 4 + 24 + q * layout.bytes_per_position() + 3 * kStageTimingBytes);
  }
}

TEST_CASE("an error StageResult carries no activations, only the allowlisted message") {
  StageResult r;
  r.status = ErrorCode::kStaleEpoch;
  r.error_message = "stale";
  // epoch, session, window, stage, status, string, has_acts=false, timing count
  CHECK(size_of(r) == 8 + 8 + 8 + 4 + 2 + (4 + 5) + 1 + 4);
  // A hostile sender cannot smuggle activations next to an error status: the codec requires has_acts == (status OK).
  ByteWriter w;
  w.u64(1);
  w.u64(1);
  w.u64(1);
  w.u32(1);
  w.u16(static_cast<std::uint16_t>(ErrorCode::kInternal));
  w.str("x");
  w.boolean(true);  // claims activations although status != OK
  clusterlm::testing::sample_activations(1, clusterlm::testing::sample_layout()).encode(w);
  w.u32(0);
  CHECK_FALSE(decode(MessageType::kStageResult, w.bytes()).is_ok());
}

TEST_CASE("the plan-scoped manifest is the only structured text a Node receives, and it is allowlisted too") {
  // ModelManifest: version u32, artifact_id, license, geometry(family + 15 u32 + kinds), shards, objects.
  const std::size_t base = empty_manifest_bytes();
  CHECK(base == 4 + 4 + 4 + (4 + 15 * 4 + 4) + 4 + 4);

  objects::ModelManifest m;
  m.artifact_id.assign(10, 'a');
  m.license.assign(11, 'l');
  m.geometry.family.assign(12, 'f');
  m.geometry.layer_kinds.assign(5, objects::LayerKind::kRecurrent);
  objects::ShardInfo shard;
  shard.file_name.assign(13, 's');
  m.shards.assign(2, shard);
  objects::ManifestObject o;
  o.name.assign(14, 'n');
  o.representation.quant_type.assign(15, 'q');
  o.source_ranges.assign(3, objects::SourceRange{});
  o.dependencies = {"dep-a", "dep-bb"};
  m.objects.assign(2, o);
  ByteWriter w;
  m.encode(w);
  constexpr std::size_t kShard = 4 + 8 + 32;  // name length prefix + size + digest
  constexpr std::size_t kObjectFixed = 4 + 1 + 1 + 4 + 1 + 4 + 4 + 4 + 4 + 1 + 4 + 8 + 4 + 8 + 32 + 32 + 4;
  const std::size_t expected = base + 10 + 11 + 12 + 5 + 2 * (kShard + 13) +
                               2 * (kObjectFixed + 14 + 15 + 3 * 20 + (4 + 5) + (4 + 6));
  CHECK(w.size() == expected);
}

TEST_CASE("every sample round-trips and uses its documented channel") {
  for (const Message& m : clusterlm::testing::sample_messages()) {
    const MessageType type = type_of(m);
    CAPTURE(std::string(to_string(type)));
    const Bytes bytes = encode(m);
    auto back = decode(type, bytes);
    REQUIRE(back.is_ok());
    CHECK(encode(back.value()) == bytes);
    const Channel c = channel_of(type);
    const bool bulk = type == MessageType::kProvisionChunk || type == MessageType::kSealObject ||
                      type == MessageType::kObjectSealed || type == MessageType::kProvisionStatus;
    const bool activation = type == MessageType::kRunWindow || type == MessageType::kStageResult;
    CHECK((c == Channel::kProvision) == bulk);
    CHECK((c == Channel::kActivation) == activation);
  }
}

TEST_CASE("no wire message type exists beyond the reviewed set (no tokens, logits, chat, command messages)") {
  // The numeric MessageType space is closed: anything not in the registry decodes to a protocol error. This pins
  // the reviewed set so that adding, say, a "SetPrompt" or "Exec" message cannot go unnoticed.
  const auto registry = build_registry();
  std::size_t accepted = 0;
  for (std::uint32_t t = 0; t < 256; ++t) {
    const auto type = static_cast<MessageType>(t);
    auto r = decode(type, Bytes{});
    const bool known = registry.count(type) != 0;
    if (!known) CHECK_FALSE(r.is_ok());
    if (known) ++accepted;
  }
  CHECK(accepted == registry.size());
}

TEST_CASE("PlanReady's timing breakdown is an optional trailing extension: old bodies decode, new bodies round-trip") {
  PlanReady full{LeaseGeneration{7}, clusterlm::testing::sample_digest(9), 1ull << 20, 123456, 40000, 50000, 33456};
  const Bytes encoded = encode(Message(full));
  REQUIRE(encoded.size() == 8 + 32 + 8 + 8 + 3 * 8);
  auto back = decode(MessageType::kPlanReady, encoded);
  REQUIRE(back.is_ok());
  const auto& r = std::get<PlanReady>(back.value());
  CHECK(r.chunk_write_ns == 40000);
  CHECK(r.seal_hash_ns == 50000);
  CHECK(r.build_ns == 33456);
  // A Node built before the extension sends only lease, plan hash, resident bytes and prepare_ns.
  const Bytes old_body(encoded.begin(), encoded.end() - 3 * 8);
  auto legacy = decode(MessageType::kPlanReady, old_body);
  REQUIRE_MESSAGE(legacy.is_ok(), legacy.status().to_string());
  const auto& l = std::get<PlanReady>(legacy.value());
  CHECK(l.prepare_ns == 123456);
  CHECK(l.chunk_write_ns == 0);
  CHECK(l.seal_hash_ns == 0);
  CHECK(l.build_ns == 0);
  // A partial extension (one or two of the three fields) is malformed, not silently accepted.
  for (std::size_t drop : {8u, 16u}) {
    const Bytes partial(encoded.begin(), encoded.end() - static_cast<std::ptrdiff_t>(drop));
    CHECK_FALSE(decode(MessageType::kPlanReady, partial).is_ok());
  }
}
