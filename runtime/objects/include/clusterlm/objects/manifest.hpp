#pragma once
// ModelManifest: Father's per-tensor description of the canonical model, and the plan-scoped subset sent to
// a Node in PreparePlan.
//
// The manifest is what makes selected-object provisioning possible without a full model copy on a Node:
// every object names its exact source byte ranges (an expert's gate/up/down slices inside larger stacked
// tensors are separate ranges), its representation, and two distinct digests — the digest of the source
// bytes as they sit in the canonical shard, and the digest of the transformed object the Node receives.
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "clusterlm/common/digest.hpp"
#include "clusterlm/objects/geometry.hpp"

namespace clusterlm::objects {

enum class ObjectKind : std::uint8_t {
  kEmbedding = 0,     // token embedding table (Father prefix only)
  kPleLookup = 1,     // per-layer-embedding / n-gram lookup (Father SSD only)
  kLayerDense = 2,    // per-layer dense/mixer/router/norm weights
  kRoutedExpert = 3,  // one routed expert (gate+up+down ranges)
  kSharedExpert = 4,
  kOutputHead = 5,    // final norm + head (Father tail only)
  kMtp = 6,           // MTP drafter weights (Father tail only)
};

// Where an object lives for the lifetime of a lease.
enum class AllocationTarget : std::uint8_t {
  kGpuResident = 0,       // uploaded through bounded RAM staging; host staging discarded
  kCpuResident = 1,       // resident in the domain's RAM arena, executed by CPU kernels
  kTemporaryBacking = 2,  // lease-scoped temporary file (mapping/conversion workspace)
  kFatherOnly = 3,        // never provisioned to a Node
};

// Representation of the object's bytes. Quantized tensors must preserve type, block layout, scales and any
// transformation; "native" never permits silent requantization.
struct Representation {
  std::string quant_type;          // e.g. "f32", "q8_0-fixture", "iq3_s"
  std::uint32_t block_size = 0;    // elements per quant block (0 = unblocked)
  std::uint32_t conversion_version = 0;  // 0 = source bytes as stored; >0 = runtime-packed layout version
  bool little_endian = true;
  friend bool operator==(const Representation&, const Representation&) = default;
};

struct SourceRange {
  std::uint32_t shard = 0;  // index into ModelManifest::shards
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
};

struct ManifestObject {
  std::string name;  // stable identity, e.g. "blk.12.exp.301" (not a vocabulary token ID)
  ObjectKind kind = ObjectKind::kLayerDense;
  std::optional<std::uint32_t> layer;
  std::optional<std::uint32_t> expert;
  Representation representation;
  std::vector<SourceRange> source_ranges;  // concatenated in order to form the transformed object
  std::uint64_t byte_size = 0;             // size of the object as provisioned
  std::uint32_t alignment = 64;
  std::uint64_t peak_workspace = 0;        // temporary bytes needed while packing/uploading
  Digest256 source_digest;                 // digest over the concatenated source ranges
  Digest256 object_digest;                 // digest of the provisioned (possibly transformed) bytes
  std::vector<std::string> dependencies;

  bool father_only() const { return kind == ObjectKind::kEmbedding || kind == ObjectKind::kPleLookup ||
                                    kind == ObjectKind::kOutputHead || kind == ObjectKind::kMtp; }
};

struct ShardInfo {
  std::string file_name;  // relative to Father's model directory; never a Node path
  std::uint64_t byte_size = 0;
  Digest256 digest;
};

struct ModelManifest {
  static constexpr std::uint32_t kFormatVersion = 1;

  std::string artifact_id;  // e.g. "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_S" or a fixture ID
  std::string license;
  ModelGeometry geometry;
  std::vector<ShardInfo> shards;
  std::vector<ManifestObject> objects;

  // Root hash over geometry, shards and all object entries; names the model identity in PreparePlan.
  Digest256 root_hash() const;
  Status validate() const;
  const ManifestObject* find(std::string_view name) const;
  // Objects owned by a layer range (dense + experts + shared), excluding Father-only objects.
  std::vector<const ManifestObject*> layer_objects(LayerRange range) const;
  std::uint64_t total_bytes(LayerRange range) const;

  void encode(ByteWriter& w) const;
  static Result<ModelManifest> decode(ByteReader& r);
  // JSON form for humans, tooling and fixtures.
  std::string to_json() const;
  static Result<ModelManifest> from_json(std::string_view json);
};

// Canonical object naming shared by manifest builders, resolvers and backends.
std::string dense_object_name(std::uint32_t layer);
std::string expert_object_name(std::uint32_t layer, std::uint32_t expert);
std::string shared_expert_object_name(std::uint32_t layer);
inline constexpr std::string_view kEmbeddingObjectName = "token_embd";
inline constexpr std::string_view kPleObjectName = "ple_lookup";
inline constexpr std::string_view kHeadObjectName = "output_head";
inline constexpr std::string_view kMtpObjectName = "mtp_drafter";

std::string_view to_string(ObjectKind kind);
std::string_view to_string(AllocationTarget target);

}  // namespace clusterlm::objects
