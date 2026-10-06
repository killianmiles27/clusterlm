#pragma once
// Builds the canonical ModelManifest directly from real GGUF artifact(s) on Father: exact byte ranges into the
// GGUF files, no requantization, no full copy anywhere else. See docs/model-manifest.md for the object model,
// naming and the quant_type grammar.
//
// The tensor naming and metadata keys follow the Qwen3.8 Flash-Next ("qwen4exp") GGUF layout as consumed by
// Strata at the pinned commit (third_party/upstream.json):
//   <arch>.block_count, .embedding_length, .expert_count, .expert_used_count, .attention.head_count,
//   .attention.head_count_kv, .ple.ngram_size           (required)
//   .attention.key_length, .expert_feed_forward_length    (optional; cross-checked against tensor shapes)
//   blk.L.{ffn_gate,ffn_up,ffn_down}_exps.weight  stacked experts, ne = [in, out, n_experts]
//   blk.L.{ffn_gate,ffn_up,ffn_down}_shexp.weight, blk.L.ffn_gate_inp_shexp.weight  shared expert
//   blk.L.*  everything else of a layer (router, GDN/QSA projections, indexer, norms, gated-residual, PLE block)
//   token_embd.weight, per_layer_token_embd.weight, output*.weight/output_hc_*.weight, mtp.* / blk.N>=n_layers.* / *nextn*
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/objects/ggml_types.hpp"
#include "clusterlm/objects/gguf.hpp"
#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::objects {

// ---- tensor classification --------------------------------------------------------------------------------------

enum class TensorClass : std::uint8_t {
  kEmbedding,
  kPleLookup,
  kOutputHead,
  kMtp,
  kLayerDense,
  kSharedExpert,
  kExpertGate,
  kExpertUp,
  kExpertDown,
  kExpertGateUp,  // fused gate+up stacked tensor (one slice then holds both)
  kUnclassified,
};

struct TensorClassification {
  TensorClass cls = TensorClass::kUnclassified;
  std::optional<std::uint32_t> layer;  // for layer-scoped classes (and MTP "blk.N" with N >= n_layers)
  std::string reason;                  // why a tensor is unclassified
};

TensorClassification classify_tensor(std::string_view name, std::uint32_t n_layers);
std::string_view to_string(TensorClass c);

// ---- quant_type grammar ----------------------------------------------------------------------------------------

// One ggml type name when every range has that type ("iq3_s"); otherwise one name per source range, in range
// order, joined by '|' ("iq3_s|iq3_s|iq4_xs").
std::string encode_quant_types(std::span<const GgmlType> per_range);
// Inverse. A single name is broadcast to `n_ranges`; a list must have exactly n_ranges entries.
Result<std::vector<GgmlType>> decode_quant_types(std::string_view quant_type, std::size_t n_ranges);

// ---- build ------------------------------------------------------------------------------------------------------

struct BuildProgress {
  enum class Phase : std::uint8_t { kHashingObjects, kHashingShards };
  Phase phase = Phase::kHashingObjects;
  std::uint64_t bytes_done = 0;
  std::uint64_t bytes_total = 0;
  std::string_view current;  // object name or shard file name (never tensor contents)
};
// Return false to cancel (the build fails with kCancelled). Called from hashing threads, serialized.
using ProgressFn = std::function<bool(const BuildProgress&)>;

struct ManifestBuildOptions {
  bool hash_objects = false;  // compute source_digest/object_digest by streaming every range (83 GB for Ultra)
  bool hash_shards = false;   // additionally digest each whole file into ShardInfo::digest
  bool allow_unclassified = false;  // collect unknown tensors into the report instead of failing
  bool collect_layouts = true;      // fill ManifestBuildReport::layouts
  std::string artifact_id;    // default: general.name, else the first file name
  std::string license;        // default: general.license
  std::string require_architecture;  // non-empty: general.architecture must equal this
  GgufLimits limits;
  unsigned hash_threads = 1;
  std::size_t hash_buffer_bytes = 4u << 20;  // per thread
  ProgressFn progress;
};

// Where each tensor of an object sits inside the object's concatenated bytes. Not part of the manifest wire
// format (ManifestObject has no per-range tensor identity); backends that need tensor boundaries use this.
struct TensorLayoutEntry {
  std::string tensor;                // GGUF tensor name; for experts the stacked tensor the slice came from
  std::vector<std::uint64_t> dims;   // per-object dims (experts: the stacked dims without the expert axis)
  GgmlType type = GgmlType::kF32;
  std::uint64_t object_offset = 0;   // within the concatenated object
  std::uint64_t length = 0;
};
struct ObjectLayout {
  std::vector<TensorLayoutEntry> entries;  // one per source range, same order
};

struct UnclassifiedTensor {
  std::string name;
  std::uint32_t shard = 0;
  GgmlType type = GgmlType::kF32;
  std::uint64_t bytes = 0;
  std::string reason;
};

struct ManifestBuildReport {
  std::string architecture;
  std::vector<UnclassifiedTensor> unclassified;
  std::vector<ObjectLayout> layouts;  // parallel to manifest.objects (empty unless collect_layouts)
  std::uint64_t tensors_total = 0;
  std::uint64_t tensor_bytes_total = 0;
  std::uint64_t tensor_bytes_in_manifest = 0;
  bool hashed_objects = false;
  bool hashed_shards = false;
};

struct BuiltManifest {
  ModelManifest manifest;
  ManifestBuildReport report;
  std::vector<std::filesystem::path> shard_paths;  // same order as manifest.shards
};

Result<BuiltManifest> build_manifest(const std::vector<std::filesystem::path>& paths, const ManifestBuildOptions& opts = {});
// From already-opened files (the inspect tool opens once and reuses the tensor directory).
Result<BuiltManifest> build_manifest(const GgufModelFiles& files, const ManifestBuildOptions& opts = {});

struct HashOptions {
  bool hash_objects = true;
  bool hash_shards = false;
  unsigned threads = 1;
  std::size_t buffer_bytes = 4u << 20;
  ProgressFn progress;
};
// Fills source_digest/object_digest (and shard digests) of an existing manifest by streaming the shard files
// with bounded buffers. Can be run later or on another thread than build_manifest.
Status compute_manifest_digests(ModelManifest& manifest, const std::vector<std::filesystem::path>& shard_paths,
                                const HashOptions& opts = {});

}  // namespace clusterlm::objects
