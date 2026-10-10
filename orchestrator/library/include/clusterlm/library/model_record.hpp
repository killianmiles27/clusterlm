#pragma once
// Host model library: records of the exact model artifacts that live on the Host (the only place weights are stored
// permanently; Workers only ever hold ephemeral, plan-scoped allocations).
//
// A ModelRecord is identity and structure, never a measurement and never a qualification claim:
//   * `id` (mdl_*) is derived from `structure_digest`: SHA-256 over the ordered file names + sizes and the whole tensor
//     directory (names, types, dims) of the GGUF file set. Two copies of the same artifact get the same id; any change
//     of tensors or quantization changes it. It is NOT a content hash: `root_hash` (the manifest root over the tensor
//     bytes) is filled only by verification, and a pin (`pinned_root`) only by an explicit user confirmation.
//   * `dir` is a Host-local path; it never leaves the Host (not exported, not logged, not sent to a Worker).
//   * `quant` is the dominant tensor type by bytes ("IQ3_S"); `tensor_types` lists every type present, so a mixed
//     quantization can never be mistaken for a single one.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::library {

inline constexpr std::string_view kModelIdPrefix = "mdl_";
inline constexpr std::size_t kMaxRecordFiles = 4096;
inline constexpr std::size_t kMaxTensorTypes = 64;

struct ModelFile {
  std::string name;  // file name only (no directory), in shard order
  std::uint64_t bytes = 0;
  friend bool operator==(const ModelFile&, const ModelFile&) = default;
};

struct TensorTypeStat {
  std::string type;  // canonical ggml name, upper case ("IQ3_S", "F32")
  std::uint64_t tensors = 0;
  std::uint64_t bytes = 0;
  friend bool operator==(const TensorTypeStat&, const TensorTypeStat&) = default;
};

struct ModelRecord {
  std::string id;            // mdl_<24 hex>; immutable
  std::string name;          // mutable display label
  std::string family;        // mutable label used for profile identity matching (default: general.basename/name)
  std::string architecture;  // general.architecture
  std::string container = "gguf";
  std::string quant;         // dominant tensor type by bytes
  std::string dir;           // Host-local directory holding `files`
  std::vector<ModelFile> files;
  bool split = false;
  std::uint64_t total_bytes = 0;
  std::uint64_t tensor_count = 0;
  std::vector<TensorTypeStat> tensor_types;
  std::optional<std::uint32_t> block_count, context_length, expert_count;
  std::string structure_digest;               // 64 hex; see above
  std::optional<std::string> root_hash;       // 64 hex; set by verification only
  std::optional<std::string> pinned_root;     // 64 hex; set by explicit user confirmation only
  std::int64_t discovered_at_unix = 0;
  friend bool operator==(const ModelRecord&, const ModelRecord&) = default;
};

bool is_model_id(std::string_view s);
// "mdl_" + first 24 hex characters of the structure digest.
std::string model_id_from_digest(std::string_view structure_digest_hex);

Status validate(const ModelRecord& r);

// JSON (one record). Parse = bounded, strict types, full validation; unknown fields ignored.
std::string to_json(const ModelRecord& r);
Result<ModelRecord> model_record_from_json(std::string_view json);

}  // namespace clusterlm::library
