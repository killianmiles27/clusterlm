#pragma once
// Bounds-checked GGUF v2/v3 reader (little-endian) for Father's manifest builder.
//
// Only the header region of a file is ever read (magic, metadata KV, tensor directory); tensor data is never
// mapped or read here. Every count and length is checked against GgufLimits BEFORE any allocation, all
// offset/size arithmetic is overflow-checked, and every tensor's byte range must lie inside the file.
//
// Layout reference (GGUF spec, as implemented by the pinned llama.cpp/ggml gguf.cpp):
//   u32 magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
//   n_kv x { string key | u32 type | value }          string = u64 length + bytes
//   n_tensors x { string name | u32 n_dims | u64 dims[n_dims] | u32 ggml_type | u64 offset }
//   padding to `general.alignment` (default 32) | tensor data (offsets relative to the data start)
// Version 1 (32-bit counts) and big-endian files are rejected.
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/objects/ggml_types.hpp"

namespace clusterlm::objects {

struct GgufLimits {
  std::uint64_t max_tensors = 1u << 20;
  std::uint64_t max_kv = 1u << 20;
  std::uint64_t max_string_bytes = 64u << 10;
  std::uint64_t max_array_elements = 16u << 20;
  std::uint64_t max_header_bytes = 256ull << 20;  // magic .. end of the tensor directory
  std::uint64_t max_alignment = 1ull << 30;
  std::uint32_t max_shards = 4096;
  // Metadata keys whose array values are retained IN FULL (instead of the first GgufValue::kRetainedArrayItems
  // items), bounded by max_retained_array_elements. The tokenizer builder asks for tokenizer.ggml.{tokens,
  // merges,token_type}; nothing else needs more than a prefix.
  std::vector<std::string> retain_full_arrays;
  std::uint64_t max_retained_array_elements = 1u << 20;
  // Accept a tensor-less file that ends right after its header without the alignment padding (llama.cpp's
  // vocab-only test files). Metadata readers only; off by default.
  bool allow_vocab_only_tail = false;
};

enum class GgufValueType : std::uint32_t {
  kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7,
  kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12,
};

struct GgufValue {
  // Only the first kRetainedArrayItems elements of an array are kept (unless GgufLimits::retain_full_arrays
  // names its key); `count` is the true length.
  static constexpr std::size_t kRetainedArrayItems = 64;

  GgufValueType type = GgufValueType::kU32;
  GgufValueType elem_type = GgufValueType::kU32;  // arrays only (never kArray: nested arrays are rejected)
  std::uint64_t count = 0;                        // arrays only
  std::uint64_t u = 0;                            // unsigned/bool payload; signed integers in `i`
  std::int64_t i = 0;
  double f = 0;
  std::string s;
  std::vector<GgufValue> items;  // retained array prefix

  bool is_array() const { return type == GgufValueType::kArray; }
  bool is_string() const { return type == GgufValueType::kString; }
  bool is_integer() const;
  bool is_float() const { return type == GgufValueType::kF32 || type == GgufValueType::kF64; }
  // Integer (signed or unsigned) value if it is >= 0.
  std::optional<std::uint64_t> as_u64() const;
  std::optional<double> as_double() const;  // integers and floats
};

struct GgufTensorInfo {
  std::string name;
  std::uint32_t n_dims = 0;
  std::array<std::uint64_t, 4> dims{1, 1, 1, 1};  // dims[0] is the contiguous axis (GGUF/ggml order)
  GgmlType type = GgmlType::kF32;
  std::uint64_t offset = 0;       // relative to GgufFile::data_start
  std::uint64_t n_elements = 0;
  std::uint64_t n_bytes = 0;      // exact payload size from type + dims
  std::uint64_t file_offset = 0;  // absolute: data_start + offset

  std::uint64_t row_bytes() const;  // bytes of one row (dims[0] elements)
};

struct GgufFile {
  std::string path;  // label only
  std::uint32_t version = 0;
  std::uint64_t file_size = 0;
  std::uint64_t header_bytes = 0;  // bytes consumed by magic .. tensor directory
  std::uint64_t alignment = 32;
  std::uint64_t data_start = 0;
  std::map<std::string, GgufValue> metadata;
  std::vector<GgufTensorInfo> tensors;  // in directory order

  const GgufValue* find(const std::string& key) const;
  const GgufTensorInfo* find_tensor(std::string_view name) const;

  std::unordered_map<std::string, std::size_t> tensor_index;  // name -> index into `tensors`
};

// Random-access byte source: a file on disk or memory (fuzzing/tests).
class GgufSource {
 public:
  virtual ~GgufSource() = default;
  virtual std::uint64_t size() const = 0;
  virtual Status read(std::uint64_t offset, void* dst, std::size_t n) = 0;  // exactly n bytes or an error
};

Result<GgufFile> parse_gguf(GgufSource& src, const GgufLimits& limits = {}, std::string label = {});
Result<GgufFile> read_gguf_file(const std::filesystem::path& path, const GgufLimits& limits = {});
// `bytes` stands in for the whole file (tensor ranges are checked against bytes.size()).
Result<GgufFile> parse_gguf_bytes(ByteSpan bytes, const GgufLimits& limits = {});

// A model = one GGUF, a split GGUF (split.no / split.count / split.tensors.count), or a primary GGUF plus
// companion files (e.g. a separate MTP GGUF) that carry no split keys.
struct GgufModelFiles {
  std::vector<std::filesystem::path> paths;  // same order as `shards`
  std::vector<GgufFile> shards;              // split shards ordered by split.no; companions in argument order
  bool is_split = false;

  const GgufFile& meta() const { return shards.front(); }  // metadata source: first shard / primary file
  struct TensorRef {
    std::uint32_t shard = 0;
    const GgufTensorInfo* info = nullptr;
  };
  std::optional<TensorRef> find_tensor(std::string_view name) const;
  std::size_t tensor_count() const;
};

// Opens and cross-validates every file. Split shards must agree on split.count / split.tensors.count, cover
// split.no 0..count-1 exactly once and hold split.tensors.count tensors in total. A tensor name may appear in
// only one file. Files with no split keys are treated as companions only when several are given.
Result<GgufModelFiles> open_gguf_model(const std::vector<std::filesystem::path>& paths, const GgufLimits& limits = {});
// Given any shard of a "<stem>-00001-of-00003.gguf" set, returns all sibling paths (existence-checked).
// A path that does not match the pattern is returned unchanged.
Result<std::vector<std::filesystem::path>> expand_split_paths(const std::filesystem::path& any_shard);

}  // namespace clusterlm::objects
