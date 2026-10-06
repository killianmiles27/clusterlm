#pragma once
// ggml tensor type table: type id -> name, elements per block, bytes per block.
//
// Values are taken from the pinned llama.cpp (third_party/upstream.json, commit 6753a033f058fbf778d282556ed9b16c78de7c71):
// ids from `enum ggml_type` in ggml/include/ggml.h, names/block sizes from `type_traits` in ggml/src/ggml.c, and
// bytes per block from sizeof(block_*) in ggml/src/ggml-common.h (verified by compiling those structs).
// Unknown or removed ids (4, 5, 31-33, 36-38, >= 43) are errors, never guessed.
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "clusterlm/common/status.hpp"

namespace clusterlm::objects {

enum class GgmlType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ8_1 = 9,
  kQ2_K = 10,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kQ8_K = 15,
  kIQ2_XXS = 16,
  kIQ2_XS = 17,
  kIQ3_XXS = 18,
  kIQ1_S = 19,
  kIQ4_NL = 20,
  kIQ3_S = 21,
  kIQ2_S = 22,
  kIQ4_XS = 23,
  kI8 = 24,
  kI16 = 25,
  kI32 = 26,
  kI64 = 27,
  kF64 = 28,
  kIQ1_M = 29,
  kBF16 = 30,
  kTQ1_0 = 34,
  kTQ2_0 = 35,
  kMXFP4 = 39,
  kNVFP4 = 40,
  kQ1_0 = 41,
  kQ2_0 = 42,
};

struct GgmlTypeInfo {
  GgmlType type;
  std::string_view name;      // lower-case canonical name, e.g. "iq3_s", "q4_k"
  std::uint32_t block_elems;  // elements per block (1 for unblocked scalar types)
  std::uint32_t block_bytes;  // bytes per block
  bool quantized;
};

// nullptr for an id that is not in the pinned table.
const GgmlTypeInfo* ggml_type_info(std::uint32_t id);
inline const GgmlTypeInfo& ggml_type_info(GgmlType t) { return *ggml_type_info(static_cast<std::uint32_t>(t)); }
// Case-insensitive; nullptr when unknown.
const GgmlTypeInfo* ggml_type_by_name(std::string_view name);
std::span<const GgmlTypeInfo> all_ggml_types();

// Bytes of one row of `ne0` elements (ne0 must be a whole number of blocks).
Result<std::uint64_t> ggml_row_bytes(GgmlType t, std::uint64_t ne0);
// Bytes of a tensor with the given dims (dims[0] is the contiguous axis). Overflow-checked; every dim must be > 0.
Result<std::uint64_t> ggml_tensor_bytes(GgmlType t, std::span<const std::uint64_t> dims);

}  // namespace clusterlm::objects
