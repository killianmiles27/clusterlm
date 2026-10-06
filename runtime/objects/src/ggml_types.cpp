#include "clusterlm/objects/ggml_types.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <string>

namespace clusterlm::objects {

namespace {

using T = GgmlType;
// Pin: llama.cpp 6753a033f058fbf778d282556ed9b16c78de7c71 (see header). Sorted by id.
constexpr std::array kTypes = std::to_array<GgmlTypeInfo>({
    {T::kF32, "f32", 1, 4, false},
    {T::kF16, "f16", 1, 2, false},
    {T::kQ4_0, "q4_0", 32, 18, true},
    {T::kQ4_1, "q4_1", 32, 20, true},
    {T::kQ5_0, "q5_0", 32, 22, true},
    {T::kQ5_1, "q5_1", 32, 24, true},
    {T::kQ8_0, "q8_0", 32, 34, true},
    {T::kQ8_1, "q8_1", 32, 36, true},
    {T::kQ2_K, "q2_k", 256, 84, true},
    {T::kQ3_K, "q3_k", 256, 110, true},
    {T::kQ4_K, "q4_k", 256, 144, true},
    {T::kQ5_K, "q5_k", 256, 176, true},
    {T::kQ6_K, "q6_k", 256, 210, true},
    {T::kQ8_K, "q8_k", 256, 292, true},
    {T::kIQ2_XXS, "iq2_xxs", 256, 66, true},
    {T::kIQ2_XS, "iq2_xs", 256, 74, true},
    {T::kIQ3_XXS, "iq3_xxs", 256, 98, true},
    {T::kIQ1_S, "iq1_s", 256, 50, true},
    {T::kIQ4_NL, "iq4_nl", 32, 18, true},
    {T::kIQ3_S, "iq3_s", 256, 110, true},
    {T::kIQ2_S, "iq2_s", 256, 82, true},
    {T::kIQ4_XS, "iq4_xs", 256, 136, true},
    {T::kI8, "i8", 1, 1, false},
    {T::kI16, "i16", 1, 2, false},
    {T::kI32, "i32", 1, 4, false},
    {T::kI64, "i64", 1, 8, false},
    {T::kF64, "f64", 1, 8, false},
    {T::kIQ1_M, "iq1_m", 256, 56, true},
    {T::kBF16, "bf16", 1, 2, false},
    {T::kTQ1_0, "tq1_0", 256, 54, true},
    {T::kTQ2_0, "tq2_0", 256, 66, true},
    {T::kMXFP4, "mxfp4", 32, 17, true},
    {T::kNVFP4, "nvfp4", 64, 36, true},
    {T::kQ1_0, "q1_0", 128, 18, true},
    {T::kQ2_0, "q2_0", 64, 18, true},
});

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
  return true;
}

}  // namespace

const GgmlTypeInfo* ggml_type_info(std::uint32_t id) {
  // Ids are sparse; binary search on the sorted table.
  auto it = std::lower_bound(kTypes.begin(), kTypes.end(), id, [](const GgmlTypeInfo& t, std::uint32_t v) {
    return static_cast<std::uint32_t>(t.type) < v;
  });
  if (it == kTypes.end() || static_cast<std::uint32_t>(it->type) != id) return nullptr;
  return &*it;
}

const GgmlTypeInfo* ggml_type_by_name(std::string_view name) {
  for (const GgmlTypeInfo& t : kTypes)
    if (iequals(t.name, name)) return &t;
  return nullptr;
}

std::span<const GgmlTypeInfo> all_ggml_types() { return kTypes; }

Result<std::uint64_t> ggml_row_bytes(GgmlType t, std::uint64_t ne0) {
  const GgmlTypeInfo* info = ggml_type_info(static_cast<std::uint32_t>(t));
  if (info == nullptr) return make_error(ErrorCode::kInvalidArgument, "ggml: unknown tensor type");
  if (ne0 % info->block_elems != 0)
    return make_error(ErrorCode::kInvalidArgument,
                      "ggml: row of " + std::to_string(ne0) + " elements is not a whole number of " +
                          std::string(info->name) + " blocks (" + std::to_string(info->block_elems) + ")");
  const std::uint64_t blocks = ne0 / info->block_elems;
  if (blocks > std::numeric_limits<std::uint64_t>::max() / info->block_bytes)
    return make_error(ErrorCode::kOutOfRange, "ggml: row byte size overflows");
  return blocks * info->block_bytes;
}

Result<std::uint64_t> ggml_tensor_bytes(GgmlType t, std::span<const std::uint64_t> dims) {
  if (dims.empty()) return make_error(ErrorCode::kInvalidArgument, "ggml: tensor has no dimensions");
  if (dims[0] == 0) return make_error(ErrorCode::kInvalidArgument, "ggml: zero-sized dimension");
  CLM_ASSIGN_OR_RETURN(std::uint64_t bytes, ggml_row_bytes(t, dims[0]));
  for (std::size_t i = 1; i < dims.size(); ++i) {
    if (dims[i] == 0) return make_error(ErrorCode::kInvalidArgument, "ggml: zero-sized dimension");
    if (bytes > std::numeric_limits<std::uint64_t>::max() / dims[i])
      return make_error(ErrorCode::kOutOfRange, "ggml: tensor byte size overflows");
    bytes *= dims[i];
  }
  return bytes;
}

}  // namespace clusterlm::objects
