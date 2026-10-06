#pragma once
// Tensor byte codecs shared by the fixture generator and the reference backend.
//
// Two representations exist in the fixture model:
//   "f32"          little-endian IEEE-754 binary32, row-major;
//   "q8_0-fixture" blocks of 32 elements: one little-endian f32 scale followed by 32 int8 values (36 bytes).
//                  Dequantization is `scale * q` in FP32 and is exact/deterministic: the backend must never
//                  requantize or approximate it.
#include <cstddef>
#include <cstdint>
#include <span>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"

namespace clusterlm::objects {

inline constexpr std::string_view kQuantF32 = "f32";
inline constexpr std::string_view kQuantQ8Fixture = "q8_0-fixture";
inline constexpr std::uint32_t kQ8BlockElems = 32;
inline constexpr std::uint32_t kQ8BlockBytes = 4 + kQ8BlockElems;

// Bytes occupied by `n_elems` elements in the given representation (0 for an unknown type or a q8 element
// count that is not a multiple of the block size).
std::uint64_t tensor_bytes(std::string_view quant_type, std::uint64_t n_elems);

// Decode `out.size()` elements from `bytes` (which must be exactly tensor_bytes(...) long).
Status decode_tensor(std::string_view quant_type, ByteSpan bytes, std::span<float> out);

// Encoders (used by the generator and by tests that need a corrupted/alternate representation).
void encode_f32(std::span<const float> values, Bytes& out);
Status encode_q8_fixture(std::span<const float> values, Bytes& out);

}  // namespace clusterlm::objects
