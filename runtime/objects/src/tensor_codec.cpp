#include "clusterlm/objects/tensor_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace clusterlm::objects {

std::uint64_t tensor_bytes(std::string_view quant_type, std::uint64_t n_elems) {
  if (quant_type == kQuantF32) return n_elems * 4;
  if (quant_type == kQuantQ8Fixture) return n_elems % kQ8BlockElems == 0 ? n_elems / kQ8BlockElems * kQ8BlockBytes : 0;
  if (quant_type == kQuantQ8_0) return n_elems % kQ8BlockElems == 0 ? n_elems / kQ8BlockElems * kQ8_0BlockBytes : 0;
  return 0;
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = std::uint32_t{h & 0x8000u} << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  std::uint32_t man = h & 0x3FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // subnormal: normalize
      std::uint32_t e = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --e;
      }
      bits = sign | (e << 23) | ((man & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

std::uint16_t float_to_half(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof bits);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t abs = bits & 0x7FFFFFFFu;
  if (abs > 0x7F800000u) return static_cast<std::uint16_t>(sign | 0x7E00u);  // NaN
  if (abs >= 0x47800000u) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow / inf
  if (abs < 0x38800000u) {  // subnormal or zero in half
    if (abs < 0x33000000u) return static_cast<std::uint16_t>(sign);
    const std::uint32_t exp = abs >> 23;
    const std::uint32_t man = (abs & 0x7FFFFFu) | 0x800000u;
    const std::uint32_t shift = 126 - exp;  // 14..24
    std::uint32_t h = man >> shift;
    const std::uint32_t rem = man & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = (((abs >> 23) - 127 + 15) << 10) | ((abs >> 13) & 0x3FFu);
  const std::uint32_t rem = abs & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // a carry correctly bumps the exponent
  return static_cast<std::uint16_t>(sign | h);
}

namespace {
float load_f32(const std::uint8_t* p) {
  const std::uint32_t bits = std::uint32_t{p[0]} | std::uint32_t{p[1]} << 8 | std::uint32_t{p[2]} << 16 |
                             std::uint32_t{p[3]} << 24;
  float v;
  std::memcpy(&v, &bits, sizeof v);
  return v;
}
}  // namespace

Status decode_tensor(std::string_view quant_type, ByteSpan bytes, std::span<float> out) {
  const std::uint64_t want = tensor_bytes(quant_type, out.size());
  if (want == 0 && !out.empty()) return make_error(ErrorCode::kInvalidArgument, "unsupported tensor representation");
  if (bytes.size() != want) return make_error(ErrorCode::kDataLoss, "tensor byte size does not match representation");
  if (quant_type == kQuantF32) {
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = load_f32(bytes.data() + 4 * i);
    return Status::ok();
  }
  if (quant_type == kQuantQ8_0) {
    for (std::size_t b = 0; b < out.size() / kQ8BlockElems; ++b) {
      const std::uint8_t* blk = bytes.data() + b * kQ8_0BlockBytes;
      const float scale = half_to_float(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
      for (std::uint32_t i = 0; i < kQ8BlockElems; ++i)
        out[b * kQ8BlockElems + i] = scale * static_cast<float>(static_cast<std::int8_t>(blk[2 + i]));
    }
    return Status::ok();
  }
  for (std::size_t b = 0; b < out.size() / kQ8BlockElems; ++b) {
    const std::uint8_t* blk = bytes.data() + b * kQ8BlockBytes;
    const float scale = load_f32(blk);
    for (std::uint32_t i = 0; i < kQ8BlockElems; ++i)
      out[b * kQ8BlockElems + i] = scale * static_cast<float>(static_cast<std::int8_t>(blk[4 + i]));
  }
  return Status::ok();
}

void encode_f32(std::span<const float> values, Bytes& out) {
  out.reserve(out.size() + values.size() * 4);
  for (float v : values) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(bits >> (8 * i)));
  }
}

Status encode_q8_fixture(std::span<const float> values, Bytes& out) {
  if (values.size() % kQ8BlockElems != 0)
    return make_error(ErrorCode::kInvalidArgument, "q8 element count must be a multiple of 32");
  for (std::size_t b = 0; b < values.size(); b += kQ8BlockElems) {
    float amax = 0.0f;
    for (std::uint32_t i = 0; i < kQ8BlockElems; ++i) amax = std::max(amax, std::fabs(values[b + i]));
    const float scale = amax / 127.0f;
    encode_f32(std::span<const float>(&scale, 1), out);
    for (std::uint32_t i = 0; i < kQ8BlockElems; ++i) {
      const float q = scale == 0.0f ? 0.0f : std::nearbyint(values[b + i] / scale);
      out.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f))));
    }
  }
  return Status::ok();
}

Status encode_q8_0_ggml(std::span<const float> values, Bytes& out) {
  if (values.size() % kQ8BlockElems != 0)
    return make_error(ErrorCode::kInvalidArgument, "q8_0 element count must be a multiple of 32");
  for (std::size_t b = 0; b < values.size(); b += kQ8BlockElems) {
    float amax = 0.0f;
    for (std::uint32_t i = 0; i < kQ8BlockElems; ++i) amax = std::max(amax, std::fabs(values[b + i]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const std::uint16_t h = float_to_half(d);
    out.push_back(static_cast<std::uint8_t>(h & 0xFFu));
    out.push_back(static_cast<std::uint8_t>(h >> 8));
    for (std::uint32_t i = 0; i < kQ8BlockElems; ++i)
      out.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(std::round(values[b + i] * id))));
  }
  return Status::ok();
}

}  // namespace clusterlm::objects
