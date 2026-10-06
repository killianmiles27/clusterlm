#include "clusterlm/objects/tensor_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace clusterlm::objects {

std::uint64_t tensor_bytes(std::string_view quant_type, std::uint64_t n_elems) {
  if (quant_type == kQuantF32) return n_elems * 4;
  if (quant_type == kQuantQ8Fixture) return n_elems % kQ8BlockElems == 0 ? n_elems / kQ8BlockElems * kQ8BlockBytes : 0;
  return 0;
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

}  // namespace clusterlm::objects
