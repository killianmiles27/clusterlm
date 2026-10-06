#include "clusterlm/common/bytes.hpp"

namespace clusterlm {

void ByteWriter::f32_array(std::span<const float> values) {
  u32(static_cast<std::uint32_t>(values.size()));
  buf_.reserve(buf_.size() + values.size() * 4);
  for (float v : values) f32(v);
}

bool ByteReader::u8(std::uint8_t& v) {
  if (failed_ || pos_ >= data_.size()) return fail();
  v = data_[pos_++];
  return true;
}

bool ByteReader::i32(std::int32_t& v) {
  std::uint32_t u;
  if (!u32(u)) return false;
  v = static_cast<std::int32_t>(u);
  return true;
}

bool ByteReader::i64(std::int64_t& v) {
  std::uint64_t u;
  if (!u64(u)) return false;
  v = static_cast<std::int64_t>(u);
  return true;
}

bool ByteReader::f32(float& v) {
  std::uint32_t bits;
  if (!u32(bits)) return false;
  std::memcpy(&v, &bits, sizeof v);
  return true;
}

bool ByteReader::f64(double& v) {
  std::uint64_t bits;
  if (!u64(bits)) return false;
  std::memcpy(&v, &bits, sizeof v);
  return true;
}

bool ByteReader::boolean(bool& v) {
  std::uint8_t b;
  if (!u8(b)) return false;
  if (b > 1) return fail();
  v = b == 1;
  return true;
}

bool ByteReader::raw(std::size_t n, ByteSpan& out) {
  if (failed_ || data_.size() - pos_ < n) return fail();
  out = data_.subspan(pos_, n);
  pos_ += n;
  return true;
}

bool ByteReader::blob(Bytes& out, std::size_t max_size) {
  std::uint32_t n;
  if (!u32(n)) return false;
  if (n > max_size) return fail();
  ByteSpan s;
  if (!raw(n, s)) return false;
  out.assign(s.begin(), s.end());
  return true;
}

bool ByteReader::str(std::string& out, std::size_t max_size) {
  std::uint32_t n;
  if (!u32(n)) return false;
  if (n > max_size) return fail();
  ByteSpan s;
  if (!raw(n, s)) return false;
  out.assign(reinterpret_cast<const char*>(s.data()), s.size());
  return true;
}

bool ByteReader::f32_array(std::vector<float>& out, std::size_t max_count) {
  std::uint32_t n;
  if (!u32(n)) return false;
  if (n > max_count || remaining() < static_cast<std::size_t>(n) * 4) return fail();
  out.resize(n);
  for (std::uint32_t i = 0; i < n; ++i) f32(out[i]);
  return ok();
}

Status ByteReader::finish(std::string_view what) const {
  if (failed_) return make_error(ErrorCode::kProtocolError, std::string(what) + ": truncated or invalid field");
  if (pos_ != data_.size())
    return make_error(ErrorCode::kProtocolError, std::string(what) + ": " + std::to_string(data_.size() - pos_) +
                                                     " trailing bytes");
  return Status::ok();
}

std::string to_hex(ByteSpan data) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(data.size() * 2);
  for (std::uint8_t b : data) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 0xF]);
  }
  return out;
}

Result<Bytes> from_hex(std::string_view hex) {
  if (hex.size() % 2 != 0) return make_error(ErrorCode::kInvalidArgument, "odd-length hex string");
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  Bytes out(hex.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    int hi = nibble(hex[2 * i]);
    int lo = nibble(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) return make_error(ErrorCode::kInvalidArgument, "invalid hex digit");
    out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
  }
  return out;
}

}  // namespace clusterlm
