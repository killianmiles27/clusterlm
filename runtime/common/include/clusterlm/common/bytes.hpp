#pragma once
// Explicit little-endian binary encoding used for every ClusterLM wire format and on-disk journal.
//
// The wire format never depends on host struct layout or host endianness: every field is written with an
// explicit width through ByteWriter and read back with bounds checking through ByteReader.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm {

using Bytes = std::vector<std::uint8_t>;
using ByteSpan = std::span<const std::uint8_t>;

class ByteWriter {
 public:
  ByteWriter() = default;
  explicit ByteWriter(std::size_t reserve) { buf_.reserve(reserve); }

  void u8(std::uint8_t v) { buf_.push_back(v); }
  void u16(std::uint16_t v) { put_le(v); }
  void u32(std::uint32_t v) { put_le(v); }
  void u64(std::uint64_t v) { put_le(v); }
  void i32(std::int32_t v) { put_le(static_cast<std::uint32_t>(v)); }
  void i64(std::int64_t v) { put_le(static_cast<std::uint64_t>(v)); }
  void f32(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    put_le(bits);
  }
  void f64(double v) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    put_le(bits);
  }
  void boolean(bool v) { u8(v ? 1 : 0); }
  void raw(ByteSpan data) { buf_.insert(buf_.end(), data.begin(), data.end()); }
  // Length-prefixed (u32) byte string.
  void blob(ByteSpan data) {
    u32(static_cast<std::uint32_t>(data.size()));
    raw(data);
  }
  void str(std::string_view s) {
    u32(static_cast<std::uint32_t>(s.size()));
    buf_.insert(buf_.end(), s.begin(), s.end());
  }
  // FP32 array as explicit little-endian values (u32 count prefix).
  void f32_array(std::span<const float> values);

  std::size_t size() const noexcept { return buf_.size(); }
  const Bytes& bytes() const& noexcept { return buf_; }
  Bytes take() && noexcept { return std::move(buf_); }

 private:
  template <typename U>
  void put_le(U v) {
    for (std::size_t i = 0; i < sizeof(U); ++i) buf_.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
  }
  Bytes buf_;
};

// Bounds-checked reader. Every accessor returns false/an error instead of reading past the end; after the
// first failure the reader is poisoned so callers may check ok() once at the end of a decode.
class ByteReader {
 public:
  explicit ByteReader(ByteSpan data) : data_(data) {}

  bool u8(std::uint8_t& v);
  bool u16(std::uint16_t& v) { return get_le(v); }
  bool u32(std::uint32_t& v) { return get_le(v); }
  bool u64(std::uint64_t& v) { return get_le(v); }
  bool i32(std::int32_t& v);
  bool i64(std::int64_t& v);
  bool f32(float& v);
  bool f64(double& v);
  bool boolean(bool& v);
  bool raw(std::size_t n, ByteSpan& out);
  bool blob(Bytes& out, std::size_t max_size);
  bool str(std::string& out, std::size_t max_size);
  bool f32_array(std::vector<float>& out, std::size_t max_count);

  std::size_t remaining() const noexcept { return failed_ ? 0 : data_.size() - pos_; }
  bool ok() const noexcept { return !failed_; }
  bool at_end() const noexcept { return !failed_ && pos_ == data_.size(); }
  // OK only if every read succeeded and the input was consumed exactly.
  Status finish(std::string_view what) const;

 private:
  template <typename U>
  bool get_le(U& v) {
    if (failed_ || data_.size() - pos_ < sizeof(U)) return fail();
    U out = 0;
    for (std::size_t i = 0; i < sizeof(U); ++i) out |= static_cast<U>(static_cast<U>(data_[pos_ + i]) << (8 * i));
    pos_ += sizeof(U);
    v = out;
    return true;
  }
  bool fail() {
    failed_ = true;
    return false;
  }
  ByteSpan data_;
  std::size_t pos_ = 0;
  bool failed_ = false;
};

std::string to_hex(ByteSpan data);
Result<Bytes> from_hex(std::string_view hex);

}  // namespace clusterlm
