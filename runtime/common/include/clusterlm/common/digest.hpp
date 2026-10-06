#pragma once
// SHA-256 content digests used for object validation, manifest identity and plan hashes.
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "clusterlm/common/bytes.hpp"

namespace clusterlm {

struct Digest256 {
  std::array<std::uint8_t, 32> bytes{};

  std::string hex() const { return to_hex(bytes); }
  static Result<Digest256> from_hex(std::string_view hex);
  bool is_zero() const noexcept;
  friend bool operator==(const Digest256&, const Digest256&) = default;
  friend auto operator<=>(const Digest256&, const Digest256&) = default;
};

class Sha256 {
 public:
  Sha256();
  ~Sha256();
  Sha256(const Sha256&) = delete;
  Sha256& operator=(const Sha256&) = delete;
  Sha256(Sha256&&) noexcept;
  Sha256& operator=(Sha256&&) noexcept;

  void update(ByteSpan data);
  void update(std::string_view s) { update(ByteSpan(reinterpret_cast<const std::uint8_t*>(s.data()), s.size())); }
  Digest256 finish();

  static Digest256 of(ByteSpan data);
  static Digest256 of(std::string_view s);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace clusterlm

template <>
struct std::hash<clusterlm::Digest256> {
  std::size_t operator()(const clusterlm::Digest256& d) const noexcept {
    std::size_t h;
    static_assert(sizeof h <= sizeof d.bytes);
    std::memcpy(&h, d.bytes.data(), sizeof h);
    return h;
  }
};
