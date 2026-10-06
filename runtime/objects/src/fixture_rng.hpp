#pragma once
// Deterministic tensor generator shared by the raw-shard fixture writer and the GGUF fixture writer, so both
// produce the same weights for the same spec + seed. Private to runtime/objects.
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace clusterlm::objects::detail {

inline std::uint64_t splitmix64(std::uint64_t& s) {
  s += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

inline std::uint64_t fnv1a(std::string_view s) {
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (char c : s) h = (h ^ static_cast<std::uint8_t>(c)) * 0x100000001B3ull;
  return h;
}

// Uniform in [-1, 1) from the top 24 bits; exactly representable in float.
inline float uniform(std::uint64_t& s) {
  return static_cast<float>(static_cast<std::int64_t>(splitmix64(s) >> 40) - (1 << 23)) / static_cast<float>(1 << 23);
}

// Each tensor draws from its own stream so tensors are independent of generation order.
class TensorRng {
 public:
  TensorRng(std::uint64_t seed, std::string_view name) : state_(seed ^ fnv1a(name)) { splitmix64(state_); }
  std::vector<float> uniform_scaled(std::size_t n, float scale) {
    std::vector<float> v(n);
    for (float& x : v) x = uniform(state_) * scale;
    return v;
  }
  std::vector<float> affine(std::size_t n, float base, float amp) {
    std::vector<float> v(n);
    for (float& x : v) x = base + amp * uniform(state_);
    return v;
  }

 private:
  std::uint64_t state_;
};

inline float fan_in_scale(std::size_t fan_in) { return 1.0f / std::sqrt(static_cast<float>(fan_in)); }

}  // namespace clusterlm::objects::detail
