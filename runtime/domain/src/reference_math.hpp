#pragma once
// Deterministic single-threaded FP32 kernels for the reference backend.
//
// Every reduction runs in a fixed, ascending index order with a float accumulator, so the same inputs give
// bitwise identical outputs regardless of how layers are split across domains or windows. The target is
// compiled with -ffp-contract=off so no FMA changes rounding between toolchains.
#include <cmath>
#include <cstddef>

namespace clusterlm::domain::refmath {

inline float dot(const float* a, const float* b, std::size_t n) {
  float acc = 0.0f;
  for (std::size_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

// y[i] = sum_j W[i*cols + j] * x[j]
inline void matvec(const float* w, std::size_t rows, std::size_t cols, const float* x, float* y) {
  for (std::size_t i = 0; i < rows; ++i) y[i] = dot(w + i * cols, x, cols);
}

// out_i = x_i * w_i / sqrt(mean(x^2) + 1e-6)
inline void rmsnorm(const float* x, const float* w, std::size_t n, float* out) {
  float ss = 0.0f;
  for (std::size_t i = 0; i < n; ++i) ss += x[i] * x[i];
  const float denom = std::sqrt(ss / static_cast<float>(n) + 1e-6f);
  for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * w[i] / denom;
}

inline float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
inline float silu(float z) { return z / (1.0f + std::exp(-z)); }

// u_i = (1/hc) * sum_j S[j][i]
inline void mean_streams(const float* s, std::size_t hc, std::size_t h, float* u) {
  const float inv = 1.0f / static_cast<float>(hc);
  for (std::size_t i = 0; i < h; ++i) {
    float acc = 0.0f;
    for (std::size_t j = 0; j < hc; ++j) acc += s[j * h + i];
    u[i] = acc * inv;
  }
}

// S_j += g_j * P for every stream j.
inline void fold(float* s, const float* pending, const float* g, std::size_t hc, std::size_t h) {
  for (std::size_t j = 0; j < hc; ++j)
    for (std::size_t i = 0; i < h; ++i) s[j * h + i] += g[j] * pending[i];
}

}  // namespace clusterlm::domain::refmath
