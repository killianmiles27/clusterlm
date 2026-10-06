#pragma once
// Public entry points to the reference backend's deterministic FP32 expert math, for ClusterLM Bench and for
// kernel-provider cross-checks. These are the same operations the reference domain executes per routed expert
// (SwiGLU: down * (silu(gate*h) ⊙ (up*h))), with the same fixed ascending reduction order.
#include <cstddef>
#include <vector>

namespace clusterlm::domain {

struct ExpertScratch {
  std::vector<float> a, b, z;
};

// y[i] = sum_j w[i*cols + j] * x[j]  (rows x cols, row-major).
void reference_matvec(const float* w, std::size_t rows, std::size_t cols, const float* x, float* y);

// out[H] = down[H][ff] * (silu(gate[ff][H] * h) ⊙ (up[ff][H] * h)) for one position.
void reference_expert_forward(const float* gate, const float* up, const float* down, std::size_t ff, std::size_t hidden,
                              const float* h, float* out, ExpertScratch& scratch);

// y[p][i] = w[i] . x[p] for `positions` inputs; each weight row is read once and applied to every position
// (the access pattern of a verification window over one expert). Per-element results are bitwise identical to
// reference_matvec applied to each position separately.
void reference_matmul_rows(const float* w, std::size_t rows, std::size_t cols, const float* x, std::size_t positions,
                           float* y);

}  // namespace clusterlm::domain
