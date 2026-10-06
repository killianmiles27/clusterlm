#include "clusterlm/domain/reference_kernels.hpp"

#include "reference_math.hpp"

namespace clusterlm::domain {

namespace rm = refmath;

void reference_matvec(const float* w, std::size_t rows, std::size_t cols, const float* x, float* y) {
  rm::matvec(w, rows, cols, x, y);
}

void reference_expert_forward(const float* gate, const float* up, const float* down, std::size_t ff, std::size_t hidden,
                              const float* h, float* out, ExpertScratch& s) {
  s.a.resize(ff);
  s.b.resize(ff);
  s.z.resize(ff);
  rm::matvec(gate, ff, hidden, h, s.a.data());
  rm::matvec(up, ff, hidden, h, s.b.data());
  for (std::size_t c = 0; c < ff; ++c) s.z[c] = rm::silu(s.a[c]) * s.b[c];
  rm::matvec(down, hidden, ff, s.z.data(), out);
}

void reference_matmul_rows(const float* w, std::size_t rows, std::size_t cols, const float* x, std::size_t positions,
                           float* y) {
  for (std::size_t i = 0; i < rows; ++i) {
    const float* row = w + i * cols;
    for (std::size_t p = 0; p < positions; ++p) y[p * rows + i] = rm::dot(row, x + p * cols, cols);
  }
}

}  // namespace clusterlm::domain
