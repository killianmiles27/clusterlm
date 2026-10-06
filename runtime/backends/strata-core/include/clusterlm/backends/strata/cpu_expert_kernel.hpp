#pragma once
// CpuExpertKernel: one routed expert on the CPU with Strata's own kernels - the CPU expert complement of a Strata
// domain (the experts that are not resident in VRAM).
//
// A native expert blob is the three GGUF slices [gate rows | up rows | down rows] (object_map.hpp). The kernel is
// Strata's `native_gu_rows` / `native_down_rows` (strata src/kernels/cpu/native_expert.cpp): the activation is
// quantized to the weight type's vec_dot type and each row is a dot product - Strata's AVX-512 / AVX2 multi-token
// i-quant kernels where they apply, ggml-cpu's vec_dot otherwise. That is exactly what Strata's expert pool computes
// inside a verify window and what llama.cpp's CPU backend computes for the same tensor.
//
//   y_t = down . q_h( silu(gate . q_x(x_t)) * (up . q_x(x_t)) )      t < tokens <= 8
//
// Built only where the pinned Strata CPU kernels are compiled (CLUSTERLM_ENABLE_STRATA or
// CLUSTERLM_ENABLE_STRATA_CPU); this header is dependency-free. It runs without a GPU.
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"

namespace clusterlm::backends::strata {

inline constexpr std::uint32_t kCpuExpertMaxTokens = 8;  // Strata cpu::MAXT (one verify window)

class CpuExpertKernel {
 public:
  virtual ~CpuExpertKernel() = default;
  // ggml type names of the gate/up and down tensors (e.g. "iq3_s", "iq4_nl").
  virtual std::string_view gate_up_type() const = 0;
  virtual std::string_view down_type() const = 0;
  virtual std::uint32_t hidden() const = 0;
  virtual std::uint32_t ff() const = 0;
  virtual std::uint64_t blob_bytes() const = 0;
  // Which code path computes `tokens` tokens' gate/up rows on this CPU ("strata-iq512", "strata-iq256", "ggml-cpu").
  virtual std::string_view path(std::uint32_t tokens) const = 0;

  // x: tokens * hidden floats; y: tokens * hidden floats.
  virtual Status run(ByteSpan blob, std::span<const float> x, std::uint32_t tokens, std::span<float> y) const = 0;
  // The first half alone: ff[t] = silu(gate . q_x(x_t)) * (up . q_x(x_t)) (tokens * ff floats) - where Strata's
  // multi-token i-quant kernels do their work, before the hidden activation is re-quantized for the down rows.
  virtual Status gate_up(ByteSpan blob, std::span<const float> x, std::uint32_t tokens, std::span<float> ff) const = 0;

  // References for verification (tests, ClusterLM Bench):
  //   * kExactGgml: ggml-cpu's own per-token vec_dot on the same quantized activations - the arithmetic llama.cpp's
  //     CPU backend performs. One token takes ggml's own dot (bit-exact); Strata's multi-token AVX kernels compute
  //     the same integer block dots and differ only in float summation order - which can flip the re-quantization
  //     of a hidden value for the down rows (Strata pin issue 152), so full outputs are compared with a tolerance.
  //   * kScalarDequant: the weights dequantized with ggml's reference (to_float) and the quantized activations
  //     dequantized, accumulated in double - an implementation-independent check of the arithmetic.
  enum class Reference { kExactGgml, kScalarDequant };
  virtual Status reference(Reference kind, ByteSpan blob, std::span<const float> x, std::uint32_t tokens,
                           std::span<float> y) const = 0;
  virtual Status gate_up_reference(Reference kind, ByteSpan blob, std::span<const float> x, std::uint32_t tokens,
                                   std::span<float> ff) const = 0;
};

// Whether this build contains the Strata CPU kernels (false: make_cpu_expert_kernel returns kHardwareUnavailable).
bool cpu_expert_kernels_available();
// The CPU's kernel tier as Strata probes it ("avx512", "avx2", "baseline").
std::string cpu_expert_isa();
Result<std::unique_ptr<CpuExpertKernel>> make_cpu_expert_kernel(std::string_view gate_up_type, std::string_view down_type,
                                                                std::uint32_t hidden, std::uint32_t ff);

}  // namespace clusterlm::backends::strata
