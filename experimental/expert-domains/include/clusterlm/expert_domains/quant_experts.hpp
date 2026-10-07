#pragma once
// QuantExperts: routed experts executed by the real quantized CPU kernels (Strata IQ3_S / IQ2_XS) instead of the
// fixture's exactly-dequantized FP32 weights. EXPERIMENTAL, like the rest of this directory.
//
// The kernels come from the bench's "strata-cpu" expert provider (bench/src/expert_providers_external.cpp), which wraps
// backends::strata::make_cpu_expert_kernel. In a build without CLUSTERLM_ENABLE_STRATA_CPU the provider is a stub and
// create() fails with kHardwareUnavailable and the stub's reason; nothing is silently replaced by FP32.
//
// The expert BLOBS are synthetic: deterministic pseudo-random blocks in the real GGUF slice layout [gate|up|down],
// seeded by (seed, layer, owner), never model data and never the fixture's expert objects. So in this mode the
// prototype measures the cost structure (kernel time, barrier, messages, bytes) with the production kernels, not
// accuracy: logits are not comparable with the FP32 reference, and the simulation says so.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::expert_domains {

struct ExpertKernelSpec {
  // "" = the fixture's FP32 experts (default). "iq3_s" / "iq2_xs" = the Strata CPU kernels on synthetic blobs.
  std::string representation;
  std::uint64_t seed = 1;
  bool quantized() const { return !representation.empty(); }
};

// Parses "fixture" | "iq3_s" | "iq2_xs" (also "strata-iq3_s", "strata-iq2_xs"). kInvalidArgument otherwise.
Result<ExpertKernelSpec> parse_expert_kernel(const std::string& text, std::uint64_t seed = 1);

class QuantExperts {
 public:
  // `n_experts` experts per layer for layers [first_layer, end_layer); `owner_key` distinguishes the owners' blobs.
  static Result<std::unique_ptr<QuantExperts>> create(const ExpertKernelSpec& spec, std::uint32_t hidden, std::uint32_t ff,
                                                      std::uint32_t first_layer, std::uint32_t end_layer,
                                                      std::uint32_t n_experts, std::uint64_t owner_key);
  ~QuantExperts();
  QuantExperts(const QuantExperts&) = delete;
  QuantExperts& operator=(const QuantExperts&) = delete;

  // y[0..hidden) += weight * expert(h), one position. Not thread-safe (one scratch context).
  Status accumulate(std::uint32_t layer, std::uint32_t local_expert, const float* h, float weight, float* y);
  std::uint64_t resident_bytes() const { return resident_bytes_; }
  std::string kernel_path() const { return kernel_path_; }  // e.g. "strata-iq512", for the result

  struct Impl;

 private:
  QuantExperts();
  std::unique_ptr<Impl> impl_;
  std::uint64_t resident_bytes_ = 0;
  std::string kernel_path_;
};

}  // namespace clusterlm::expert_domains
