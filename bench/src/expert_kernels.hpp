#pragma once
// ExpertKernelProvider: the pluggable CPU routed-expert kernel interface measured by `clusterlm-bench cpu`.
//
// A provider turns a tensor representation ("f32", "q8_0-fixture", later "iq3_s", "iq2_xs") into an
// ExpertBank: a set of experts that can be executed for 1..q positions. The reference provider wraps the
// deterministic reference-backend math. The Strata CPU IQ kernels (AVX2/AVX-512, built by another workstream)
// register under the id "strata-cpu" through ExpertKernelRegistry::register_factory; in a build without the
// Strata CPU kernels that id resolves to a stub that reports kHardwareUnavailable with a precise reason.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::bench {

struct ExpertShape {
  std::uint32_t hidden = 2560;  // H (Flash-Next width)
  std::uint32_t ff = 640;       // routed expert intermediate width
};

// Wall time split of one expert execution. `dequant_ns` is 0 for representations that are consumed in place.
struct ExpertTiming {
  std::uint64_t dequant_ns = 0;
  std::uint64_t gemv_ns = 0;
};

// Per-thread scratch owned by the bank that created it.
class ExpertContext {
 public:
  virtual ~ExpertContext() = default;
};

class ExpertBank {
 public:
  virtual ~ExpertBank() = default;
  virtual std::size_t experts() const = 0;
  // Bytes one expert occupies in its stored representation (gate + up + down).
  virtual std::uint64_t bytes_per_expert() const = 0;
  virtual std::unique_ptr<ExpertContext> make_context() const = 0;
  // Which kernel path executes `positions` positions on this CPU (e.g. Strata's "strata-iq512" / "strata-iq256" /
  // "ggml-cpu"); empty when the provider has a single path. Recorded in bench results.
  virtual std::string kernel_path(std::size_t /*positions*/) const { return {}; }
  // Executes expert `e` over `positions` inputs (positions*hidden floats) into `out` (positions*hidden floats).
  // Thread-safe for distinct contexts. `timing` may be null.
  virtual Status run(std::size_t e, std::size_t positions, const float* in, float* out, ExpertContext& ctx,
                     ExpertTiming* timing) const = 0;
};

class ExpertKernelProvider {
 public:
  virtual ~ExpertKernelProvider() = default;
  virtual std::string id() const = 0;           // registry key, e.g. "reference", "strata-cpu"
  virtual std::string description() const = 0;  // one line, recorded in results
  virtual std::string isa() const = 0;          // e.g. "scalar-fp32", "avx2", "avx512"
  virtual std::vector<std::string> representations() const = 0;
  // Builds `n_experts` experts with deterministic pseudo-random weights (never model data).
  virtual Result<std::unique_ptr<ExpertBank>> make_bank(const std::string& representation, ExpertShape shape,
                                                        std::size_t n_experts, std::uint64_t seed) const = 0;
};

using ExpertProviderFactory = std::function<Result<std::unique_ptr<ExpertKernelProvider>>()>;

class ExpertKernelRegistry {
 public:
  static ExpertKernelRegistry& instance();
  // Registers (or replaces) the factory for `id`.
  void register_factory(const std::string& id, ExpertProviderFactory factory);
  Result<std::unique_ptr<ExpertKernelProvider>> create(const std::string& id) const;
  std::vector<std::string> ids() const;

 private:
  ExpertKernelRegistry() = default;
  std::vector<std::pair<std::string, ExpertProviderFactory>> factories_;
};

// "reference" (always) and the "strata-cpu" stub. Idempotent.
void register_builtin_expert_providers();
// Registration point for providers built by other workstreams (Strata CPU IQ kernels). Defined in
// expert_providers_external.cpp; that file is the only place such a workstream needs to touch.
void register_external_expert_providers();

}  // namespace clusterlm::bench
