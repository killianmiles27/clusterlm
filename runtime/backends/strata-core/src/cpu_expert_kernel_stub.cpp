// The CpuExpertKernel factory of a build without the Strata CPU kernels (neither CLUSTERLM_ENABLE_STRATA nor
// CLUSTERLM_ENABLE_STRATA_CPU): it reports the kernels as unavailable instead of failing to link.
#include "clusterlm/backends/strata/cpu_expert_kernel.hpp"

namespace clusterlm::backends::strata {

bool cpu_expert_kernels_available() { return false; }
std::string cpu_expert_isa() { return "unavailable"; }

Result<std::unique_ptr<CpuExpertKernel>> make_cpu_expert_kernel(std::string_view, std::string_view, std::uint32_t,
                                                                std::uint32_t) {
  return make_error(ErrorCode::kHardwareUnavailable,
                    "Strata CPU expert kernels are not built (configure with -DCLUSTERLM_ENABLE_STRATA_CPU=ON or "
                    "-DCLUSTERLM_ENABLE_STRATA=ON after python3 scripts/fetch_upstream.py --apply-patches strata strata-ggml)");
}

}  // namespace clusterlm::backends::strata
