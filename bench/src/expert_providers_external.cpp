// Registration point for expert kernel providers contributed by other workstreams.
//
// The Strata CPU IQ kernels (IQ3_S / IQ2_XS, AVX2 and AVX-512) replace the "strata-cpu" stub like this:
//
//   ExpertKernelRegistry::instance().register_factory("strata-cpu", [] {
//     return make_strata_cpu_provider();   // Result<std::unique_ptr<ExpertKernelProvider>>
//   });
//
// and link their library into clusterlm_bench_harness. Nothing else in the bench needs to change: the `cpu`
// command, `calibrate` and the HardwareProfile writer select the provider by id and key the resulting
// cpu.expert_bytes_per_s entries by the representation names the provider reports.
#include "expert_kernels.hpp"

namespace clusterlm::bench {

void register_external_expert_providers() {}

}  // namespace clusterlm::bench
