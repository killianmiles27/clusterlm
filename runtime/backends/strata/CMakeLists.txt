# Strata backend, the parts built against the pinned, patched Strata checkout (see runtime/backends/CMakeLists.txt):
#   clusterlm_strata_cpu      CpuExpertKernel over Strata's CPU kernels  (CLUSTERLM_ENABLE_STRATA_CPU or _STRATA)
#   clusterlm_backend_strata  the CUDA engine + BackendAdapter            (CLUSTERLM_ENABLE_STRATA)
if(NOT CLUSTERLM_ENABLE_STRATA AND NOT CLUSTERLM_ENABLE_STRATA_CPU)
  return()
endif()

clusterlm_module(clusterlm_strata_cpu
  SOURCES src/cpu_expert_kernel.cpp
  DEPS clusterlm_strata_core
  PRIVATE_DEPS strata_kernels_cpu)

if(CLUSTERLM_ENABLE_STRATA)
  clusterlm_module(clusterlm_backend_strata
    SOURCES src/strata_backend.cpp
    DEPS clusterlm_strata_core clusterlm_strata_cpu clusterlm_domain
    PRIVATE_DEPS strata_engine CUDA::cudart)
  target_compile_definitions(clusterlm_backend_strata PRIVATE CLUSTERLM_STRATA_PIN="${CLUSTERLM_STRATA_PIN}")
endif()
