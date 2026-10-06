# Strata backend, the parts built against the pinned, patched Strata checkout (see runtime/backends/CMakeLists.txt):
#   clusterlm_strata_cpu      CpuExpertKernel over Strata's CPU kernels, and the Father-side model conversion
#                             (Strata pack + GGUF -> ClusterLM model directory)   CLUSTERLM_ENABLE_STRATA_CPU or _STRATA
#   clusterlm_backend_strata  the CUDA engine + BackendAdapter                     CLUSTERLM_ENABLE_STRATA
#   clusterlm-strata          probe / cpu-experts / convert, and with CUDA requirements / numerics
if(NOT CLUSTERLM_ENABLE_STRATA AND NOT CLUSTERLM_ENABLE_STRATA_CPU)
  return()
endif()

clusterlm_module(clusterlm_strata_cpu
  SOURCES src/cpu_expert_kernel.cpp src/convert.cpp
  DEPS clusterlm_strata_core clusterlm_objects
  PRIVATE_DEPS clusterlm_platform strata_kernels_cpu strata_artifact)

if(CLUSTERLM_ENABLE_STRATA)
  clusterlm_module(clusterlm_backend_strata
    SOURCES src/strata_backend.cpp
    DEPS clusterlm_strata_core clusterlm_strata_cpu clusterlm_domain
    PRIVATE_DEPS strata_engine CUDA::cudart)
  target_compile_definitions(clusterlm_backend_strata PRIVATE CLUSTERLM_STRATA_PIN="${CLUSTERLM_STRATA_PIN}")
endif()

add_executable(clusterlm-strata tools/clusterlm_strata.cpp)
target_link_libraries(clusterlm-strata PRIVATE clusterlm_strata_cpu clusterlm_strata_core clusterlm_objects clusterlm_domain
                      clusterlm_platform clusterlm_third_party)
target_compile_definitions(clusterlm-strata PRIVATE CLUSTERLM_STRATA_PIN="${CLUSTERLM_STRATA_PIN}")
if(CLUSTERLM_ENABLE_STRATA)
  target_link_libraries(clusterlm-strata PRIVATE clusterlm_backend_strata CUDA::cudart)
  target_compile_definitions(clusterlm-strata PRIVATE CLUSTERLM_STRATA_HAVE_CUDA=1)
endif()
clusterlm_set_warnings(clusterlm-strata)
