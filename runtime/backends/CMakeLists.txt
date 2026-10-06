# Optional backend adapters. NOTE: the top-level CMakeLists.txt module list must include runtime/backends for
# this directory to be configured; each adapter is additionally gated by its own option.
#
# Strata (docs/backends/strata-port.md):
#   * strata-layout, strata-core: always built, no CUDA, no Strata sources (hand-off transpose, object mapping,
#     the StrataDomain state machine, the MTP drafter adapter). Tested in every build with a fake engine.
#   * -DCLUSTERLM_ENABLE_STRATA_CPU=ON: the pinned, patched Strata CPU expert kernels (strata_kernels_cpu + ggml-cpu)
#     and ClusterLM's CpuExpertKernel over them. No CUDA needed.
#   * -DCLUSTERLM_ENABLE_STRATA=ON (top-level option): additionally the CUDA engine (strata_engine) and the Strata
#     BackendAdapter. Needs the CUDA toolkit (nvcc >= 12.0; with Ubuntu's nvcc 12.0 pass
#     -DCMAKE_CUDA_HOST_COMPILER=g++-12). Architectures default to 86;89 (RTX 3060; RTX 4060 Ti / 4070 Laptop).
# Both need `python3 scripts/fetch_upstream.py --apply-patches strata strata-ggml` first.
option(CLUSTERLM_ENABLE_STRATA_CPU "Build the Strata CPU expert kernels (no CUDA) against the pinned, patched checkout" OFF)

if(CLUSTERLM_ENABLE_STRATA OR CLUSTERLM_ENABLE_STRATA_CPU)
  include(${CMAKE_CURRENT_SOURCE_DIR}/strata-upstream.cmake)
endif()

foreach(adapter strata-layout strata-core strata llama factory)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${adapter}/CMakeLists.txt)
    add_subdirectory(${adapter})
  endif()
endforeach()
