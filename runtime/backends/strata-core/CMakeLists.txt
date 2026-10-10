# Strata backend, platform-independent part: object mapping (strata-dense containers, expert formats, Father-side
# conversion), the StrataDomain state machine over a StrataEngine, and the MTP drafter adapter. No CUDA, no Strata
# sources: built and tested in every configuration.
set(_sources src/object_map.cpp src/strata_domain.cpp src/mtp_drafter.cpp)
if(NOT CLUSTERLM_ENABLE_STRATA AND NOT CLUSTERLM_ENABLE_STRATA_CPU)
  list(APPEND _sources src/cpu_expert_kernel_stub.cpp)  # the real kernels live in runtime/backends/strata
endif()
clusterlm_embed_text(_strata_descriptor_cpp ${PROJECT_SOURCE_DIR}/runtime/backends/strata/descriptor/strata-hybrid.descriptor.json
  clusterlm::backends::strata strata_hybrid_descriptor_json)
clusterlm_module(clusterlm_strata_core
  SOURCES ${_sources} ${_strata_descriptor_cpp}
  DEPS clusterlm_domain clusterlm_windows clusterlm_strata_layout clusterlm_objects clusterlm_common)
