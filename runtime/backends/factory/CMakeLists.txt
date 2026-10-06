# Backend selection by name ("reference", "strata") for the Node worker and the Coordinator. "strata" is linked in only
# when the CUDA backend is built (CLUSTERLM_ENABLE_STRATA); otherwise selecting it reports kUnimplemented.
clusterlm_module(clusterlm_backend_factory
  SOURCES src/backend_factory.cpp
  DEPS clusterlm_domain clusterlm_strata_core clusterlm_common)
if(TARGET clusterlm_backend_strata)
  target_link_libraries(clusterlm_backend_factory PUBLIC clusterlm_backend_strata)
  target_compile_definitions(clusterlm_backend_factory PUBLIC CLUSTERLM_FACTORY_HAS_STRATA=1)
endif()
if(TARGET clusterlm_backend_llama)
  target_link_libraries(clusterlm_backend_factory PUBLIC clusterlm_backend_llama)
  target_compile_definitions(clusterlm_backend_factory PUBLIC CLUSTERLM_FACTORY_HAS_LLAMA=1)
endif()
