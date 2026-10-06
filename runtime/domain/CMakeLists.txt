clusterlm_module(clusterlm_domain_api
  SOURCES src/boundary.cpp
  DEPS clusterlm_objects)

clusterlm_module(clusterlm_domain
  SOURCES src/reference_domain.cpp src/reference_backend.cpp src/drafter.cpp
  DEPS clusterlm_domain_api clusterlm_windows clusterlm_objects)

# Bitwise-reproducible FP32: forbid fused multiply-add contraction in the reference kernels.
if(MSVC)
  target_compile_options(clusterlm_domain PRIVATE /fp:precise)
else()
  target_compile_options(clusterlm_domain PRIVATE -ffp-contract=off)
endif()
