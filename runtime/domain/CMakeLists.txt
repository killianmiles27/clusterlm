clusterlm_module(clusterlm_domain_api
  SOURCES src/boundary.cpp src/backend_descriptor.cpp src/builtin_descriptors.cpp
  DEPS clusterlm_objects
  PRIVATE_DEPS clusterlm_third_party)

clusterlm_module(clusterlm_domain
  SOURCES src/reference_domain.cpp src/reference_backend.cpp src/drafter.cpp src/sampling.cpp src/reference_kernels.cpp
  DEPS clusterlm_domain_api clusterlm_windows clusterlm_objects)

# Bitwise-reproducible FP32: forbid fused multiply-add contraction in the reference kernels.
if(MSVC)
  target_compile_options(clusterlm_domain PRIVATE /fp:precise)
else()
  target_compile_options(clusterlm_domain PRIVATE -ffp-contract=off)
endif()
