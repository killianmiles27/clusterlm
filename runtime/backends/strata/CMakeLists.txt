# Strata backend adapter. Built only with -DCLUSTERLM_ENABLE_STRATA=ON; requires CUDA and the fetched
# upstream checkout (python3 scripts/fetch_upstream.py). Not part of the default Linux CI build.
if(NOT CLUSTERLM_ENABLE_STRATA)
  return()
endif()

set(_strata_src ${PROJECT_SOURCE_DIR}/third_party/upstream/strata)
if(NOT EXISTS ${_strata_src}/CMakeLists.txt)
  message(FATAL_ERROR "CLUSTERLM_ENABLE_STRATA=ON but the pinned Strata checkout is missing at ${_strata_src}. "
                      "Run: python3 scripts/fetch_upstream.py strata")
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CLUSTERLM_ENABLE_STRATA=ON requires the CUDA toolkit (no CUDA compiler found). "
                      "Strata's kernels cannot be built without it; use the reference backend instead.")
endif()

file(READ ${PROJECT_SOURCE_DIR}/third_party/upstream.json _pins)
string(JSON _strata_pin GET "${_pins}" pins strata commit)

# Strata sets global compile definitions/standard in its top-level CMakeLists; its targets are built from a
# subdirectory and excluded from `all` so only what the adapter links is compiled.
set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(STRATA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${_strata_src} ${CMAKE_BINARY_DIR}/third_party/strata EXCLUDE_FROM_ALL)

clusterlm_module(clusterlm_backend_strata
  SOURCES src/strata_backend.cpp
  DEPS clusterlm_common clusterlm_domain
  PRIVATE_DEPS strata_core strata_kernels CUDA::cudart)
target_compile_definitions(clusterlm_backend_strata PRIVATE
  CLUSTERLM_STRATA_HAVE_CUDA=1 CLUSTERLM_STRATA_PIN="${_strata_pin}")
