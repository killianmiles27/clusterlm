# The pinned, patched Strata checkout as a CMake subproject (included by runtime/backends/CMakeLists.txt).
#
# Strata's targets are built from a subdirectory excluded from `all`: only what ClusterLM links is compiled
# (strata_kernels_cpu + ggml for the CPU kernels; strata_engine and its CUDA libraries for the engine). Their headers
# are SYSTEM includes for ClusterLM targets, so ClusterLM's -Werror policy applies to ClusterLM code only.
if(CMAKE_VERSION VERSION_LESS 3.28)
  message(FATAL_ERROR "The Strata backend needs CMake >= 3.28 (upstream test registrations are disabled per directory)")
endif()

set(CLUSTERLM_STRATA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/upstream/strata CACHE PATH
    "The pinned Strata checkout (python3 scripts/fetch_upstream.py --apply-patches strata)")
set(CLUSTERLM_STRATA_GGML_DIR ${PROJECT_SOURCE_DIR}/third_party/upstream/strata-ggml CACHE PATH
    "The llama.cpp checkout Strata takes ggml from (python3 scripts/fetch_upstream.py strata-ggml)")
set(_strata_src ${CLUSTERLM_STRATA_SOURCE_DIR})
foreach(_req ${_strata_src}/CMakeLists.txt ${CLUSTERLM_STRATA_GGML_DIR}/ggml/CMakeLists.txt)
  if(NOT EXISTS ${_req})
    message(FATAL_ERROR "Strata build: ${_req} is missing. Run: python3 scripts/fetch_upstream.py --apply-patches strata strata-ggml")
  endif()
endforeach()

# The ClusterLM patches must be applied (third_party/patches/strata, applied by fetch_upstream.py --apply-patches).
# The last patch's marker in verify.hpp is the cheap configure-time check; ctest runs the exact one
# (fetch_upstream.py --check --apply-patches: the tree must equal pin + every patch).
file(READ ${_strata_src}/include/strata/core/verify.hpp _strata_verify_hpp)
if(NOT _strata_verify_hpp MATCHES "ClusterLM patch 0006")
  message(FATAL_ERROR "Strata build: the checkout at ${_strata_src} is not patched. "
                      "Run: python3 scripts/fetch_upstream.py --apply-patches strata")
endif()

file(READ ${PROJECT_SOURCE_DIR}/third_party/upstream.json _clm_pins)
string(JSON CLUSTERLM_STRATA_PIN GET "${_clm_pins}" pins strata commit)

if(CLUSTERLM_ENABLE_STRATA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # sm_86: RTX 3060; sm_89: RTX 4060 Ti, RTX 4070 Laptop (the three target machines).
    set(CMAKE_CUDA_ARCHITECTURES "86;89" CACHE STRING "CUDA architectures for the Strata engine")
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CLUSTERLM_ENABLE_STRATA=ON requires the CUDA toolkit (no CUDA compiler found). "
                        "Use -DCLUSTERLM_ENABLE_STRATA_CPU=ON for the CPU kernels alone.")
  endif()
  set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
else()
  set(STRATA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
endif()
set(STRATA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(STRATA_NATIVE_EXPERTS ON CACHE BOOL "" FORCE)
set(STRATA_GGML_DIR ${CLUSTERLM_STRATA_GGML_DIR} CACHE PATH "" FORCE)
# CUDA::cudart and friends are found inside Strata's directory; make them visible to the adapter next door.
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)
add_subdirectory(${_strata_src} ${CMAKE_BINARY_DIR}/third_party/strata EXCLUDE_FROM_ALL)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL OFF)

foreach(_t strata_artifact strata_plan strata_kernels_cpu ggml ggml-base ggml-cpu strata_core strata_kernels strata_engine)
  if(TARGET ${_t})
    set_target_properties(${_t} PROPERTIES SYSTEM ON)
  endif()
endforeach()

# Upstream registers some ctest entries unconditionally, most of them for a GPU or the 34 GB model pack; with
# STRATA_BUILD_TESTS off their executables are not built. Disable every upstream registration (ctest reports them as
# disabled rather than failing on executables that do not exist) and register the two upstream CPU parity checks
# that need neither a GPU nor a model: the AVX-2 i-quant expert kernels against each other and ggml-cpu's dot, and
# the router lookahead's dot against a double reference.
get_property(_strata_tests DIRECTORY ${_strata_src} PROPERTY TESTS)
foreach(_t IN LISTS _strata_tests)
  set_property(TEST ${_t} DIRECTORY ${_strata_src} PROPERTY DISABLED TRUE)
endforeach()
if(CLUSTERLM_BUILD_TESTS AND TARGET iq_avx2_parity AND TARGET router_dot_parity)
  add_custom_target(clusterlm_strata_upstream_cpu_tests ALL DEPENDS iq_avx2_parity router_dot_parity)
  add_test(NAME strata_upstream_iq_avx2_parity COMMAND iq_avx2_parity)
  add_test(NAME strata_upstream_router_dot_parity COMMAND router_dot_parity)
  set_tests_properties(strata_upstream_iq_avx2_parity strata_upstream_router_dot_parity PROPERTIES LABELS strata-upstream)
endif()
