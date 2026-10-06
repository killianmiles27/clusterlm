# Real llama.cpp backend (Fast tier, Father-local) and the P0-A RPC baseline support.
#
# Built only with -DCLUSTERLM_ENABLE_LLAMA=ON, against the pinned llama.cpp checkout
# (python3 scripts/fetch_upstream.py llama.cpp). The pinned library is built from source as static libraries
# (CPU backend always; CUDA with -DCLUSTERLM_LLAMA_CUDA=ON). Nothing from upstream is vendored or committed.
option(CLUSTERLM_ENABLE_LLAMA "Build the llama.cpp backend (Fast tier) against the pinned llama.cpp source" OFF)
option(CLUSTERLM_LLAMA_CUDA "Build the pinned llama.cpp with GGML_CUDA (nvcc 12.x, archs 86;89); needs CLUSTERLM_ENABLE_LLAMA" OFF)
if(NOT CLUSTERLM_ENABLE_LLAMA)
  return()
endif()

set(_llama_src ${PROJECT_SOURCE_DIR}/third_party/upstream/llama.cpp)
if(NOT EXISTS ${_llama_src}/CMakeLists.txt)
  message(FATAL_ERROR "CLUSTERLM_ENABLE_LLAMA=ON but the pinned llama.cpp checkout is missing at ${_llama_src}. "
                      "Run: python3 scripts/fetch_upstream.py llama.cpp")
endif()
file(READ ${PROJECT_SOURCE_DIR}/third_party/upstream.json _pins)
string(JSON _llama_pin GET "${_pins}" pins llama.cpp commit)

# Only the library: no tools, examples, tests, server or libcommon. Static libraries, so the test binaries and
# clusterlm-bench carry their own backend and need no rpath. GGML_NATIVE=OFF keeps the build independent of the
# CI machine's CPU (ggml still enables AVX2 and friends through its own defaults).
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
# The RPC client/server code is needed for the P0-A baseline harness; it is inert unless an endpoint is added.
set(GGML_RPC ON CACHE BOOL "" FORCE)
set(GGML_RPC_RDMA OFF CACHE BOOL "" FORCE)

if(CLUSTERLM_LLAMA_CUDA)
  # nvcc 12.0 supports host compilers up to GCC 12; select it explicitly (the default g++ may be newer).
  if(NOT CMAKE_CUDA_HOST_COMPILER AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 12)
    find_program(_clm_gxx12 NAMES g++-12)
    if(_clm_gxx12)
      set(CMAKE_CUDA_HOST_COMPILER ${_clm_gxx12} CACHE FILEPATH "" FORCE)
    endif()
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86;89" CACHE STRING "RTX 30xx (86) and 40xx (89)" FORCE)
  endif()
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(${_llama_src} ${CMAKE_BINARY_DIR}/third_party/llama.cpp EXCLUDE_FROM_ALL)

# The pinned rpc-server (tools/rpc/rpc-server.cpp) compiled on its own: the upstream tools tree would pull in
# libcommon and every other tool. Cache persistence is a runtime flag (-c) that is off by default; the baseline
# harness never passes it (docs/backends/llama-rpc.md).
add_executable(clusterlm-llama-rpc-server ${_llama_src}/tools/rpc/rpc-server.cpp)
target_link_libraries(clusterlm-llama-rpc-server PRIVATE ggml)
target_compile_features(clusterlm-llama-rpc-server PRIVATE cxx_std_17)
set_target_properties(clusterlm-llama-rpc-server PROPERTIES OUTPUT_NAME ggml-rpc-server)

# The pinned llama.cpp's C API for code that drives it directly (tests, the P0-A baseline harness): headers as
# SYSTEM includes so the project's warning policy does not apply to them.
add_library(clusterlm_llama_client INTERFACE)
target_link_libraries(clusterlm_llama_client INTERFACE llama ggml)
target_include_directories(clusterlm_llama_client SYSTEM INTERFACE ${_llama_src}/include ${_llama_src}/ggml/include)
target_compile_definitions(clusterlm_llama_client INTERFACE CLUSTERLM_LLAMA_PIN="${_llama_pin}")

# Writers for a tiny random-weight llama-architecture GGUF (tests and the bench proof run). Pure GGUF bytes:
# no llama.cpp dependency.
clusterlm_module(clusterlm_llama_tiny_model
  SOURCES src/llama_tiny_model.cpp
  DEPS clusterlm_common clusterlm_objects)

clusterlm_module(clusterlm_backend_llama
  SOURCES src/llama_backend.cpp src/llama_engine.cpp src/llama_manifest.cpp
  DEPS clusterlm_common clusterlm_domain clusterlm_objects clusterlm_windows
  PRIVATE_DEPS llama ggml)
target_compile_definitions(clusterlm_backend_llama PRIVATE CLUSTERLM_LLAMA_PIN="${_llama_pin}")
# llama.h and ggml headers are third-party: include as SYSTEM so project warnings do not apply to them.
target_include_directories(clusterlm_backend_llama SYSTEM PRIVATE ${_llama_src}/include ${_llama_src}/ggml/include)
if(CLUSTERLM_LLAMA_CUDA)
  target_compile_definitions(clusterlm_backend_llama PUBLIC CLUSTERLM_LLAMA_CUDA=1)
endif()
