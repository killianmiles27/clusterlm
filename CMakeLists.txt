# ClusterLM — distributed heterogeneous local LLM inference for Windows.
#
# The product targets native Windows (MSVC). Platform-independent components also build on Linux so the
# runtime/protocol/placement core can be developed and tested without the physical cluster. Windows-only
# adapters live under runtime/platform/windows and are compiled only for WIN32.
cmake_minimum_required(VERSION 3.24)
project(clusterlm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# All executables side by side: clusterlm-bench and the integration tests launch clusterlm-node from here.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(CLUSTERLM_BUILD_TESTS "Build unit and integration tests" ON)
option(CLUSTERLM_RUN_SLOW_TESTS "Run tests labelled slow (multi-process / integration); ctest -LE slow also skips them" ON)
option(CLUSTERLM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(CLUSTERLM_ENABLE_STRATA "Build the Strata backend adapter against the pinned Strata source (requires CUDA)" OFF)

include(cmake/ClusterLMWarnings.cmake)

# Fuzzing: every library is instrumented (ASan + UBSan + libFuzzer coverage) so the targets in tests/fuzz see
# inside the code under test. Declared here because the flags must precede every add_subdirectory.
option(CLUSTERLM_FUZZ "Build libFuzzer targets (clang only; instruments all libraries with ASan+UBSan)" OFF)
if(CLUSTERLM_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "CLUSTERLM_FUZZ requires clang (libFuzzer)")
  endif()
  add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)
find_package(OpenSSL 3.0 REQUIRED)
if(WIN32)
  # A static OpenSSL (vcpkg x64-windows-static, used for the installers) needs these system libraries; a dynamic one
  # does not mind them.
  set_property(TARGET OpenSSL::Crypto APPEND PROPERTY INTERFACE_LINK_LIBRARIES crypt32 ws2_32 user32 advapi32)
endif()

add_library(clusterlm_third_party INTERFACE)
target_include_directories(clusterlm_third_party SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
# doctest's forward declaration of std::basic_ostream is incompatible with recent MSVC STLs; use the real headers.
target_compile_definitions(clusterlm_third_party INTERFACE DOCTEST_CONFIG_USE_STD_HEADERS)

if(CLUSTERLM_BUILD_TESTS)
  enable_testing()
endif()

# Module order follows the dependency graph: common -> objects -> domain/windows -> transport -> lease-store
# -> placement -> protocol -> apps/bench. Each module is a static library with an explicit include root.
foreach(module
    runtime/common
    runtime/platform
    runtime/objects
    runtime/windows
    runtime/domain
    runtime/transport
    runtime/protocol
    runtime/backends
    node/lease-store
    orchestrator/placement
    orchestrator/coordinator
    orchestrator/planning
    orchestrator/catalog
    orchestrator/father-service
    orchestrator/diagnostics
    node/worker
    node/service
    bench
    apps
    experimental/expert-domains)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${module}/CMakeLists.txt)
    add_subdirectory(${module})
  endif()
endforeach()

if(CLUSTERLM_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()

# install() rules for the Windows packages (components "father" and "node"; see packaging/ and docs/packaging.md).
include(cmake/ClusterLMInstall.cmake)
