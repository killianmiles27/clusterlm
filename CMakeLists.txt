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
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(CLUSTERLM_BUILD_TESTS "Build unit and integration tests" ON)
option(CLUSTERLM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(CLUSTERLM_ENABLE_STRATA "Build the Strata backend adapter against the pinned Strata source (requires CUDA)" OFF)

include(cmake/ClusterLMWarnings.cmake)

find_package(Threads REQUIRED)
find_package(OpenSSL 3.0 REQUIRED)

add_library(clusterlm_third_party INTERFACE)
target_include_directories(clusterlm_third_party SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

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
    node/lease-store
    orchestrator/placement
    orchestrator/coordinator
    node/worker
    bench
    apps)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${module}/CMakeLists.txt)
    add_subdirectory(${module})
  endif()
endforeach()

if(CLUSTERLM_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
