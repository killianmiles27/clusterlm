# ClusterLM Bench: harness library (also used by integration tests) and the clusterlm-bench executable.
#
# GPU measurements need the CUDA runtime and cuBLAS. They are compiled in only with -DCLUSTERLM_BENCH_CUDA=ON AND a
# found CUDA toolkit (plain C++ against cudart/cuBLAS: no nvcc, no device code). Without it `clusterlm-bench gpu`
# reports "built without CUDA" and exits 3. Off by default so CI and CUDA-less hosts build unchanged; the target
# machines configure with it ON.
option(CLUSTERLM_BENCH_CUDA "Build the CUDA GPU probe into clusterlm-bench (requires the CUDA toolkit)" OFF)
set(CLUSTERLM_BENCH_CUDA_ENABLED OFF)
if(CLUSTERLM_BENCH_CUDA)
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    set(CLUSTERLM_BENCH_CUDA_ENABLED ON)
  else()
    message(WARNING "CLUSTERLM_BENCH_CUDA=ON but the CUDA toolkit was not found; building the bench without the GPU probe")
  endif()
endif()

set(CLUSTERLM_BENCH_HARNESS_SOURCES
  src/host_probe.cpp src/result.cpp src/local_cluster.cpp
  src/expert_kernels.cpp src/expert_providers_external.cpp src/cpu_bench.cpp
  src/memory_probe.cpp src/gpu_probe.cpp src/transport_bench.cpp
  src/bench_profile.cpp src/bench_common.cpp src/schema_check.cpp src/command_specs.cpp)
if(CLUSTERLM_BENCH_CUDA_ENABLED)
  list(APPEND CLUSTERLM_BENCH_HARNESS_SOURCES src/gpu_probe_cuda.cpp)
endif()

clusterlm_module(clusterlm_bench_harness
  SOURCES ${CLUSTERLM_BENCH_HARNESS_SOURCES}
  DEPS clusterlm_coordinator clusterlm_platform clusterlm_transport clusterlm_placement clusterlm_domain clusterlm_objects
       clusterlm_third_party Threads::Threads)
target_include_directories(clusterlm_bench_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/apps/common)
if(CLUSTERLM_BENCH_CUDA_ENABLED)
  target_compile_definitions(clusterlm_bench_harness PUBLIC CLUSTERLM_BENCH_HAS_CUDA=1)
  target_link_libraries(clusterlm_bench_harness PUBLIC CUDA::cudart CUDA::cublas)
endif()
# The real Strata CPU kernels (CLUSTERLM_ENABLE_STRATA_CPU / CLUSTERLM_ENABLE_STRATA) back the "strata-cpu" provider.
if(TARGET clusterlm_strata_cpu)
  target_link_libraries(clusterlm_bench_harness PUBLIC clusterlm_strata_cpu)
  target_compile_definitions(clusterlm_bench_harness PUBLIC CLUSTERLM_BENCH_HAS_STRATA_CPU=1)
endif()
if(WIN32)
  target_link_libraries(clusterlm_bench_harness PUBLIC psapi)
endif()

add_executable(clusterlm-bench src/main.cpp src/cmd_basic.cpp src/cmd_transport.cpp src/cmd_cluster.cpp
  src/cmd_cpu.cpp src/cmd_gpu.cpp src/cmd_calibrate.cpp src/cmd_placement_inputs.cpp src/cmd_baseline_llama.cpp)
target_link_libraries(clusterlm-bench PRIVATE clusterlm_bench_harness clusterlm_planning clusterlm_father_service)
target_compile_definitions(clusterlm-bench PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
clusterlm_set_warnings(clusterlm-bench)
# `baseline llama-rpc` drives the pinned llama.cpp in-process and spawns the pinned rpc-server (built beside us).
if(TARGET clusterlm_llama_client)
  target_link_libraries(clusterlm-bench PRIVATE clusterlm_llama_client)
  target_compile_definitions(clusterlm-bench PRIVATE CLUSTERLM_BENCH_HAS_LLAMA=1)
  add_dependencies(clusterlm-bench clusterlm-llama-rpc-server)
endif()
# clusterlm-bench spawns clusterlm-node from its own directory.
add_dependencies(clusterlm-bench clusterlm-node)
