# ClusterLM Bench: harness library (also used by integration tests) and the clusterlm-bench executable.
clusterlm_module(clusterlm_bench_harness
  SOURCES src/host_probe.cpp src/result.cpp src/local_cluster.cpp
  DEPS clusterlm_coordinator clusterlm_platform clusterlm_transport clusterlm_third_party)
target_include_directories(clusterlm_bench_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/apps/common)

add_executable(clusterlm-bench src/main.cpp src/cmd_basic.cpp src/cmd_transport.cpp src/cmd_cluster.cpp)
target_link_libraries(clusterlm-bench PRIVATE clusterlm_bench_harness clusterlm_placement clusterlm_domain)
target_compile_definitions(clusterlm-bench PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
clusterlm_set_warnings(clusterlm-bench)
# clusterlm-bench spawns clusterlm-node from its own directory.
add_dependencies(clusterlm-bench clusterlm-node)
