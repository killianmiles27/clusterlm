# EXPERIMENTAL grouped expert-domain topology (P0-C). Isolated: nothing under runtime/, node/, orchestrator/ or
# apps/ may depend on this directory, so it can be deleted if the layer-domain design wins.
clusterlm_module(clusterlm_expert_domains
  SOURCES src/wire.cpp src/assignment.cpp src/domain_server.cpp src/father_executor.cpp src/analytic.cpp
          src/rig.cpp src/simulation.cpp src/quant_experts.cpp src/peer.cpp
  DEPS clusterlm_domain clusterlm_objects clusterlm_transport clusterlm_common Threads::Threads
       clusterlm_bench_harness clusterlm_coordinator clusterlm_third_party)
# Same reproducibility rule as the reference backend: no FMA contraction, so sums round identically.
if(MSVC)
  target_compile_options(clusterlm_expert_domains PRIVATE /fp:precise)
else()
  target_compile_options(clusterlm_expert_domains PRIVATE -ffp-contract=off)
endif()

add_executable(clusterlm-expert-domain-bench src/bench_main.cpp)
target_link_libraries(clusterlm-expert-domain-bench PRIVATE clusterlm_expert_domains clusterlm_cli)
target_compile_definitions(clusterlm-expert-domain-bench PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
clusterlm_set_warnings(clusterlm-expert-domain-bench)
# The layer-domain comparison spawns clusterlm-node processes from this executable's directory.
add_dependencies(clusterlm-expert-domain-bench clusterlm-node)
