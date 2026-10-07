# ClusterLM Bench tests.
#
# Unit tests (kernels, CPU/memory measurement logic, profile merge, GPU orchestration over a fake probe, schema
# validation, registry command lines) plus quick smoke runs of the real clusterlm-bench commands whose outputs
# the test_bench executable then validates against bench/schema/benchmark-result.schema.json.
set(BENCH_RESULTS ${CMAKE_CURRENT_BINARY_DIR}/results)
file(MAKE_DIRECTORY ${BENCH_RESULTS})

clusterlm_test(test_bench
  SOURCES test_expert_kernels.cpp test_cpu_memory.cpp test_profile_merge.cpp test_gpu_schema.cpp test_bench_outputs.cpp test_system_observers.cpp
  DEPS clusterlm_bench_harness clusterlm_placement)
target_compile_definitions(test_bench PRIVATE
  CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  CLUSTERLM_BENCH_RESULTS_DIR="${BENCH_RESULTS}")
set_tests_properties(test_bench PROPERTIES FIXTURES_REQUIRED bench_results TIMEOUT 300)

set(BENCH $<TARGET_FILE:clusterlm-bench>)
set(FATHER_FIXTURE ${PROJECT_SOURCE_DIR}/fixtures/profiles/Father-4060Ti-7600.json)

function(clusterlm_bench_smoke name)
  cmake_parse_arguments(ARG "" "PASS_REGEX" "ARGS" ${ARGN})
  add_test(NAME bench_smoke_${name} COMMAND ${BENCH} ${ARG_ARGS})
  set_tests_properties(bench_smoke_${name} PROPERTIES FIXTURES_SETUP bench_results TIMEOUT 300 LABELS bench)
  if(ARG_PASS_REGEX)
    # Hardware-dependent commands exit 3 where the hardware is absent; the output is what is checked.
    set_tests_properties(bench_smoke_${name} PROPERTIES PASS_REGULAR_EXPRESSION "${ARG_PASS_REGEX}")
  endif()
endfunction()

clusterlm_bench_smoke(cpu ARGS cpu --quick --quiet --out ${BENCH_RESULTS}/result-cpu.json)
clusterlm_bench_smoke(cpu_sustained
  ARGS cpu --quick --quiet --sustained --minutes 0.02 --sample-s 0.2 --out ${BENCH_RESULTS}/result-cpu-sustained.json)
if(TARGET clusterlm_strata_cpu)
  # -DCLUSTERLM_ENABLE_STRATA_CPU=ON (or _STRATA=ON): the real Strata CPU kernels back the "strata-cpu" provider.
  clusterlm_bench_smoke(cpu_strata
    ARGS cpu --provider strata-cpu --representations iq3_s --q 1,2 --threads 1 --quick --quiet
         --out ${BENCH_RESULTS}/result-cpu-strata.json
    PASS_REGEX "wrote .*result-cpu-strata.json")
else()
  clusterlm_bench_smoke(cpu_strata_stub
    ARGS cpu --provider strata-cpu --quick --quiet --out ${BENCH_RESULTS}/result-cpu-strata.json
    PASS_REGEX "wrote .*result-cpu-strata.json")
endif()
clusterlm_bench_smoke(memory ARGS memory --quick --out ${BENCH_RESULTS}/result-memory.json)
clusterlm_bench_smoke(gpu ARGS gpu --quick --out ${BENCH_RESULTS}/result-gpu.json PASS_REGEX "wrote .*result-gpu.json")
clusterlm_bench_smoke(pcie ARGS pcie --quick --out ${BENCH_RESULTS}/result-pcie.json PASS_REGEX "wrote .*result-pcie.json")
clusterlm_bench_smoke(transport_loopback
  ARGS transport --simulate-nodes 2 --iterations 5 --burst-mib 2 --network-out ${BENCH_RESULTS}/network-loopback.json
       --machine-id smoke-father --out ${BENCH_RESULTS}/result-transport.json)
clusterlm_bench_smoke(transport_node_to_node
  ARGS transport --node-to-node --iterations 5 --burst-mib 2 --out ${BENCH_RESULTS}/result-node-to-node.json)
clusterlm_bench_smoke(calibrate
  ARGS calibrate --quick --quiet --machine-id smoke-father --role father --base ${FATHER_FIXTURE} --out-dir ${BENCH_RESULTS}
       --out ${BENCH_RESULTS}/result-calibrate.json)
clusterlm_bench_smoke(cluster_routing
  ARGS cluster --max-new 8 --repeat 1 --q 2 --drafter scripted:0.4 --compare-routing
       --out ${BENCH_RESULTS}/result-cluster-routing.json)
clusterlm_bench_smoke(cluster_prepare
  ARGS cluster --phase prepare --repeat 2 --tier strong --out ${BENCH_RESULTS}/result-cluster-prepare.json)

# WP20 observers: sweep in one invocation, subset of fault scenarios, NVML and cache census.
clusterlm_bench_smoke(cluster_contexts
  ARGS cluster --tier strong --contexts 16,32 --q 1,2 --repeat 2 --max-new 8 --drafter scripted:0.4 --census
       --out ${BENCH_RESULTS}/result-cluster-contexts.json)
clusterlm_bench_smoke(faults_subset
  ARGS faults --only ready,release_cycles --release-cycles 2 --out ${BENCH_RESULTS}/result-faults-subset.json)
clusterlm_bench_smoke(nvml ARGS nvml --minutes 0.01 --sample-s 0.2 --out ${BENCH_RESULTS}/result-nvml.json
  PASS_REGEX "wrote .*result-nvml.json")
clusterlm_bench_smoke(storage_census_before
  ARGS storage-census --before --snapshot ${BENCH_RESULTS}/census-snapshot.json --exclude ${BENCH_RESULTS}
       --out ${BENCH_RESULTS}/result-census-before.json)
clusterlm_bench_smoke(storage_census_after
  ARGS storage-census --after --snapshot ${BENCH_RESULTS}/census-snapshot.json --exclude ${BENCH_RESULTS}
       --out ${BENCH_RESULTS}/result-census-after.json)
# The "after" run reads the snapshot the "before" run wrote.
set_tests_properties(bench_smoke_storage_census_after PROPERTIES DEPENDS bench_smoke_storage_census_before)

# The command surface rejects flags the command does not implement.
add_test(NAME bench_rejects_unknown_flag COMMAND ${BENCH} cpu --no-such-flag)
set_tests_properties(bench_rejects_unknown_flag PROPERTIES WILL_FAIL TRUE LABELS bench)
