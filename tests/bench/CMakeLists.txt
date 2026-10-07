# ClusterLM Bench tests.
#
# Unit tests (kernels, CPU/memory measurement logic, profile merge, GPU orchestration over a fake probe, schema
# validation, registry command lines) plus quick smoke runs of the real clusterlm-bench commands whose outputs
# the test_bench executable then validates against bench/schema/benchmark-result.schema.json.
set(BENCH_RESULTS ${CMAKE_CURRENT_BINARY_DIR}/results)
file(MAKE_DIRECTORY ${BENCH_RESULTS})

clusterlm_test(test_bench
  SOURCES test_expert_kernels.cpp test_cpu_memory.cpp test_profile_merge.cpp test_gpu_schema.cpp test_bench_outputs.cpp test_system_observers.cpp
          test_real_model_harness.cpp
  DEPS clusterlm_bench_harness clusterlm_placement clusterlm_strata_core clusterlm_father_service)
# test_real_model_harness.cpp includes the Strata fake engine (tests/backends) and the synthetic GGUF vocabulary
# (tests/father_service) by relative path: they drive the bench's own pass.
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
# WP21: real backends and real models. The reference backend through the new flags (Synthetic, labelled), the correctness
# reference skipped, and the placement search's top candidates run through the cluster harness (predicted vs measured).
clusterlm_bench_smoke(cluster_backend
  ARGS cluster --tier strong --backend reference --q 1,2 --max-new 8 --repeat 1 --drafter scripted:0.4
       --out ${BENCH_RESULTS}/result-cluster-backend.json)
clusterlm_bench_smoke(cluster_noref
  ARGS cluster --tier strong --q 1 --max-new 8 --repeat 1 --no-reference --out ${BENCH_RESULTS}/result-cluster-noref.json)
clusterlm_bench_smoke(placement_validate
  ARGS placement-validate --top 4 --repeat 2 --max-new 8 --prompt-len 24 --out ${BENCH_RESULTS}/result-placement-validate.json)
# The supervised fault scenarios run the Nodes under clusterlm-node-service (HQ-REL-01); labelled integration as well.
clusterlm_bench_smoke(faults_supervised
  ARGS faults --only supervised --max-new 8 --out ${BENCH_RESULTS}/result-faults-supervised.json)
set_tests_properties(bench_smoke_faults_supervised PROPERTIES LABELS "bench;integration")

# A backend this build does not have is a clear error before anything is spawned (exit 3, the message names the CMake option).
if(NOT TARGET clusterlm_backend_strata)
  add_test(NAME bench_cluster_strata_not_built
    COMMAND ${BENCH} cluster --backend strata --tier fast --out ${BENCH_RESULTS}/result-cluster-strata-missing.json)
  set_tests_properties(bench_cluster_strata_not_built PROPERTIES LABELS bench
    PASS_REGULAR_EXPRESSION "built without CLUSTERLM_ENABLE_STRATA")
  add_test(NAME bench_faults_strata_not_built
    COMMAND ${BENCH} faults --backend strata --only stall --out ${BENCH_RESULTS}/result-faults-strata-missing.json)
  set_tests_properties(bench_faults_strata_not_built PROPERTIES LABELS bench
    PASS_REGULAR_EXPRESSION "built without CLUSTERLM_ENABLE_STRATA")
  add_test(NAME bench_placement_validate_strata_not_built
    COMMAND ${BENCH} placement-validate --backend strata --out ${BENCH_RESULTS}/result-pv-strata-missing.json)
  set_tests_properties(bench_placement_validate_strata_not_built PROPERTIES LABELS bench
    PASS_REGULAR_EXPRESSION "built without CLUSTERLM_ENABLE_STRATA")
endif()
add_test(NAME bench_cluster_unknown_backend COMMAND ${BENCH} cluster --backend vulkan-magic --tier fast)
set_tests_properties(bench_cluster_unknown_backend PROPERTIES LABELS bench WILL_FAIL TRUE)

# HQ-MTP-01 real text: the corpus tokenized by the Father tokenizer from the llama.cpp qwen2 vocabulary (the real tokenizer
# code on a real 150k vocabulary; the fixture model folds the ids, so the result is Synthetic). Registered only when the pinned
# upstream checkout has the vocabulary (python3 scripts/fetch_upstream.py llama.cpp).
set(QWEN2_VOCAB ${PROJECT_SOURCE_DIR}/third_party/upstream/llama.cpp/models/ggml-vocab-qwen2.gguf)
if(EXISTS ${QWEN2_VOCAB})
  file(WRITE ${BENCH_RESULTS}/corpus-sample.txt
    "Distributed inference across idle machines keeps every model file on one computer. The quick brown fox jumps over the lazy dog.\n"
    "A mixture of experts activates only a few experts per token, so most of the weights stay cold in memory.\n"
    "int main() { std::puts(\"hello\"); return 0; }\n"
    "Explain why the sky is blue in two sentences, then list three practical consequences for photography.\n")
  clusterlm_bench_smoke(cluster_corpus_tokenizer
    ARGS cluster --tier strong --q 1,2 --corpus ${BENCH_RESULTS}/corpus-sample.txt --tokenizer-gguf ${QWEN2_VOCAB}
         --prompt-len 16 --max-new 8 --repeat 1 --drafter scripted:0.4 --out ${BENCH_RESULTS}/result-cluster-corpus.json)
  clusterlm_bench_smoke(placement_inputs_tokenizer
    ARGS placement-inputs --tokenizer-gguf ${QWEN2_VOCAB} --q 1,2 --max-new 8 --routing-out ${BENCH_RESULTS}/routing-qwen2.json
         --work ${BENCH_RESULTS}/placement-inputs-work --out ${BENCH_RESULTS}/result-placement-inputs-tokenizer.json)
  target_compile_definitions(test_bench PRIVATE CLUSTERLM_QWEN2_VOCAB_PRESENT=1)
endif()

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
