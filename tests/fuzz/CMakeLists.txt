# Fuzzing (libFuzzer) and corpus replay.
#
# Every fuzz target is one translation unit exporting LLVMFuzzerTestOneInput. It is built twice:
#   * fuzz_<name>          - a libFuzzer executable, only with -DCLUSTERLM_FUZZ=ON under clang (the top-level
#                            CMakeLists instruments every library with ASan+UBSan and fuzzer-no-link then);
#   * fuzz_replay_<name>   - always: the same code driven by replay_main.cpp over the committed corpus in
#                            tests/fuzz/corpus/<name>/ plus a bounded deterministic mutation pass, registered
#                            in ctest. CI's clang sanitizer job therefore runs the corpus under ASan+UBSan.
#
#   cmake -S . -B build-fuzz -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCLUSTERLM_FUZZ=ON
#   cmake --build build-fuzz
#   build-fuzz/bin/fuzz_protocol_decode tests/fuzz/corpus/protocol_decode -max_total_time=60

# CLUSTERLM_FUZZ is declared in the top-level CMakeLists.txt (its instrumentation flags must precede every module).
add_library(clusterlm_fuzz_replay STATIC replay_main.cpp)
target_link_libraries(clusterlm_fuzz_replay PUBLIC clusterlm_common)
clusterlm_set_warnings(clusterlm_fuzz_replay)

# clusterlm_fuzz_target(<name> SOURCE <file> DEPS <libs>)
function(clusterlm_fuzz_target name)
  cmake_parse_arguments(ARG "" "SOURCE" "DEPS" ${ARGN})
  set(corpus ${CMAKE_CURRENT_SOURCE_DIR}/corpus/${name})
  add_executable(fuzz_replay_${name} ${ARG_SOURCE})
  target_link_libraries(fuzz_replay_${name} PRIVATE clusterlm_fuzz_replay ${ARG_DEPS} clusterlm_third_party)
  clusterlm_set_warnings(fuzz_replay_${name})
  add_test(NAME fuzz_replay_${name} COMMAND fuzz_replay_${name} ${corpus})
  set_tests_properties(fuzz_replay_${name} PROPERTIES LABELS fuzz)
  if(CLUSTERLM_FUZZ)
    add_executable(fuzz_${name} ${ARG_SOURCE})
    target_link_libraries(fuzz_${name} PRIVATE ${ARG_DEPS} clusterlm_third_party)
    target_compile_options(fuzz_${name} PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_${name} PRIVATE -fsanitize=fuzzer)
  endif()
endfunction()

clusterlm_fuzz_target(frame_stream SOURCE fuzz_frame_stream.cpp DEPS clusterlm_protocol clusterlm_transport)
clusterlm_fuzz_target(protocol_decode SOURCE fuzz_protocol_decode.cpp DEPS clusterlm_protocol)
clusterlm_fuzz_target(stage_activations SOURCE fuzz_stage_activations.cpp DEPS clusterlm_domain_api clusterlm_protocol)
clusterlm_fuzz_target(manifest_decode SOURCE fuzz_manifest_decode.cpp DEPS clusterlm_objects)
clusterlm_fuzz_target(manifest_json SOURCE fuzz_manifest_json.cpp DEPS clusterlm_objects)
clusterlm_fuzz_target(lease_journal SOURCE fuzz_lease_journal.cpp DEPS clusterlm_lease_store)
clusterlm_fuzz_target(catalog_json SOURCE fuzz_catalog_json.cpp DEPS clusterlm_catalog)
clusterlm_fuzz_target(profile_json SOURCE fuzz_profile_json.cpp DEPS clusterlm_placement)
clusterlm_fuzz_target(text_parsers SOURCE fuzz_text_parsers.cpp DEPS clusterlm_coordinator clusterlm_transport)
if(TARGET clusterlm_expert_domains)
  clusterlm_fuzz_target(expert_wire SOURCE fuzz_expert_wire.cpp DEPS clusterlm_expert_domains)
endif()
# GGUF header parser + manifest builder (runtime/objects/gguf*).
clusterlm_fuzz_target(gguf SOURCE fuzz_gguf.cpp DEPS clusterlm_objects)

# GGUF seed generator: header-only GGUF files from the writer (`gen_gguf_seeds tests/fuzz/corpus/gguf`).
add_executable(gen_gguf_seeds gen_gguf_seeds.cpp)
target_link_libraries(gen_gguf_seeds PRIVATE clusterlm_objects)
clusterlm_set_warnings(gen_gguf_seeds)

# Seed-corpus generator: builds valid encodings of every protocol message, manifest, journal, ... Not run by
# ctest; regenerate with `clusterlm_fuzz_gen_corpus tests/fuzz/corpus` after changing a wire format.
add_executable(clusterlm_fuzz_gen_corpus gen_corpus.cpp)
target_link_libraries(clusterlm_fuzz_gen_corpus PRIVATE clusterlm_protocol clusterlm_objects clusterlm_lease_store
                      clusterlm_transport clusterlm_domain_api clusterlm_catalog clusterlm_placement
                      clusterlm_third_party)
if(TARGET clusterlm_expert_domains)
  target_link_libraries(clusterlm_fuzz_gen_corpus PRIVATE clusterlm_expert_domains)
  target_compile_definitions(clusterlm_fuzz_gen_corpus PRIVATE CLUSTERLM_FUZZ_HAVE_EXPERT_DOMAINS=1)
endif()
target_compile_definitions(clusterlm_fuzz_gen_corpus PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
clusterlm_set_warnings(clusterlm_fuzz_gen_corpus)
