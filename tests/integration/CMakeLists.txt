# Multi-process integration tests: real clusterlm-node processes on localhost, driven through the protocol by
# clusterlm-bench. Labelled so they can be selected (ctest -L integration) or skipped (ctest -LE integration).
add_test(NAME integration_cluster_tls_direct
  COMMAND clusterlm-bench cluster --max-new 24 --repeat 1 --q 1,3 --drafter scripted:0.3 --out ${CMAKE_CURRENT_BINARY_DIR}/cluster.json)
add_test(NAME integration_cluster_relay_impaired
  COMMAND clusterlm-bench cluster --max-new 16 --repeat 1 --q 2 --drafter scripted:0.5 --relay --impair gige-simulated --out ${CMAKE_CURRENT_BINARY_DIR}/cluster-relay.json)
add_test(NAME integration_faults
  COMMAND clusterlm-bench faults --max-new 12 --release-cycles 3 --out ${CMAKE_CURRENT_BINARY_DIR}/faults.json)
set_tests_properties(integration_cluster_tls_direct integration_cluster_relay_impaired integration_faults
  PROPERTIES LABELS integration TIMEOUT 300)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME qualification_registry_in_sync
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/gen_qualification_md.py --check)
endif()
