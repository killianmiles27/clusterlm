# Dev end-to-end: real clusterlm-node-service x2 + clusterlm-father-agent + scripted IPC client (labelled
# integration; also slow). Uses the explicit --dev-fixture-model override; nothing here is a measurement.
clusterlm_test(test_dev_e2e SLOW SOURCES test_dev_e2e.cpp DEPS clusterlm_objects clusterlm_transport clusterlm_platform)
target_compile_definitions(test_dev_e2e PRIVATE
  CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  CLUSTERLM_NODE_BINARY="$<TARGET_FILE:clusterlm-node>"
  CLUSTERLM_NODE_SERVICE_BINARY="$<TARGET_FILE:clusterlm-node-service>"
  CLUSTERLM_FATHER_AGENT_BINARY="$<TARGET_FILE:clusterlm-father-agent>")
add_dependencies(test_dev_e2e clusterlm-node clusterlm-node-service clusterlm-father-agent)
set_tests_properties(test_dev_e2e PROPERTIES LABELS "integration;slow" TIMEOUT 600)
