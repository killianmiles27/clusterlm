# Privacy assertions: the protocol schema, real captured traffic, logs, error replies and the diagnostics bundle
# must carry no prompts, responses, token arrays, logits or activation dumps to or about a Node.
clusterlm_test(test_privacy
  SOURCES test_message_schema.cpp test_traffic.cpp test_logs.cpp test_error_replies.cpp test_diagnostics.cpp
  DEPS clusterlm_father_service clusterlm_node_worker clusterlm_coordinator clusterlm_protocol clusterlm_transport
       clusterlm_domain clusterlm_objects clusterlm_diagnostics)
target_compile_definitions(test_privacy PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# The CLI entry point writes a bundle and exits 0; the content rules are covered by test_privacy.
add_test(NAME privacy_father_diagnostics_cli
  COMMAND clusterlm-father diagnostics --out ${CMAKE_CURRENT_BINARY_DIR}/diagnostics-bundle.json --bench-results results/example.json)
set_tests_properties(privacy_father_diagnostics_cli PROPERTIES PASS_REGULAR_EXPRESSION "wrote diagnostics bundle")
