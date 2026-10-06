clusterlm_test(test_node_service SOURCES test_supervisor.cpp test_supervisor_power.cpp
  DEPS clusterlm_node_service clusterlm_coordinator clusterlm_objects)
target_compile_definitions(test_node_service PRIVATE CLUSTERLM_NODE_BINARY="$<TARGET_FILE:clusterlm-node>")
add_dependencies(test_node_service clusterlm-node)
