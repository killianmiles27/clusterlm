clusterlm_test(test_ipc
  SOURCES test_ipc.cpp test_helper_activity.cpp test_key_acl.cpp
  DEPS clusterlm_platform clusterlm_transport clusterlm_common)

# Father agent transport (no worker process needed).
clusterlm_test(test_father_agent SOURCES test_father_agent.cpp DEPS clusterlm_father_agent)

# ServiceCore with a real worker process and a real local pipe (like tests/node_service).
clusterlm_test(test_service_core SOURCES test_service_core.cpp DEPS clusterlm_node_service)
target_compile_definitions(test_service_core PRIVATE CLUSTERLM_NODE_BINARY="$<TARGET_FILE:clusterlm-node>")
add_dependencies(test_service_core clusterlm-node)
