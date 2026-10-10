clusterlm_test(test_ui_father_viewmodel SOURCES test_father_viewmodel.cpp DEPS clusterlm_ui_father)
clusterlm_test(test_ui_node_viewmodel SOURCES test_node_viewmodel.cpp DEPS clusterlm_ui_node)
clusterlm_test(test_ui_clients SOURCES test_clients.cpp DEPS clusterlm_ui_common)
clusterlm_test(test_ui_draw_smoke SOURCES test_draw_smoke.cpp DEPS clusterlm_ui_father_draw clusterlm_ui_node_draw)
# End to end: InProcessFatherClient over a real FatherService on the fixture model (tests/father_service pattern).
clusterlm_test(test_ui_e2e SOURCES test_e2e.cpp DEPS clusterlm_ui_father clusterlm_node_worker clusterlm_domain)
target_compile_definitions(test_ui_e2e PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# IpcFatherClient against a real clusterlm-father-agent (--dev-fixture-model); like tests/father_agent/test_dev_e2e.
clusterlm_test(test_ui_ipc_father SLOW SOURCES test_ipc_father_e2e.cpp
  DEPS clusterlm_ui_father clusterlm_objects clusterlm_platform)
target_compile_definitions(test_ui_ipc_father PRIVATE
  CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  CLUSTERLM_FATHER_AGENT_BINARY="$<TARGET_FILE:clusterlm-father-agent>")
add_dependencies(test_ui_ipc_father clusterlm-father-agent)
set_tests_properties(test_ui_ipc_father PROPERTIES LABELS "integration;slow" TIMEOUT 300)

# Host management views (workstream E)
clusterlm_test(test_ui_host_viewmodels SOURCES test_host_viewmodels.cpp DEPS clusterlm_ui_host)
clusterlm_test(test_ui_host_draw_smoke SOURCES test_host_draw_smoke.cpp DEPS clusterlm_ui_host_draw)
