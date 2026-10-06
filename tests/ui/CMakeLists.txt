clusterlm_test(test_ui_father_viewmodel SOURCES test_father_viewmodel.cpp DEPS clusterlm_ui_father)
clusterlm_test(test_ui_node_viewmodel SOURCES test_node_viewmodel.cpp DEPS clusterlm_ui_node)
clusterlm_test(test_ui_clients SOURCES test_clients.cpp DEPS clusterlm_ui_common)
clusterlm_test(test_ui_draw_smoke SOURCES test_draw_smoke.cpp DEPS clusterlm_ui_father_draw clusterlm_ui_node_draw)
# End to end: InProcessFatherClient over a real FatherService on the fixture model (tests/father_service pattern).
clusterlm_test(test_ui_e2e SOURCES test_e2e.cpp DEPS clusterlm_ui_father clusterlm_node_worker clusterlm_domain)
target_compile_definitions(test_ui_e2e PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
