clusterlm_test(test_planning SOURCES test_planning.cpp
  DEPS clusterlm_planning clusterlm_node_worker clusterlm_domain)
target_compile_definitions(test_planning PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
