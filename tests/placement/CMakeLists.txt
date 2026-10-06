clusterlm_test(test_placement SOURCES test_placement.cpp test_search_space.cpp DEPS clusterlm_placement)
target_compile_definitions(test_placement PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
