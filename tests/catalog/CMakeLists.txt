clusterlm_test(test_catalog SOURCES test_catalog.cpp test_readiness.cpp DEPS clusterlm_catalog)
target_compile_definitions(test_catalog PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
