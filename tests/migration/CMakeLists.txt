clusterlm_test(test_migration SOURCES test_migration.cpp DEPS clusterlm_migration)
target_compile_definitions(test_migration PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
