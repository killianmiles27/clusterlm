clusterlm_test(test_library SOURCES test_library.cpp DEPS clusterlm_library clusterlm_objects)
target_include_directories(test_library PRIVATE ${PROJECT_SOURCE_DIR}/tests/objects)
