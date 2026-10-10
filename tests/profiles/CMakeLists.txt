clusterlm_test(test_profiles SOURCES test_profile.cpp test_topology.cpp test_compat.cpp DEPS clusterlm_profiles)
target_include_directories(test_profiles PRIVATE ${PROJECT_SOURCE_DIR}/tests/objects)
target_compile_definitions(test_profiles PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
