clusterlm_test(test_profiles SOURCES test_profile.cpp test_topology.cpp DEPS clusterlm_profiles)
target_compile_definitions(test_profiles PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
