clusterlm_test(test_platform
  SOURCES test_fs_safety.cpp test_mapped_file.cpp test_adapters.cpp test_firewall.cpp test_paths.cpp
          test_helper_startup.cpp test_service_host.cpp
  DEPS clusterlm_platform clusterlm_common)
