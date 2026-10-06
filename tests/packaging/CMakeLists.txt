# Installer support: the uninstaller's `clusterlm-node-service --cleanup`, and static consistency checks of the
# packaging sources (WiX, PowerShell, install rules, CI) against the code they refer to.
clusterlm_test(test_packaging SOURCES test_cleanup.cpp DEPS clusterlm_lease_store clusterlm_platform)
target_compile_definitions(test_packaging PRIVATE
  CLUSTERLM_NODE_SERVICE_BINARY="$<TARGET_FILE:clusterlm-node-service>"
  CLUSTERLM_CRASH_HELPER_BINARY="$<TARGET_FILE:lease_store_crash_helper>")
add_dependencies(test_packaging clusterlm-node-service lease_store_crash_helper)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME packaging_sources_consistent
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/packaging/check_packaging.py)
endif()
