clusterlm_module(clusterlm_diagnostics
  SOURCES src/diagnostics.cpp
  DEPS clusterlm_common clusterlm_platform
  PRIVATE_DEPS clusterlm_third_party)
target_compile_definitions(clusterlm_diagnostics PRIVATE
  CLUSTERLM_VERSION_STRING="${PROJECT_VERSION}"
  CLUSTERLM_BUILD_TYPE_STRING="$<CONFIG>")
