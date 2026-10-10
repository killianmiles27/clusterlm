clusterlm_module(clusterlm_config
  SOURCES src/settings.cpp src/store.cpp
  DEPS clusterlm_common clusterlm_platform clusterlm_transport clusterlm_profiles clusterlm_library clusterlm_migration
  PRIVATE_DEPS clusterlm_third_party)
