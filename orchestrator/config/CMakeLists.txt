clusterlm_module(clusterlm_config
  SOURCES src/settings.cpp src/store.cpp
  DEPS clusterlm_common clusterlm_platform clusterlm_transport
  PRIVATE_DEPS clusterlm_third_party)
