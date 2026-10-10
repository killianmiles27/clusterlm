clusterlm_module(clusterlm_migration
  SOURCES src/example_set.cpp src/tier_to_profile.cpp src/settings_v1_to_v2.cpp
  DEPS clusterlm_common clusterlm_catalog clusterlm_profiles clusterlm_library clusterlm_domain_api
  PRIVATE_DEPS clusterlm_third_party)
