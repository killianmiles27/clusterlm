clusterlm_module(clusterlm_profiles
  SOURCES src/strict_json.cpp src/profile.cpp src/validate.cpp src/topology.cpp src/compat.cpp
  DEPS clusterlm_common clusterlm_library clusterlm_domain_api
  PRIVATE_DEPS clusterlm_third_party)
