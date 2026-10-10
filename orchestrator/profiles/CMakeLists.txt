clusterlm_module(clusterlm_profiles
  SOURCES src/strict_json.cpp src/profile.cpp src/validate.cpp src/topology.cpp
  DEPS clusterlm_common clusterlm_library
  PRIVATE_DEPS clusterlm_third_party)
