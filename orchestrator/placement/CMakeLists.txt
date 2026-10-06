clusterlm_module(clusterlm_placement
  SOURCES src/profile.cpp src/model_inputs.cpp src/placement.cpp
  DEPS clusterlm_common
  PRIVATE_DEPS clusterlm_third_party)
