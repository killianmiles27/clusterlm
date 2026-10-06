clusterlm_module(clusterlm_catalog
  SOURCES src/catalog.cpp src/readiness.cpp
  DEPS clusterlm_common
  PRIVATE_DEPS clusterlm_third_party)
