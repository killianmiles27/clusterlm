clusterlm_module(clusterlm_library
  SOURCES src/model_record.cpp src/scanner.cpp src/library.cpp
  DEPS clusterlm_common clusterlm_objects
  PRIVATE_DEPS clusterlm_third_party)
