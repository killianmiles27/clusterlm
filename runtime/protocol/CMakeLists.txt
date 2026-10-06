clusterlm_module(clusterlm_protocol
  SOURCES src/messages.cpp src/wire.cpp
  DEPS clusterlm_common clusterlm_objects clusterlm_domain_api clusterlm_transport)
