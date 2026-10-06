clusterlm_module(clusterlm_pairing
  SOURCES src/spake2.cpp src/pairing.cpp
  DEPS clusterlm_common clusterlm_transport Threads::Threads
  PRIVATE_DEPS OpenSSL::Crypto)
