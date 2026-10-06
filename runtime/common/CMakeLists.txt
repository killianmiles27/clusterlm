clusterlm_module(clusterlm_common
  SOURCES src/status.cpp src/bytes.cpp src/digest.cpp src/log.cpp
  DEPS Threads::Threads
  PRIVATE_DEPS OpenSSL::Crypto)
