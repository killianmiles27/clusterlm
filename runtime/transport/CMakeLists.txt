clusterlm_module(clusterlm_transport
  SOURCES src/socket.cpp src/stream.cpp src/tls.cpp src/framed.cpp src/transport.cpp src/security.cpp
          src/fault_injector.cpp src/impairment.cpp
  DEPS clusterlm_common Threads::Threads
  PRIVATE_DEPS OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(clusterlm_transport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(WIN32)
  target_link_libraries(clusterlm_transport PRIVATE ws2_32)
endif()
