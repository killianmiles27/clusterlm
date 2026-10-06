# Security regressions: TLS policy, peer identity binding, replay, allocation bounds, staging paths, process launch.
clusterlm_test(test_security_hardening
  SOURCES test_tls_policy.cpp test_peer_binding.cpp test_replay.cpp test_hardening.cpp
  DEPS clusterlm_father_service clusterlm_node_worker clusterlm_coordinator clusterlm_protocol clusterlm_transport
       clusterlm_domain clusterlm_objects clusterlm_catalog clusterlm_platform clusterlm_lease_store)
target_compile_definitions(test_security_hardening PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
if(NOT WIN32)
  target_link_libraries(test_security_hardening PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
