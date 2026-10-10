# Tests for the llama.cpp backend; present only when the pinned llama.cpp is built (-DCLUSTERLM_ENABLE_LLAMA=ON).
if(NOT TARGET clusterlm_backend_llama)
  return()
endif()
clusterlm_test(test_llama_backend
  SOURCES test_llama_backend.cpp
  DEPS clusterlm_backend_llama clusterlm_llama_tiny_model clusterlm_coordinator clusterlm_domain clusterlm_objects clusterlm_llama_client)

clusterlm_test(test_llama_architectures
  SOURCES test_llama_architectures.cpp
  DEPS clusterlm_backend_llama clusterlm_llama_tiny_model clusterlm_coordinator clusterlm_domain clusterlm_objects clusterlm_llama_client)

clusterlm_test(test_llama_tensor_types
  SOURCES test_llama_tensor_types.cpp
  DEPS clusterlm_backend_llama clusterlm_llama_tiny_model clusterlm_coordinator clusterlm_domain clusterlm_objects clusterlm_llama_client)

clusterlm_test(test_baseline_llama_rpc SLOW
  SOURCES test_baseline_llama_rpc.cpp
  DEPS clusterlm_backend_llama clusterlm_llama_tiny_model clusterlm_coordinator clusterlm_platform clusterlm_objects clusterlm_llama_client)
add_dependencies(test_baseline_llama_rpc clusterlm-bench)
