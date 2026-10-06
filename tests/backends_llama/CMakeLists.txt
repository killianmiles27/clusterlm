# Tests for the llama.cpp backend; present only when the pinned llama.cpp is built (-DCLUSTERLM_ENABLE_LLAMA=ON).
if(NOT TARGET clusterlm_backend_llama)
  return()
endif()
clusterlm_test(test_llama_backend
  SOURCES test_llama_backend.cpp
  DEPS clusterlm_backend_llama clusterlm_llama_tiny_model clusterlm_coordinator clusterlm_domain clusterlm_objects llama)
# llama.h is a third-party header: keep the project's warning policy off it.
get_target_property(_llama_inc llama INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(test_llama_backend SYSTEM PRIVATE ${_llama_inc})
