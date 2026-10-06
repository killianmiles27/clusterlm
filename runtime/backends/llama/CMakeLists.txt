# llama.cpp comparison-arm adapter skeleton. Built only with -DCLUSTERLM_ENABLE_LLAMA=ON. It does not link
# llama.cpp (the arm drives external processes), so no checkout is required to compile it.
option(CLUSTERLM_ENABLE_LLAMA "Build the llama.cpp comparison-arm adapter skeleton" OFF)
if(NOT CLUSTERLM_ENABLE_LLAMA)
  return()
endif()

file(READ ${PROJECT_SOURCE_DIR}/third_party/upstream.json _pins)
string(JSON _llama_pin GET "${_pins}" pins llama.cpp commit)

clusterlm_module(clusterlm_backend_llama
  SOURCES src/llama_backend.cpp
  DEPS clusterlm_common clusterlm_domain)
target_compile_definitions(clusterlm_backend_llama PRIVATE CLUSTERLM_LLAMA_PIN="${_llama_pin}")
