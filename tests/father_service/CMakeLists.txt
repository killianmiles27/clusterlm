clusterlm_test(test_father_service SOURCES test_father_service.cpp test_tokenizer.cpp test_gguf_bpe.cpp
  DEPS clusterlm_father_service clusterlm_node_worker clusterlm_domain)
target_compile_definitions(test_father_service PRIVATE CLUSTERLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# Verification of GgufBpeTokenizer against llama.cpp's own vocab tests (qwen2 + qwen35). The vocab files come from
# the pinned llama.cpp checkout (python3 scripts/fetch_upstream.py llama.cpp); they are never committed, so the test
# is only registered when they exist. Point CLUSTERLM_LLAMACPP_VOCAB_DIR elsewhere to use another checkout.
set(CLUSTERLM_LLAMACPP_VOCAB_DIR "${PROJECT_SOURCE_DIR}/third_party/upstream/llama.cpp/models" CACHE PATH
    "Directory with llama.cpp's ggml-vocab-qwen2/qwen35 .gguf/.inp/.out files")
set(_vocab_files "")
foreach(_v qwen2 qwen35)
  foreach(_ext gguf gguf.inp gguf.out)
    list(APPEND _vocab_files "${CLUSTERLM_LLAMACPP_VOCAB_DIR}/ggml-vocab-${_v}.${_ext}")
  endforeach()
endforeach()
set(_vocab_present TRUE)
foreach(_f ${_vocab_files})
  if(NOT EXISTS "${_f}")
    set(_vocab_present FALSE)
  endif()
endforeach()
if(_vocab_present)
  clusterlm_test(test_tokenizer_llamacpp_vocab SOURCES test_vocab_llamacpp.cpp DEPS clusterlm_father_service)
  target_compile_definitions(test_tokenizer_llamacpp_vocab PRIVATE CLUSTERLM_LLAMACPP_VOCAB_DIR="${CLUSTERLM_LLAMACPP_VOCAB_DIR}")
  set_tests_properties(test_tokenizer_llamacpp_vocab PROPERTIES LABELS vocab)
else()
  message(STATUS "llama.cpp vocab files not found in ${CLUSTERLM_LLAMACPP_VOCAB_DIR}: tokenizer vocab test not registered "
                 "(run: python3 scripts/fetch_upstream.py llama.cpp)")
endif()
