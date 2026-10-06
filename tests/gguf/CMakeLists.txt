clusterlm_test(test_gguf
  SOURCES test_ggml_types.cpp test_gguf_parser.cpp test_gguf_manifest.cpp test_gguf_fixture.cpp
  DEPS clusterlm_domain)
# The GGUF fixture test drives the reference pipeline through the shared in-process harness.
target_include_directories(test_gguf PRIVATE ${PROJECT_SOURCE_DIR}/tests/domain)
