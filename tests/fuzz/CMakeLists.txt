# libFuzzer targets. Only built with -DCLUSTERLM_FUZZ=ON and a clang toolchain (-fsanitize=fuzzer,address).
# Other work packages add their targets here. Each fuzz target compiles the sources it exercises directly so they are
# coverage- and sanitizer-instrumented; the rest of the tree links normally.
option(CLUSTERLM_FUZZ "Build libFuzzer targets (requires clang)" OFF)

if(CLUSTERLM_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "CLUSTERLM_FUZZ=ON requires clang (configure with -DCMAKE_CXX_COMPILER=clang++)")
  endif()
  set(CLM_FUZZ_COMPILE -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  set(CLM_FUZZ_LINK -fsanitize=fuzzer,address,undefined)

  # fuzz_gguf: GGUF header parser + manifest builder (WP1).
  set(OBJ ${PROJECT_SOURCE_DIR}/runtime/objects)
  add_executable(fuzz_gguf fuzz_gguf.cpp
    ${OBJ}/src/gguf.cpp ${OBJ}/src/ggml_types.cpp ${OBJ}/src/gguf_manifest.cpp ${OBJ}/src/manifest.cpp ${OBJ}/src/geometry.cpp)
  target_include_directories(fuzz_gguf PRIVATE ${OBJ}/include)
  target_link_libraries(fuzz_gguf PRIVATE clusterlm_common clusterlm_third_party Threads::Threads)
  target_compile_options(fuzz_gguf PRIVATE ${CLM_FUZZ_COMPILE} -g -O1)
  target_link_options(fuzz_gguf PRIVATE ${CLM_FUZZ_LINK})
  clusterlm_set_warnings(fuzz_gguf)

  # Seed corpus generator (plain build; writes header-only GGUF files from the writer).
  add_executable(gen_gguf_seeds gen_gguf_seeds.cpp)
  target_link_libraries(gen_gguf_seeds PRIVATE clusterlm_objects)
  clusterlm_set_warnings(gen_gguf_seeds)
endif()
