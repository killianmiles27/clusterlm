clusterlm_test(test_strata_layout SOURCES test_strata_handoff.cpp DEPS clusterlm_strata_layout)

# StrataDomain state machine (fake engine), object mapping and Father-side conversion: every build.
clusterlm_test(test_strata_core SOURCES test_strata_objects.cpp test_strata_domain.cpp DEPS clusterlm_strata_core)

# fetch_upstream.py's patch application (idempotence, --check, refusal of foreign edits) on a synthetic upstream.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME test_fetch_upstream_patches COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/test_fetch_upstream.py)
endif()

if(CLUSTERLM_ENABLE_STRATA OR CLUSTERLM_ENABLE_STRATA_CPU)
  # The real checkout is exactly pin + third_party/patches/strata (and the ggml pin is clean).
  if(Python3_Interpreter_FOUND)
    add_test(NAME strata_checkout_is_pin_plus_patches
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/fetch_upstream.py --check --apply-patches strata strata-ggml)
  endif()
  # Strata's CPU expert kernels (IQ3_S / IQ2_XS gate-up, IQ4_NL / Q2_0 down) against ggml-cpu and a scalar reference.
  clusterlm_test(test_strata_cpu_kernels SOURCES test_strata_cpu_kernels.cpp DEPS clusterlm_strata_cpu)
endif()

if(CLUSTERLM_ENABLE_STRATA)
  # The CUDA backend: sizing from Strata's own arithmetic, hardware detection, prepare without a device. Cases that
  # need a GPU report a skip message and pass.
  clusterlm_test(test_strata_cuda SOURCES test_strata_cuda.cpp DEPS clusterlm_backend_strata)
endif()
