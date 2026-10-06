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
  # The same on Strata's AVX-2 kernels (Ryzen 5 5600-class / Zen 3 has no AVX-512); a no-op split on AVX-2-only hosts.
  add_test(NAME test_strata_cpu_kernels_avx2 COMMAND test_strata_cpu_kernels)
  set_tests_properties(test_strata_cpu_kernels_avx2 PROPERTIES ENVIRONMENT "STRATA_NO_IQ512=1;STRATA_FORCE_AVX2=1")
  # Father-side conversion (Strata pack + GGUF -> ClusterLM model directory) on a synthetic model written with
  # Strata's own GGUF test fixture.
  clusterlm_test(test_strata_convert SOURCES test_strata_convert.cpp DEPS clusterlm_strata_cpu clusterlm_objects)
  target_link_libraries(test_strata_convert PRIVATE strata_artifact)
  target_include_directories(test_strata_convert SYSTEM PRIVATE ${CLUSTERLM_STRATA_SOURCE_DIR}/tests/core)
  # The tool runs: host probe and a short CPU kernel throughput pass.
  add_test(NAME clusterlm_strata_probe COMMAND clusterlm-strata probe)
  add_test(NAME clusterlm_strata_cpu_experts
           COMMAND clusterlm-strata cpu-experts --types iq3_s+iq4_nl --tokens 1,4 --reps 1)
endif()

if(CLUSTERLM_ENABLE_STRATA)
  # The CUDA backend: sizing from Strata's own arithmetic, hardware detection, prepare without a device. Cases that
  # need a GPU report a skip message and pass.
  clusterlm_test(test_strata_cuda SOURCES test_strata_cuda.cpp DEPS clusterlm_backend_strata)
  target_link_libraries(test_strata_cuda PRIVATE strata_engine CUDA::cudart)  # Strata's sizing arithmetic, cudart
endif()

# The Strata backend wired into Father, Node and Coordinator (fake engine through the guarded backend_factory seam,
# real Node workers over loopback), the backend factory, and refusal of unknown backends: every build, no GPU.
clusterlm_test(test_strata_wiring SOURCES test_strata_wiring.cpp
  DEPS clusterlm_coordinator clusterlm_node_worker clusterlm_strata_core clusterlm_backend_factory clusterlm_domain clusterlm_objects)
