# 0152 The GPU probe is plain C++ over cudart and cuBLAS, behind an interface

## Context

The bench needs device enumeration, VRAM, allocation overhead, pageable and pinned transfer bandwidth, a pinned limit
and cuBLAS GEMV/GEMM timings. The development host has the CUDA 12.0 toolkit (nvcc needs `g++-12` as host compiler)
but no GPU. The probe must compile everywhere the toolkit exists, never crash without a device, and its logic must be
testable without hardware.

## Decision

- All GPU measurement sits behind `GpuProbe` (`bench/src/gpu_probe.hpp`). The CUDA implementation
  (`gpu_probe_cuda.cpp`) is ordinary C++ using the runtime API and cuBLAS: no device code, so no `enable_language(CUDA)`,
  no nvcc and no host-compiler pinning. It is compiled only with `-DCLUSTERLM_BENCH_CUDA=ON` and a found
  `CUDAToolkit`; the option is OFF by default so CI and CUDA-less hosts build unchanged.
- Without CUDA, or with CUDA but no device, `make_gpu_probe()` yields a probe whose every call returns
  `kHardwareUnavailable` with the reason. `gpu` / `pcie` then exit 3 and list the pending HQ items.
- Orchestration (`run_gpu_bench`), metric emission and profile mapping are probe-independent and tested with a fake
  probe. The CUDA path itself is only exercised on a machine with a GPU; it is compile-checked here.
- Backend-specific kernels (Strata) register `GpuKernelCase`s in `GpuKernelRegistry`; the probe times them with
  device events on its own stream.
- DXGI budget comes from `platform::GpuBudgetProbe` (Windows only) and is preferred for `vram_budget`.

## Consequences

- The runtime behaviour of `gpu_probe_cuda.cpp` is unverified until the first run on a GPU machine. Every transfer,
  pinned and GEMM sub-measurement that fails is reported as a note in the result, never silently dropped.
- cuBLAS shapes bound what a GPU can do; they are not the Strata layer kernels. `dense_layer_ms`,
  `gpu_expert_bytes_per_s` and `prefill_tokens_per_s` therefore stay Synthetic in profiles (HQ-GPU-01).
