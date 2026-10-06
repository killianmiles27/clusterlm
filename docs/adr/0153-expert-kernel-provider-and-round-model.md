# 0153 Pluggable expert kernels and the round model of the CPU benchmark

## Context

HQ-CPU-01 needs effective routed-expert throughput (dequant + GEMV) for the exact representations, at q = 1, 2, 4,
with expert-union selection, thread scaling and sustained behaviour. The production CPU kernels (Strata IQ3_S/IQ2_XS,
AVX2/AVX-512) are built by another workstream and cannot be linked yet. The measurement logic must not depend on which
kernels sit underneath.

## Decision

- `ExpertKernelProvider` / `ExpertBank` (`bench/src/expert_kernels.hpp`): a provider builds a bank of experts for a
  representation and executes one expert over 1..q positions with a per-thread context, reporting dequant and GEMV
  time separately. Providers register by id in `ExpertKernelRegistry`.
  `register_external_expert_providers()` (`expert_providers_external.cpp`) is the single hook another workstream edits;
  until then `strata-cpu` resolves to a stub returning `kHardwareUnavailable` with the reason (exit code 3).
- The reference provider wraps the reference backend's math. This required one small public header in
  `runtime/domain`: `reference_kernels.hpp` (`reference_matvec`, `reference_expert_forward`,
  `reference_matmul_rows`) over the existing private `refmath` functions, compiled with the domain's no-FMA flags.
  `reference_matmul_rows` streams each weight row once for all positions of a window, which is the access pattern of
  verifying q positions through one expert; it is bitwise identical per element to q separate matvecs (unit-tested).
- A benchmark round is the union of the experts selected by q positions, spread over worker threads (separate experts
  per thread, caller included). Effective bytes/s counts each union expert's stored bytes once. Selection patterns:
  `same` (union = active, the basis of `q_scaling`), `overlap` (half shared with the previous position), `disjoint`.
- `q_scaling` is the mean of `(t_q/t_1 - 1)/(q-1)` over q > 1 on the `same` pattern at the best thread count, matching
  `HardwareProfile.cpu.q_scaling` ("time *= 1 + q_scaling*(q-1)"); weight traffic of the union is a separate placement term.
- Repetitions interleave across configurations; the working set defaults to 1 GiB so one round exceeds the last-level cache.

## Consequences

- Reference numbers are real measurements of scalar FP32 kernels, an order of magnitude below production kernels. They
  are recorded under provider id `reference` and HQ-CPU-01 stays pending until `strata-cpu` registers.
- Threads are not pinned to cores. On hybrid or SMT machines the sweep reports what the OS scheduler gives; pinning is
  a follow-up once the production kernels exist.
