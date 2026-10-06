# Development status (cloud phase)

This document maps the cloud-development completion definition to evidence in the repository. Each "Evidence"
entry is a test or command that runs in CI or on a development host. Nothing here is a measurement of the
Father, G14 or 3060 machines.

## Cloud completion definition

| Criterion | State | Evidence |
|---|---|---|
| Builds reproducibly | Done (Linux); Windows via CI | `cmake --build`; `.github/workflows/ci.yml`; `scripts/check_windows_compile.sh` (MinGW proxy for MSVC) |
| Father + multiple Node processes form a local cluster | Done | `ctest -R integration_cluster`; `clusterlm-bench cluster` |
| Execution domains have independent ownership | Done (reference backend) | Partial domains load only their own objects (`test_domain`); Nodes hold only plan-assigned objects (`provisioned_only_assigned_layers` check) |
| Stage boundary serialized and transported | Done | `kResidualHandoffF32V1` codec; split-vs-unsplit bitwise equality (`test_domain`); bounded activation messages (`activation_messages_bounded_by_boundary_abi`) |
| Selected objects provisioned without a full model copy | Done | Plan-scoped manifests; chunk and object digests; expert = 3 source ranges (`test_objects`) |
| Node state ephemeral and cleanup proven | Done (Linux) | Lease journal, orphan recovery, crash at 8 phases (`test_lease_store`); `clusterlm-bench faults` census = 0 after every scenario |
| Failure, cancellation and epoch handling exercised | Done | `WindowLedger` rules (`test_windows`); crash, link loss, stall, local activity and stale plan (`clusterlm-bench faults`) |
| Synthetic profiles drive the placement scheduler | Done | `fixtures/profiles/*` (all fields `Synthetic`); `test_placement` scenario directions; `test_planning` (placement → executable plan → execution) |
| Network constraints emulated | Done | `transport::impair` presets, shared `SimulatedLink`, `FaultInjector`; `integration_cluster_relay_impaired` |
| Bench emits machine-readable qualification data | Done | `bench/schema/benchmark-result.schema.json`; provenance enforced (`Synthetic` for any simulated input) |
| Remaining hardware questions are executable tests | Done | `bench/qualification/experiments.json` → `HARDWARE-QUALIFICATION.md` (CI checks they stay in sync) |

## Verified properties

- **Correctness of distribution.** The tokens are identical across all of these runs, for every acceptance length:
  - Father-only reference execution.
  - Four-domain split (prefix → Node A → Node B → tail).
  - Direct peer forwarding and Father relay.
  - Mutual TLS and the simulated 1GbE preset.
  - Speculative windows q = 1…4, with every rejection position.
- **Token-free Nodes.** The middle-stage API takes only `StageActivations`, and no protocol message can carry token
  IDs, logits or candidates. Bench checks that every activation message is bounded by the boundary ABI.
- **Ephemeral storage.**
  - Crashes during transfer, hashing, mapping, allocation, Ready, inference and cleanup all recover to 0 staged
    bytes on restart, and a fresh lease then works.
  - Local activity releases the lease without consulting Father, and Father refuses to run the stale plan.
- **Sanitizers.** The full suite, including multi-process clusters and fault runs, is clean under ASan+UBSan and
  ThreadSanitizer on the development host.

## Implemented as interfaces, pending real platforms

| Item | State |
|---|---|
| Strata CUDA backend | `runtime/backends/strata`: adapter skeleton behind `CLUSTERLM_ENABLE_STRATA`; returns `kHardwareUnavailable`/`kUnimplemented`. The handoff transpose is implemented and tested (`test_strata_layout`). The port plan is in `docs/backends/strata-port.md` |
| llama.cpp RPC baseline | `runtime/backends/llama` skeleton; analysis in `docs/backends/llama-rpc.md` (P0-A) |
| Windows adapters | Real Win32 implementations for activity, power, Job Objects, DXGI budget and file mapping, plus mocks for tests. They compile only on Windows |
| Node service supervisor, tray, installer, firewall rules, session helper | Not yet implemented. These are product surfaces after the runtime lifecycle; see below |

## Known gaps / next engineering steps (no hardware required)

1. **Node service supervisor.** Run `clusterlm-node` as a worker inside a Job Object and enforce the 2 s cancellation
   deadline by terminating the job. Route session-helper activity to the worker.
2. **Prefill chunk pipelining across stages.** Chunks currently flow sequentially, which is correct but not overlapped.
3. **Streamed provisioning reads on Father.** Objects are read whole before chunking. Bounded streaming reads are
   needed for very large dense objects.
4. **Grouped expert-domain prototype (P0-C)** for measured comparison against the stage design.
5. **Stochastic speculative acceptance.** Greedy verification only so far.
6. **A runtime-reported allocation ledger from the real backend** (`describe_requirements`) feeding placement
   admission.
7. **Product surfaces** (Father UI, tray, installer, pairing UX) after the lifecycle above is exercised on hardware.

## Pending hardware qualification

Every item in [HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md) is pending, including:

- Fast, Strong and Ultra tok/s, and the 20+ tok/s Ultra target.
- Layer boundaries and expert residency.
- The CPU/GPU crossover.
- MTP acceptance.
- VRAM margins.
- Laptop thermals.
- Real network latency.
- Provisioning speed.
- Release latency under real drivers.
- Windows pinned-memory behaviour.
