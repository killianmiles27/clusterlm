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
| Node service supervisor | `node/service`: policy (idle, lock, AC, power saver), cooperative revocation with 2 s deadline, Job Object termination + relaunch with orphan recovery |
| Windows service host, session helper, named-pipe IPC, power/session notifications, firewall rules, key/staging ACLs | Implemented against Win32/COM (SCM host, `\\.\pipe` IPC with DACL and peer checks, WTS/power notifications, `INetFwPolicy2`, owner-only DACLs); logic tested on Linux with mocks and Unix sockets, Win32 code type-checked with MinGW/MSVC only. Real behaviour is pending `HQ-WIN-01`..`HQ-WIN-04`. See `docs/windows-architecture.md` |
| Installer | Two WiX MSI packages, signing script and CI install smoke test: `docs/packaging.md` (CI job `windows-packaging`, not yet run on a runner; `HQ-INSTALL-01` pending) |
| Pairing, persistent settings, Father agent API, production providers | Implemented and tested on Linux (SPAKE2 pairing with TLS channel binding, versioned owner-only settings, JSON IPC, config-driven placement, live readiness; dev end-to-end on the fixture model). No real inference backend is built, so tiers are never Ready without `--dev-fixture-model`. Real LAN behaviour pending `HQ-PAIR-01`; see `docs/pairing.md`, `docs/father-ipc.md` |
| Node product glue (WP18) | Implemented and tested on Linux: Node settings read/saved through the service over the helper pipe (validated, persisted, applied; caps restart the worker), the worker's lease state and counts in `StatusReply` and in `NodeViewModel`, a rate-limited pairing-mode request behind the Node UI's Pair button, `caps.threads` passed to the worker (offer + `DomainSpec::cpu_threads`; the reference backend is single-threaded), Father-side unpair that notifies a reachable Node (`UnpairNotice`, accepted only from the pinned Father), Coordinator prepare progress (bytes, objects, phases) through the service, the agent API and the Father UI, and the installed-layout catalog default. Real Windows behaviour pending `HQ-UI-01`, `HQ-PAIR-01`, `HQ-WIN-02`. `start_with_system` is persisted only |
| Father tokenizer (WP19) | Implemented and verified on Linux: `GgufBpeTokenizer` (GGUF byte-level BPE, `pre` qwen2/qwen35, hand-written pre-tokenizer over a generated Unicode table) matches llama.cpp's own vocab tests exactly (qwen2 46/46, qwen35 50/50 cases; CI step in the gcc and llama jobs), native Qwen ChatML template with the `enable_thinking` toggle, stop tokens for the service, per-tier wiring (a tier whose tokenizer cannot be built is Unavailable with the reason), libFuzzer target. Not implemented: other pre-tokenizers, non-ChatML templates (no Jinja engine), tool/multimodal turns, llama.cpp's tokenizer for the Fast tier. See `docs/tokenizer.md` |
| Bench tool gaps (WP20) | Implemented and tested on Linux: `faults --only`, `cluster --contexts`, per-cycle/per-round RSS/commit/VRAM series, Father NIC byte counters, dynamically loaded NVML telemetry (`nvml`), power plan/governor in the profile, Father-read vs. transfer vs. Node hash/write split of provisioning (`PlanReady` extension), inter-token gap distribution, cache/temp census (`storage-census`), expert-domain LAN peer mode and Strata IQ kernel mode. Windows paths are compile-checked only; real NVML/NIC/plan values and the IQ kernels at target scale are pending the corresponding `HQ-*` entries (ADR 0340) |

## Known gaps / next engineering steps (no hardware required)

1. ~~Node service supervisor~~ — done: `node/service` (`NodeSupervisor`, `clusterlm-node-service`), tested with a real worker incl. forced termination at the deadline. Windows service host, session helper and IPC are now implemented (see `docs/windows-architecture.md`); real-Windows verification pending.
2. **Prefill chunk pipelining across stages.** Chunks currently flow sequentially, which is correct but not overlapped.
3. **Streamed provisioning reads on Father.** Objects are read whole before chunking. Bounded streaming reads are
   needed for very large dense objects.
4. **Grouped expert-domain prototype (P0-C)** for measured comparison against the stage design.
5. **Stochastic speculative acceptance.** Greedy verification only so far.
6. **A runtime-reported allocation ledger from the real backend** (`describe_requirements`) feeding placement
   admission.
7. ~~Windows key-file ACL~~ — done: the key is written owner-only (owner account + SYSTEM, protected DACL) via `platform::write_owner_only_file`; ADR 0133. Real ACL inspection pending `HQ-WIN-04`. Further checks: HQ-SEC-01, `docs/security/threat-model.md`.
8. **Product surfaces** (Father UI, Node UI and tray, installer, pairing UX): built and tested on Linux (WP13, WP14, WP18); exercising them on Windows hardware is `HQ-UI-01`, `HQ-PAIR-01`, `HQ-WIN-02`, `HQ-INSTALL-01`.

## Pending hardware qualification

Every item in [HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md) is pending, including:

- Fast, Strong and Ultra tok/s, and the 20+ tok/s Ultra target.
- Layer boundaries and expert residency.
- The CPU/GPU crossover.
- MTP acceptance.
- VRAM margins.
- Laptop thermals.
- Real network latency.
- Provisioning speed (the progress counters of WP18 only observe a transfer; they are not a measurement of it).
- Release latency under real drivers.
- Windows pinned-memory behaviour.
