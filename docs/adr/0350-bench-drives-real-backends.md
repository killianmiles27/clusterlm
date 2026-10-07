# 0350: The bench drives real backends and real models

Status: accepted (WP21)

## Context

Most runtime qualification entries carried the gap "`clusterlm-bench cluster` runs the reference backend on the fixture
model". The pieces to close it exist (backend factory, `--backend` on `clusterlm-father`/`clusterlm-node`,
`Coordinator::make_drafter`, the Father tokenizer, the bench observers), but nothing here has a GPU, a CUDA build or a
model artifact, so what can be built and tested is the wiring, the provenance rules and every measurement that does not
need the hardware.

## Decision

- **One flag grammar.** `apps/common/backend_cli.hpp` parses `--backend` and the Strata/llama options for Father and for the
  bench, and builds the flags forwarded to every spawned Node (engine options only; the PLE table, MTP directory and llama
  never reach a Node). A backend that is not built is refused before anything is spawned (exit 3, CMake option named).
- **`--model DIR` goes to Father only.** A Node never receives a model path: it is handed plan-assigned objects, which is the
  product's privacy and storage rule. References run before the cluster starts (one model in device memory at a time) and
  `--no-reference` exists for models that do not fit on Father alone.
- **Provenance is computed from what is real.** `Measured` requires a real model, a real backend, `--on-target`, the real
  drafter, no `--impair`, no folded token ids and no loopback Node. Nodes the bench spawns are localhost processes and stay
  Synthetic; `--node` / `--endpoint` attach to Nodes already running on other machines (real links), and a Father-only plan
  has no links. Every simulated input is named under `simulated`; `finish()` overrides any attempt to say `Measured`.
- **The Strata fake engine is reachable from the bench only in-process.** A spawned `clusterlm-node` builds its backend
  through the factory and has no test seam (and `strata` is not built without `CLUSTERLM_ENABLE_STRATA`, the CUDA option;
  `CLUSTERLM_ENABLE_STRATA_CPU` adds only the kernels). The bench's pass therefore lives in the harness library
  (`cluster_pass.cpp`) and `tests/bench/test_real_model_harness.cpp` runs it over `StrataDomain` with the fake engine on a
  Father-only plan; Node stages through Strata need a CUDA host (HQ-NUM-01, HQ-MTP-02).
- **`ReleaseComplete` carries per-domain allocation sizes** (trailing optional list: stage id, state-bytes peak, window-bytes
  peak) read from `DomainMetrics` after every OpenSession and at release; Father's own domains are read the same way. Reference
  domains report their real buffers, StrataDomain reports Strata's `session_bytes` and `Verifier::init_bytes`.
- **Supervised faults use the real service process.** `faults --supervised` runs Node 0 under `clusterlm-node-service
  --console --simulate-activity` and counts its events. A hung worker is a forced termination (the service relaunches it as part
  of the same event); a worker that exits on its own is `worker_restarted`. The Job Object is Windows-only.

## Consequences

Every closed gap has code, tests and a registry entry naming the real flags. What stays pending needs the hardware or the
artifact: real throughput, real acceptance with the real MTP head, Strata kernels through spawned Nodes, a real NIC, the
Windows Job Object. Those results will be `Measured` only if every input above is real; nothing is promoted to `Qualified`.
