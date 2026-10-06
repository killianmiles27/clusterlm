# ClusterLM architecture

ClusterLM runs one LLM across several Windows PCs on a wired LAN. **Father** is the user-facing coordinator,
the primary inference machine and the only permanent model store. **Nodes** are idle PCs that receive a
lease-scoped subset of model objects, execute complete layers on their own CPU, RAM, GPU and VRAM, and delete
everything when the lease ends.

This document describes the code as it exists. For the hardware experiments that still have to be run, see
[HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md). For the Strata and llama.cpp port analysis, see
[backends/strata-port.md](backends/strata-port.md) and [backends/llama-rpc.md](backends/llama-rpc.md).

## Pipeline

```
Father prefix (embedding, PLE, layers 0..a)  --activations-->  Node A (layers a..b)
      ^                                                             | direct peer link
      | CommitWindow(accepted) to every domain                      v
Father tail (layers c..L, head, sampling, MTP)  <--activations--  Node B (layers b..c)
```

- Each stage is an **execution domain** that owns complete layers: their weights, sequence state, scratch and
  expert residency. Nothing is exchanged per layer or per expert.
- Stages exchange a fixed **boundary ABI**, `kResidualHandoffF32V1`. It holds `hc·H + H + hc` FP32 values per
  position: the residual streams, the pending block output and the injections. That is 51,216 B per position for
  Flash-Next. The layout is explicit little-endian and position-major. See
  `runtime/domain/include/clusterlm/domain/boundary.hpp`.
- **Token IDs never leave Father.**
  - The prefix consumes tokens and owns the token-dependent PLE history.
  - The tail produces logits and does the sampling.
  - Middle stages have a token-free API (`run_window(request, StageActivations)`).
  - The protocol has no message type that can carry tokens, logits or candidates.
- Father hosts the prefix and the tail as two domains in one process, with disjoint layer state.

## Speculative windows

A round verifies `q` positions: the next token followed by `q−1` drafts from the MTP drafter on Father.

1. Father runs the prefix, sends `RunWindow` to the first Node, and receives the last Node's `StageResult`.
   Nodes forward directly to their peer, so Father does not relay.
2. Father runs the tail and decides the accepted prefix `n` by greedy verification.
3. Father sends `CommitWindow(n)` to every domain and waits for every `CommitAck` before the next window.

Every domain delegates admission to `windows::WindowLedger`, so the rules are identical in-process and remote:

- Epochs.
- Strictly increasing window IDs.
- At most one outstanding window.
- An exact base position and state version.
- An accepted length in `[1, q]`.
- Idempotent commit replay.
- Abort that invalidates the session.

A worker failure, a stale epoch or an unknown commit outcome aborts the distributed session and advances the
epoch. Father keeps the conversation and the caller decides whether to retry or downgrade. `RunWindow` is never
replayed blindly.

## Modules

| Module | Target | Responsibility |
|---|---|---|
| `runtime/common` | `clusterlm_common` | Status/Result, explicit LE codec, SHA-256, strong IDs, privacy-safe logging |
| `runtime/objects` | `clusterlm_objects` | `ModelGeometry`, `ModelManifest`, `ObjectResolver`, `CanonicalModelStore`, deterministic fixture models |
| `runtime/domain` | `clusterlm_domain_api`, `clusterlm_domain` | `StageBoundary`, `ExecutionDomain`, `BackendAdapter`, CPU reference backend, drafters |
| `runtime/windows` | `clusterlm_windows` | Speculative-window ledger: epochs, window IDs, state versions, commit/abort |
| `runtime/transport` | `clusterlm_transport` | Framed TCP, mutual TLS 1.3 with pinned device identities, network impairment and fault injection |
| `runtime/protocol` | `clusterlm_protocol` | Father/Node messages (§9 of the addendum), bounded decoding, per-channel `MessageStream` |
| `runtime/platform` | `clusterlm_platform` | File mapping, durable journal files, safe deletion, child processes, Windows adapters (activity, power, Job Objects, DXGI) |
| `runtime/backends` | optional | Strata and llama.cpp adapter skeletons (CUDA, behind CMake options) |
| `node/lease-store` | `clusterlm_lease_store` | Ephemeral lease object store, content-free journal, orphan recovery |
| `node/worker` | `clusterlm_node_worker` | Node service: lease state machine, provisioning, domain hosting, peer forwarding, release |
| `orchestrator/placement` | `clusterlm_placement` | Hardware profiles with provenance, cost model, placement search, Pareto frontier |
| `orchestrator/coordinator` | `clusterlm_coordinator` | Father orchestration: connect, prepare, generate, release |
| `bench` | `clusterlm-bench` | ClusterLM Bench: profiles, transport, local clusters, faults, placement, qualification registry |
| `apps` | `clusterlm-node`, `clusterlm-father` | Executables |

## Channels and transport

Each Father↔Node pair uses three authenticated connections:

- **control**: plan, commit/abort, release.
- **activation**: `RunWindow`/`StageResult`.
- **provision**: `ProvisionChunk`/`SealObject`.

Keeping them separate means bulk transfer never blocks cancellation or commits. Frames have a 24-byte header,
and the payload limit is checked before anything is allocated.

- In `kMutualTls` mode both sides present self-signed certificates. Each is pinned by its SHA-256 fingerprint,
  which is the device ID.
- A direct Node→Node link is trusted only for one lease, after Father sends `AuthorizePeer` to both ends.
- `kInsecureLoopbackOnly` refuses any non-loopback address. It exists for tests.

`transport::impair()` wraps any connection with simulated bandwidth, latency and jitter.

- Bandwidth can be shared through a `SimulatedLink`, which models Father's single NIC.
- `FaultInjector` rules close, stall, delay or corrupt traffic at the n-th frame of a given type.
- The `gige-simulated` preset (110 MB/s, 0.15 ms) is a simulation parameter only.

## Ephemeral provisioning

Father builds a **plan-scoped manifest** for each Node. It contains the geometry plus only that Node's layer
objects: dense weights, routed experts and the shared expert. It never includes the embedding, PLE lookup, head,
MTP drafter, tokenizer or unrelated layers.

- A routed expert is described by three source ranges inside stacked GGUF-style tensors.
- Objects are sent in bounded chunks. Each chunk carries its own SHA-256 digest, and each object is sealed
  against its whole-object digest.
- The Node's `LeaseStore` journals each lease and each file *before* creating it.
- On release, the Node unmaps views, closes handles and deletes everything under its app-owned root without
  following links. It verifies by census that zero bytes remain, then reports resource release and storage
  cleanup separately.
- On restart, orphan recovery runs before a new lease can begin.
- Lease generations increase strictly across restarts, so stale traffic stays stale.

Node availability follows the addendum §11: Busy → Available → Preparing → Ready ⇄ Inferencing → Releasing →
Busy/Available, or CleanupPending. Local activity wins without consulting Father.

## Placement

`orchestrator/placement` works only from `HardwareProfile`s and `ModelCostInputs`. Every quantity carries a
provenance: `Synthetic`, `Measured` or `Qualified`.

- The synthetic development profiles for the three target machines live in `fixtures/profiles/`, not in product
  logic.
- A plan's provenance is the weakest of its inputs.
- `require_qualified()` refuses any plan that used a synthetic or merely measured input.
- The search enumerates Father-only, one-Node and two-Node contiguous plans, in both node orders and at
  four-layer granularity. It applies memory admission, fills each domain's VRAM with experts by critical-path
  time saved per byte, and predicts decode, prefill and preparation time with lease amortization. The output is
  the feasible set, the rejected plans with reasons, a Pareto frontier and a recommendation.

## Development without the cluster

| Stage | How it is exercised now |
|---|---|
| Reference execution | `clusterlm_domain` CPU FP32 reference over generated fixture models (bitwise deterministic) |
| Split domains | Split-versus-unsplit tests: logits are bitwise identical |
| Separate processes | `clusterlm-node` processes on 127.0.0.1, driven by the Coordinator through the real protocol |
| Authenticated transport | Mutual TLS on localhost |
| 1GbE emulation | `--impair gige-simulated` and the other presets |
| Faults | `--crash-at <phase>`, fault rules, local-activity revocation, restart and orphan recovery |
| Physical Nodes | Change the endpoints. Real measurements replace synthetic profiles through ClusterLM Bench |

The Strata CUDA backend is the leading production candidate. It needs the hardware, so its adapter is a
skeleton that returns `kHardwareUnavailable`/`kUnimplemented`. It sits behind the same `BackendAdapter`
interface the reference backend uses.
