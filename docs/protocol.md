# ClusterLM Father/Node protocol

The authoritative definitions are `runtime/protocol/include/clusterlm/protocol/messages.hpp` (messages and codec
limits) and `runtime/transport/include/clusterlm/transport/transport.hpp` (framing). This document explains the
rules that those types do not state on their own. Version: `protocol::kProtocolVersion = 1`.

## Transport and channels

Every Father↔Node pair uses three authenticated connections. Each starts with `Hello`/`HelloAck`.

| Channel | Messages | Why separate |
|---|---|---|
| control | `OfferResources`, `PreparePlan`/`PlanAccepted`, `PlanReady`, `AuthorizePeer`, `OpenSession`/`SessionOpened`, `CommitWindow`/`CommitAck`, `AbortWindow`/`WindowAborted`, `AbortSession`, `ReleaseLease`/`ReleaseComplete`, `UnpairNotice` (Father to Node, answered by `Pong`; docs/pairing.md), `Ping`/`Pong`, `Error` | Latency-critical; must never queue behind bulk data |
| activation | `RunWindow` → `StageResult` | Bounded activation payloads |
| provision | `ProvisionStatus`, `ProvisionChunk`, `SealObject` → `ObjectSealed` | Bulk, lowest priority; opened only while preparing |

- **Frames.** Each frame has a 24-byte little-endian header: magic `CLMF`, version, type, channel, flags, a u64
  correlation id, and a u32 payload length.
  - The payload limit is checked before any allocation.
  - A message on the wrong channel is a protocol error, and the connection is closed.
- **Security.** Both sides use mutually authenticated TLS 1.3 with no session tickets. Each peer's certificate is
  pinned by its SHA-256 fingerprint, which is its device id from pairing.
  - A direct Node→Node activation channel is trusted only for one lease, after Father sends `AuthorizePeer` to both
    ends.
  - The Node revokes that trust on release.
  - `kInsecureLoopbackOnly` refuses non-loopback addresses and exists for tests.
- **Correlation.** Requests carry a non-zero correlation id and replies echo it. Correlation 0 marks unsolicited
  messages: `OfferResources`, `PlanReady`, a self-initiated `ReleaseComplete`, and errors not tied to a request.

## Privacy invariants

These are enforced by type design, Node admission checks and tests.

- No message carries vocabulary token ids, decoded text, logits, sampling state, candidate tokens, chat roles or
  prompts.
- `RunWindow` and `StageResult` carry only identities, positions, dimensions, timing counters and boundary
  activations (`kResidualHandoffF32V1`).
- Nodes accept only token-free middle stages. `PreparePlan` with a prefix or tail stage is refused with
  `PERMISSION_DENIED` before any transfer.
- Plan-scoped manifests never contain Father-only objects (embedding, PLE/lookup, output head, MTP). The Node also
  rejects such assignments.

## Identities and staleness

| Identity | Owner | Rule |
|---|---|---|
| `LeaseGeneration` | Node | Strictly increasing across restarts (the lease journal persists the highest). Every lease-scoped message names it; any other value is `STALE_EPOCH`. |
| `Epoch` | Father | Advanced whenever a distributed session is established or invalidated. Window/commit/abort messages naming another epoch are `STALE_EPOCH`. |
| `SessionId` | Father | One conversation's sequence state in every domain. |
| `WindowId` | Father | Strictly increasing per session. A `RunWindow` for an old or repeated id is refused, never replayed. |
| `StateVersion` | Domain | Incremented by every commit. `RunWindow` and `CommitWindow` must name the current version. |

## Lifecycle

```
Node: Busy ──idle policy──▶ Available ──PreparePlan──▶ Preparing ──all objects sealed + check──▶ Ready
Ready ⇄ Inferencing (per window)   Ready/Preparing/Inferencing ──activity/pause/fault/Father loss──▶ Releasing
Releasing ──storage census 0──▶ Busy|Available      Releasing ──deletion blocked──▶ CleanupPending
```

1. **Offer.** On control connect, an Available Node sends `OfferResources`: its lease generation and safe
   RAM/VRAM/disk under its live policy.
2. **Prepare.** `PreparePlan` carries the following:
   - lease, model root hash, backend build and plan hash;
   - the stage assignments and caps;
   - a plan-scoped manifest with per-object targets (CPU-resident, GPU-resident or temporary backing).

   The Node checks the backend build, roles, Father-only objects and budgets against its own allowances. It then
   journals the lease and reserves objects, and replies `PlanAccepted`.
3. **Provision.** Father opens the provision channel, and the Node first sends `ProvisionStatus` listing the objects
   already sealed under this lease.
   - Father streams `ProvisionChunk`s. Each chunk is digested, and the source is read with bounded reads.
   - Father then sends `SealObject` for each object, and the Node verifies the whole-object digest.
   - If the bulk connection breaks while the lease is valid, Father reconnects and resumes. Sealed objects are kept,
     and partial objects are reset and resent whole.
   - Once every object is sealed, the Node builds its domains, runs a synthetic execution check and sends
     `PlanReady`.
4. **Peers.** In direct mode Father sends `AuthorizePeer` to both ends of every Node→Node hop.
5. **Sessions.** `OpenSession(epoch, session)` opens a session on every domain. A conversation keeps its session,
   with KV, recurrent and PLE state, between turns.
6. **Windows** are described in the next section.
7. **Release.** A release is triggered in one of three ways:
   - `ReleaseLease` from Father;
   - Node-local activity or a pause, without consulting Father, followed by an unsolicited `ReleaseComplete`;
   - Father loss, when the control connection drops.

   The Node stops new work and closes the data channels. It waits for the cooperative safe point, releases the
   domains, and deletes every lease-owned file and object. It then reports `resources_released` and
   `storage_cleaned` separately, with residual bytes. The next lease uses a new generation.

## Windows (speculative and prefill)

A window is `q` consecutive positions starting at the session's committed position.

- **Prefill.** `RunWindow.auto_commit = true`, and every stage commits all positions as soon as it computes them.
  - Father keeps up to `prefill_inflight` chunks in the pipeline. Successive chunks overlap across stages, and each
    domain processes its chunks strictly in window order.
  - A transport chunk may exceed a stage's `max_local_batch`; that stage executes it in local sub-batches.
- **Decode / verification.** The window is `[next_token, d1 … d_{q−1}]`, where the drafts come from Father's MTP
  drafter.
  1. Father decides how many positions to accept, between 1 and q, by greedy prefix match or by exact speculative
     sampling.
  2. Father sends `CommitWindow(accepted)` to every Node in parallel, commits its local domains, and waits for every
     `CommitAck`.
  3. All acks must agree on the committed position and state version.
- **Direct routing.** `forward_to_peer` makes each Node send its output straight to the next Node's activation
  channel. The last Node returns the `StageResult` to Father, carrying the timings of every traversed stage.
- **Abort.** `AbortWindow` discards the outstanding window and keeps the committed state. It is used for
  cancellation; repeating it is harmless, and naming a window that was never admitted is refused.
- **Duplicate commits.** Re-sending the identical `CommitWindow` replays the identical ack. A different request for
  the same window is refused.
- **Lost acknowledgements.** Father waits for the reply to either the original or the resent commit, backing off ×2
  on each retry.
- **Failure.** If any stage fails, a deadline expires, an epoch is stale, or a commit outcome stays unknown, the
  distributed session is invalidated: `AbortSession` goes everywhere and the epoch advances. Father keeps the
  conversation and the caller retries or downgrades.

## Decode limits (bounded before allocation)

| Field | Default limit |
|---|---|
| Frame payload | 64 MiB (`SecurityConfig::max_payload`) |
| Activation positions per message | 4096 (`DecodeLimits::max_window_positions`) |
| Provision chunk | 8 MiB |
| Manifest objects / assignments / sealed list | 1,048,576 |
| Strings | 64 KiB |
| Stages per plan / timing entries | 64 |
