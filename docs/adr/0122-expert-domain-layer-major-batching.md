# 0122. Layer-major batching: one exchange per domain per layer per window

## Context

Sending one RPC per selected expert would cost K round trips per layer. The window has q positions and the router
output for a position depends only on that position's mixer result, never on another position's MoE output at the
same layer (recurrent state and KV are written from the mixer input `x`).

## Decision

Father executes a window layer by layer: all q mixers and routers for layer L, then ONE `ExpertBatch` to each
participating domain carrying the q activations and each position's routes for that domain, then its own experts and
the shared expert while waiting, then one `ExpertResult` per domain, then the combine. A domain with no selected
expert at that layer for any position is not contacted. There are therefore at most `domains x layers` request/response
pairs per window (about 48 sequential barriers per target pass), independent of K and q.

Positions with no route to a domain still travel in the batch (an empty route list). The cost is bytes, not messages;
dropping them would need a position index in the wire and is left out until the real routing statistics show it matters.

Domains are stateless, so there is no commit or abort traffic: an epoch field rejects stale batches, window ids must
not decrease, and Father keeps all sequence state (recurrent state, KV, history). This removes the layer-domain
design's `CommitWindow` broadcast, which is a real difference in the cost comparison.

## Consequences

- Bytes per layer grow linearly with q (activations both ways); the expert union per domain grows sub-linearly (see
  `expert_union_growth` in the bench result).
- A failed domain fails the window and breaks the executor; the caller rebuilds it. There is no partial retry because
  retrying a layer would need the domain to be idempotent per (epoch, window, layer), which it is not required to be.
