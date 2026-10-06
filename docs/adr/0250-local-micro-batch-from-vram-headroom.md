# ADR 0250: Per-domain local micro-batch from VRAM headroom

**Status:** Accepted.

## Context

The pipeline moves prefill in chunks of `prefill_chunk` tokens, and every domain needs activation workspace
proportional to the tokens it processes at once. If one chunk size had to fit every GPU, the smallest GPU
(the 8 GB laptop) would force a small chunk on the whole cluster, multiplying boundary transfers and
pipeline bubbles for stages that have plenty of room.

## Decision

The chunk stays a cluster-wide pipeline quantity. Each domain runs it as `ceil(chunk / local_batch)`
micro-batches, where `local_batch` is chosen per (domain, layer ranges) from that domain's own VRAM headroom:

- `headroom = vram_budget - (dense + state + scratch)`; at most `batch_headroom_fraction` (default 25%) of it
  may be reserved for workspace: `batch * batch_scratch_bytes_per_token`.
- `local_batch` is the requested chunk if it fits, else the largest power of two that does. A result below
  `max(min_local_batch, q)` makes the domain inadmissible, with a reason.
- The workspace is reserved BEFORE the GPU expert fill (it is in `MemoryLedger::batch_workspace` and
  `vram_used()`), so the ledger invariant `vram_used <= vram_budget` holds and the greedy fill still leaves
  less than one expert of slack.
- Cost: each micro-batch streams the CPU-resident experts its tokens touch, so a smaller local batch raises
  that stage's prefill chunk time (not decode time, which is a q-position union).

`batch_scratch_bytes_per_token` is a model-side input, Synthetic in the planning estimate (HQ-PLACE-02); 0 means
"not modelled" and never constrains a domain.

## Consequences

A tight GPU degrades only its own stage's prefill. The 25% cap is a policy knob, not a measurement; HQ-PLACE-01
decides it. Plan hashes include the batch (domain digest), so plans differing only by batch are distinct.
