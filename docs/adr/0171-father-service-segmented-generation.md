# 0171: Father service drives the Coordinator in short segments

## Context
`Coordinator::generate` returns only when the whole generation finishes: no streaming, no cancellation hook, and a
failure discards nothing but exposes no partial progress. The Father UI needs streamed tokens, responsive cancel,
and a conversation that survives a node loss. Coordinator/Node internals are out of scope for this work package.

## Decision
The service generates in segments of `ServiceOptions::segment_tokens` (default 4). Each segment is a fresh
Coordinator session whose prompt is the conversation tokens plus the tokens already emitted; the segment's tokens
are emitted as one TokensEvent and the cancel flag is checked between segments. Greedy decoding is deterministic,
so segmentation yields exactly the tokens of one long generation (tested against a Father-only reference).
A failed segment invalidates only that session; emitted tokens stay valid, so the same answer can continue on
another tier or after a retry by re-prefilling.

Fallback follows the catalog policy and never substitutes silently: retry the same tier up to `max_retries` only
while the tier is still observed viable, then downgrade to the first lower tier that is Ready or Available, else
stop with an error. Every retry/downgrade is a FallbackEvent naming both models; each TokensEvent and the final
stats name the model that produced them. The service owns the "plan ready" fact itself and invalidates it on any
failed segment.

## Consequences
Cancel latency is bounded by one segment. Each segment re-prefills, a cost of one prefill per segment that is
acceptable for a UI path and measured only on hardware. A future Coordinator streaming/cancel API (a small
callback in `generate`) would remove the re-prefill; the service API does not change.
