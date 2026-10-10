# 0402: Exact model ids never reroute; routing aliases are explicit

## Context
The spec requires automatic selection without silent substitution. Today fallback happens only in the Host chat with a visible
event naming both models (ADR 0171). An API invites clients that cannot see UI events.

## Decision
- A request for a profile's `api_model_id` runs that exact model or returns a structured error. No implicit fallback.
- A **routing alias** (`alias_*`, own `api_model_id`) lists candidate profiles with conditions; selection is deterministic and
  recorded in the ticket at enqueue; if the recorded candidate stops being viable before start the request is rejected.
- Every response states what ran: `x_clusterlm.{profile_id, api_model_id, model, quant, routed_via?}`.
- `on_worker_loss.then = fallback-profile` is a Host built-in-chat policy with a visible event; it never applies to API
  requests. In-flight API requests are never switched mid-stream.
- Aliases do not start prepare jobs unless `allow_prepare` is set; they route among prepared profiles by default.

## Consequences
Auto-selection rules in the spec ("Host only → A; Host + Worker A → B …") are expressed as aliases or as the built-in chat's
selected-profile rules (C). Clients wanting predictable output pin an exact id.
