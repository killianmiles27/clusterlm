# ADR 0171: Father service generation (segmented sessions — superseded)

**Status:** Superseded by conversation-based generation (this revision).

## Context

The first Father service implementation could not stream, cancel mid-answer, or keep sequence state between turns,
because `Coordinator::generate` ran one one-shot session per call. It generated in short segments, each a fresh
session that re-prefilled the whole history.

## Decision

The Coordinator now provides `Conversation`s: a distributed session whose KV, recurrent and PLE state stays
allocated in every domain between turns, plus `GenerationRequest::on_tokens` (streaming) and
`GenerationRequest::cancel` (cooperative cancellation; drains prefill, aborts an outstanding decode window with
`AbortWindow`). The Father service keeps one conversation per active tier:

- each answer is one `generate` call over the conversation, feeding only the tokens the conversation does not hold;
- tokens stream to the UI as each decode round emits them;
- cancel reaches the in-flight window directly;
- if the rendered history is not an extension of what the conversation holds (edited history, reset), or after any
  session loss / tier change, a new conversation is opened and the full history is prefilled.

## Consequences

No per-segment re-prefill; follow-up turns prefill only their new tokens; cancel latency is one window instead of
one segment. Fallback still re-prefills the full history on the new tier (its domains hold no state for it).
