# ADR 0201: Local sub-batches of a Strata domain commit provisionally and can only be committed whole

Status: accepted (WP6)

## Context

`DomainSpec::max_local_batch` lets a stage execute a large transport chunk (a prefill chunk of e.g. 128 positions)
in local sub-batches. Strata's Verifier handles at most `kVerifyMaxT` = 8 positions per window and keeps the window
tentative in a way that does not chain: K/V and indexer keys are appended, but the GDN recurrent and conv state is
advanced only by `commit()` replaying stored rows. A second Verifier window started before the first is committed
would compute its GDN layers from the old state.

## Decision

`StrataDomain` runs a window of `q` positions as consecutive engine windows of at most
`local_batch = min(max_local_batch or max_window, 8)` positions. Every sub-batch except the last is committed to the
engine immediately (provisionally); the last stays tentative. `commit_window(accepted)`:

* `q <= local_batch` (one engine window): any accepted prefix 1..q, exactly as Strata's commit;
* split window: only `accepted == q` is honoured (it commits the last sub-batch). Anything else - a partial commit,
  an `abort_window`, a failure in a later sub-batch - cannot restore the committed state, so the session is
  invalidated (ledger + engine) and `kAborted` is returned; Father re-opens and re-prefills from its own history.

## Consequences

* Prefill chunks (always committed whole: `RunWindow::auto_commit`) pipeline at any transport size while the GPU
  never holds more than 8 positions of window state.
* Speculative windows must satisfy `q <= local_batch` to keep partial acceptance; with Strata that is `q <= 8`,
  already the Verifier limit. The domain never returns a silently wrong state: an impossible request invalidates.
* A later engine path (Strata's batched `Prefill`) could replace sub-batching for long prompts; the contract above
  stays the same.
