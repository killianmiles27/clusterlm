# ADR 0202: Abort a Strata window without ending the session (Verifier::abort_window)

Status: accepted (WP6)

## Context

The ExecutionDomain contract requires `abort_window`: drop the outstanding window, keep the committed session.
Pinned Strata has no such path - a verify window always keeps at least token 0, so `commit(n_keep)` requires
`n_keep >= 1` (strata-port.md finding 3). Before this decision the adapter had to map every window abort to
`abort_session`, discarding the conversation's state on all domains.

## Decision

Strata patch 0003 adds `Verifier::abort_window()`. It restores exactly what a window mutates:

* the QSA indexer key tail, from the window's own pre-window snapshot (`tail_snap_`, the snapshot commit already
  restores before re-appending kept keys - abort re-appends none);
* the PLE history (the stage holding layer 1), from a new pre-window snapshot captured in the window graph;
* nothing else: GDN state is untouched by a window, `ple_prev` advances only at commit, and K/V cells past the
  committed position are rewritten before any query reads them (the rule partial commits rely on).

The one-token window's self-commit (it advances GDN state inside the window graph) would make that window
un-abortable, so the patch makes it a per-verifier switch (`set_self_commit`) that ClusterLM turns off; one-token
windows then commit through the commit graph like every other window. A window that was already committed or
aborted makes `abort_window` a no-op, matching the ledger's idempotence.

## Consequences

* A stale epoch after a partial window, a failed downstream stage or a cancelled round no longer costs the
  conversation; only the outstanding window is dropped.
* One-token windows pay one commit-graph launch they previously skipped (decode with q = 1; Father normally runs
  q > 1 with MTP).
* Correctness on the GPU is qualification item HQ-GPU-05 (state after abort equals a never-run window, bit for bit).
