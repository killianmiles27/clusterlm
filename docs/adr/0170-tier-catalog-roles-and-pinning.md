# 0170: Tier catalog expresses roles, not machines, and refuses unpinned models

## Context
Fast, Strong and Ultra differ in model, backend and topology. The catalog ships before Father has downloaded any
artifact, so it cannot know manifest hashes, and the hardware cannot yet qualify context lengths or throughput.
Machine names in a shipped catalog would break as soon as a user's pairing differs.

## Decision
- The catalog (`fixtures/catalog/clusterlm-catalog.json`) holds exactly three tiers. Topology is a list of ROLES
  (`father`, `node:laptop-class`, `node:designated-3060`). A user/pairing-level `TierAssignment` binds roles to
  concrete machines; a machine may fill at most one role per tier. Ultra with a third node role, or any unknown
  role, fails validation (Node 3 is excluded from Ultra).
- Model identity carries `expected_root_hash = null` and `pin_status = "unpinned"` until Father has inspected the
  artifact. Readiness refuses an unpinned tier unless the user confirmed exactly the manifest found on disk; a
  pinned tier requires the hash to match. The Father service re-checks the real Coordinator manifest root before
  preparing.
- Backend is `llama-local` (Father-only Fast) or `strata-hybrid` (Strong, Ultra); `llama-local` cannot span nodes.
- Every context profile is `qualified: false`; performance targets (Ultra 20 tok/s median) are
  `pending_qualification`. Both are recorded by experiment HQ-TIER-01 / HQ-PERF-*.
- Parsing is bounded: 1 MiB cap, nesting depth cap before parse, non-throwing parse, strict field validation.

## Consequences
Shipping the catalog is safe before any download. Pinning is a deliberate later edit gated by HQ-TIER-01. Adding a
fourth role or tier needs a schema change, which is intended.
