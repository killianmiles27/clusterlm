# 0271 — Versioned, owner-only, self-recovering settings documents

## Context

Pairing results, tier assignments and Node policy must survive restarts, be readable only by the owning
account (they name machines and model locations), never crash the product when damaged, and survive upgrades.

## Decision

`orchestrator/config` provides two JSON documents (`father-settings.json`, `node-settings.json`) behind
`SettingsStore<T>`:

* Writes: owner-only temp file → flush → rename over the target → directory sync. Readers never see a mixture.
* Loads are bounded (1 MiB) and strictly validated. Unknown fields are ignored (forward compatible within a
  version); wrong types or out-of-range values fail validation.
* A document that fails parsing or validation is **renamed** to `<file>.corrupt-<stamp>` and defaults are used.
  Nothing is deleted. A newer-than-supported `version` is refused outright and left untouched.
* Older versions go through an explicit migration hook (one version per call); the original is kept as
  `<file>.v<N>.bak`. Version 1 is current, so no hook is registered yet.
* Permissions: a document readable by others is tightened on load and the load report says so.
* Role → machine assignments are global (catalog `TierAssignment` semantics), not per tier; a tier uses the
  subset of roles it lists.

## Consequences

Recovery is silent for the user but visible in `LoadReport` (surfaced through diagnostics). Settings never
hold secrets, prompts or tokens. The worker does not yet enforce `caps.threads`; it is stored for when it does.
