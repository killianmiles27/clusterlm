# Execution profile schema v1

Frozen v1.1: ADR 0400 (profiles), 0402 (routing vs substitution), revised by ADR 0407. Schema: [profile-v1](schemas/profile-v1.schema.json),
[routing-alias-v1](schemas/routing-alias-v1.schema.json). Examples: [examples/](examples/).

An **execution profile** says *what* to run (exact model + quantization), *how* (backend, context, speculation), *where*
(topology slots that Workers may fill), *when it is usable* (readiness over its slots) and *what to do when things change*
(lifecycle, Worker loss, exposure). It replaces the hard-coded tiers `fast/strong/ultra` of `orchestrator/catalog`.
It never contains measurements, qualification claims, pairing identities (on export), credentials, or host paths.

## 1. Objects

| Object | Id | Persisted in | Notes |
|---|---|---|---|
| Profile | `prof_*` | Host settings v2 (`profiles[]`) | stable id; `revision` increments on every edit |
| Routing alias | `alias_*` | Host settings v2 (`aliases[]`) | explicit candidate list; see §6 |
| Machine binding | name (string) | Host settings v2 (`machine_bindings`) | name → device fingerprint; supersedes `assignments` (§5) |
| Model record | `mdl_<24 hex>` | Host library (workstream A, `orchestrator/library`) | exact identity of an artifact on the Host; profiles reference it by identity |

### 1a. Document rules (all contract objects)

* **Unknown fields are rejected** at every nesting level, on import and on local load (strict, typo-safe; a security-relevant
  field such as `exposure` can never be silently ignored). Compatibility for later optional additions comes from the writer
  side: a writer **omits every optional field that equals its default**, so a document that does not use a newer feature stays
  readable by an older build, and one that does is refused with "field X needs a newer ClusterLM" rather than misread.
* A profile or alias that fails validation on load is **quarantined individually**: kept verbatim in settings (never deleted,
  never rewritten), listed with state `unavailable` and blocker `invalid_document`, and excluded from the API. It never causes
  the whole settings file to be treated as corrupt (that would drop pairing records).
* `revision` increments on every accepted edit; the scheduler pins `id@revision` per ticket (scheduler-admission-v1 §4).

## 2. Topology: slots, not roles

`topology.slots[]` is the pipeline in stage order. Slot 0 is the single `host` slot (the Host always owns the prefix and tail
domains, per the ExecutionDomain contract). Each `worker` slot has a **selector**:

| Selector | Meaning | Exported? |
|---|---|---|
| `binding` | named binding resolved through the Host's `machine_bindings` (the old `assignments` keys such as `node:laptop-class` survive as binding names) | yes, as a label; recipient must bind it |
| `machine` | one specific paired Worker by device fingerprint | **removed** (becomes `binding` named after the Worker's display name, or an unbound slot) |
| `requirements` | any paired Worker meeting resource requirements (vendor, VRAM, RAM, compute capability, link speed, CPU-only OK) | yes |

`min_workers`/`max_workers` bound how many worker slots must/may be filled. Defaults: `min_workers` = number of non-optional
worker slots, `max_workers` = number of worker slots; always `non-optional ≤ min_workers ≤ max_workers ≤ worker slots ≤ 8`.
A slot marked `optional` may be left empty when the count stays within bounds (a lower-capacity re-plan, never a different
model). A Worker that is Busy, on battery when policy forbids, offline, paused or releasing does **not** satisfy a selector
("a busy machine is not available compute"). **One machine fills at most one slot of a profile** (as `TierAssignment` enforces
today): `requirements` slots are resolved after `binding`/`machine` slots, in slot order, over machines not already chosen,
picking deterministically (highest free VRAM, then fingerprint order). A `binding` whose name has no entry in
`machine_bindings` leaves the slot unsatisfied with reason "Worker slot <label> is not assigned" (never auto-bound).
Layer numbering is never exposed in the UI; `placement.manual` is an advanced override honoured only if the backend
descriptor sets `manual_layer_ranges` and the same admission checks pass.

## 3. Validation (three levels, all must pass before "ready" can ever be reported)

1. **Schema + semantic (offline, no hardware).** The JSON Schema plus: exactly one host slot at index 0; unique slot names;
   the worker-count rule of §2; no two slots naming the same binding or machine; `default_tokens ≤ max_tokens`;
   `offered_profiles` ascending with `max_tokens` as its largest entry; `fallback_profile_id` resolves, is not the profile
   itself and the chain is acyclic; `api_model_id` unique across profiles and aliases (also for profiles with `api: false`),
   never starting with the reserved prefix `clusterlm-`, and `auto`/`default` only on aliases; `prepare_when_available` not
   with `preparation: manual`; manual stages name existing slots and are contiguous from layer 0; limits on array sizes as in
   the schema. `validate_examples.py` implements this level for the examples.
2. **Compatibility (Host library + backend descriptor).** Model identity resolves to exactly one library record
   (`ModelLibrary::resolve`): a pinned identity only to a record whose verified `root_hash` or user `pinned_root` equals it; an
   unpinned one by family + quant and then needs the user's confirmation; several candidates are `ambiguous` (never "first
   wins"). `model.library_id`, when present, selects among candidates but must itself pass the same identity/pin check — it
   never bypasses it, and a stale id is a finding, not a fallback. Then `check_model` (backend-capability-v1 §3) over the
   record's full `tensor_types` and `architecture`; `topology.max_workers ≤ CompatReport.max_workers` (0 for a model that is
   not distributable); worker slots require `execution.cross_machine`; `speculation`: effective q ≤ descriptor `max_q`, q > 1
   needs `window_commit_abort`; `backend.options` match the descriptor's `options_schema`; a `development-only` backend is
   refused for product profiles; `backend.id` is known to this build (an unknown id is a readiness blocker, not a parse error);
   the backend is built into this binary and its runtime is present (`BackendInfo.hardware_available` is reported separately,
   never hidden). An `Experimental` label additionally needs the user's per-model opt-in, stored on the library record (A),
   and restricts the profile to Host-only.
   Failures are returned **before** any provisioning (spec: "reject incompatible model/backend pairs before expensive
   provisioning"), each with a stable machine code and a human sentence.
3. **Dry-run placement (needs Worker capability data).** `profiles.dry_run(profile_id, {context_tokens, bindings?})` runs
   the existing `placement::search_placements` over the current Host and the candidate Workers and returns
   `DryRunReport{feasible, per_machine_bytes{cpu,gpu,state,scratch,staging}, preparation_bytes, bottlenecks[], plan_summary,
   quantities[]}`; every quantity is a `Quantity{value, provenance, source}`; with no benchmark behind it the provenance is
   `synthetic` and the UI/API must say so. A dry run **never** reports a throughput as a benchmark. It never reaches Workers
   beyond reading their last advertised capability descriptor.

## 4. Fields that deliberately do not exist

* `qualified`, `compatibility`, `tokens_per_second`, `ready` — computed, evidence-backed, never stored in a profile.
  (`goals[]` carries the old catalog's `performance_targets` as non-binding, `pending_qualification` data;
  `qualification_experiments[]` names the HQ experiments that *would* qualify this model+topology — a reference, not a claim.)
* Machine names or addresses (only bindings/fingerprints locally, never on export).
* Weights, paths, URLs. Models are identified by `{family, quant, expected_root_hash}`; the Host library maps identity to files.
  `expected_files[]` carries bare file names and approximate sizes only (size estimates before import), never a path.
* Secrets or API keys.

## 5. Migration: Fast/Strong/Ultra (and settings v1 → v2)

Catalog tier → profile (all three become the **example set**, `example_of: "tier:<id>"`, deterministic ids so the
migration is idempotent and example imports are recognised):

| Catalog / settings v1 field | Profile v1 field |
|---|---|
| `tiers[].id` `fast`/`strong`/`ultra` | `id` `prof_example_fast` / `_strong` / `_ultra`; `example_of: tier:<id>` |
| `display_name` | `name` |
| `model.family`, `display_name`, `quant`, `artifact_id`, `expected_root_hash` | `model.identity.*` (null hash stays null = unpinned) |
| `model.expected_files[] {role, name, approx_bytes}` | `model.identity.expected_files[]` (verbatim) |
| `model.pin_status` | derived from `expected_root_hash` (as today) |
| `backend` `llama-local` / `strata-hybrid` | `backend.id` (same strings) |
| `roles[]` (`father`, `node:laptop-class`, `node:designated-3060`) | `topology.slots`: `host`; `w1` / `w2` with `select: {mode: binding, binding: <old role string>}` |
| (count of node roles) | `min_workers = max_workers = count` (these tiers have no optional slots) |
| `contexts[].tokens` where `offered` | `context.offered_profiles`; `max_tokens` = largest offered; `default_tokens` 4096 |
| `contexts[]` with `offered: false` | not listed (derivable: absent from `offered_profiles`) |
| `contexts[].requirements` (`feasible_placement`, `qualification_required`) | not stored: feasible placement is always enforced by readiness; qualification notes are computed |
| `contexts[].qualified` (all false) | dropped; qualification is evidence-derived (HQ-TIER-01) |
| `performance_targets[]` | `goals[]` (`pending_qualification`, experiment id kept) |
| `on_session_loss {max_retries, then: stop|downgrade}` | `on_worker_loss {max_retries, then: stop|fallback-profile}`; `downgrade` → `fallback-profile` with `fallback_profile_id` = next lower tier in `fallback_order` (`ultra→strong→fast`) |
| `qualification_experiments[]` | `qualification_experiments[]` (verbatim; the experiment definitions stay in `bench/qualification/experiments.json`) |
| `fallback_order` | encoded by the `fallback_profile_id` chain; the global list disappears |
| (new) `lifecycle`, `exposure` | from settings v1 `keep_ready` (below); API off, LAN off; `api_model_id` `example-<tier>` |

| Settings v1 (`father-settings.json`) | Settings v2 |
|---|---|
| `selected_tier` (`fast`...) | `selected_profile_id` (`prof_example_fast`...) |
| `assignments {role → fingerprint}` | `machine_bindings {role → fingerprint}` (same keys, same values) |
| `context_tokens` | `context_tokens` (unchanged; per-request default) |
| `keep_ready {enabled, release_after_idle_minutes}` | applied to **all three** example profiles (v1 applied it to whichever tier was prepared): `enabled: false` → `preparation: on-demand`, `release_after_idle_seconds: 60` (v1's fixed 60 s, `father_service_api.cpp` watchdog); `enabled: true` → `preparation: keep-ready`, seconds = minutes × 60 (0 stays 0 = no idle release) |
| `model_dirs {tier → dir}` | Host library `scan_roots` (+ per-record `dir`); each existing dir is scanned on first v2 load; the record found there that matches the tier identity sets that example profile's local `model.library_id` |
| `confirmed_model_roots {tier → root}` | library record `pinned_root` of the record in `model_dirs[tier]` whose manifest root equals it (this *is* the user's earlier confirmation, carried over, so a working install stays usable); profile `expected_root_hash` stays null (the catalog value; never auto-promoted) |
| (both maps, until resolved) | kept verbatim under settings v2 `legacy_v1 {model_dirs, confirmed_model_roots}` and re-applied on each rescan until every entry has found its record, so a dir that is offline at migration time loses nothing |
| `paired_nodes`, `advanced` | unchanged |
| (new) `profiles[]`, `aliases[]`, `library`, `server` | seeded with the example set (not selected for API exposure) |

Migration rules: a backup of the v1 file is written once next to it before the first v2 save (owner-only ACL, same as the
original; the store's existing `<file>.v1.bak`); migration is a pure function `v1 → v2` with a test that round-trips every
field above and asserts that every v1 value is either mapped, kept under `legacy_v1`, or listed above as derivable; v2 files are refused by a
v1 binary (the version check already exists in the store); a migrated installation keeps working with `tiers.*` IPC ops
mapped to the three example ids until the UI moves to profile ops. **The example set is also shipped as importable files**
(`fixtures/profiles-example/`, produced by A from [examples/](examples/)) so a fresh install can import it.

## 6. Routing aliases and substitution (ADR 0402)

* A request addressed to a profile's `api_model_id` runs **that profile's exact model** or fails with a structured error
  (`model_not_ready`, `workers_unavailable`, `insufficient_memory`, ...). There is no implicit fallback, ever.
* A **routing alias** is a separate `api_model_id` whose candidates the user listed. Selection is deterministic
  (`first-viable`: first viable in list order; `best-ready`: first already-prepared candidate, else first viable), evaluated
  against live readiness at enqueue and recorded in the ticket; if the recorded candidate is no longer viable at start the
  request is rejected, not re-routed (ADR 0402). The alias never starts a prepare job unless `allow_prepare` is true.
* **An alias never widens access.** A candidate is considered only if the candidate profile is itself visible to the calling
  key (`exposure.api`, `exposure.allow_lan` for a LAN key, the key's `allowed_models`); the alias itself needs `allow_lan`
  to be visible to LAN keys and must be in `allowed_models` (or `*`).
* Every response reports `x_clusterlm` naming what **actually** ran; the field set and which chunks carry it are normative in
  scheduler-admission-v1 §6 (this section's v1.0 list was a subset).
* Built-in-chat fallback (`on_worker_loss.then = fallback-profile`) is a Host-UI behaviour with a visible event, kept from
  `FatherService`; it is a user-chosen policy on the profile, not an API behaviour. As today, a fallback target whose
  tokenizer vocabulary differs from the failing model's cannot continue an in-flight answer (it may serve the next turn).
* The model of an in-flight request never changes mid-stream for API clients. A Worker loss ends the stream with an
  OpenAI-shaped error event (scheduler-admission-v1 §4 "Cancellation"); no non-standard `finish_reason` is ever emitted.

## 7. Export / import

`profiles.export(id)` emits the profile JSON with: `model.library_id` removed; `machine` selectors converted to `binding`s
named `worker:<slug>` where slug = the Worker's display name lower-cased, every run of characters outside `[a-z0-9]` replaced
by `-`, trimmed to 32 characters (`worker:worker` if empty; `-2`, `-3`… on collision within the profile), **and the
fingerprint dropped**; no paths; no keys; `revision` and `id` kept so a re-import is recognised (same id + same content =
no-op; same id + different content = user chooses replace/duplicate). `profiles.import(doc)` is bounded (1 MiB, depth 16),
validates fully (level 1 always; level 2 once the model exists in the library), assigns a fresh id on "duplicate", never
auto-binds an unbound slot, never auto-enables API exposure or LAN visibility (those reset to `false`), refuses an
`api_model_id` that collides with an existing one unless the user picks a new one, and lists what the user must still do
(import the model, bind the Worker).

## 8. IPC / API surface (Host-local management; scopes in auth-scopes-v1)

`profiles.list | get | create | update | delete | duplicate | export | import | validate | dry_run | select | prepare | release`,
`aliases.*`, `bindings.list | set`, `library.*` (workstream A defines the library ops). The old `tiers.*` ops remain as thin
aliases for the example ids during the transition and are removed by E after the UI migrates (CMR-0003).

## Amendments

* v1.1 (2026-10-10, ADR 0407, Opus architecture review): strict unknown-field policy + per-document quarantine (§1a);
  worker-count defaults and one-machine-per-slot (§2); library resolution and `library_id` precedence (§3); lossless
  migration (`expected_files`, `qualification_experiments`, `keep_ready` → `release_after_idle_seconds`, `legacy_v1`); alias
  visibility; export slug rule; `api_model_id` required when `api: true`; `backend.id` as a pattern; selector fields exact per
  mode; bounded numeric fields. Schema `schema_version` stays `1`: no v1.0 document was ever persisted by a product build.
