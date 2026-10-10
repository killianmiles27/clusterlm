# Execution profile schema v1

Frozen by ADR 0400 (profiles), 0402 (routing vs substitution). Schema: [profile-v1](schemas/profile-v1.schema.json),
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
| Model record | `mdl_*` | Host library (workstream A) | exact identity of an artifact on the Host; profiles reference it by identity |

## 2. Topology: slots, not roles

`topology.slots[]` is the pipeline in stage order. Slot 0 is the single `host` slot (the Host always owns the prefix and tail
domains, per the ExecutionDomain contract). Each `worker` slot has a **selector**:

| Selector | Meaning | Exported? |
|---|---|---|
| `binding` | named binding resolved through the Host's `machine_bindings` (the old `assignments` keys such as `node:laptop-class` survive as binding names) | yes, as a label; recipient must bind it |
| `machine` | one specific paired Worker by device fingerprint | **removed** (becomes `binding` named after the Worker's display name, or an unbound slot) |
| `requirements` | any paired Worker meeting resource requirements (vendor, VRAM, RAM, compute capability, link speed, CPU-only OK) | yes |

`min_workers`/`max_workers` bound how many worker slots must/may be filled; a slot marked `optional` may be left empty when
the count stays within bounds (a lower-capacity re-plan, never a different model). A Worker that is Busy, on battery when
policy forbids, offline or releasing does **not** satisfy a selector ("a busy machine is not available compute").
Layer numbering is never exposed in the UI; `placement.manual` is an advanced override honoured only if the backend
descriptor sets `manual_layer_ranges` and the same admission checks pass.

## 3. Validation (three levels, all must pass before "ready" can ever be reported)

1. **Schema + semantic (offline, no hardware).** The JSON Schema plus: exactly one host slot at index 0; unique slot names;
   `min_workers ≤ max_workers ≤ #worker slots`; `default_tokens ≤ max_tokens`; every `offered_profiles` ≤ `max_tokens`;
   `fallback_profile_id` resolves and the chain is acyclic; `api_model_id` unique across profiles and aliases and different from
   every reserved name (`auto` is allowed only as an alias); `placement.manual` only if the descriptor allows; limits on
   array sizes as in the schema.
2. **Compatibility (Host library + backend descriptor).** Model identity resolves to a library record; the record's
   tensor formats and family are executable by `backend.id` at `min_contract_version`; `topology.max_workers ≤
   descriptor.execution.validated_max_workers`; a cross-machine topology requires `execution.cross_machine`; the backend is
   built into this binary and its runtime is present (`BackendInfo.hardware_available` is reported separately, never hidden).
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
  (`goals[]` carries the old catalog's `performance_targets` as non-binding, `pending_qualification` data.)
* Machine names or addresses (only bindings/fingerprints locally, never on export).
* Weights, paths, URLs. Models are identified by `{family, quant, expected_root_hash}`; the Host library maps identity to files.
* Secrets or API keys.

## 5. Migration: Fast/Strong/Ultra (and settings v1 → v2)

Catalog tier → profile (all three become the **example set**, `example_of: "tier:<id>"`, deterministic ids so the
migration is idempotent and example imports are recognised):

| Catalog / settings v1 field | Profile v1 field |
|---|---|
| `tiers[].id` `fast`/`strong`/`ultra` | `id` `prof_example_fast` / `_strong` / `_ultra`; `example_of: tier:<id>` |
| `display_name` | `name` |
| `model.family`, `display_name`, `quant`, `artifact_id`, `expected_root_hash` | `model.identity.*` (null hash stays null = unpinned) |
| `backend` `llama-local` / `strata-hybrid` | `backend.id` (same strings) |
| `roles[]` (`father`, `node:laptop-class`, `node:designated-3060`) | `topology.slots`: `host`; `w1` / `w2` with `select: {mode: binding, binding: <old role string>}` |
| (count of node roles) | `min_workers = max_workers = count` (these tiers have no optional slots) |
| `contexts[].tokens` where `offered` | `context.offered_profiles`; `max_tokens` = largest offered; `default_tokens` 4096 |
| `contexts[].qualified` (all false) | dropped; qualification is evidence-derived (HQ-TIER-01) |
| `performance_targets[]` | `goals[]` (`pending_qualification`, experiment id kept) |
| `on_session_loss {max_retries, then: stop|downgrade}` | `on_worker_loss {max_retries, then: stop|fallback-profile}`; `downgrade` → `fallback-profile` with `fallback_profile_id` = next lower tier in `fallback_order` (`ultra→strong→fast`) |
| `qualification_experiments[]` | stay in `bench/qualification/experiments.json`, referenced via descriptor `hardware_experiments` |
| `fallback_order` | encoded by the `fallback_profile_id` chain; the global list disappears |
| (new) `lifecycle`, `exposure` | defaults: `on-demand`, 30 min idle release, API off, LAN off |

| Settings v1 (`father-settings.json`) | Settings v2 |
|---|---|
| `selected_tier` (`fast`...) | `selected_profile_id` (`prof_example_fast`...) |
| `assignments {role → fingerprint}` | `machine_bindings {role → fingerprint}` (same keys, same values) |
| `context_tokens` | `context_tokens` (unchanged; per-request default) |
| `keep_ready {enabled, release_after_idle_minutes}` | applied to the *selected* profile's `lifecycle` (`keep-ready` / idle minutes); not a global any more |
| `model_dirs {tier → dir}` | Host library `scan_roots` (+ per-record `dir` hint); each existing dir is scanned on first v2 load |
| `confirmed_model_roots {tier → root}` | library record `pinned_root` for the matching identity + profile `expected_root_hash` stays null until the user re-confirms (pin never auto-promoted) |
| `paired_nodes`, `advanced` | unchanged |
| (new) `profiles[]`, `aliases[]`, `library`, `server` | seeded with the example set (not selected for API exposure) |

Migration rules: a backup of the v1 file is written once next to it before the first v2 save (owner-only ACL, same as the
original); migration is a pure function `v1 → v2` with a test that round-trips every field above; v2 files are refused by a
v1 binary (the version check already exists in the store); a migrated installation keeps working with `tiers.*` IPC ops
mapped to the three example ids until the UI moves to profile ops. **The example set is also shipped as importable files**
(`fixtures/profiles-example/`, produced by A from [examples/](examples/)) so a fresh install can import it.

## 6. Routing aliases and substitution (ADR 0402)

* A request addressed to a profile's `api_model_id` runs **that profile's exact model** or fails with a structured error
  (`model_not_ready`, `workers_unavailable`, `insufficient_memory`, ...). There is no implicit fallback, ever.
* A **routing alias** is a separate `api_model_id` whose candidates the user listed. Selection is deterministic
  (`first-viable`, or `best-ready`), evaluated against live readiness. The alias never starts a prepare job unless
  `allow_prepare` is true. Every response and stream chunk reports `x_clusterlm: {profile_id, api_model_id, model, quant,
  routed_via?}` naming what **actually** ran (field set in scheduler-admission-v1 §6).
* Built-in-chat fallback (`on_worker_loss.then = fallback-profile`) is a Host-UI behaviour with a visible event, kept from
  `FatherService`; it is a user-chosen policy on the profile, not an API behaviour.
* The model of an in-flight request never changes mid-stream for API clients. A Worker loss ends the stream with a structured
  error and finish reason `error`.

## 7. Export / import

`profiles.export(id)` emits the profile JSON with: `model.library_id` removed; `machine` selectors converted to `binding`s
named after the Worker's display name, **and the fingerprint dropped**; no paths; no keys; `revision` and `id` kept so a
re-import is recognised (same id + same content = no-op; same id + different content = user chooses replace/duplicate).
`profiles.import(doc)` validates fully (level 1 always; level 2 once the model exists in the library), assigns a fresh id on
"duplicate", never auto-binds an unbound slot, never auto-enables API exposure or LAN visibility (those reset to `false`), and
lists what the user must still do (import the model, bind the Worker).

## 8. IPC / API surface (Host-local management; scopes in auth-scopes-v1)

`profiles.list | get | create | update | delete | duplicate | export | import | validate | dry_run | select | prepare | release`,
`aliases.*`, `bindings.list | set`, `library.*` (workstream A defines the library ops). The old `tiers.*` ops remain as thin
aliases for the example ids during the transition and are removed by E after the UI migrates (CMR-0003).
