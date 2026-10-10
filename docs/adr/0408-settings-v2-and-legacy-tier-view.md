# ADR 0408: Settings v2 keeps a legacy tier view

Status: accepted (workstream A, 2026-10-10)

## Context
ADR 0400 replaces the hard-coded Fast/Strong/Ultra tiers with execution profiles. The father-service, coordinator and UI still
read `selected_tier`, `keep_ready`, `model_dirs`, `confirmed_model_roots` and `assignments` from `FatherSettings`. Rewriting those
consumers belongs to C and E, and the project rule is to preserve existing engineering.

## Decision
- `kFatherSettingsVersion` becomes 2. New persisted fields: `selected_profile_id`, `profiles`, `aliases`, `library`,
  `examples_seeded`, plus a `quarantine` list for documents that fail validation (kept verbatim, never silently dropped).
- The tier-era fields stay in the struct as a "legacy tier view" and are persisted under `legacy_v1`. `assignments` is persisted
  as `machine_bindings`. Unknown top-level fields are preserved on rewrite.
- The settings store runs the v1 to v2 migration by default, so no caller wiring is needed. Migration is a pure text to text
  function (`orchestrator/migration`), tested against a real v1 file emitted by the pre-change code (commit 9b76ef6).
- Tiers become `prof_example_<tier>` profiles with `example_of: "tier:<id>"`. The same three profiles ship as the importable
  example set (`fixtures/profiles-example/`); a test pins the embedded set to the catalog-derived profiles.
- `tiers.select` and `settings.set` keep their wire shape and write through the legacy view.

## Consequences
- C and E can move to profiles at their own pace, then drop the legacy fields (CMR-0003, CMR-0004, CMR-0008).
- Reconciling `legacy_v1` roots into the model library (`library_id`, user pin) is I/O and is not done yet (CMR-0009).
- Profile `speculation.max_q` is independent of the legacy `advanced.default_q`, which stays the legacy path's control.
