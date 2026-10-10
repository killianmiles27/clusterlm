# 0400: Execution profiles replace the hard-coded tier catalog

## Context
ClusterLM models its product as three fixed tiers (`fast/strong/ultra`) in `orchestrator/catalog`, with topology roles tied
to one user's machines (`node:laptop-class`, `node:designated-3060`), per-tier model directories and a fixed `fallback_order`
(ADR 0170). The mission is a configurable platform. ADR 0170's good ideas (roles not machine names, unpinned models refused
until confirmed, bounded strict parsing, qualification as data) must survive.

## Decision
- The unit of configuration is the **execution profile** (`docs/interfaces/profile-schema-v1.md`): stable `prof_*` id, mutable
  name, exact model identity, backend id, context, topology **slots** with selectors (`binding`, `machine`, `requirements`),
  resource margins, placement mode, speculation, lifecycle, Worker-loss policy, exposure.
- Roles become **named machine bindings** (same strings, same fingerprint values) so existing `assignments` migrate losslessly
  and ADR 0170's "roles, not machines" stays true for shipped/exported profiles.
- Profiles hold **no** qualification, compatibility or performance claims (`goals[]` excepted, always `pending_qualification`):
  those are computed from backend descriptors and evidence (ADR 0401).
- Fast/Strong/Ultra become the **example set** with deterministic ids (`prof_example_*`), produced by a tested migration and
  shippable as importable files. The catalog code and JSON remain only as the migration source until A removes the tier
  vocabulary from product code.
- The unpinned-model rule is kept verbatim: `expected_root_hash: null` is refused by readiness until the Host user confirms
  the inspected manifest; confirmation is never inferred or auto-promoted.
- API clients address a profile by `api_model_id`, independent of `name` and `id`.

## Consequences
A is a large workstream (sequenced in `docs/plan.md` §4). Old IPC `tiers.*` ops remain as aliases over the example ids until E
migrates the UI. Hardware-qualification experiment ids stay in `bench/qualification/experiments.json` and are referenced from
descriptors, not from profiles. Rejected alternative: keep tiers and add "custom tiers" — it would perpetuate the three-profile
assumptions in readiness, fallback order and IPC.
