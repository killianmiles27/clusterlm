# 0407: Interface review and freeze v1.1

## Context
Thread 0 (Sonnet) wrote the v1 contracts in `docs/interfaces/` (ADRs 0400–0406) without an Opus review. Before B, C and F
build on them, the Opus architecture reviewer checked the profile schema, backend capability contract, scheduler/admission
API, auth scopes and readiness state machine against the spec's non-negotiable guarantees and against the existing code
(`orchestrator/catalog`, `orchestrator/config`, `orchestrator/father-service`, `apps/father-agent/father_service_api.cpp`,
`runtime/domain`, `runtime/objects`, `runtime/backends/strata-core`, `orchestrator/placement`, the in-progress
`orchestrator/library` of workstream A, `docs/ARCHITECTURE.md`, `docs/tiers.md`). No contract had been implemented, so every
fix below is applied directly as contract revision **v1.1**; document `schema_version`/`contract_version` stay `1` because no
v1.0 document was ever persisted by a product build. File names are unchanged.

## Decision: changes (what, why)

### Cross-cutting
1. **Unknown-field policy.** v1 said both "rejected on import, ignored on local load" (schema description) and
   `additionalProperties: false` (schema), and ADR 0406 allowed "compatible additions" that strict readers would reject. Now:
   reject everywhere; writers omit optional fields equal to their default (so additions stay readable until used); a profile or
   alias that fails validation inside settings is quarantined individually (state `unavailable`, `invalid_document`) instead of
   the settings store moving the whole file aside and losing pairing records. *Why:* typo-safety for security fields, no silent
   data loss, one consistent rule.
2. **Bounds.** Every array/integer in the three schemas now has a maximum (revision, MiB fields, layer indices, link speed,
   options, evidence lists, families). *Why:* "bounded allocations" applies to parsers (ADR 0233).

### Profile schema / migration (profile-schema-v1, profile-v1.schema.json, routing-alias-v1.schema.json, examples)
3. **Lossless Fast/Strong/Ultra migration.** v1 dropped `model.expected_files` (incl. Ultra's published shard sizes) and the
   per-tier `qualification_experiments` (HQ-STRONG-01, HQ-PERF-02..04 appear nowhere else as tier mappings). Added
   `model.identity.expected_files[]` and top-level `qualification_experiments[]` (references, not claims). This supersedes ADR
   0400's "experiment ids are referenced from descriptors, not profiles" for topology-specific experiments.
4. **`keep_ready` mapping was lossy and changed behaviour.** v1 settings release an idle plan after a fixed **60 s** when
   `keep_ready.enabled` is false (`FatherServiceApi::watchdog`) and apply the policy to whichever tier is prepared; v1 contracts
   mapped it to the selected profile only, with a 30-minute default (30× longer Worker memory retention). Replaced
   `lifecycle.release_after_idle_minutes` with `release_after_idle_seconds` (default 60, max 7 days as the v1 validator), mapped
   onto all three example profiles; `keep-ready` no longer implies automatic preparation (only `prepare_when_available` does).
5. **Root confirmations survive.** v1 said the profile pin "stays null until the user re-confirms", i.e. a working install
   would stop being ready. The v1 confirmation is carried over as the library record's `pinned_root`; `model_dirs` and
   `confirmed_model_roots` are also kept verbatim under settings v2 `legacy_v1` until every entry has found its record.
6. **Worker-count defaults.** `min_workers`/`max_workers` defaulted to 0, which made a profile with worker slots that omits
   them unusable. Defaults are now "non-optional slots" / "all worker slots", with `non-optional ≤ min ≤ max ≤ slots ≤ 8` and
   `max_workers ≤ CompatReport.max_workers`.
7. **One machine per slot**, deterministic resolution order for `requirements` selectors, unassigned bindings never auto-bound
   (v1 lost `TierAssignment::validate_for`'s "a machine fills at most one role").
8. **Selector exactness.** A selector could carry `binding` and `machine` together; now exactly the fields of its mode.
   `gpu_vendor: "any"` removed (ambiguous with other vendors; absent = any). Host slot cannot carry `select`/`optional`.
9. **`exposure.api: true` requires `api_model_id`**; reserved ids defined (`auto`/`default` only for aliases, prefix
   `clusterlm-` reserved); uniqueness also for unexposed profiles; `fallback-profile` requires `fallback_profile_id`, not self.
10. **`backend.id` is a pattern, not an enum**, so a profile for a backend this build lacks loads and reports `unavailable`
    instead of failing schema validation (and being quarantined).
11. **`backend.options` / `options_schema`.** v1 let a descriptor embed arbitrary JSON Schema; the C++ tree has no JSON Schema
    engine (third_party: doctest, imgui, nlohmann). Restricted to scalar options with `type/minimum/maximum/enum`. Removed
    llama's `n_gpu_layers` from the example: GPU offload is a placement decision; options must not bypass memory admission.
12. **`library_id`** pattern aligned with A's `mdl_<24 hex>`; resolution rules written down (pinned/unpinned/ambiguous as in
    `ModelLibrary::resolve`; `library_id` disambiguates but never bypasses the pin).
13. **Speculation**: effective q = `max_q` when enabled; must be ≤ descriptor `max_q`; q > 1 needs `window_commit_abort`.
14. **Export slug.** Machine selectors were exported as bindings "named after the display name", which violates the binding
    pattern (`G14`, `Gaming PC`); now `worker:<slug>` with a defined slug and collision rule. Import refuses colliding
    `api_model_id`s and is bounded.
15. **Aliases cannot widen access.** A key allowed only an alias could reach profiles it may not see, and a LAN key could reach
    LAN-hidden profiles. Candidates are now filtered by the caller's visibility; aliases gained `allow_lan`. `best-ready` was
    undefined; both policies are now defined deterministically. Workers-available slot names and candidate uniqueness are
    validated.
16. **Generic example** used `Q4_K_M` on `strata-hybrid` (not in its tensor formats, family Unsupported) and a compute
    capability below the backend's minimum; corrected so the example passes its own level-2 rules.

### Backend capability contract
17. **Required capabilities.** §5 said "no maybe values" but `serving`, `per_client_isolation`, `max_sessions_per_domain`,
    `max_q`, `requires_runtime`, `role_in_product`, `hardware_experiments` were optional (the reference example omitted
    `max_sessions_per_domain`, which the scheduler's concurrency formula needs). All required now; `cross_machine: false` forces
    `validated_max_workers: 0`, and `window_commit_abort: false` forces `max_q: 1`; `id`↔`factory_name` pairs enforced.
18. **Tensor formats.** The llama example's `pinned-llama.cpp-supported` placeholder could not be evaluated; Strata listed only
    `IQ2_XS`/`IQ3_S`, so every real model (F32 norms etc.) would be `Unsupported`. Rule 1 now checks the record's full
    `tensor_types`; examples list the pinned ggml tables (upper bounds for F to narrow).
19. **Architecture.** The Strata example's `architecture: qwen3.8-flash-next` is a manifest family name; the GGUF
    `general.architecture` is `qwen4exp` (`gguf_manifest.hpp`, `tests/gguf/mini_model_core.hpp`). Family matching now also
    requires the architecture when an entry names one (the library's family is a user-editable label).
20. **`unlisted_family_status`** can no longer be a Supported label; `max_workers` is 0 for non-distributable models;
    development-only backends (the fixture-"qualified" `reference`) are excluded from product profiles and `/v1/models`
    (resolves the contradiction with "fixture results never raise a label"); Experimental opt-in is stored on the library record.
21. **Build hash** mismatch Host/Worker is a readiness input (v1 required the match for admission but readiness ignored it).

### Scheduler / admission
22. **Non-standard finish reasons.** v1 emitted `finish_reason: "cancelled"` / `"error"`, which unmodified OpenAI clients do not
    know (spec: no per-app special cases). Server-ended streams now send one OpenAI-shaped `error` event and close without
    `[DONE]`; only `stop`/`length`/`tool_calls` are used.
23. **Response metadata.** Three different field sets (ADR 0402, profile §6, scheduler §6); and the standard `model` field was
    to carry the alias's chosen *profile id*. Scheduler §6 is normative; `model` is the actual profile's `api_model_id`; carried
    on non-streaming responses, first and last stream chunks, and error bodies.
24. **Alias re-routing contradiction.** §3 (and ADR 0402) rejected a ticket whose recorded candidate became non-viable; §5 let
    it move to the next candidate. Aligned on ADR 0402 (reject).
25. **Cancellation could hang and free memory early.** Added a ticket state machine (cancel valid in every state, including the
    `starting` race), `cancel_timeout` (10 s) after which the plan is invalidated (epoch advance, lease release, Host domain
    destruction) and the slot is freed only when no domain can still write the session's state. Non-streaming disconnects also
    cancel.
26. **Swaps and memory ownership.** At most `max_prepared_profiles` (default 1, as today); a swap releases the old plan and
    waits for confirmed lease release and Host domain destruction before preparing; dispatch stays in queue order (removes v1's
    contradictory "queue behind the swap" sentence). One running prepare job; job states separated from Coordinator phases
    (v1 omitted `father-domains` and reused the profile state name `ready`); a shared job is not cancelled by one waiting
    request's timeout; bounded job history.
27. **Context check** compared against the *prepared* plan, rejecting requests that fit the profile; now checked against the
    profile, with a re-prepare for the larger context class (test-matrix "context growth forcing re-plan").
28. **Revision pinning.** A plan records the profile revision it was built for; a dequeue re-check rejects tickets whose key was
    revoked, whose model was hidden, or whose profile's model/backend changed while queued.
29. **Warm KV retention** reuse requires same key + conversation hint + exact token-prefix extension and is evicted before
    another client's session opens (it must not block other clients or hold memory against them).
30. **Error table.** `FAILED_PRECONDITION → 409/503` and `queue_full → 429/503` were ambiguous; one table now fixes code, HTTP
    status and OpenAI `type`. Inference-only keys get generic reasons (no machine names).
31. **Management API.** `GET /queue` referenced a non-existent `admin` scope; now `status:read` with anonymous positions. HTTP
    diagnostics export is the shareable bundle (ADR 0232's local bundle contains bench result paths).

### Auth scopes
32. **DNS rebinding / CSRF.** No Host-header or Origin validation was specified, and the "no-auth on loopback" option would
    have let any web page drive the API through a rebound name or a `text/plain` POST. Added mandatory Host allowlist (421),
    Origin refusal, JSON-only POST bodies, bearer-only credentials; the no-key loopback option is limited to `inference` on
    exposed models.
33. **Key handling.** Exact key format and parse rule; hash domain-separated and bound to the key id; identical work and
    response for unknown ids (no id-existence timing oracle); concrete per-address failure throttle; TLS private key storage.
34. **Escalation.** `keys:admin` could mint keys with wider `allowed_models`, LAN reach or longer expiry than its own and revoke
    stronger keys; subset rules added. `allowed_models` now filters every scope; admin scopes require `*`. `source` renamed
    `network` with enforced semantics (`loopback-only` default). Expiry vs revocation semantics defined. `/health` explicitly
    unauthenticated; `/ready?model=` returns 404 for invisible models.

### Readiness
35. **`compatible` vs `unavailable`** for a busy/offline Worker was contradictory (table vs diagram). Rule: blockers ⇒
    `unavailable` (with `reached`), transient conditions ⇒ `compatible`. `installed` = verification pending. Readiness is per
    context class; `preparing` includes a queued job; `/ready` booleans defined exactly; plan revision, build hash, document
    problems and library ambiguity are inputs.

### Tooling
36. `validate_examples.py` now enforces the level-1 rules and the descriptor-based level-2 checks (compat label, max workers, q,
    options, manual stages, compute capability, dev-only backends), alias references and reserved/unique API ids, and refuses
    documents over 1 MiB. Result at freeze: `OK` (exit 0); a mutation pass confirmed it rejects each defect class above.

## Frozen v1.1
The five contracts — profile schema (with routing alias and migration), backend capability contract, scheduler/admission API,
auth scopes, readiness state machine — are **Frozen v1.1** as of 2026-10-10. v1.1 replaces v1 outright (nothing implemented
v1). Further changes follow ADR 0406. Each contract carries an "Amendments" line for v1.1. Superseded statements: ADR 0400
"experiment ids … not from profiles" (item 3), ADR 0402 `x_clusterlm` field list (item 23), ADR 0403 key record `source` (item
34).

## Residual risks (deliberately left)
* **Family label matching** remains possible for descriptor entries without `architecture` (the llama example entry); F must
  add architectures when confirming descriptors (backend-capability-v1 §6). Tensor-format and pin checks still apply.
* **Example descriptor contents** (tensor lists, family lists, rollback modes, isolation) are upper bounds from code tables,
  not verified execution support; F owns them. No label was raised.
* **Settings v2 top-level** still follows the v1 store rule "unknown fields ignored", so a downgraded binary that rewrites
  settings could drop top-level fields a newer one added; mitigated by the store refusing newer `version`s. A owns settings v2.
* **Timing side channel on warm KV reuse** within one key (prefix-reuse latency) is accepted: reuse never crosses keys.
* **Prepare triggered by `inference` keys** on `on-demand` profiles can make Workers provision; bounded by API exposure being
  opt-in, rate limits, one running job and idle release. Owners wanting stricter control choose `manual`.
* **Self-signed LAN TLS** relies on clients pinning or trusting the shown fingerprint; enrollment UX is B/E's (ADR 0403).
* **`max_slot_seconds` fairness** is a coarse heuristic; C may refine it within the same knobs.
* **`docs/status.md`** rows for this revision are left to the lead/thread that commits it (this review edited only
  `docs/interfaces/**` and this ADR); `docs/handoff/0-architecture-2026-10-10.md` still says "Frozen interface v1".

## Consequences
A implements v1.1 directly (no v1 code to migrate). B gains a precise error/metadata contract and Host/Origin checks; C gets an
explicit cancellation, swap and memory-ownership protocol; F gets a stricter descriptor shape and two corrected facts
(architecture, tensor formats). The cost is a larger contract surface, all of it validated by `validate_examples.py`.
