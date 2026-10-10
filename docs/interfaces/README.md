# ClusterLM frozen interfaces (v1.1)

Owner: Thread 0. Status: **Frozen v1.1** as of 2026-10-10 (v1 by Thread 0; v1.1 after the Opus architecture review,
[ADR 0407](../adr/0407-interface-review-and-freeze-v1.md), which lists every change and the residual risks). "Frozen" is a
process state (ADR 0406): a change needs an ADR in `docs/adr/` and a note in `docs/status.md`; additive, backward-compatible
clarifications may use the lightweight path described there. No implementation of these contracts existed when v1.1 was cut,
so v1.1 replaces v1 outright (document `schema_version`/`contract_version` stay `1`; file names are unchanged). Workstream A is
the first to exercise them.

| Revision | Date | Summary |
|---|---|---|
| v1 | 2026-10-10 | Thread 0 drafts (ADRs 0400–0406) |
| v1.1 | 2026-10-10 | Review fixes: unknown-field policy, lossless Fast/Strong/Ultra migration, alias visibility, scheduler cancellation/swap/memory ownership, standard finish reasons, auth hardening (DNS rebinding/CSRF, escalation, timing), readiness blocker/transient rule, required descriptor capabilities (ADR 0407) |

| Interface | Document | Machine-readable | ADR |
|---|---|---|---|
| Execution profile schema v1 + routing alias + migration from Fast/Strong/Ultra | [profile-schema-v1.md](profile-schema-v1.md) | [profile-v1](schemas/profile-v1.schema.json), [routing-alias-v1](schemas/routing-alias-v1.schema.json), [examples](examples/) | 0400, 0402 |
| Backend capability contract v1 | [backend-capability-v1.md](backend-capability-v1.md) | [backend-descriptor-v1](schemas/backend-descriptor-v1.schema.json) | 0401 |
| Scheduler / admission API v1 (incl. serving-layer rules) | [scheduler-admission-v1.md](scheduler-admission-v1.md) | | 0405 |
| API auth scopes v1 | [auth-scopes-v1.md](auth-scopes-v1.md) | | 0403 |
| Readiness state machine v1 | [readiness-state-machine-v1.md](readiness-state-machine-v1.md) | | 0404 |

`python3 -I docs/interfaces/validate_examples.py` validates every example against its schema plus the cross-field and
cross-document rules the schemas cannot express (profile level-1 rules, descriptor-based level-2 checks, alias references;
needs `pip install jsonschema`). It is a design-time guard; the C++ validator is workstream A's.

## Conventions shared by all contracts

* **Versioning.** Every persisted or wire object carries `schema` and `schema_version` (or `contract_version`). A reader
  accepts its own version and older ones through a migration; a newer version is refused with a clear error, never guessed at.
* **Unknown fields.** Rejected at every level, on import and on local load (all schemas use `additionalProperties: false`).
  Writers omit optional fields equal to their default, so an addition made through ADR 0406's lightweight path stays readable
  by older readers until a document actually uses it. A contract document that fails validation inside a settings file is
  quarantined on its own (profile-schema-v1 §1a); it never makes the whole settings file "corrupt".
* **Stable IDs.** `prof_*` (profile), `alias_*` (routing alias), `mdl_<24 hex>` (library model, derived from the structure
  digest), `key_<16>` (API key), `job_*` (prepare job), `req_*` (request). IDs are opaque, immutable, lowercase `[a-z0-9_]`,
  and are never reused (a deleted profile's id stays reserved in settings). Display names are
  mutable labels. API clients address models by `api_model_id`, which is also independent of the display name.
* **Provenance.** Any numeric result shown or returned (throughput, memory, time-to-ready, latency) carries
  `provenance ∈ {synthetic, measured, qualified}` with a `source` string; aggregation takes the weakest input
  (`placement::Provenance`). No contract here may return a bare number.
* **Honest labels.** Compatibility labels are exactly: `Supported and qualified`, `Supported, awaiting hardware
  qualification`, `Experimental`, `Unsupported`. They are computed from descriptor + evidence, never typed in a profile.
* **Errors.** Structured: `{code, message, details?}` with the `ErrorCode` names the code base already uses
  (`INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, `PERMISSION_DENIED`, `NOT_FOUND`, ...). The OpenAI
  surface maps these to OpenAI-shaped errors with one fixed table (scheduler-admission-v1 §7); only standard OpenAI
  `finish_reason` values are ever emitted.
* **Privacy.** No contract carries prompts, completions, token IDs, tool arguments or file paths into logs, diagnostics,
  exports, MCP responses or any Worker-bound message. Token IDs and text stay Host-local (existing guarantee, unchanged).
* **Bounded.** Every parser states size and depth limits (default: 1 MiB document, depth 16, string 4 KiB unless stated);
  every array and integer in the schemas has an explicit maximum.
* **Naming.** User-facing text says Host and Worker; code keeps Father/Node (ADR 0406 §display names).
