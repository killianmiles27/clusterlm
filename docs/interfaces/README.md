# ClusterLM frozen interfaces (v1)

Owner: Thread 0. Status: **Frozen v1** as of 2026-10-10. "Frozen" is a process state (ADR 0406): a change needs an ADR in
`docs/adr/` and a note in `docs/status.md`; additive, backward-compatible clarifications may use the lightweight path
described there. No implementation of these contracts exists yet; workstream A is the first to exercise them.

| Interface | Document | Machine-readable | ADR |
|---|---|---|---|
| Execution profile schema v1 + routing alias + migration from Fast/Strong/Ultra | [profile-schema-v1.md](profile-schema-v1.md) | [profile-v1](schemas/profile-v1.schema.json), [routing-alias-v1](schemas/routing-alias-v1.schema.json), [examples](examples/) | 0400, 0402 |
| Backend capability contract v1 | [backend-capability-v1.md](backend-capability-v1.md) | [backend-descriptor-v1](schemas/backend-descriptor-v1.schema.json) | 0401 |
| Scheduler / admission API v1 (incl. serving-layer rules) | [scheduler-admission-v1.md](scheduler-admission-v1.md) | | 0405 |
| API auth scopes v1 | [auth-scopes-v1.md](auth-scopes-v1.md) | | 0403 |
| Readiness state machine v1 | [readiness-state-machine-v1.md](readiness-state-machine-v1.md) | | 0404 |

`python3 -I docs/interfaces/validate_examples.py` validates every example against its schema plus the cross-field rules the
schemas cannot express (needs `pip install jsonschema`). It is a design-time guard; the C++ validator is workstream A's.

## Conventions shared by all contracts

* **Versioning.** Every persisted or wire object carries `schema` and `schema_version` (or `contract_version`). A reader
  accepts its own version and older ones through a migration; a newer version is refused with a clear error, never guessed at.
* **Stable IDs.** `prof_*` (profile), `alias_*` (routing alias), `mdl_*` (library model), `key_*` (API key), `job_*`
  (prepare job), `req_*` (request). IDs are opaque, immutable, lowercase `[a-z0-9_]`, and are never reused. Display names are
  mutable labels. API clients address models by `api_model_id`, which is also independent of the display name.
* **Provenance.** Any numeric result shown or returned (throughput, memory, time-to-ready, latency) carries
  `provenance ∈ {synthetic, measured, qualified}` with a `source` string; aggregation takes the weakest input
  (`placement::Provenance`). No contract here may return a bare number.
* **Honest labels.** Compatibility labels are exactly: `Supported and qualified`, `Supported, awaiting hardware
  qualification`, `Experimental`, `Unsupported`. They are computed from descriptor + evidence, never typed in a profile.
* **Errors.** Structured: `{code, message, details?}` with the `ErrorCode` names the code base already uses
  (`INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, `PERMISSION_DENIED`, `NOT_FOUND`, ...). The OpenAI
  surface maps these to OpenAI-shaped errors (scheduler-admission-v1 §7).
* **Privacy.** No contract carries prompts, completions, token IDs, tool arguments or file paths into logs, diagnostics,
  exports, MCP responses or any Worker-bound message. Token IDs and text stay Host-local (existing guarantee, unchanged).
* **Bounded.** Every parser states size and depth limits (default: 1 MiB document, depth 16, string 4 KiB unless stated).
* **Naming.** User-facing text says Host and Worker; code keeps Father/Node (ADR 0406 §display names).
