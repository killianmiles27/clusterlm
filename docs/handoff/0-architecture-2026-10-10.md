# Handoff: Workstream 0 · Architecture & Interfaces — 2026-10-10

Branch: `claude/architecture-interfaces-wpeya6`. Thread: "00 · Architecture & Interfaces". Documentation only; no product code,
build or test run.

## 1. Implemented (files)

| Deliverable | Files |
|---|---|
| Spec reference copy (+ Part 2 knowledge split) | `docs/spec/` (`clusterlm-project-spec.md` = unmodified original; `00`–`07` cut from it by line range, no wording changes); copy at `/mnt/project-files/spec/clusterlm-project-spec.md` |
| Dependency map, workstream plan, file ownership, A sequencing, risks | `docs/plan.md` |
| Frozen interface v1 | `docs/interfaces/README.md`, `profile-schema-v1.md`, `backend-capability-v1.md`, `scheduler-admission-v1.md`, `auth-scopes-v1.md`, `readiness-state-machine-v1.md` |
| Machine-readable contracts | `docs/interfaces/schemas/{profile-v1,routing-alias-v1,backend-descriptor-v1}.schema.json` |
| Examples (Fast/Strong/Ultra migrated + generic + alias + 3 descriptors) | `docs/interfaces/examples/*.json` |
| Example/schema guard | `docs/interfaces/validate_examples.py` |
| Requirement ledger | `docs/status.md` |
| Decisions | `docs/adr/0400`–`0406` |
| Cross-module requests | `docs/cross-module-requests.md` (CMR-0001..0007) |

Key decisions: tiers → profiles with slots/selectors and named machine bindings (migration lossless, ids deterministic);
descriptors beside `BackendAdapter` with computed labels; exact model ids never reroute, aliases are explicit; API keys/scopes are
a separate system from pairing; 7-state readiness over the existing pure evaluator; scheduler with admission, async prepare
jobs, concurrency 1 by default, Coordinator remains sole owner of domain state.

## 2. Tested (real) vs mocked

* **Real:** `python3 -I docs/interfaces/validate_examples.py` → `OK` (all 8 example documents validate against their JSON Schemas
  plus the cross-field rules it implements: one host slot first, unique slots, min ≤ max ≤ worker slots, context bounds,
  fallback chain resolves and is acyclic). A deliberate bad edit (invalid id, max_workers 3 with 2 slots) was confirmed to
  fail with exit 1, then reverted. jsonschema 4.26.0.
* **Not run:** the C++ build, ctest, any CI job. Thread 0 changed no code, so none was needed, but **nothing in this handoff
  asserts the build or CI is green on this branch**. The "Implemented and tested" rows in `status.md` restate the
  repository's own documentation; owning threads must re-confirm with a real run.
* **Unexercised:** no implementation consumes these contracts yet. They are designs. Expect amendments (ADR 0406 describes the path).

## 3. Pending hardware qualification

Unchanged: all 41 entries of `HARDWARE-QUALIFICATION.md`. The example backend descriptors are labelled `Supported, awaiting
hardware qualification` and list the HQ ids that gate promotion. The Ultra 20 tok/s target is carried only as a
`pending_qualification` goal.

## 4. Interface changes and requests

New contracts only (nothing existing changed). Notable: `catalog::TierState` → seven-state readiness; `FatherService` single-job gate →
scheduler (A/C refactor with a bounded A edit, CMR-0002); `config` settings v1 → v2 with migration; IPC `tiers.*` kept as
aliases. Requests: CMR-0001..0007 (F descriptors, C/A directory hand-offs, E alias removal, G CI/README, B template/tool parsing
heads-up, A topology heads-up).

## 5. Subagents and model substitutions

None used. Thread 0 ran as a single Sonnet 5.5 session (configured model `claude-sonnet-5-5`); the project policy assigns
architecture/interfaces to Opus 5.5 lead. **Substitution to report:** these architecture decisions were not Opus-reviewed.
Recommendation: have an Opus-led Architecture review (or Thread Z / a "00 (2)" thread) read the interface docs and ADRs 0400–0406
before B, C and F rely on them, especially scheduler-admission-v1 (concurrency/cancellation/memory ownership), auth-scopes-v1
(security design) and the descriptor semantics for Strata. A's schema/migration review by Opus is already required by its kickoff.

## 6. Known issues and next step

* Descriptor contents are provisional (family lists, rollback mode, isolation) and based on reading `docs/backends/*` and
  headers, not on running the backends.
* `validated_max_workers: 2` for Strata reflects tested topologies only (reference backend); no code limit was found in
  headers, so the real cap is unknown.
* Open design choices left deliberately to implementers: LAN TLS certificate enrollment UX (B/E), chat-template engine
  (B, CMR-0006), descriptor loader/evidence format (F), exact wait-policy caps (C).
* The root `README.md` still carries the old status paragraph and does not link the new docs (owned by G, CMR-0005).

**Next step:** start **A · Foundation** (kickoff in `docs/spec/07-thread-kickoffs.md`): step 1 (`orchestrator/profiles`) first.
Commit/handoff at each step boundary in `docs/plan.md` §4 so B/C/F can start as soon as profile types and capability types merge.
G may begin CI/doc scaffolding immediately (CMR-0005). Do not start B, C, F, D, E before A merges.
