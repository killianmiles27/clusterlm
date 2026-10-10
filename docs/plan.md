# ClusterLM generalization plan

Author: Thread 0 (Architecture & Interfaces), 2026-10-10. Requirements: [spec/](spec/README.md). Progress ledger:
[status.md](status.md). Frozen contracts: [interfaces/](interfaces/README.md). Decisions: [adr/](adr/) (0400 series).

## 1. Where the repository is today

The repository is a complete, CI-verified, pre-hardware implementation of a *three-tier, personalized* product
([DEVELOPMENT-STATUS.md](DEVELOPMENT-STATUS.md): nine CI jobs green including MSVC build, WiX MSIs smoke-tested on a Windows
runner; 41 hardware-qualification entries pending; Strata CUDA compiled but never run on a GPU). The engineering to
**preserve** is the runtime (ExecutionDomain, WindowLedger, StageBoundary ABI, Coordinator, provisioning, placement search,
pairing, Windows service stack). The parts that are *personalized* and must be generalized are few and well localized:

| Hard-coded today | Where | Generalized into |
|---|---|---|
| Exactly three tiers `fast/strong/ultra`, fixed `fallback_order` | `orchestrator/catalog` (`TierEntry`, `kRole*` constants, validation rejects any other role or a 3rd node), `fixtures/catalog/clusterlm-catalog.json` | Execution profiles (schema v1) |
| Roles `node:laptop-class`, `node:designated-3060` | `catalog.hpp` `kRoleLaptop`, `kRole3060`; `FatherSettings::assignments` | Topology slots + named machine bindings + resource selectors |
| Model identity inside the tier (`ModelIdentity`, per-tier `model_dirs`, `confirmed_model_roots`) | `catalog.hpp`, `config/settings.hpp` | Host model library records (exact hash, quant, files) referenced by profiles |
| Backend enum `llama-local` / `strata-hybrid`, `llama-local` = Father-only | `BackendKind`, `backend_factory.cpp`, `BackendInfo` | Backend capability descriptors (data) beside the existing `BackendAdapter` |
| UI/IPC speak tiers (`tiers.list`, `tiers.select`, `tiers.prepare`) | `apps/father-agent/.../father_service_api.cpp`, `docs/father-ipc.md`, `ui/father` | Profile ops (old ops kept as a deprecated alias for the migrated example set during the transition) |
| `FatherService`: one job at a time, one conversation, Father-local chat roles | `orchestrator/father-service` | Scheduler (queue, admission, per-client sessions) with the service as its first client |
| No network API at all | n/a | Authenticated OpenAI-compatible server on the Host (new) |
| Readiness is 4 states per tier | `catalog/readiness.hpp` | Readiness state machine v1 (7 states, profile-level) |

What is **already generic** and must not be rewritten: `ExecutionDomain`/`BackendAdapter` (roles prefix/middle/tail,
window commit/abort, `DomainRequirements`), `placement::search_placements` + `Provenance` (Synthetic < Measured < Qualified),
`objects::ModelManifest` (hash-addressed, plan-scoped manifests), the Coordinator/Node leases, SPAKE2 pairing, owner-only
settings store, and the privacy contracts (token IDs/text never cross to Nodes, counts-only diagnostics).

## 2. Architectural stance (one paragraph)

Introduce **three new data concepts and two new runtime services**, and nothing else structural. Data: the *Model record*
(Host library; exact artifact identity), the *Execution Profile* (what to run, where, when, on what policy), and the *Backend
descriptor* (what a backend honestly supports). Services: the *Serving layer* (authenticated HTTP API on the Host, a client
of the scheduler) and the *Scheduler* (admission, bounded queue, per-client sessions, async prepare jobs) which replaces
`FatherService`'s single-job gate while reusing its Coordinator-driving internals. Tiers become the *example profile set*
produced by migration. Workers stay dumb: no new Worker-side concept except honest capability reporting and the existing
lease/release lifecycle. See ADRs 0400-0406.

## 3. Workstreams, file ownership and dependency map

```
            ┌────────────────────────────────────────────────────────────────┐
   0 ──────►│ A Foundation (schema impl, model library, migration, topology) │
 (this)     └───────┬───────────────────────┬───────────────────┬────────────┘
                    │                       │                   │
                    ▼                       ▼                   ▼
            B Serving API            C Cluster mgmt        F Runtime & compat
            (HTTP, SSE, tools)       (scheduler, policy)   (descriptors, llama, Strata, packaging)
                    │                       │                   │
                    └───────┬───────────────┘                   │
                            ▼                                   │
                  D MCP (needs B+C status APIs)                 │
                            │                                   │
          E UX (needs A; consumes B/C/F as they land) ◄─────────┘
                            │
          G Verification & docs: continuous from A onward
                            ▼
                      Z Final review
```

Start rules (from the spec): B, C and F start only after A has **merged** its profile-schema implementation and the
backend-capability types; D after B and C expose stable status APIs; E after A (it can use the draft IPC of A while B/C land);
G continuously; Z last.

### Initial file ownership (no two threads edit the same file concurrently)

| Path | Owner | Notes |
|---|---|---|
| `docs/spec/`, `docs/plan.md`, `docs/interfaces/`, `docs/adr/04*` | 0 | Frozen; changes by ADR through the owning thread of the change, reviewed by an Architecture thread |
| `docs/status.md` | all (append own rows) | Each thread edits only its own rows; Z/0 reconciles |
| `orchestrator/catalog/`, `orchestrator/config/` (Father side), new `orchestrator/profiles/`, new `orchestrator/library/`, `fixtures/catalog/`, new `fixtures/profiles-example/` | A | `config/settings.hpp` NodeSettings section is handed to C **after A merges** (cross-module request CMR-0004) |
| `apps/father-agent/` IPC ops (`father_service_api.*`), `docs/father-ipc.md` | A for profile ops until A merges, then E (consumer) requests changes via CMR | B adds its listener in new files only |
| new `orchestrator/serving/` (HTTP server, OpenAI schema, tool-call parsing, SSE), `orchestrator/father-service/{chat_template,tokenizer,tier_tokenizer,gguf_bpe_tokenizer}.*` | B | chat template/tool parser work touches the tokenizer files |
| new `orchestrator/scheduler/`, `orchestrator/father-service/{service,production,node_notify}.*`, `orchestrator/coordinator/`, `node/`, Node policy in `config/` (after A) | C | |
| new `orchestrator/mcp/` (or `apps/mcp-server/`) | D | stdio first |
| `ui/`, `packaging/wix` UI strings | E | |
| `runtime/` (domain, backends, platform, transport), `third_party/`, `experimental/`, `packaging/` build scripts, `bench/` | F | Capability descriptor implementations live in each backend dir |
| `.github/`, `tests/conformance/`, `tests/clients/`, user-facing docs (`README.md`, install, matrices, security policy, etc.) | G | Each thread writes tests for its own code under its own `tests/<area>/`; G owns cross-cutting suites and CI |

A thread needing a file it does not own files a request in [cross-module-requests.md](cross-module-requests.md).

## 4. Sequencing inside A (so B, C, F can start on A's merge)

1. `orchestrator/profiles`: profile + routing-alias types, strict bounded parser/validator/serializer matching
   `interfaces/schemas/profile-v1.schema.json`; stable-ID rules; export/import redaction. Pure library, no I/O.
2. `orchestrator/library`: model records (scan dirs, GGUF identification via the existing GGUF reader, hashes, split GGUF,
   family/quant, compatibility report skeleton), persisted in Host settings v2.
3. Capability types: `BackendDescriptor` + registry in `runtime/domain` (type only; F fills descriptors for real backends,
   A ships descriptors for `reference` and a provisional one per existing backend marked "needs F review").
   *Exception to the F ownership of `runtime/`: A adds only the new header and registry; recorded in CMR-0001.*
4. Settings v1→v2 migration (tiers→profiles, `assignments`→machine bindings, `model_dirs`/`confirmed_model_roots`→library) with
   the survival test the kickoff demands; ship the Fast/Strong/Ultra example set as importable fixtures.
5. Topology resolution: slots + selectors → concrete `Deployment` for the existing `DeploymentProvider` (dynamic topology).
6. Profile IPC ops (`profiles.list/get/create/update/delete/duplicate/export/import/validate/dry_run`), keeping `tiers.*` as aliases
   that map onto the three example profile IDs until E migrates the UI.

A is the largest workstream; if its context runs long it hands off at step boundaries (1-2 are enough for B to begin
transport work against the schema; 3 is enough for F; 4-5 gate C and E).

## 5. Principal risks and how the plan handles them

| Risk | Handling |
|---|---|
| Generalizing the profile schema promises more distributed models than backends can run | Backend descriptor declares *validated* limits; profile validation rejects a topology the descriptor does not list; compat labels are computed from descriptor + qualification evidence, never typed by a user (interfaces/backend-capability-v1) |
| API serving blocks on 10+ minute prepare | Prepare is an async job; `/ready` distinguishes loadable vs prepared; completions never carry custom events; bounded wait policy then structured 503 (interfaces/scheduler-admission-v1) |
| Silent substitution via "fallback" | Exact model IDs never rerouted; routing aliases are explicit; response metadata names the profile/model actually used (ADR 0402) |
| LAN API exposes Worker control | API keys are a separate credential family from pairing identities; no scope grants Worker control (ADR 0403) |
| Strata correctness unprovable here | All GPU claims stay `pending`; descriptors carry `qualification` evidence references to `HARDWARE-QUALIFICATION.md` IDs |
| Parallel threads diverge on a contract | Interfaces frozen by ADR 0406 process; schema JSON files are machine-checked in CI (G) |
| Existing users' config breaks | Migration is idempotent, versioned, tested, and writes a backup of the v1 document before first v2 save |

## 6. Honesty carry-over

Nothing in Thread 0 changed code or ran hardware. No schema in `interfaces/` has been exercised by an implementation yet except
the JSON examples, which are validated against the JSON Schemas (see handoff for the exact command and result). Interfaces are
"frozen" in the process sense (changes need an ADR); A is expected to find defects while implementing and should amend through
ADR 0406's lightweight path rather than silently deviating.
