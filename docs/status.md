# Requirement ledger

The single place that says where every requirement of the [spec](spec/README.md) stands. Rows are edited by the owning
workstream only (column **WS**); Thread Z reconciles at the end. Update the row, the evidence and the date in the same commit
as the work.

**States:** `Implemented and tested` · `Implemented but hardware-unqualified` · `Partially implemented` · `Not implemented` ·
`Deferred with rationale`.

**Basis of the initial states (2026-10-10, Thread 0).** Thread 0 read the code and docs; it did **not** build the project, run
tests or run CI on this branch. "Implemented and tested" below therefore means *the repository's own documents
([DEVELOPMENT-STATUS.md](DEVELOPMENT-STATUS.md), per-feature docs, ADRs) report it implemented with a named test or CI job*;
each owning thread must re-confirm with a real run before relying on or changing it. Nothing is a measurement of the target
machines; every hardware item stays pending ([HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md), 41 entries).

## 1. Non-negotiable guarantees

| ID | Guarantee | WS | State | Evidence / note |
|---|---|---|---|---|
| G-01 | Host-only permanent model storage | A,F | Implemented and tested | provisioning docs/tests: no persistent cache, census 0 B after every fault scenario. Library (A) must keep it true for imported models |
| G-02 | Ephemeral Worker allocations, no persistent Worker caches | C,F | Implemented and tested | `docs/provisioning-lifecycle.md`; fault suite; HQ-REL-01/HQ-STORE-01 pending on hardware |
| G-03 | Automatic release when a local user returns | C | Implemented but hardware-unqualified | service/helper/WTS paths tested via mocks on Linux and type-checked on Windows; real sessions: HQ-WIN-01..04 |
| G-04 | CPU+RAM+GPU+VRAM as one heterogeneous domain | F | Implemented but hardware-unqualified | reference backend accounts GPU targets but executes on CPU; CUDA engine never run (HQ-GPU-01..03) |
| G-05 | Correct layer and sequence-state ownership | F,C | Implemented and tested | `ExecutionDomain` contract, bitwise-identical logits for every split (reference backend). Scheduler ownership rules now frozen in scheduler-admission-v1 §1 |
| G-06 | Strata Flash-Next and llama.cpp behind capability interfaces | A,F | Partially implemented | Real descriptors for llama-local and strata-hybrid ship embedded (ADR 0408, checked by scripts/check_backend_descriptors.py); BackendDescriptor parsing/registry is A (CMR-0010) (F, 2026-10-10) |
| G-07 | Explicit speculative-window commit/abort | F | Implemented but hardware-unqualified | `WindowLedger` tested for every accept length 0..q; Strata abort on GPU: HQ-P0D-01 |
| G-08 | Authenticated secure LAN; pairing unchanged | B,C | Partially implemented | Host↔Worker mTLS 1.3 + SPAKE2 pairing implemented and tested; client-facing API auth/TLS not implemented (auth-scopes-v1) |
| G-09 | Token-free middle-stage communication where supported | F | Implemented but hardware-unqualified | reference backend tested; privacy/traffic tests; Strata on GPU pending |
| G-10 | No silent model or quantization substitution | A,B,C | Partially implemented | tiers: FallbackEvent names both models; no requantization (manifest refuses conversion). API-level exact-model rule and response metadata not implemented (ADR 0402) |
| G-11 | measured/synthetic/qualified provenance on every result | A,C,E,G | Partially implemented | placement/bench carry `Provenance`; must extend to dry-run, API metadata, UI, MCP |
| G-12 | Offline operation | G | Partially implemented | no network dependency known in product code; no explicit offline test yet |
| G-13 | Bounded allocations | all | Implemented and tested | ADR 0233; 13 fuzz targets replayed in CI; every new parser must state limits (interfaces/README) |
| G-14 | Windows process isolation, services, installer | E,F,G | Implemented but hardware-unqualified | Job Objects, LocalService, MSIs smoke-tested on Windows runner (DEVELOPMENT-STATUS); HQ-WIN-*, HQ-INSTALL-01 pending |

## 2. Product requirements

| ID | Requirement | WS | State | Evidence / note |
|---|---|---|---|---|
| R-01 | Execution profiles: versioned schema, stable ids, create/rename/duplicate/export/import | 0,A | Partially implemented | schema + examples frozen (interfaces/profile-schema-v1); **no implementation** |
| R-02 | Profile fields: model+quant, backend, context, worker selection/resource requirements, min/max workers, margins, preparation/retention, worker-loss policy, API exposure | A | Not implemented | schema covers all; code still tier-based |
| R-03 | Fast/Strong/Ultra migrate to an importable example set; no hard-coded three tiers | A | Not implemented | mapping specified (profile-schema-v1 §5); `orchestrator/catalog` still hard-codes three tiers and roles |
| R-04 | Automatic selection rules + manual selection with session lock | A,C | Not implemented | routing alias schema frozen; evaluation is C |
| R-05 | Routing vs substitution: exact ids never re-routed; aliases explicit; actual model reported | 0,B | Not implemented | ADR 0402, scheduler-admission-v1 §3, §6 |
| R-06 | Generic placement on real budgets, policy, availability, context | A,C | Partially implemented | `placement::search_placements` generic over measured inputs, Pareto, replan, workload sweep; slot/selector-based inputs and "busy ≠ available" integration pending |
| R-07 | Manual placement for advanced users; automatic placement with explanations | A,E | Partially implemented | search + frontier exist; `placement.manual` schema frozen; explanation surface (dry-run report) not implemented |
| R-08 | Model library: import GGUF/split, scan dirs, family/quant/metadata, exact hashes, backend detection, size/runtime estimates, missing deps, compat warnings, one canonical artifact across profiles, early rejection | A,F | Partially implemented | GGUF v2/v3 reader, manifest builder with hashes, plan-scoped manifests exist; library records, scanning, family/quant identification UI-facing, compat report not implemented |
| R-09 | Detect tool-calling / reasoning / vision when reliably known; select chat templates | B,F | Not implemented | tokenizer supports ChatML + qwen2/qwen35 pre-tokenizers only; no Jinja engine (known issue 5) |
| R-10 | Backend capability contract (versioned) | 0,F | Partially implemented | Descriptors + evidence record + checker done; registry/check_model in C++ pending A (F, 2026-10-10) |
| R-11 | llama.cpp broad Host compatibility; investigate distributed extension; never claim all GGUFs distributable | F | Partially implemented | Real CPU run: 8 architecture layouts incl. MoE and recurrent, 28 tensor types, decode equals llama.cpp (test_llama_architectures, test_llama_tensor_types); 153-name arch table. CUDA, real checkpoints, hybrid SSM, RPC investigation unchanged and pending (F, 2026-10-10) |
| R-12 | Strata/Flash-Next retained and improved (MTP distribution, batched prefill) | F | Implemented but hardware-unqualified | Retained; expert type admission added (no std::exit path); other improvements need a GPU, see handoff F (F, 2026-10-10) |
| R-13 | Future backend adapter extensibility without fake flags | F | Partially implemented | Descriptor rule enforced by checker; adding a backend = descriptor + adapter + factory + ADR (F, 2026-10-10) |
| R-14 | OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions` (stream + non-stream) | B | Not implemented | no network server exists |
| R-15 | `GET /health`, `GET /ready` (loadable vs prepared), authenticated management endpoints | B,C | Not implemented | specified: readiness-state-machine-v1 §4, scheduler-admission-v1 §8 |
| R-16 | `/v1/responses` evaluation; Anthropic Messages adapter evaluation | B | Deferred with rationale | spec says evaluate after core is correct; revisit after B's conformance suite |
| R-17 | API correctness: message roles, SSE ordering, ids, usage, finish reasons, stop sequences, cancellation, context enforcement, errors, SDK conformance | B,G | Not implemented | |
| R-18 | Works with unmodified clients (Pi, OpenCode, Open WebUI, Continue, SillyTavern) without special cases | B,G | Not implemented | |
| R-19 | Tool calling (declared tools, ids, streaming deltas, results, choices) via real chat template + parser; server never executes tools | B | Not implemented | |
| R-20 | Pi and OpenCode documented configs + scripted multi-turn agentic test; real-model quality measured separately | B,G | Not implemented | executable-client run depends on availability; mark pending if not runnable |
| R-21 | MCP server (stdio first), read-only tools then scoped privileged tools | D | Not implemented | scopes frozen (auth-scopes-v1 §4) |
| R-22 | Optional opt-in Host-side MCP client for built-in chat | D | Not implemented | |
| R-23 | Document API vs MCP distinction | D,G | Not implemented | |
| R-24 | UI nav: Chat, Models, Profiles, Machines, Connections, Performance, Settings | E | Partially implemented | Dear ImGui Host/Worker UI with tiers, readiness, pairing, diagnostics exists; new views not implemented |
| R-25 | First-run wizard (Host/Worker → … → Chat) | E | Not implemented | |
| R-26 | UI quality bar (progress, actionable errors, no fabricated metrics, never false "ready", DPI, keyboard, a11y, Unicode, dark/light, persistence) | E,G | Partially implemented | honest readiness, redacted diagnostics exist; Latin-only bundled font, no screen-reader support (known issue 9) |
| R-27 | Resource management: idle-only, AC-only, battery, max RAM/VRAM, threads, storage cap, pause/resume | C | Partially implemented | NodeSettings has idle/AC/storage/paused/RAM/VRAM caps (threads recorded, not enforced); battery thresholds, temperature/power limits, schedules, continuous-idle retention: Not implemented |
| R-28 | Local user always wins; API requests never bypass protections | C,B | Implemented but hardware-unqualified | release on activity exists (see G-03); API path not implemented, rule frozen (scheduler §5) |
| R-29 | Automatic preparation opt-in/profile-driven | A,C | Not implemented | `lifecycle.prepare_when_available` frozen |
| R-30 | Scheduler: one active generation default, bounded queue, ids, fairness, cancellation, queue limits, priorities, admission, per-client isolation, monopoly protection, Worker-loss handling, warm retention | C | Partially implemented | `FatherService` has one-job gate, cancel, fallback policy, `keep_ready`; queue/fairness/clients/admission not implemented |
| R-31 | Async prepare jobs with progress via management APIs; no custom events in completion streams | C,B | Partially implemented | `PrepareProgress/PrepareDetail` + IPC events exist; HTTP job API not implemented |
| R-32 | Security: localhost default, LAN opt-in, per-client keys with rotation/revocation, owner-only secrets, TLS, separate scopes, size/rate limits, CORS, redacted logs, no prompt logging | B,G | Partially implemented | owner-only key/ACL storage, redacting diagnostics, no-prompt-logging tests exist; everything API-facing is Not implemented |
| R-33 | Pairing not weakened; API token never grants Worker control | B,G | Partially implemented | pairing intact; negative test for API→Worker path not yet writable |
| R-34 | No cloud account/telemetry/crash upload | all | Implemented and tested | none exists; keep it (non-goal) |
| R-35 | Profile validation; dry-run placement (never synthetic as benchmark); compatibility reports | A,E | Not implemented | `placement-validate` exists for predicted-vs-measured; profile dry-run is new |
| R-36 | Profile export/import without identities, credentials or private paths | A | Not implemented | rules frozen (profile-schema-v1 §7) |
| R-37 | Config/library-metadata backup separate from weights | A,E | Not implemented | settings store has corruption recovery; user backup/restore feature absent |
| R-38 | Hardware/driver recalibration; stale-profile detection | A,C,F | Partially implemented | `profile.hpp` provenance + merge; staleness rule frozen (readiness §2); detection of driver change not implemented |
| R-39 | Protocol version negotiation with safe refusal | F | Partially implemented | `kProtocolVersion = 1`, frame version; extension caveat in known issue 6; negotiation across versions not verified by Thread 0 |
| R-40 | Reproducible diagnostics report (hashes, versions, hardware, placement, network timing, errors, no prompts) | C,E,G | Partially implemented | diagnostics bundle + log sink (ADR 0232), counts-only snapshots; must add backend versions/placement per new surfaces |
| R-41 | Readiness states (installed … unavailable) everywhere; never falsely "ready" | A,C | Partially implemented | 4-state tier readiness implemented and tested; 7-state machine frozen, not implemented |
| R-42 | API observability without leaking prompts/tool args | B,G | Not implemented | |
| R-43 | Packaging: CPU / CUDA / dev variants, runtime detection, never advertise an absent backend | F | Partially implemented | two MSIs, static runtime, honest signing; no CUDA runtime bundled, no variants, backend-presence gating via descriptors not implemented |
| R-44 | Portable core, Windows primary | all | Implemented and tested | Linux gcc/clang + Windows MSVC CI jobs (per DEVELOPMENT-STATUS) |
| R-45 | Open-source readiness docs (README, install, matrices, architecture, contributing, issue templates, security policy, changelog, release procedure, CI, attribution, provenance) | G | Partially implemented | README, architecture, threat model, CI, packaging licenses folder exist; others absent |
| R-46 | License comparison (MIT vs Apache-2.0) **without choosing**; no "open source" claim | G | Not implemented | owner decision; do not add a LICENSE file |
| R-47 | Strata: layer-domain exec, domain-local CPU expert pools, hybrid expert residency, distributed MTP, token-free stages, ephemeral selected-tensor provisioning, instrumentation | F | Implemented but hardware-unqualified | per DEVELOPMENT-STATUS; GPU pending |
| R-48 | Strata investigations: draft distribution/acceptance, batched prefill, exact on-device accounting, real GGUF/quant compat, CUDA correctness, WDDM | F | Not implemented | all need GPU/real artifacts (HQ-GPU-*, HQ-NUM-01, HQ-MTP-*) |
| R-49 | Original three-PC Ultra setup remains a supported example and performance goal | A,F | Implemented but hardware-unqualified | the 20 tok/s target stays `pending_qualification` (HQ-PERF-01) |
| R-50 | Naming: UI says Host/Worker; internals keep Father/Node | E | Not implemented | UI currently says Father/Node; display-name layer only |

## 3. Test matrix (spec 04)

| ID | Item | WS | State | Note |
|---|---|---|---|---|
| T-01 | Single machine: import generic llama.cpp model, select in GUI, generate | A,E,F,G | Not implemented | needs library + profiles |
| T-02 | Custom distributed profile: pair, assign, provision selected pieces, infer, clean up | A,C,G | Partially implemented | runtime integration tests exist on reference backend; custom-profile path new |
| T-03 | Topology logic cases (host only, +1, +2 workers, busy worker, replacement, insufficient memory, missing model, unsupported backend, conflicts, context growth re-plan) | A,C,G | Partially implemented | replan/recosting, busy handling and faults exist per tier; generic-topology versions new |
| T-04 | API clients (Pi/OpenCode, streaming, tools, cancel, errors, long context, SDK conformance) | B,G | Not implemented | |
| T-05 | MCP discovery, schemas, permissions, refusal | D,G | Not implemented | |
| T-06 | Failure suite preserved and extended | C,G | Implemented and tested | existing fault-injection + supervised faults |
| T-07 | Windows GUI/service/process supervision/pairing/installers on Windows CI | E,G | Partially implemented | `windows-msvc`, `windows-packaging` jobs; interactive GUI/pairing over LAN: HQ-UI-01, HQ-PAIR-01 |
| T-08 | GPU qualification (never synthetic) | F | Implemented but hardware-unqualified | experiments + commands written; all pending |

## 4. Acceptance (spec 06) and CI

| Item | State |
|---|---|
| All feasible tests and CI pass | CI workflow exists (9 jobs). **Not verified on this branch by Thread 0** (docs only changed). |
| Hardware-dependent features stay unqualified until measured | Holding: no qualification result exists. |
| Complete real Strata CUDA cluster | Needs physical GPUs and artifacts; no excuse to leave the rest unfinished. |

## 5. Workstream status

| WS | Name | State | Latest handoff |
|---|---|---|---|
| 0 | Architecture & Interfaces | Done 2026-10-10 (docs only) | [handoff/0-architecture-2026-10-10.md](handoff/0-architecture-2026-10-10.md) |
| A | Foundation | In progress: interfaces reviewed by Opus and frozen v1.1 (ADR 0407); library module landed; profiles, descriptors, topology, migration next | |
| B | Serving API | Blocked on A (schema impl + capability types merged) | |
| C | Cluster management | Blocked on A | |
| D | MCP | Blocked on B, C status APIs | |
| E | Desktop UX | Blocked on A | |
| F | Runtime & compatibility | Blocked on A (capability types) | |
| G | Verification & docs | May start CI/doc scaffolding now on files it owns; full suites follow A | |
| Z | Final review | Last | |
