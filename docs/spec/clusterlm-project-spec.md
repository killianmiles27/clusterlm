# ClusterLM Claude Project Pack

This replaces the single monolithic brief. Set it up like this:

| Layer | Where it goes | Why |
|---|---|---|
| Part 1: Project Instructions | Project "custom instructions" field | Loaded in every thread, so keep it short and stable |
| Part 2: Knowledge files | Upload to Project knowledge | Reference material threads pull on demand, not repeated per prompt |
| Part 3: Thread kickoffs | One new thread per workstream | Each thread gets a focused context and one job |
| Part 4: Handoff protocol | Followed by every thread | Threads can't rely on each other's memory |

Rule of thumb: **the repository is the source of truth, not any chat.** Threads read and write `docs/` in the repo. Project knowledge holds the stable spec, not progress.

---

# PART 1: PROJECT INSTRUCTIONS (paste into the Project)

You are lead architect, integration engineer and final reviewer for **ClusterLM**: *Distributed local LLM inference and model serving.*

## Mission
Turn ClusterLM from a personal three-PC, three-model tool (Fast/Strong/Ultra) into a configurable, general-purpose distributed local-LLM inference platform. Preserve the existing engineering. Do not build another personalized wrapper. Proceed with implementation, not more specification.

## How work is organized
- Each thread in this Project owns **one workstream** (A to G, see knowledge file `03-workstreams.md`). Stay in your workstream. If you need a change outside it, record it in `docs/cross-module-requests.md` instead of making it.
- Before starting any thread: read `docs/status.md`, `docs/interfaces/` and the latest `docs/handoff/*.md` in the repo. Use past-chat search within this Project for decisions and context.
- Interfaces (profile schema, backend capability contract, scheduler API, auth scopes) are defined in Thread 0 **before** parallel threads begin. Do not change a frozen interface without writing an ADR in `docs/adr/` and flagging it to the Architecture thread.
- Before ending any thread: follow the handoff protocol in `05-handoff-protocol.md`.

## Subagent policy (inside a thread / Claude Code session)
Use subagents to reduce usage without reducing quality. Delegate aggressively but not carelessly.
- **Haiku 5.5**: repo exploration and code search, docs/README, simple UI components, config schemas and examples, small isolated tests, boilerplate migrations, research summaries, compatibility inventories.
- **Sonnet 5.5**: substantial independent features, API endpoints and interop tests, model catalog and profile management, UI screens, MCP integrations, packaging/installers, Windows compatibility, backend adapter extensions, independent reviews of finished modules.
- **Opus 5.5** (lead): architecture and interfaces, distributed-runtime risk, concurrency/cancellation/memory ownership, CUDA/Strata decisions, security design, API correctness, cross-module disputes, final integration and sign-off.
- Use separate agents to implement and to review a feature. Never let two agents edit the same files without coordination. Cheaper agents must not make architectural decisions.
- If a requested model is unavailable, use the closest alternative and **report the substitution**.
- Priority is engineering quality per unit of model usage, not minimum cost.

## Non-negotiable guarantees (never weaken for generality)
Host-only permanent model storage; Worker allocations ephemeral, no persistent Worker caches; automatic release when a local user returns (local user always has priority); CPU+RAM+GPU+VRAM as one heterogeneous execution domain; correct layer and sequence-state ownership; Strata-derived Flash-Next backend and llama.cpp backend, both behind capability interfaces; explicit speculative-window commit/abort; authenticated, secure LAN communication and unchanged pairing protocol; token-free middle-stage comms where supported; **no silent model or quantization substitution**; measured vs synthetic vs qualified provenance on every result; offline operation; bounded allocations; Windows process isolation, services and installer support.

## Honesty rules
- Never fake test results, claim CI ran when it didn't, or report an integration as working when only mocks passed.
- Never present synthetic estimates as benchmarks. Never present fixture results as proof of CUDA inference.
- Never label a model "fully compatible" because its GGUF parses. Use: Supported and qualified / Supported, awaiting hardware qualification / Experimental / Unsupported.
- If hardware is unavailable, implement the test and mark it **pending**.
- Do not choose a license. Do not describe the project as open source until I decide.

## Autonomy
You may refactor, add interfaces and tests, extend backends, implement APIs, change the Windows apps, add config formats, improve packaging, write docs, delegate, commit and push to the established branch. Don't force-push over unrelated changes or discard my work. Don't ask about routine decisions the spec already settles. Ask only for genuine user preference, legal authorization, or physical-hardware access.

## Non-goals
Cloud marketplace; accounts, subscriptions, telemetry; permanent Worker replicas; "every GGUF" support; MCP as the inference protocol; remote code execution; Kubernetes-style orchestration; becoming an agent framework; automatic fallback that compromises exact-model execution; auto-downloading huge models to every machine; rewriting working subsystems without concrete benefit.

---

# PART 2: KNOWLEDGE FILES (upload these to Project knowledge)

Split the spec into these files. Contents below are the stable requirements, regrouped from the original brief.

## `01-product-vision.md`

**ClusterLM combines compatible compute resources across multiple computers into a configurable local model-serving platform.**

User flow: install on each computer → choose one Host → pair others as Workers → import supported models to the Host library → create execution profiles → define which model runs when which Workers are available → let ClusterLM measure and optimize placement → chat in the built-in UI or connect external clients via standard APIs → optionally manage via MCP → keep privacy, predictability and automatic resource release.

Useful for one, two or several heterogeneous computers. Support zero Workers as a normal deployment. Keep extensible toward more Workers, with **explicit validated per-backend limits**. The original 4060 Ti / 4070 Laptop / 3060 three-PC setup and its Swift/Flash-Next quantizations become an **importable example profile set**, not product logic. The Host might be a laptop; the same software should work with a 3090, 4090 or other supported NVIDIA GPU.

Naming: product **ClusterLM**, repo `clusterlm`, description *Distributed local LLM inference and model serving.* UI says **Host** and **Worker**; internal Father/Node terms stay (display-name layer only, no mass rename). Use plain terms: Models, Profiles, Workers, Runtime, API, Benchmarks, Diagnostics. No invented futuristic names. No unverified throughput claims.

## `02-requirements-detail.md`

### Execution profiles (replace hard-coded Fast/Strong/Ultra)
First-class, versioned-schema objects with **stable IDs** (renaming never breaks API clients). Users can: name profiles; pick model and exact quantization; pick a compatible backend; set context limits; select specific Workers or resource-based requirements; set min/max Worker counts within backend limits; set GPU/CPU/RAM/VRAM preferences and safety margins; choose preparation and warm-retention behavior; define what happens if a required Worker disappears; choose external-API exposure; rename, duplicate, export, import.

Example (User A): Quick Chat = Host only / Qwen 27B IQ3_S; Coding = Host + Worker A / Swift Flash-Next IQ2_XS; Maximum Quality = Host + A + B / Flash-Next IQ3_S. Another user: Gemma 12B on Host; a 32B on Host + any compatible Worker; a 70B on Host + two Workers, each subject to real backend support and memory.

**Automatic selection** via rules (Host only → A; Host+Worker A → B; two suitable Workers → C; laptop on battery → stop using it; Worker lost → profile's fallback policy). Manual selection with session lock.

**Routing vs substitution:** automatic routing ≠ silent substitution. Use explicit routing aliases with user-configured candidates. A request for an exact model ID never runs another model; if resources are missing, return a structured error unless the request names a routing policy allowing fallback. Always report the actual model used in response metadata.

### Generic hardware and placement
Placement operates on hardware capabilities, real memory budgets, kernel/quantization support, backend capabilities, network performance, user policy, availability, expected preparation time, context needs and the model execution graph. Manual placement (specific Workers; advanced users may set supported layer placement) and automatic placement (measured eligibility, feasible plan, compare latency/prep cost/performance, explain decisions). A busy machine is not available compute. Users never need to know layer numbering.

### Model library
Import local GGUF and split GGUF; scan configured Host directories; identify family, quantization, metadata; record exact hashes; detect supported backends; show size and runtime estimates; report missing dependencies and unsupported tensor formats; detect tool-calling, reasoning, vision when reliably known; select chat templates; show compatibility warnings before profile creation; reuse one canonical artifact across profiles; reject incompatible model/backend pairs before expensive provisioning. No online marketplace for now; offline stays first-class.

### Backend capability contract (versioned)
Each backend declares: model families, GGUF/tensor representations, CPU/GPU support, device types, single- vs cross-machine execution, layer partitioning, CPU/GPU hybrid, context/state requirements, speculative/MTP support, tool/structured-output capabilities, limitations, qualification status.
- **llama.cpp**: broad local compatibility on the Host. Investigate distributed extension; never claim every compatible GGUF is distributable.
- **Strata / Flash-Next**: retained and improved (MTP draft distribution, batched prefill). Don't flatten it into basic RPC.
- **Future backends** (AMD, Intel, CPU-only Workers, other families): adapter architecture stays genuinely extensible; no fake support flags.

### Inference server
Host-hosted **OpenAI-compatible API**; the built-in chat is just one client. Required: `GET /v1/models`, `POST /v1/chat/completions` (streaming + non-streaming), `GET /health`, `GET /ready` (distinguish *loadable* from *prepared*), separate authenticated management endpoints (prepare, progress, availability, diagnostics). Evaluate `/v1/responses` next; evaluate an Anthropic Messages adapter after the core is correct.
Real compatibility means: correct system/developer/user/assistant/tool message handling, model selection, context management, SSE chunk ordering, stable request/response IDs, usage accounting, finish reasons, stop sequences, cancellation on client disconnect, context-length enforcement, accurate errors for unsupported parameters, sampling, structured output only when genuinely supported, SDK conformance tests. Must work with unmodified clients (Pi, OpenCode, Open WebUI, Continue, SillyTavern, scripts, agent frameworks), with **no per-app special cases**.

### Tool calling
Declared tools, names, JSON Schema args, tool-choice policies where supported, multiple calls when supported, stable call IDs, streaming deltas, tool-result messages, follow-up turns, correct finish reasons. Use the model's real chat template and parser (not "please output JSON"). The server **never executes tools**; the external agent owns tools and permissions.
Acceptance: connect Pi and OpenCode via documented custom-provider config; run a real multi-turn task (inspect file → tool → result → edit → tool → run test → interpret → finish); verify exact request/response structures. Synthetic backend tests transport only; measure real tool-following quality separately. Ship version-appropriate config examples for both; no custom Pi extension if a standard endpoint works.

### MCP (separate from serving)
- **A. ClusterLM as MCP server** (current official spec; stdio first, optionally authenticated Streamable HTTP). Read-only first: `clusterlm_list_models`, `_list_profiles`, `_list_nodes`, `_get_status`, `_get_capabilities`, `_get_benchmarks`, `_get_active_profile`. Privileged actions (prepare profile, request change, release session, pause Worker) need explicit scoped authorization. No MCP connection may interrupt workloads, take over Workers or alter pairing without permission. No generic shell tool.
- **B. ClusterLM as MCP client** (optional, opt-in, Host-side only) for the built-in chat: per-server permissions, visible tool calls, approval for side effects. Workers never become remote-execution agents.
- **C. External agents** use the serving API and manage their own MCP servers. Document this distinction.

### Desktop UI
Keep the native architecture; no cosmetic rewrite. Navigation: **Chat, Models, Profiles, Machines, Connections, Performance, Settings** (merge pages where sensible). First-run wizard: Host or Worker → explain → detect capabilities → pair or skip → import/choose model → suggest profile → optional API access → validation → open Chat. No GGUF or layer knowledge required. Quality bar: progress and preparation estimates, actionable errors, human machine names, no stale or fabricated metrics, never show "ready" falsely, high-DPI, keyboard nav, accessibility, Unicode fonts, dark/light, reliable persistence, diagnostics without private conversations. Worker UI stays minimal.

### Resource management
Idle-only and AC-only participation, battery thresholds, max RAM/VRAM, CPU/thread limits, temperature/power limits when supported, temporary storage cap, Pause/Resume, schedules, optional retention while continuously idle. Local user always wins. Automatic preparation is opt-in or profile-driven, never filling every paired PC. No persistent Worker weight caches. Profiles may reuse objects only during a continuous lease and only where exact representations and ownership rules allow. Temporary material cleaned up per existing release guarantees. API requests never bypass these protections.

### Scheduling
Preserve one active generation as the default concurrency limit unless the backend supports more. Add a bounded queue, request IDs, client fairness, cancellation, queue-time limits, optional priorities, resource admission, per-client session isolation, monopoly protection, safe Worker-loss handling, clear preparation states, warm-session retention/release policy. No cross-client cache/KV mixing. A model swap never silently kills another client's active request. Long preparations are async jobs with progress via management APIs; standard completion streams never carry proprietary events (wait within bounded policy or return a clear availability error). Keep it simple.

### Security
Localhost by default; LAN only by explicit action; auth for LAN clients; per-client keys with rotation and revocation; owner-only secret storage; TLS for remote access; inference and admin permissions separate; request-size limits; rate limits; conservative CORS; no accidental internet exposure; redacted logs; no raw prompt/response logging by default; no cross-client conversation access; no arbitrary shell via management or MCP. Don't weaken pairing; an API token never grants Worker control. Remote access beyond the LAN needs its own explicit design. No cloud account; no telemetry or crash upload without consent.

### Lifecycle and operations
Profile validation; dry-run placement (machines, memory, preparation bytes, bottlenecks; never synthetic numbers as benchmarks); compatibility reports; profile export/import without pairing identities, credentials or private paths; config/library-metadata backup separate from weights; hardware/driver recalibration with stale-profile detection; protocol version negotiation with safe refusal; reproducible diagnostics report (hashes, backend versions, hardware, placement, network timing, errors, no prompts); readiness states (installed, compatible, loadable, preparing, ready, busy, unavailable); API observability without leaking prompts or tool arguments. Engineered as cohesive features, not toggles.

### Packaging
A user must not install a "complete" product and discover GPU inference isn't included. Variants as needed: CPU, NVIDIA CUDA, development/qualification. Prefer one installer with runtime dependency detection and clear feature availability. Bundle redistributables legally. No compilers, Git, Python or CUDA toolkit required. Never advertise a profile whose backend isn't present. Windows primary; keep core portable without a cross-platform rewrite.

### Open-source readiness
README, install docs, platform matrix, model/backend matrix, architecture overview, contributing, issue templates, security policy, changelog, versioning and release procedure, reproducible builds, CI, third-party license attribution, binary provenance. **Do not pick a license.** Explain MIT vs Apache-2.0 and dependency compatibility, then leave the decision to me.

### Flash-Next / Strata (core asset)
Preserve layer-domain execution, domain-local CPU expert pools, hybrid expert residency, distributed MTP, token-free Worker stages, ephemeral selected-tensor provisioning, numerical-correctness requirements, detailed instrumentation. Investigate: draft distribution/acceptance, faster batched prefill, exact on-device allocation accounting, real GGUF and quantization compatibility, CUDA correctness, Windows/WDDM constraints. Don't declare the real backend complete on fixtures. The original three-PC Ultra setup stays a supported example and performance goal.

## `03-workstreams.md`

| ID | Workstream | Scope | Primary tier |
|---|---|---|---|
| 0 | Architecture & Interfaces | Frozen contracts, ADRs, dependency map, status ledger | Opus |
| A | Foundation | Generalized models, execution profiles, Worker capabilities, dynamic topology, **Fast/Strong/Ultra migration** | Opus design, Sonnet impl |
| B | Serving interfaces | OpenAI API, discovery, streaming, tool calling, client compat | Sonnet, Opus review |
| C | Cluster management | Profile scheduling, availability, resource policies, request coordination, fallback, cancellation | Opus for runtime, Sonnet for rest |
| D | MCP | Read-only server, scoped admin ops, optional client | Sonnet |
| E | UX | Model library, profile editor, Machines, Connections, wizard, diagnostics | Sonnet, Haiku for components |
| F | Runtime & compat | llama.cpp breadth, capability detection, Strata improvements, packaging/backend availability | Opus for Strata/CUDA, Sonnet otherwise |
| G | Verification & docs | Unit/integration, external-client smoke tests, security tests, CI, installer tests, docs | Sonnet, Haiku for docs |

Dependencies: 0 → A → (B, C, F in parallel) → (D, E) → G continuously. Threads B, C and F must not start before A's profile schema and the backend capability contract are frozen.

## `04-test-matrix.md`

- **Single machine:** import a generic llama.cpp-supported model, select via GUI, generate on the Host.
- **Custom distributed profile:** pair a compatible Worker, assign, provision selected pieces, infer, clean up.
- **Topology logic:** Host only; +1 Worker; +2 Workers; a Worker goes busy; a different eligible Worker replaces it where supported; insufficient memory; missing model; unsupported backend; profile conflicts; context growth forcing re-plan.
- **API clients:** actual Pi and OpenCode configs where executable; streaming, normal chat, tool calls, tool results, cancellation, errors, long context; SDK conformance.
- **MCP:** discovery, schemas, permissions, read-only access, authorization refusal.
- **Failures:** preserve and extend the existing fault-injection suite.
- **Windows:** GUI, service behavior, process supervision, pairing, installers on Windows CI where possible.
- **GPU qualification:** never substitute synthetic results; mark pending if no hardware.

## `05-handoff-protocol.md`

See Part 4 below; upload it as a separate file so every thread can cite it.

## `06-acceptance-criteria.md`

Successful when: no hard-coded three profiles; users create named profiles and define the Workers/hardware conditions that enable them; the Host runs a broader set of genuinely supported GGUFs; distributed support is advertised only for validated backends; the UI can configure and manage a cluster; an OpenAI-compatible endpoint exists; Pi and OpenCode work through documented, tested configs; tool-calling semantics are implemented and tested; optional permissioned MCP exists; multi-client scheduling is safe and predictable; idle/cleanup/privacy guarantees are intact; capabilities are discoverable and errors understandable; installation needs no repo knowledge; Fast/Strong/Ultra configs migrate; all feasible tests and CI pass; hardware-dependent features stay unqualified until measured. A complete real Strata CUDA cluster needs the physical GPUs and artifacts. That dependency is no excuse to leave the rest unfinished.

---

# PART 3: THREAD KICKOFF PROMPTS

Create threads in this order. Name each thread exactly as shown so past-chat search finds it.

### Thread: `00 · Architecture & Interfaces` (run first, alone)
> Review the current repo, CI, docs and implementation. Produce: (1) a dependency map and workstream plan in `docs/plan.md`; (2) frozen interface drafts in `docs/interfaces/`: profile schema v1 (with migration mapping from Fast/Strong/Ultra), backend capability contract v1, scheduler/admission API, API auth scopes, readiness state machine; (3) `docs/status.md` as the requirement ledger using these states: Implemented and tested / Implemented but hardware-unqualified / Partially implemented / Not implemented / Deferred with rationale. Record decisions as ADRs. Then write the handoff note. Don't implement features yet.

### Thread: `A · Foundation`
> Read the 00 handoff and interfaces. Implement generalized model records, execution profiles, Worker capability descriptors, dynamic topology, and the config migration from Fast/Strong/Ultra, with a test proving existing configs survive and the old three profiles ship as an importable example set. Use Sonnet subagents for implementation and a separate Sonnet reviewer; Opus reviews the schema and migration. Handoff when done.

### Thread: `B · Serving API`
> After A is merged. Implement the OpenAI-compatible server (models, chat completions with SSE, health, ready, management endpoints), tool calling with template-correct parsing, usage, finish reasons, cancellation, error semantics. Add SDK conformance tests, a synthetic backend for transport tests, and Pi and OpenCode config examples with a scripted multi-turn agentic test. Mark real-model tool quality as measured separately.

### Thread: `C · Cluster Management`
> After A. Implement the scheduler (bounded queue, fairness, cancellation, admission, session isolation, warm retention), profile auto-selection rules, availability and Worker-loss handling, resource policies (idle, AC, battery, RAM/VRAM caps, schedules, pause). Opus owns concurrency, cancellation and memory-ownership design; extend the fault-injection suite.

### Thread: `D · MCP`
> After B and C expose stable status APIs. Implement the MCP server (stdio first, current spec) with read-only tools, then scoped privileged tools behind explicit authorization. Optional opt-in Host-side MCP client for the built-in chat. Test discovery, schemas, permission refusal. Document that agents use the serving API for inference and manage their own MCP servers.

### Thread: `E · Desktop UX`
> After A. Build Models, Profiles, Machines, Connections, Performance and Settings views, and the first-run wizard, on the existing UI framework. Haiku for components, Sonnet for workflows. Include dry-run placement display, compatibility reports, copyable client setup snippets, dark/light, DPI, keyboard and accessibility checks.

### Thread: `F · Runtime & Compatibility`
> After A. (1) llama.cpp breadth on the Host plus capability detection and honest compatibility statuses; investigate distributed extension. (2) Strata: MTP draft distribution, batched prefill, allocation accounting, GGUF/quant compatibility; Opus owns CUDA/Strata decisions. (3) Packaging: CPU, CUDA and dev variants, runtime dependency detection, never advertising an absent backend. Mark all real-GPU work pending when hardware is absent.

### Thread: `G · Verification & Docs`
> Continuous. Build CI, the test matrix in `04-test-matrix.md`, security tests, installer tests, and documentation: README, install, platform and model/backend matrices, architecture, contributing, issue templates, security policy, changelog, release procedure, third-party attributions, provenance, and a license comparison (MIT vs Apache-2.0, dependency compatibility) **without choosing**. Include an ordinary-user walkthrough: install, import a model, create a profile, connect Workers, connect an external coding agent.

### Thread: `Z · Final Review` (last)
> Review every requirement in the Project knowledge against the repo and `docs/status.md`. Classify each as Implemented and tested / Implemented but hardware-unqualified / Partially implemented / Not implemented / Deferred with rationale. Continue any feasible remaining work. Produce the final deliverable: branch and commit; architectural summary; new profile/model capabilities; model/backend matrix; API docs; working Pi and OpenCode configs; MCP setup; Windows install steps; tests and CI results (real, not claimed); migration details; known limitations; hardware qualification requirements; my next actions once the physical machines are available.

---

# PART 4: HANDOFF PROTOCOL (every thread)

Threads in a Project can search each other via past-chat search, but that is a convenience, not a contract. Durable state lives in the repo.

**On start:** read `docs/status.md`, `docs/interfaces/`, the latest relevant `docs/handoff/*.md`, and `docs/cross-module-requests.md`. State in one line what you are about to build.

**On finish (or when context runs long), write `docs/handoff/<workstream>-<YYYY-MM-DD>.md` containing:**
1. What was implemented, with commits and files.
2. What was tested, with real results, and what was only mocked.
3. What is pending hardware qualification.
4. Interface changes made (with ADR links) and requests for other workstreams.
5. Subagent usage and any model substitutions.
6. Known issues and the next concrete step.

Then update `docs/status.md`, commit, push, and end with a short summary for me.

**Context hygiene:** if a thread is getting long, stop, write the handoff, and tell me to start a fresh thread of the same name suffixed `(2)`. Don't carry on with degraded context.

**Conflict rule:** two threads must not edit the same files concurrently. If a thread needs a file owned by another workstream, it files a cross-module request instead.

---

## Notes on what changed from the original

- The 23-section brief is now split by volatility: standing rules (Instructions), stable requirements (Knowledge), per-job tasks (Threads). The original repeated everything in one prompt.
- Subagents stay (inside a Claude Code session), but cross-workstream parallelism now happens through **threads**, with an explicit Thread 0 to freeze interfaces first. This fixes the original's main risk of parallel agents making incompatible architecture decisions.
- Handoffs go through the repo because threads in a Project don't share live context. Past-chat search within the Project helps recall but isn't relied on.
- Section 21's final audit became the dedicated `Z · Final Review` thread.
- Content condensed in wording only; every requirement from the original is kept. I didn't verify exact Project limits (knowledge size, instruction length), so check those when you upload.
