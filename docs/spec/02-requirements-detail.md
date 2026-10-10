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

