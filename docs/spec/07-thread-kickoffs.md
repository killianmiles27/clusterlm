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

