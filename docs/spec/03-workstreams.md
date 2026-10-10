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

