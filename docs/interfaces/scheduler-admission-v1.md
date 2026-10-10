# Scheduler / admission API v1 (and serving-layer rules)

Frozen v1.1: ADR 0405, revised by ADR 0407. Implemented by workstream C (scheduler, policies) with B (HTTP surface). Replaces
the single-job gate of `father::FatherService` ("one job at a time") with an explicit scheduler, keeping that service's
Coordinator-driving internals.

## 1. Ownership (who may touch what)

| Resource | Single owner | Others |
|---|---|---|
| Domain sequence state (KV / recurrent / indexer) and speculative windows | the **Coordinator session** for that request (via `ExecutionDomain` + `WindowLedger`) | scheduler never calls a domain; it calls the Coordinator |
| Leases / plan on Workers | **Coordinator** (acquire, release, epoch) | scheduler requests `prepare`/`release` jobs; Workers refuse anything else |
| Token IDs and text | Host process only (serving layer + tokenizer) | never in scheduler queue records beyond counts; never logged |
| Queue, tickets, client records, jobs | **Scheduler** | serving layer enqueues and cancels by ticket |
| Profile/alias definitions | profiles store (A) | scheduler reads immutable snapshots `id@revision`; an edit applies to tickets admitted after it (§3 dequeue re-check) |
| Host resource policy (what Workers may be used) | Host settings + Node reports (C) | local user activity on a Worker is authoritative and immediate |

Rules that follow: one request owns one Coordinator session; a session's state is released on finish/cancel/abort unless warm
retention applies (§4); no state, cache or prefix is shared between clients; the scheduler holds no pointer into a domain.
A prepared plan records the profile `id@revision` and context it was built for; it serves only tickets whose pinned revision has
the same model identity, backend, topology and options (an edit to any of those makes the plan stale, never silently reused).

## 2. Entities

`Client` (API key id, or `local-ui`, or `mcp:<key id>`; carries limits) → `Request` (`req_*`; tokens counted not stored; target
`api_model_id`; resolved `profile_id` + `profile_revision`; state) → `Ticket` (queue position, enqueue time, deadline).

Ticket states: `queued → starting → running → finishing → done`, plus `cancelling`. Every transition is a single atomic step
under the scheduler lock; `cancel` is valid in every state and idempotent (§4 Cancellation).

`Job` (`job_*`; kinds `prepare`, `release`; `state ∈ queued | running | succeeded | failed | cancelled`; while `running`,
`phase` = the Coordinator's `PreparePhase` string (`father-domains | provisioning | node-preparing | node-ready`) with progress
as in `PrepareDetail`). The v1.0 job states mixed phases with outcomes and used `ready`, which is a profile state; v1.1
separates them. At most **one** job runs at a time (default; bounded); others wait `queued` in FIFO order. Finished job
records are kept for at most 64 jobs or 1 h, whichever is smaller. `Lease` is the Coordinator's existing notion.

## 3. Admission: a pure function, evaluated at enqueue and re-evaluated at dequeue

`admit(request, profile@revision, readiness, queue_state, client_limits, policy) → Admit | Queue{position} | Reject{code,details}`

Checks, in order (first failure wins; every rejection is structured and names the reason; codes in §7):

| # | Check | Reject code |
|---|---|---|
| 1 | Auth and scope (`inference`), key not revoked/expired, model visible to key (auth-scopes-v1 §3) | `invalid_api_key` / `insufficient_scope` / `model_not_found` |
| 2 | Body size, parameters supported by backend + template (`n>1`, logprobs, response_format…) | `request_too_large` / `unsupported_parameter` |
| 3 | Context: prompt tokens + completion budget ≤ profile `context.max_tokens` (completion budget = request `max_tokens`/`max_completion_tokens`, else the rest of the context, capped by policy `max_completion_tokens_cap`) | `context_length_exceeded` |
| 4 | Client rate / concurrency / per-client queue share | `rate_limit_exceeded` (with `Retry-After`) |
| 5 | Readiness of the target profile **for the required context class** (smallest offered context ≥ need) | `model_not_ready` (details: state, reasons, `job_id?`), `workers_unavailable`, `insufficient_memory` |
| 6 | Resource policy: required Workers allowed now (idle/AC/battery/schedule/pause) | `workers_unavailable` |
| 7 | Global queue capacity | `queue_full` |
| 8 | Server state | `shutting_down` |

Step 3 is checked against the *profile*, not the currently prepared plan: a request that fits the profile but not the prepared
plan's context makes the profile `loadable` for that context class, i.e. it needs a re-prepare at the larger context (same model,
same rules as a swap in §4). It is never truncated and never silently served by a smaller plan.

**Dequeue re-check** (`queued → starting`): the ticket is rejected, never re-routed, if the key was revoked or lost the model,
the profile was deleted, its pinned revision's model identity/backend changed, `exposure.api` was turned off, or (alias) the
recorded candidate is no longer viable. A revision change that touches none of these (e.g. a rename) does not affect the ticket.

Exact model ids are never re-routed (ADR 0402). Only a routing alias evaluates its candidates (step 5 is applied per candidate
over the candidates visible to the key; the chosen one is *recorded in the ticket* at enqueue; if it is no longer viable at
dequeue the request is rejected with the reason, not rerouted).

## 4. Concurrency, queue, fairness, retention

* **Concurrency** per prepared plan = `min(descriptor.state.max_sessions_per_domain, profile/plan limit)`, and > 1 only when
  the descriptor also declares `per_client_isolation: true`; default and current maximum for all shipped backends is
  **1 active generation**. Raising it requires the descriptor to say so, tests, and an ADR.
* **Prepared plans** are bounded: at most `max_prepared_profiles` (default **1**, as today) are prepared at once; more only if
  Host and Worker memory admission admits all of them simultaneously. Preparing another profile when the bound is reached is a
  *swap*.
* **Queue** is bounded (`max_queue_depth` default 32, `max_queue_per_client` default 8, `queue_timeout` default 120 s, all
  configurable within hard caps 256 / 64 / 600 s). Order: per-priority FIFO across clients with **round-robin between
  clients** (no client can monopolize by submitting many requests). Priority classes (optional, off by default): `interactive`
  (Host built-in chat) above `api`; within a class, fairness as above. Monopoly protection: `max_completion_tokens_cap`
  (default: profile `max_tokens`) bounds every request, and a client that has held the generation slot for more than
  `max_slot_seconds` (default 300) in the last 10 minutes is placed last in its round-robin turn.
* **Model swap** (the head of the queue needs a profile/context that is not prepared): dispatch stays in queue order (no
  reordering to avoid swaps, so neither side starves). The swap waits for the active request to finish; it never kills it. It
  is then: release the old plan → wait until the Coordinator confirms every lease released and the Host-side domains are
  destroyed → start the prepare job. Allocations of two plans never overlap on a machine unless memory admission admitted both.
* **Prepare is an async job.** Started by: an explicit `prepare` call (scope `profiles:prepare`), a request to a `loadable`
  profile whose `lifecycle.preparation` is `on-demand`/`keep-ready`, a request routed through an alias with `allow_prepare`, or
  `prepare_when_available`. Never started for `manual` (requests then get `model_not_ready`). `POST :prepare` is idempotent per
  profile while a job for it is queued or running. A job is shared by every request waiting on it and is **not** cancelled when a
  waiting request is cancelled or times out; it ends by success, failure, explicit job cancel, local-user return/policy, or
  shutdown. API requests wait at most `prepare_wait_ms` (default 0 for API clients = fail fast with `model_not_ready`,
  `job_id` and `Retry-After`; per-profile/per-key opt-in up to a bounded maximum, default cap 60 s); completion streams never
  carry custom events or comments while waiting; the Host built-in chat shows progress through its own IPC events.
* **Warm retention.** After a request finishes the lease and plan are kept for `lifecycle.release_after_idle_seconds` unless:
  the local user returns on a Worker (immediate release), a policy forbids it (battery/AC/schedule/pause), the Host is
  shutting down, or a swap needs the resources. A retained *plan* is reusable by any client allowed to use the profile.
  Retained *session state* (KV) is reusable only by the same client (key id) **and** the same conversation hint
  (`session_id`/`user`), only when the new prompt's tokens extend the retained tokens exactly; otherwise it is dropped at
  request end. Retained session state never delays another client: with `max_sessions_per_domain = 1` it is aborted
  (`abort_session`) before another client's session opens. No cross-client KV mixing, ever. Only exact-representation objects
  may be reused across profiles during a *continuous* lease (existing provisioning rule); nothing persists on a Worker after
  release.
* **Cancellation.** Client disconnect (streaming **and** non-streaming), explicit cancel, queue timeout, key revocation and
  shutdown all call `cancel(ticket)`: if `queued`, dequeue; if `starting`, mark it so that the start completes into an
  immediate abort; if `running`, the Coordinator drains in-flight prefill chunks, issues `abort_window` for the outstanding
  window, then `abort_session`. The generation slot is freed only after the Coordinator confirms the abort. If that
  confirmation does not arrive within `cancel_timeout` (default 10 s), the scheduler invalidates the plan: the Coordinator
  advances the epoch (late Worker work is then rejected as stale), releases the leases (Workers drop all state with the lease)
  and destroys the Host-side domains; the slot is freed when the Host domains are gone, and the profile re-evaluates
  readiness (it is no longer `ready`). The slot is never freed while a domain may still write that session's state.
  Cancel is idempotent. A request ended by the server (Worker loss, revocation, `cancel_timeout`, shutdown, management cancel)
  ends a stream with one OpenAI-shaped error event `data: {"error":{...},"x_clusterlm":{...}}` and closes **without**
  `data: [DONE]`; a non-streaming request gets the error body. A disconnected client receives nothing further. Finish reasons
  are only the standard ones (`stop`, `length`, `tool_calls`); v1.0's `cancelled`/`error` finish reasons are withdrawn because
  unmodified OpenAI clients do not know them.

## 5. Failure handling

| Event | Behavior |
|---|---|
| Worker lost / local user returns / lease invalidated mid-generation | active request ends with `worker_lost` (stream: error event then close; non-stream: 503); conversation state for API clients is **not** preserved (they own history); Host built-in chat applies the profile's `on_worker_loss` with a visible event |
| Worker lost while queued | requests for affected profiles re-run admission at dequeue; those not satisfiable are rejected with `workers_unavailable`. An alias ticket whose recorded candidate is no longer viable is rejected too (ADR 0402; v1.0's "next candidate" wording contradicted §3 and is withdrawn) |
| Prepare job fails | job `failed` with `last_error`; profile state per the state machine; waiting requests get `model_not_ready` with the error |
| Host resource pressure | admission step 6; running request continues unless the *Worker* policy invalidates the plan |
| Shutdown | stop admitting, cancel queued with `shutting_down`, let running finish up to a grace period (default 10 s), then cancel; release all leases (existing release guarantees) |

## 6. Response metadata (serving layer)

`x_clusterlm: {request_id, profile_id, api_model_id, model: {family, quant, root_hash?}, backend, routed_via?}` is carried by
every non-streaming response, the **first and the last** chunk of every stream, and every error body. This is the normative
field set (ADR 0402 and profile-schema-v1 §6 listed a subset). The standard OpenAI `model` field is the `api_model_id` of the
profile that actually ran; when routed through an alias, `routed_via` is the alias's `api_model_id`. (v1.0 said "the alias's
chosen profile id", which would have leaked an internal `prof_*` id into a field clients compare against model ids.) Standard
`usage` (prompt/completion tokens counted by the Host tokenizer) is included as OpenAI defines it. Throughput/latency numbers
are never included unless measured on that request, labelled `measured`.

## 7. Error mapping

Error bodies are `{"error":{"message","type","param","code"}, "x_clusterlm":{request_id, reasons[], job_id?}}`. Messages never
include prompts, tool arguments, paths or fingerprints. For keys without `status:read`, `reasons[]` are generic sentences
("a required Worker is not available") without machine names; detailed reasons need `status:read` (auth-scopes-v1 §5).

| `code` | HTTP | OpenAI `type` | Internal `ErrorCode` |
|---|---|---|---|
| `invalid_api_key` (also unknown, revoked, expired) | 401 | `authentication_error` | unauthenticated |
| `insufficient_scope` | 403 | `permission_error` | `PERMISSION_DENIED` |
| `model_not_found` (also: not visible to the key) | 404 | `invalid_request_error` | `NOT_FOUND` |
| `request_too_large` | 413 | `invalid_request_error` | `INVALID_ARGUMENT` |
| `unsupported_parameter`, `context_length_exceeded`, `invalid_request` | 400 | `invalid_request_error` | `INVALID_ARGUMENT` |
| `conflict` (management only, e.g. release refused while a request is active) | 409 | `invalid_request_error` | `FAILED_PRECONDITION` |
| `rate_limit_exceeded` (per key, incl. per-client queue share) | 429 + `Retry-After` | `rate_limit_error` | `RESOURCE_EXHAUSTED` |
| `queue_full` (global) | 503 + `Retry-After` | `server_error` | `RESOURCE_EXHAUSTED` |
| `model_not_ready`, `workers_unavailable`, `insufficient_memory`, `worker_lost`, `shutting_down`, `request_cancelled` | 503 (+ `Retry-After` where a job is running) | `server_error` | `UNAVAILABLE` / `FAILED_PRECONDITION` |
| `queue_timeout` | 504 | `server_error` | `DEADLINE_EXCEEDED` |
| `internal_error` | 500 (request id only, no internals) | `server_error` | `INTERNAL` |

## 8. Management API (authenticated; also surfaced over local IPC and MCP)

Prefix `/clusterlm/v1/`. All JSON; scopes from auth-scopes-v1. Every listing is filtered by the key's `allowed_models`.

| Method + path | Scope | Purpose |
|---|---|---|
| `GET /profiles`, `GET /profiles/{id}` | `status:read` | profiles with readiness |
| `POST /profiles/{id}:prepare` → `{job_id}` | `profiles:prepare` | start async prepare (idempotent while a job exists) |
| `GET /jobs/{id}`, `POST /jobs/{id}:cancel` | `status:read` / `profiles:prepare` | progress (`PrepareDetail`, estimates flagged), cancel |
| `POST /profiles/{id}:release` | `profiles:prepare` | release lease (`conflict` while a request is active unless `force` and scope `profiles:admin`; a forced release ends that request with `request_cancelled`) |
| `GET /machines` | `status:read` | Workers: display name, state, advertised capability, availability reason (no fingerprints/addresses) |
| `GET /queue` | `status:read` | depth and per-priority counts; the caller's own tickets with positions; other clients' tickets only as anonymous positions (no key id, no content) |
| `GET /availability` | `status:read` | which profiles could be prepared now and why not |
| `POST /diagnostics:export` | `status:read` | the **shareable** redacted bundle: hashes, backend versions, hardware, placement, network timing, errors; no prompts, no paths (bench result paths included in the local bundle are dropped), no fingerprints/addresses. The full local bundle (ADR 0232) is available only over local IPC |
| profile/alias CRUD, `validate`, `dry_run`, `export`, `import` | `profiles:admin` | as profile-schema §8 |
| key management | `keys:admin` | auth-scopes-v1 |

The local IPC (existing pipe, interactive user only) remains the owner's root path and exposes the same operations plus the
ones HTTP never exposes (pairing, LAN-bind, key bootstrap, full diagnostics).

## Amendments

* v1.1 (2026-10-10, ADR 0407): ticket state machine; job states vs phases, one running job, bounded job history; plan pinned to
  profile revision; dequeue re-check; context check against the profile with re-prepare; bounded prepared plans and
  release-before-prepare swaps; prepare jobs not cancelled by waiting requests; bounded cancellation with plan invalidation;
  standard finish reasons only (error event instead); `x_clusterlm` carriage and `model` field; alias rule aligned with ADR
  0402; full error-code table; `GET /queue` scope; shareable diagnostics over HTTP.
