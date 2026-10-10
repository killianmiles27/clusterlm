# Scheduler / admission API v1 (and serving-layer rules)

Frozen by ADR 0405. Implemented by workstream C (scheduler, policies) with B (HTTP surface). Replaces the single-job gate of
`father::FatherService` ("one job at a time") with an explicit scheduler, keeping that service's Coordinator-driving internals.

## 1. Ownership (who may touch what)

| Resource | Single owner | Others |
|---|---|---|
| Domain sequence state (KV / recurrent / indexer) and speculative windows | the **Coordinator session** for that request (via `ExecutionDomain` + `WindowLedger`) | scheduler never calls a domain; it calls the Coordinator |
| Leases / plan on Workers | **Coordinator** (acquire, release, epoch) | scheduler requests `prepare`/`release` jobs; Workers refuse anything else |
| Token IDs and text | Host process only (serving layer + tokenizer) | never in scheduler queue records beyond counts; never logged |
| Queue, tickets, client records | **Scheduler** | serving layer enqueues and cancels by ticket |
| Profile/alias definitions | profiles store (A) | scheduler reads snapshots; edit during a job applies to the next job (revision pinned per ticket) |
| Host resource policy (what Workers may be used) | Host settings + Node reports (C) | local user activity on a Worker is authoritative and immediate |

Rules that follow: one request owns one Coordinator session; a session's state is released on finish/cancel/abort unless warm
retention applies (§4); no state, cache or prefix is shared between clients; the scheduler holds no pointer into a domain.

## 2. Entities

`Client` (API key id, or `local-ui`, or `mcp:<key>`; carries limits) → `Request` (`req_*`; tokens counted not stored; target
`api_model_id`; resolved `profile_id` + `profile_revision`; state) → `Ticket` (queue position, enqueue time, deadline).
`Job` (`job_*`; kinds `prepare`, `release`; states `queued | provisioning | node-preparing | ready | failed | cancelled`; progress
as in `PrepareDetail`). `Lease` is the Coordinator's existing notion.

## 3. Admission: a pure function, evaluated at enqueue and re-evaluated at dequeue

`admit(request, profile@revision, readiness, queue_state, client_limits, policy) → Admit | Queue{position} | Reject{code,details}`

Checks, in order (first failure wins; every rejection is structured and names the reason):

| # | Check | Reject code (HTTP / OpenAI `code`) |
|---|---|---|
| 1 | Auth and scope (`inference`), model visible to key | `invalid_api_key` 401 / `insufficient_scope` 403 / `model_not_found` 404 |
| 2 | Body size, parameters supported by backend + template (`n>1`, logprobs, response_format…) | `request_too_large` 413 / `unsupported_parameter` 400 |
| 3 | Context: prompt tokens + `max_tokens` ≤ profile `max_tokens` and ≤ the *prepared* plan's context | `context_length_exceeded` 400 |
| 4 | Client rate / concurrency / queue-share limits | `rate_limit_exceeded` 429 (with `Retry-After`) |
| 5 | Readiness of the target profile (state machine) | `model_not_ready` 503 (details: state, reasons, `job_id?`), `workers_unavailable` 503, `insufficient_memory` 503 |
| 6 | Resource policy: required Workers allowed now (idle/AC/battery/schedule/pause) | `workers_unavailable` 503 |
| 7 | Queue capacity | `queue_full` 429/503 |
| 8 | Server state | `shutting_down` 503 |

Exact model ids are never re-routed (ADR 0402). Only a routing alias evaluates its candidates (step 5 is applied per candidate;
the first viable one is *recorded in the ticket* so a later readiness change cannot silently change the model mid-queue; if the
recorded candidate becomes non-viable before start the request is rejected, not rerouted).

## 4. Concurrency, queue, fairness, retention

* **Concurrency** per prepared plan = `min(descriptor.state.max_sessions_per_domain, profile/plan limit)`; default and current
  maximum for all shipped backends is **1 active generation**. Raising it requires the descriptor to say so, tests, and an ADR.
* **Queue** is bounded (`max_queue_depth` default 32, `max_queue_per_client` default 8, `queue_timeout` default 120 s, all
  configurable). Order: per-priority FIFO across clients with **round-robin between clients** (no client can monopolize by
  submitting many requests). Priority classes (optional, off by default): `interactive` (Host built-in chat) above `api`;
  within a class, fairness as above. Monopoly protection: a client that has held the generation slot for more than
  `max_slot_seconds` cannot be re-queued ahead of others, and `max_tokens` is capped by policy.
* **Model swap** (a queued request needs a different prepared profile): the swap job waits for the active request to finish; it
  never kills it. While a swap is pending, new requests for the old model still queue behind it in order (no starvation either
  way); a swap that would disturb another client's *queued* requests of the current model is performed only when the queue head
  needs it (policy `swap_when: head-of-queue` default).
* **Prepare is an async job.** Started by: an explicit `prepare` call (scope `profiles:prepare`), a request to a `loadable`
  model whose `lifecycle.preparation` is `on-demand`/`keep-ready`, or `prepare_when_available`. Never started for
  `manual`. API requests wait at most `prepare_wait_ms` (default 0 for API clients = fail fast with `model_not_ready` and
  `job_id`; per-profile/per-key opt-in up to a bounded maximum, default cap 60 s); completion streams never carry custom events
  or comments while waiting; the Host built-in chat shows progress through its own IPC events.
* **Warm retention.** After a request finishes the lease and plan are kept for `release_after_idle_minutes` (profile) unless:
  the local user returns on a Worker (immediate release), a policy forbids it (battery/AC/schedule/pause), the Host is
  shutting down, or another profile needs the Workers (swap). Retained *plan* is reusable by any client; retained *session
  state* (KV) is reusable only by the same client+conversation (explicit `session_id`/`user` hint) — otherwise state is dropped
  at request end. No cross-client KV mixing, ever. Only exact-representation objects may be reused across profiles during a
  *continuous* lease (existing provisioning rule); nothing persists on a Worker after release.
* **Cancellation.** Client disconnect, explicit cancel, queue timeout and shutdown all call `cancel(ticket)`: dequeue if queued;
  if running, the Coordinator drains in-flight prefill chunks, issues `abort_window` for the outstanding window, then
  `abort_session`; slot freed only after the Coordinator confirms. Cancel is idempotent. A stream cancelled by the server (not by a client disconnect) ends with a final chunk whose `finish_reason` is
  `cancelled`; a disconnected client receives nothing further.

## 5. Failure handling

| Event | Behavior |
|---|---|
| Worker lost / local user returns / lease invalidated mid-generation | active request ends with structured error `worker_lost` (stream: error object then close; non-stream: 503); conversation state for API clients is **not** preserved (they own history); Host built-in chat applies the profile's `on_worker_loss` with a visible event |
| Worker lost while queued | requests for affected profiles re-run admission; those not satisfiable are rejected with `workers_unavailable` (alias: next candidate **only if** the alias policy re-evaluates before start, and the ticket records the change — response metadata names it) |
| Prepare job fails | job `failed` with `last_error`; profile returns to `loadable`/`unavailable` per state machine; waiting requests get `model_not_ready` with the error |
| Host resource pressure | admission step 6; running request continues unless the *Worker* policy invalidates the plan |
| Shutdown | stop admitting, cancel queued with `shutting_down`, let running finish up to a grace period, then cancel; release all leases (existing release guarantees) |

## 6. Response metadata (serving layer)

Every response (including streamed chunks' final usage chunk and error bodies where applicable) carries
`x_clusterlm: {request_id, profile_id, api_model_id, model: {family, quant, root_hash?}, routed_via?, backend, provenance_note?}`
and standard `usage` (prompt/completion tokens counted by the Host tokenizer). `model` in the standard OpenAI field is the
`api_model_id` the client asked for (or the alias's chosen profile id when routed via an alias, with `routed_via` naming the
alias). Throughput/latency numbers are never included unless measured on that request, labelled `measured`.

## 7. Error mapping

Internal `ErrorCode` → HTTP/OpenAI: `INVALID_ARGUMENT`→400 `invalid_request_error`; `PERMISSION_DENIED`→403; unauthenticated→401;
`NOT_FOUND`→404; `RESOURCE_EXHAUSTED`→429 (rate/queue) or 503 (memory); `FAILED_PRECONDITION`→409/503 (state); `UNAVAILABLE`→503;
`DEADLINE_EXCEEDED`→504 (`queue_timeout`); `INTERNAL`→500 with a request id and no internals. Error bodies are
`{"error":{"message","type","param","code"}, "x_clusterlm":{request_id, reasons[], job_id?}}`. Messages never include prompts,
tool arguments, paths or fingerprints.

## 8. Management API (authenticated; also surfaced over local IPC and MCP)

Prefix `/clusterlm/v1/`. All JSON; scopes from auth-scopes-v1.

| Method + path | Scope | Purpose |
|---|---|---|
| `GET /profiles`, `GET /profiles/{id}` | `status:read` | profiles with readiness |
| `POST /profiles/{id}:prepare` → `{job_id}` | `profiles:prepare` | start async prepare (idempotent while a job exists) |
| `GET /jobs/{id}`, `POST /jobs/{id}:cancel` | `status:read` / `profiles:prepare` | progress (`PrepareDetail`, estimates flagged), cancel |
| `POST /profiles/{id}:release` | `profiles:prepare` | release lease (refused while a request is active unless `force` and scope `profiles:admin`) |
| `GET /machines` | `status:read` | Workers: display name, state, advertised capability, availability reason (no fingerprints/addresses) |
| `GET /queue` | `status:read` | depth, own-client positions only unless `status:read` + `admin` |
| `GET /availability` | `status:read` | which profiles could be prepared now and why not |
| `POST /diagnostics:export` | `status:read` | redacted bundle (hashes, backend versions, hardware, placement, network timing, errors; no prompts) |
| profile/alias CRUD, `validate`, `dry_run`, `export`, `import` | `profiles:admin` | as profile-schema §8 |
| key management | `keys:admin` | auth-scopes-v1 |

The local IPC (existing pipe, interactive user only) remains the owner's root path and exposes the same operations plus the
ones HTTP never exposes (pairing, LAN-bind, key bootstrap).
