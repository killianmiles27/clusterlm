# 0405: A scheduler with admission replaces FatherService's single-job gate

## Context
`FatherService` runs one job at a time for one local conversation. Multiple API clients need bounded queueing, fairness,
isolation and async preparation without touching the working Coordinator/domain logic.

## Decision
- Introduce a Scheduler owning queue, tickets, clients and prepare jobs; the Coordinator remains the sole owner of leases and
  domain sequence state; the scheduler never calls a domain. `FatherService` becomes the scheduler's internal runner and the
  local-UI client.
- Default concurrency stays 1 active generation per prepared plan; more only if the backend descriptor says so, tests exist and an ADR is written.
- Pure `admit()` with an ordered, enumerated rejection list; bounded queue; round-robin fairness between clients; model swaps
  wait for the active request; prepare is an async job (`job_*`) with progress and cancellation; API requests fail fast with
  `model_not_ready` + job id unless the key/profile opts into a bounded wait; completion streams never carry custom events.
- KV/session state is never shared across clients; warm retention keeps the plan, and session state only for the same client.
- Worker loss/local-user return ends in-flight API requests with a structured error; the Host chat applies profile policy.
- Management API under `/clusterlm/v1/`, mirroring local IPC.

## Consequences
C owns concurrency and cancellation design (Opus-led) and extends the fault-injection suite. B only enqueues/cancels tickets.
A "simple" queue was chosen over priorities-by-default: priorities exist but are off unless configured.
