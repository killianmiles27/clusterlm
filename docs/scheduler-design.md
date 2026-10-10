# Scheduler design: concurrency, cancellation, memory ownership (workstream C)

Status: design for implementation, 2026-10-10. Implements `docs/interfaces/scheduler-admission-v1.md` (frozen v1.1, "the
contract"), ADR 0405, `readiness-state-machine-v1.md`. Library `orchestrator/scheduler` (`clusterlm_scheduler`), namespace
`clusterlm::scheduler`, C++20. It builds on the value types already in `types.hpp`, `admission.hpp`, `status.hpp`
(`Code`, `TicketState`, `JobKind`, `JobState`, `ProfileSnapshot`, `ClientInfo`, `Decision`, `StatusProvider`) and adds:

```
include/clusterlm/scheduler/clock.hpp      ClockSource (injectable), SteadyClockSource, ManualClock (tests)
include/clusterlm/scheduler/backend.hpp    ExecutionBackend port + BackendEvents + command/event value types
include/clusterlm/scheduler/fair_queue.hpp FairQueue (pure data structure, no locks)
include/clusterlm/scheduler/scheduler.hpp  Scheduler (public API), TicketObserver, SchedulerOptions
src/fair_queue.cpp src/jobs.cpp src/scheduler.cpp src/outbox.cpp
tests/scheduler/fake_backend.hpp           deterministic FakeBackend (+ ThreadedFakeBackend for TSAN)
```

Section 9 lists contract defects; everything else below is consistent with the contract or names the section it refines.

## 1. Threading model and lock hierarchy

### 1.1 Threads

| Thread | Owner | Does | Never does |
|---|---|---|---|
| Caller threads (HTTP, IPC, MCP, Node-report thread) | callers | `enqueue`, `cancel`, `prepare`, `cancel_job`, `release`, input pushes, status queries: take `mu_`, mutate, append *actions* to `actions_`, notify dispatcher, return | call the backend, call a sink |
| **Dispatcher** (1, scheduler-owned) | scheduler | pops actions in FIFO order, runs them **without** `mu_` (backend commands, sink deliveries via outbox); sleeps on `cv_` until `next_deadline()` of the injected clock; runs `pump()` after every wake | block in the backend (commands are non-blocking by contract, §6.3) |
| **Delivery** (1, scheduler-owned) | scheduler | drains the `Outbox` and calls `TicketObserver`/`SchedulerObserver` callbacks, serialized, in enqueue order | hold any scheduler lock while calling out |
| Backend threads (any number) | backend | call `BackendEvents::*` (§6.3); stream tokens into the ticket's `GatedSink` | hold a backend lock while calling `BackendEvents` |

Why a dispatcher instead of "act on the caller thread": two threads that each computed actions under the lock could execute
them out of order (thread A computed `start_request(t)`, thread B then computed `abort_request(t)`; B runs first and the
abort is a no-op, A's start then runs unaborted). A single FIFO action executor gives every backend command a total order
equal to the order in which the decisions were made under `mu_`. Token data never goes through either thread.

`SchedulerOptions::threading = kThreaded | kManual`. `kManual` starts no threads: actions and outbox entries accumulate and
the test thread runs them with `Scheduler::run_pending()` (executes actions, then delivers events, repeats until both are
empty, returns the count). With `ManualClock` and `FakeBackend` this makes every test single-threaded and bit-reproducible.
The same `pump()`/action code runs in both modes; only the executor differs.

### 1.2 Locks (total order; a thread may only acquire downward)

| Level | Lock | Protects | May call while held |
|---|---|---|---|
| L1 | `Scheduler::mu_` (one `std::mutex`) | every record: tickets, clients, plans, slots, jobs, queue, timers, `actions_`, world snapshot | `Outbox::push` (L2), pure functions (`admit`, `FairQueue`), clock `now()` |
| L2 | `Outbox::mu_` | the delivery deque | nothing |
| L3 | `GatedSink::mu_` (one per ticket) | gate state + the serving layer's token sink call | the serving layer's token sink only |

Rules (each is a test assertion or a code-review checklist item):

1. **No backend call under L1.** Backend methods are invoked only by the dispatcher from an action, with no lock held.
2. **No callback under L1/L2.** Observers are invoked only by the delivery thread with no lock held. Observers may call
   `cancel`, `enqueue`, queries (they take L1 normally); they must not call `shutdown()` or destroy the scheduler — doing
   so from the delivery thread returns `kFailedPrecondition` (checked by thread id) instead of deadlocking.
3. **`BackendEvents` handlers take L1 only**, never block, never call the backend (they append actions).
4. **L3 is never taken while L1 is held.** `GatedSink::close()` is an action executed by the dispatcher. Backend token
   threads take L3 alone; they never hold L3 when calling `BackendEvents`.
5. The backend may call `BackendEvents` synchronously from inside a command (the dispatcher holds no lock), so no
   re-entrancy deadlock exists; a handler running on the dispatcher thread simply appends to `actions_`.

### 1.3 `pump()` — the only place decisions are made

`pump()` runs under L1 after every state change (event, API call, timer). Order, fixed for determinism:
1. `now = clock.now()` clamped to `max(now, last_now_)` (a clock that goes backwards is treated as not moving).
2. Fire due timers in `(deadline, seq)` order: cancel/cleanup timeouts → queue/prepare-wait deadlines → shutdown grace →
   idle-retention release → job-history GC.
3. Job runner: if no job is `running` and the head job is runnable (§4), start it.
4. Dispatch loop (§3.3) until the selected head cannot advance.
5. Recompute the readiness overlay (§7.3) if anything it depends on changed; enqueue a `ReadinessChanged` outbox entry.

### 1.4 Shutdown ordering (`Scheduler::shutdown(grace)`; destructor calls `shutdown(options.shutdown_grace)` if needed)

1. L1: `server_state = kDraining`. From now `enqueue` → `shutting_down` (§9 D4 for the check order). `prepare` → `shutting_down`.
2. L1: every `queued` ticket → `done(shutting_down)` (terminal delivered). Every `queued` job → `cancelled`. Running prepare
   job → `cancel_prepare(job)`; it ends `cancelled` on `on_prepare_done`.
3. Running/starting/finishing tickets continue; timer `grace` (default 10 s). On expiry each still active → `cancel(shutting_down)`
   (normal §2 path, so `cancel_timeout` and invalidation still apply).
4. When no ticket is active and no prepare is running: drop retained sessions, then issue `release(plan)` for every plan
   directly (no job queue; it is empty). Wait for `on_plan_gone` for each, bounded by `release_timeout` (default 30 s);
   a plan not gone by then is reported in `ShutdownReport.problems` and treated as invalidated (`invalidate_plan`).
5. Wait until `outstanding_ == 0` (commands without their completion event) or `release_timeout` again; then
   `backend->detach()` (blocks until no `BackendEvents` call is in progress and guarantees none starts).
6. Stop dispatcher (drain remaining actions), stop delivery (drain outbox — every terminal is delivered), join both.
Upper bound: `grace + cancel_timeout + invalidate_timeout + 2*release_timeout`. `shutdown` returns `ShutdownReport`.

## 2. Ticket state machine

### 2.1 Records (counts only, never tokens or text — §6.1)

```cpp
using TicketId = std::uint64_t;  // internal, monotonic, also the dispatch tie-break `seq`
using PlanId   = std::uint64_t;  // scheduler-assigned, never reused within a process
using JobSeq   = std::uint64_t;  // job_* ids map 1:1
enum class StartStep : std::uint8_t { kNone, kDroppingRetained, kStartIssued };
enum class EndCause  : std::uint8_t { kNone, kClientGone, kClientCancel, kQueueTimeout, kPrepareWaitExpired,
                                      kKeyRevoked, kShutdown, kManagement, kWorkerLost, kBackendError, kRejectedAtDequeue };
struct TicketRecord {
  TicketId id; std::string request_id; std::string client_id; Priority prio;
  std::string profile_id; std::uint32_t profile_revision; std::string plan_fingerprint; std::string routed_via;
  std::uint32_t context_class, prompt_tokens, completion_budget;
  TimePoint enqueued, queue_deadline; std::optional<TimePoint> prepare_deadline;
  TicketState state; StartStep step = StartStep::kNone;
  EndCause cause = EndCause::kNone;              // first cause wins
  bool terminal_sent = false;                    // exactly one TicketObserver::on_terminal
  PlanId plan = 0; std::optional<JobSeq> waiting_job;
  RetentionKey retention;                        // {client_id, hint_digest} — §5
  std::shared_ptr<GenerationInput> input;        // opaque (incomplete type here); moved into start_request
  std::shared_ptr<GatedSink> sink;               // token path, owned jointly with backend while running
  std::shared_ptr<TicketObserver> observer;
  std::optional<TimePoint> cleanup_deadline;     // cancel_timeout / finish cleanup timeout
  std::uint32_t completion_tokens = 0;           // count, from on_request_output_done
};
```

### 2.2 Slot

Per plan, `slots.size() == concurrency` (= `ProfileSnapshot::max_sessions`, 1 for every shipped backend). A slot is one of:

| Slot state | Meaning | Counts as busy (readiness `busy`, §9 D12) | May start a ticket |
|---|---|---|---|
| `free` | no session state of any client in the domains for this slot | no | yes |
| `reserved{ticket}` | ticket in starting/running/finishing/cancelling | yes | no |
| `retained{key, session, tokens}` | a finished request's KV kept (§5) | no | same key: reuse; other: after drop confirmed |
| `dropping{key, for_ticket?}` | `drop_session` issued, not confirmed | yes | no |
| `quarantined` | plan invalidation issued; domains may still write | yes | never (plan is going away) |

**Slot-free rule (the invariant the whole design protects):** a slot becomes `free` only on (a) `on_request_ended` with
disposition `closed`, (b) `on_session_dropped{ok}`, or (c) `on_plan_gone` for its plan (Host domains destroyed). A timer
never frees a slot; a timer only escalates (to `invalidate_plan`). `retained` is entered only on
`on_request_ended{retained}`.

### 2.3 Transition table (every row is one atomic step under L1; "→ act" lists actions appended, executed later in order)

| # | From | Input | Guard | To | Actions |
|---|---|---|---|---|---|
| T1 | — | `enqueue` → `Decision::kQueue` | | `queued` | FairQueue push; deadlines armed |
| T2 | — | `enqueue` → `kAdmit` | | `queued`, then same pump dispatches | (as T1; admit == queue at depth 0 — one code path) |
| T3 | `queued` | dispatch selects it, re-check fails | | `done(cause=RejectedAtDequeue)` | terminal error (§3.2 codes) |
| T4 | `queued` | dispatch, plan ready, slot `free` or `retained` by same key | | `starting/kStartIssued` | slot→`reserved`; `sink.open`; `start_request(cmd)`; arm `start_timeout` (= cancel_timeout) |
| T5 | `queued` | dispatch, plan ready, slot `retained` by other key | | `starting/kDroppingRetained` | slot→`dropping{for=t}`; `drop_session` |
| T6 | `starting/kDroppingRetained` | `on_session_dropped{ok}` | | `starting/kStartIssued` | as T4 |
| T7 | `starting/kStartIssued` | `on_request_running` | | `running` | `observer.on_started(PlanInfo)` (x_clusterlm data) |
| T8 | `running` | `on_request_output_done` | | `finishing` | `sink.close()`; terminal success `{finish, usage counts}`; arm cleanup timer |
| T9 | `finishing` | `on_request_ended{closed\|retained}` | | `done(completed)` | slot→`free`/`retained`; client slot-time accounting stops |
| T10 | `starting`/`running` | `cancel(cause)` | | `cancelling` | `sink.close()`; terminal (error, or silent if `kClientGone`/`kClientCancel`); if `kStartIssued`: `abort_request(t)` and arm `cleanup_deadline = now+cancel_timeout` |
| T10b | `starting/kDroppingRetained` | `cancel(cause)` | | `cancelling` (step stays) | `sink.close()`; terminal; no abort (nothing started); drop continues |
| T11 | `cancelling/kDroppingRetained` | `on_session_dropped{ok}` | | `done(cause)` | slot→`free` (the start is never issued) |
| T12 | `cancelling` | `on_request_ended{any}` | | `done(cause)` | slot→`free` (aborted sessions are never retained; backend contract §6.3) |
| T13 | `cancelling`/`finishing`/`starting` | `cleanup_deadline`/`start_timeout` passes | plan not yet invalidating | unchanged | plan→`invalidating`; all its slots→`quarantined`; `invalidate_plan(plan, kAbortNotConfirmed)`; arm `invalidate_timeout` |
| T14 | any active on plan P except T17's case | `on_plan_gone(P)` | | `done(cause or WorkerLost)` | slot→`free` (plan record removed); terminal if not yet sent (`worker_lost`) |
| T15 | `running`/`starting` | `on_request_ended{failed, worker_lost\|backend_error}` (no prior cancel) | | `done(WorkerLost\|BackendError)` | `sink.close()`; terminal `worker_lost`/`internal_error`; plan→`invalidating`; `invalidate_plan` (the session is invalid; leases must go, §7.2) — slot stays `quarantined` until `on_plan_gone` (T14 frees it) |
| T16 | `running` | `on_request_ended` without prior `output_done` and without cancel | disposition closed | `done(BackendError)` | treated as T15 (contract violation by backend is logged and counted) |
| T17 | `starting/kDroppingRetained` | `on_plan_gone(P)` (drop escalated to invalidation, or Worker lost) | | `queued` at its original `TicketId` position | none — no session of this ticket was ever opened, so nothing is lost; next dispatch follows §3.3 (§9 D2) |

Notes:
* T15 ordering: the ticket becomes `done` for the client immediately (terminal sent) but the **slot** is not freed until
  `on_plan_gone`. Ticket `done` and slot `free` are distinct facts; tests assert on both.
* `finishing` exists so the client gets its response as soon as the last token is emitted (T8) while the slot is held until
  the backend confirms the session closed or retained (T9). `cancel` in `finishing` is a no-op returning `kOk` (output is
  complete, nothing to abort); the cleanup timer (T13) still protects the slot.
* Stale events: every handler first looks up `(ticket id)`/`(plan id)`; an event for a `done` ticket or a removed plan, or
  one not allowed from the current state, is ignored and increments `counters.stale_events` (never a state change).
  `on_request_ended` for a ticket in `done` after T14 is exactly that case.

### 2.4 `cancel(TicketId, EndCause) -> Status`

| State at call | Effect | Return |
|---|---|---|
| `queued` | removed from FairQueue (O(1) via per-client list iterator); `done(cause)`; waiter count of its job decremented; **job untouched** | ok |
| `starting/kDroppingRetained` | T10b | ok |
| `starting/kStartIssued` | T10 — abort issued immediately. Backend contract: an `abort_request` after `start_request` is valid at any point, including before the session opened; the run must then end with `on_request_ended{aborted}` without writing domain state beyond what it aborts | ok |
| `running` | T10 | ok |
| `finishing` | none | ok |
| `cancelling` | none (first cause kept; timer not re-armed) | ok |
| `done` (record still in history) | none | ok |
| unknown / GC'd | none | `kNotFound` |

Cancel "during prepare/swap": the ticket is `queued` (waiting tickets never leave `queued` until dispatched), so it is
dequeued and the job continues (contract §4). Cancel "during starting": covered by T10/T10b; the race "start completes
while cancel is in flight" is resolved by the dispatcher's total order: `start_request` is always executed before the
`abort_request` for the same ticket, and the backend must accept the abort in any phase (§6.3 E4).

### 2.5 Cause → terminal (what the serving layer sends)

| Cause | Terminal | Wire `code` |
|---|---|---|
| `kClientGone`, `kClientCancel` | `silent` (nothing more is sent) | — |
| `kQueueTimeout` | error | `queue_timeout` |
| `kPrepareWaitExpired` | error + `job_id` + `Retry-After` | `model_not_ready` |
| `kKeyRevoked` | error (stream: error event, no `[DONE]`) | `invalid_api_key` (§9 D9) |
| `kShutdown` | error | `shutting_down` |
| `kManagement` (forced release, management cancel) | error | `request_cancelled` |
| `kWorkerLost` | error | `worker_lost` |
| `kBackendError` | error (request id only) | `internal_error` |
| `kRejectedAtDequeue` | error | per §3.2 |

`cancel_timeout` is never a client-visible cause: the terminal was already sent at cancel time (T10).

## 3. Queue

### 3.1 `FairQueue` (pure, unit-tested in isolation; owned by the scheduler, used under L1)

```cpp
class FairQueue {
 public:
  struct Entry { TicketId id; std::string client; Priority prio; };
  void push(const Entry& e);                           // O(log C)
  bool erase(TicketId id);                             // O(1) via index
  // The ticket that dispatch must consider next, or nullopt. Pure in (state, usage, now).
  std::optional<TicketId> head(const SlotUsage& usage, TimePoint now) const;
  void served(const std::string& client);             // advance this class's RR cursor past `client`
  std::size_t depth() const; std::size_t depth(Priority) const; std::size_t client_depth(const std::string&) const;
  std::vector<TicketId> order(const SlotUsage&, TimePoint) const;  // full dispatch order, for GET /queue positions
};
```

* One ring per priority class (`interactive` before `api`, only when priorities are enabled; otherwise all tickets use
  `kApi`). A ring is the list of clients with ≥ 1 queued ticket in that class, ordered by the `TicketId` (monotonic seq) of
  the ticket that made the client non-empty — **never by time or string compare**, so equal enqueue times cannot reorder.
  Each client has a FIFO of its tickets (by `TicketId`).
* `head()`: highest non-empty class; in it, starting at the cursor (the client after the last served), the first client
  that is not *demoted*; if all are demoted, the first client from the cursor. Return that client's oldest ticket.
* Demotion (contract §4 `max_slot_seconds`): `SlotUsage` keeps per client a deque of `[start,end)` intervals during which
  a slot was `reserved` for that client's tickets (start at T4/T5, end at `done`; an open interval counts up to `now`),
  pruned to the last 10 minutes. Demoted ⇔ sum > `max_slot_seconds`. `retained` and `dropping` do not count (§9 D11).
* `served(client)` is called only when the head ticket leaves `queued` (dispatch or dequeue-reject). A head that cannot
  advance (waiting for a slot, a job, or a swap) does **not** move the cursor, so the decision is reproducible.
* Positions for `GET /queue` are indices in `order()`, 1-based; other clients' tickets are reported as positions only.

### 3.2 Dequeue re-check (`queued → starting`, contract §3)

Pure function `recheck(const TicketRecord&, const WorldSnapshot&) -> std::optional<Code>` over the **latest pushed**
world snapshot (§7.1; never a pull under L1). Rejects (T3), in order: key revoked/expired or model no longer visible →
`invalid_api_key` / `model_not_found`; profile deleted or `exposure.api` off → `model_not_found`; pinned revision's
`plan_fingerprint` identity changed → `model_not_ready` with reason "the model was changed" (§9 D8); alias candidate no
longer viable → its readiness code; readiness for the context class below `loadable` → `readiness_reject_code`
(`workers_unavailable` for transient, `model_not_ready` for blockers). Revision changes that keep the fingerprint pass.

### 3.3 Dispatch loop (inside `pump`, step 4)

```
while (auto t = queue.head(usage, now)):
  if (auto code = recheck(t, world)) { reject(t, *code); queue.served(t.client); continue; }
  P = plan with fingerprint == t.fingerprint && context >= t.context_class && state == ready
  if P:
    s = free slot of P, else slot retained by t.retention, else slot retained by another key, else none
    none            -> break                       // wait for the slot (head-of-line, contract §4)
    retained(other) -> T5; queue.served; break     // one drop at a time
    free/retained(same) -> T4; queue.served; continue
  if a job J (queued|running) prepares this fingerprint for >= t.context_class: attach t (waiting_job=J); break
  if prepare not allowed for t (manual, or wait budget 0): reject(model_not_ready + job_id?); served; continue
  ensure_prepare(t.profile, t.context_class)       // §4: may enqueue a swap (release then prepare)
  attach t; break
```

`break` = strict head-of-line: no ticket behind a blocked head is dispatched (contract §4: "dispatch stays in queue order").

### 3.4 Deadlines and clock

* `queue_deadline = enqueued + queue_timeout` (default 120 s, cap 600 s) → T-cancel with `kQueueTimeout` if still `queued`.
* `prepare_deadline = enqueued + min(client.prepare_wait_ms, profile opt-in, cap 60 s)`, set only when the ticket is
  attached to a job; on expiry while still attached → `kPrepareWaitExpired`. When both fire in one pump, the earlier
  deadline wins; equal ⇒ `kPrepareWaitExpired` (more useful: carries the job id).
* `class ClockSource { public: virtual TimePoint now() const = 0; virtual ~ClockSource() = default; };`
  `SteadyClockSource` (production) and `ManualClock{ void advance(Duration); void set(TimePoint); }` (tests; `set` may
  go backwards to test clamping). Timers are a `std::multimap<std::pair<TimePoint, std::uint64_t seq>, TimerKind+id>`;
  a large forward jump fires every due timer in `(deadline, seq)` order in one pump. The dispatcher waits with
  `cv_.wait_until(next_deadline)` in threaded mode; in manual mode `run_pending()` calls `pump()` once first.
* The scheduler never uses wall time; job history age and the 10-minute usage window use the same `ClockSource`.

## 4. Jobs (prepare / release)

```cpp
struct JobRecord {
  JobSeq seq; std::string job_id; JobKind kind; JobState state;
  std::string profile_id; std::uint32_t profile_revision; std::string plan_fingerprint; std::uint32_t context_tokens;
  PlanId plan;                         // prepare: the PlanId pre-assigned for the result; release: the plan released
  std::optional<JobSeq> after;         // swap: the release job this prepare must follow
  std::uint32_t waiters = 0; bool cancel_requested = false;
  PrepareProgressView progress;        // counts only (phase, bytes, objects, per-Node display names)
  std::string last_error; TimePoint created, finished;
};
```

* **One running job**, FIFO by `seq` (`max_running_jobs` option, default and cap 1 in v1). Head job runnability:
  `prepare` — its `after` job (if any) succeeded, prepared-plan count < `max_prepared_profiles`, and no plan is
  `releasing`/`invalidating` (allocations never overlap); `release(P)` — P has no slot `reserved`/`dropping`/`quarantined`.
  A non-runnable head blocks later jobs (FIFO; release heads become runnable when the active request ends, which the swap
  rule requires anyway).
* **Idempotent prepare**: `ensure_prepare(profile, ctx)` returns the existing job if one with the same `plan_fingerprint`
  is `queued`/`running` and its `context_tokens >= ctx`; if a *queued* one has a smaller context it is raised to `ctx`;
  if a *running* one is smaller, a new job follows it (§9 D7). Same for `POST :prepare` (returns the same `job_id`).
* **Swap** = `release(P_old)` job + `prepare(new)` job with `after = release.seq`, both appended atomically. If the release
  fails, the prepare fails with "the previous model could not be released" (allocations must not overlap). Counted in
  `counters.swaps`.
* **Waiters do not own jobs.** A ticket's cancel/timeout only decrements `waiters`. A job ends by: success, failure,
  `cancel_job`, invalidation input (Worker of the job's plan lost / local user returns / policy: `cancel_prepare`, ends
  `failed` with the reason), shutdown (`cancelled`). A blocker appearing mid-job (readiness `unavailable`) → `cancel_prepare`.
* `cancel_job(id)`: `queued` → `cancelled` immediately (and a dependent swap prepare is cancelled too; a queued release job
  is cancelled only by explicit request); `running` → `cancel_prepare(job)`, state stays `running` with
  `cancel_requested` until `on_prepare_done` → `cancelled` (the backend guarantees the partial lease is released, §6.3 E2).
  A running `release` job cannot be cancelled (`kFailedPrecondition`).
* On prepare end, attached tickets: success → nothing (next pump dispatches them); failure/cancel → each is rejected
  `model_not_ready` with `last_error` (or "preparation was cancelled") — §9 D10.
* **History**: finished jobs stay queryable while `count <= 64` and `age <= 1 h` (evict oldest first when either is
  exceeded; GC in pump step 2 and on every job end).
* `release(profile_id, force)` (management): `conflict` if a slot of its plan is `reserved` and `!force`; with `force`
  (scope checked by the serving layer) the active ticket is cancelled with `kManagement` and the release job queued.
  A `retained` slot is not "active" and never causes `conflict`.

## 5. Warm retention and session isolation

* **Retained plan**: after the last slot of plan P becomes `free`/`retained` and no queued ticket targets P's fingerprint,
  arm `idle_deadline = now + release_after_idle_seconds` (0 = none). Any dispatch to P disarms it. On expiry → release job.
  Immediate release (no job queue, not delayed by a running prepare): local user returns on a Worker of P, Worker policy
  forbids P's Workers, Host shutdown → `invalidate_plan` (§7.2). Swap → release job (§4).
* **Retained session state** (KV): `RetentionKey{ client_id, hint_digest }`, `hint_digest = HMAC-SHA256(process_key,
  session_id or user)` truncated to 128 bits; the raw hint is never stored by the scheduler (it may be a user identifier).
  No hint ⇒ no retention (`retain_on_finish = false`).
  - At T4 the scheduler passes `reuse = RetainedSessionRef` only when the slot is `retained` with an equal key.
  - **The exact-extension check is done by the backend** (only it may see token ids, §6.1): it compares the retained
    committed+pending tokens with the new prompt; extension ⇒ reuse and report `reused_prefix_tokens`; otherwise it closes
    (`abort_session`) the retained session first and opens a fresh one, still inside the same start. Either way exactly one
    session exists for the slot.
  - At T9 the backend retains only if `retain_on_finish` and the request completed normally; never after cancel, error,
    or a request that ended at `length` with the context full.
  - Another key needs the slot ⇒ T5 (`drop_session` = `abort_session` + confirm) before its session opens. Same client,
    different hint ⇒ also T5. Retained state is dropped on: plan release/invalidation, idle expiry of the plan, key revoked
    (`on_key_revoked` → drop if retained by that key), shutdown.
  - `drop_session` not confirmed within `cancel_timeout` ⇒ T13 (invalidate).
* No state, cache or prefix crosses `client_id`: enforced by the scheduler (key equality) **and** by the backend (the
  reuse ref names the session; the backend refuses a ref whose recorded key differs — defence in depth, tested).
* Local-UI (`local-ui` client) retains with hint = the Host conversation id, so the built-in chat keeps today's
  prefill-only-new-tokens behaviour of `FatherService::generate_loop`.

## 6. Memory ownership and the `ExecutionBackend` port

### 6.1 Who owns what

| Data | Owner | Scheduler holds |
|---|---|---|
| Prompt tokens, sampling params, stop tokens, tool schema | serving layer, as `GenerationInput` (defined in the serving/adapter library) | `std::shared_ptr<GenerationInput>` to an **incomplete type** (forward-declared in `backend.hpp`): scheduler code cannot read, log or serialize it. Moved into `StartCommand` at T4; reset at `done` if never started |
| Generated tokens/text | backend → `GatedSink` → serving layer | nothing (the `GatedSink` passes spans through under L3; it stores none) |
| Retained session tokens | backend adapter (Coordinator `Conversation`) | `RetainedSessionRef{PlanId, std::uint64_t session_seq}` + token **count** |
| Conversation hint | serving layer | 128-bit keyed digest |
| Leases, epochs, domains, `Conversation` | Coordinator inside the backend adapter | `PlanId` only; no pointer into a domain or Coordinator |
| Tickets, clients, slots, plans (as records), jobs, queue | scheduler | — |
| Profile definitions | profile store (A) | `ProfileSnapshot` by value (immutable per revision) |

`GatedSink` (in `backend.hpp`): `class GatedSink { public: void tokens(std::span<const std::int32_t>, std::string_view text);
void close(); bool closed() const; };` — `tokens` takes L3, drops the call if closed, else forwards to the serving
layer's `TokenSink`. `close()` takes L3 (so it waits for an in-progress delivery) and sets closed. The scheduler closes the
gate **before** queueing the terminal, so no token is ever delivered after a terminal (property P3).

### 6.2 Port definition

```cpp
namespace clusterlm::scheduler {
struct GenerationInput;                               // incomplete here
struct PlanInfo { std::uint32_t max_context = 0, concurrency = 1; bool per_client_isolation = false;
                  std::vector<std::string> worker_ids; std::string backend_id, model_identity; };
struct PrepareCommand { JobSeq job; PlanId plan; ProfileSnapshot profile; std::uint32_t context_tokens; };
struct StartCommand { TicketId ticket; PlanId plan; std::shared_ptr<GenerationInput> input;
                      std::shared_ptr<GatedSink> sink; std::optional<RetainedSessionRef> reuse;
                      RetentionKey key; bool retain_on_finish; std::uint32_t completion_budget; };
enum class Disposition : std::uint8_t { kClosed, kRetained, kAborted, kFailed };
enum class FailKind   : std::uint8_t { kNone, kWorkerLost, kSessionInvalidated, kBackendError };
struct OutputSummary { enum class Finish : std::uint8_t { kStop, kLength } finish; std::uint32_t prompt_tokens,
                       completion_tokens, reused_prefix_tokens; };
struct RequestEnd { Disposition disposition; FailKind fail = FailKind::kNone; Status status;
                    std::optional<RetainedSessionRef> retained; std::uint32_t retained_tokens = 0; };
enum class GoneCause : std::uint8_t { kReleased, kInvalidated, kLost };
struct PlanGone { GoneCause cause; bool clean = true; std::string problems; };  // problems: status text, no paths
enum class InvalidateReason : std::uint8_t { kAbortNotConfirmed, kWorkerLost, kLocalUserReturned, kPolicy,
                                             kSessionInvalidated, kShutdown };

class BackendEvents {                                  // implemented by the scheduler; thread-safe; non-blocking
 public:
  virtual void on_prepare_progress(JobSeq, const PrepareProgressView&) = 0;
  virtual void on_prepare_done(JobSeq, PlanId, Result<PlanInfo>) = 0;
  virtual void on_request_running(TicketId) = 0;
  virtual void on_request_output_done(TicketId, const OutputSummary&) = 0;
  virtual void on_request_ended(TicketId, RequestEnd) = 0;
  virtual void on_session_dropped(PlanId, RetainedSessionRef, Status) = 0;
  virtual void on_plan_gone(PlanId, PlanGone) = 0;
  virtual void on_plan_lost(PlanId, InvalidateReason) = 0;   // spontaneous: backend saw lease loss while idle
 protected: ~BackendEvents() = default;
};

class ExecutionBackend {
 public:
  virtual ~ExecutionBackend() = default;
  virtual void attach(BackendEvents* events) = 0;     // once, before any command
  virtual void detach() = 0;                          // blocks until no event call is running; none after
  virtual void prepare(PrepareCommand) = 0;           // -> progress*, then exactly one on_prepare_done
  virtual void cancel_prepare(JobSeq) = 0;            // no own completion; affects on_prepare_done
  virtual void release(PlanId) = 0;                   // -> exactly one on_plan_gone(kReleased|...)
  virtual void invalidate_plan(PlanId, InvalidateReason) = 0;  // -> on_plan_gone(kInvalidated) (one per plan in total)
  virtual void start_request(StartCommand) = 0;       // -> [running] [output_done] then exactly one on_request_ended
  virtual void abort_request(TicketId) = 0;           // no own completion; affects on_request_ended
  virtual void drop_session(PlanId, RetainedSessionRef) = 0;  // -> exactly one on_session_dropped
  virtual BackendDiagnostics diagnostics() const = 0; // counts only; called only outside L1
};
}
```

### 6.3 Contract of the port (each clause is a FakeBackend check and an adapter conformance test)

* **E1 Non-blocking.** Every command returns promptly (no network I/O, no waiting for Coordinator calls, no domain
  compute on the caller thread). Commands are invoked from one thread (the dispatcher), so they are serialized; the
  backend may still be called concurrently with its own event delivery.
* **E2 Exactly-once completions** as annotated. `prepare` failure/cancel ⇒ every partial lease released and Host domains
  destroyed **before** `on_prepare_done` (as `Coordinator::cancel_prepare` + `release` today).
* **E3 Per-entity order.** For a ticket: `running` ≺ `output_done` ≺ `ended` (either of the first two may be absent). For a
  plan: every `on_request_ended`/`on_session_dropped` of that plan ≺ its `on_plan_gone`; nothing for that plan after
  `on_plan_gone`. No order across entities.
* **E4 Abort in any phase.** `abort_request` before the session opened ⇒ the session is never opened (or is closed before
  any window), `ended{kAborted}`. During prefill ⇒ in-flight chunks drained, then `abort_session`. During decode ⇒
  `abort_window` then `abort_session`. After `output_done` ⇒ no-op (session close proceeds). Unknown/finished ticket ⇒
  no-op. Aborted sessions are never retained. `kAborted` is reported **only after** the domains confirmed (no domain may
  write that session afterwards). Repeated aborts are no-ops.
* **E5 Invalidation.** `invalidate_plan` (idempotent; may arrive while a release, start or prepare for the plan is in
  progress): advance the epoch so late Worker work is rejected as stale, interrupt any in-progress generate/prepare, release
  leases (Workers drop all state), destroy Host-side domains, end any running request with `ended{kFailed,
  kSessionInvalidated}`, then `on_plan_gone(kInvalidated)`. `on_plan_gone` is emitted **only when Host domains are
  destroyed**; if a Host domain is stuck in compute, it is late — never early.
* **E6 Failure semantics.** No exceptions cross the port (adapter catches, reports `kBackendError`). Unknown ids are no-ops.
  A generate that fails because a stage/Worker was lost ⇒ `ended{kFailed, kWorkerLost}` (the adapter maps
  `kUnavailable`/`kStaleEpoch`/`kAborted` from the Coordinator to it); other failures ⇒ `kBackendError`. The scheduler
  invalidates the plan after either (T15) — the adapter must not "repair" a plan on its own.
* **E7 Privacy.** Events carry counts, ids, `Status` text without prompt/token/path content.

### 6.4 Coordinator adapter (later; constrains the port, not implemented here)

`CoordinatorBackend` owns, per plan, one `PlanRunner` thread with a work deque (the `Coordinator` is not thread-safe:
`generate`, `close_conversation`, `release` run only there), the `Coordinator`, the `Deployment`, and retained
`Conversation`s. It reuses `FatherService`'s internals: `ensure_tier`'s identity check + `connect` + `prepare` with progress
mapping (→ `PrepareProgressView`), `generate_loop`'s held-prefix check (→ the exact-extension check of §5), and
`teardown_active` (→ release). `abort_request` sets the ticket's `std::shared_ptr<std::atomic<bool>>` that is
`GenerationRequest::cancel`; when `generate` returns `cancelled`, the runner calls `close_conversation` (= `AbortSession`
everywhere) and then emits `ended{kAborted}`. `invalidate_plan` needs a **thread-safe Coordinator interrupt** (§9 D1).
`FatherService` becomes a thin `local-ui` client over the scheduler (prepare/chat/cancel map to `prepare`/`enqueue`/`cancel`;
its fallback policy re-enqueues a new ticket for the fallback profile — the scheduler never re-routes).

## 7. Worker loss, local-user return, policy

### 7.1 Inputs (pushed; never pulled under L1)

```cpp
struct WorldSnapshot { std::uint64_t version; TimePoint as_of;
  std::map<std::string, ProfileSnapshot> profiles; std::map<std::string, AliasSnapshot> aliases;
  std::map<std::string, ReadinessView> readiness_by_profile_ctx;  // key "prof_x@ctx", from A's evaluator
  std::map<std::string, KeyState> keys; std::map<std::string, WorkerObservation> workers; };
enum class WorkerEventKind : std::uint8_t { kLost, kLocalUserActive, kPolicyForbids, kLeaseRevoked, kBack };
struct WorkerEvent { std::string worker_id; WorkerEventKind kind; TimePoint observed_at; std::string reason; };
void Scheduler::update_world(std::shared_ptr<const WorldSnapshot>);  // replaces atomically; pump
void Scheduler::worker_event(const WorkerEvent&);                      // edge-triggered fast path; pump
void Scheduler::key_revoked(const std::string& client_id);             // cancels that client's tickets (kKeyRevoked)
```

Node reports (Node state events, power, local activity, lease revoked; ADR 0134 fail-closed) arrive through the Host's
readiness board, which calls `worker_event` immediately for regressions and `update_world` with the recomputed snapshot.
`enqueue` evaluates `admit()` against the current snapshot (taken under L1 — it is a pointer copy, no I/O).

### 7.2 Handling (all in one pump; order fixed)

| Event | Plans using the Worker (`PlanInfo::worker_ids`) | Running/starting ticket on such a plan | Retained session | Queued tickets | Prepare job touching the Worker |
|---|---|---|---|---|---|
| `kLost`, `kLeaseRevoked`, `kLocalUserActive`, `kPolicyForbids` | → `invalidating`; slots `quarantined`; `invalidate_plan(reason)`; readiness overlay `plan_ready=false` **now** | terminal `worker_lost` now (gate closed); ticket `cancelling` cause `kWorkerLost`; `done` at `on_plan_gone` (T14) | gone with the plan | untouched; re-run §3.2 at dequeue with the new snapshot → `workers_unavailable` if not satisfiable | running: `cancel_prepare` → `failed` "<Worker display name> is no longer available"; queued: left (re-evaluated when runnable: a prepare whose profile is below `loadable` fails with the readiness reason) |
| `kBack` | none | — | — | — | `prepare_when_available` profiles get a prepare job if loadable and no swap is needed |
| `on_plan_lost(P)` from backend | as row 1 for P | as row 1 | as row 1 | as row 1 | — |

* Invalidation is **not** a job: it must not wait behind a running prepare (contract "immediate release"; §9 D6).
* `invalidate_timeout` (default 30 s) without `on_plan_gone`: the plan stays `invalidating`, its slots stay `quarantined`
  (never freed), `counters.stuck_invalidations++`, a `SchedulerObserver::on_fault` is emitted, readiness overlay adds
  reason "the previous session has not stopped yet" (profile at most `compatible`), and no prepare job is runnable while it
  persists (allocations must not overlap). Only process restart clears it. Tested (F7).
* The Host built-in chat applies `on_worker_loss` by re-enqueueing (new ticket); API clients get the error only.

### 7.3 Readiness overlay (output)

After any change, the scheduler publishes per `(profile, ctx)`: `active_requests` (= slots `reserved|dropping|quarantined`),
`concurrency_limit`, `plan {profile_revision, plan_fingerprint, context, plan_ready = state==ready}`, `job {id, state, phase,
last_error}`. A's pure evaluator produces the state; the scheduler never asserts `ready` itself.

## 8. Fault-injection and stress test plan

### 8.1 Fakes

`FakeBackend` (manual): records every command in `std::vector<Cmd>`; tests complete them explicitly
(`fake.running(t)`, `fake.output_done(t, {...})`, `fake.ended(t, ...)`, `fake.gone(P, ...)`, `fake.prepared(job, info)`,
`fake.fail_prepare(job, st)`, `fake.dropped(P, ref)`, `fake.lost(P)`). It enforces E1–E6 and records a **domain-write
ledger**: for each slot, whether a session "may still write" (from `start_request` until the test sends `ended`, or from
`invalidate_plan` until `gone`). Modes: `never_confirm_abort`, `never_gone`, `sync_completion` (completes inside the command),
`late_events` (re-delivers old events after `gone`). `ThreadedFakeBackend`: same semantics, but completions happen on 1–4
backend threads after seeded random delays (0–2 ms) and tokens are streamed into `GatedSink` from those threads.

### 8.2 Invariants (checked after every `run_pending()` in all scenario/property tests)

* **P1** Per plan, slots `reserved|dropping|quarantined` ≤ concurrency; tickets in `starting|running|finishing|cancelling`
  map 1:1 to `reserved` slots.
* **P2** No slot is `free` while the fake's write ledger says a session of that slot may write. (The core safety property.)
* **P3** Exactly one `on_terminal` per ticket; zero tokens reach the serving sink after the terminal (counted at the fake
  serving sink); silent causes produce no client-visible output after cancel.
* **P4** ≤ 1 job `running`; jobs start in `seq` order; a prepare is never issued while a plan is `releasing|invalidating`
  or the prepared count is at `max_prepared_profiles`.
* **P5** Queue depth ≤ `max_queue_depth`; per client ≤ `max_queue_per_client`.
* **P6** Every `start_request` gets exactly one `ended` before or with `gone` (E2/E3) — validates the fake and the adapter.
* **P7** No record/diagnostic/log contains a canary token id sequence or prompt string (the test prompt carries a unique
  canary; the scheduler's `diagnostics()`, `StatusProvider` outputs and captured logs are searched).
* **P8** Determinism: two runs with the same seed produce byte-identical traces (commands + events + terminals).
* **P9** Liveness: if the fake eventually answers every command, every ticket reaches `done` and every slot is `free` or
  `retained` once input stops (bounded steps).

### 8.3 Scenarios (manual mode, `ManualClock`; expected observable outcome)

| # | Scenario | Expected |
|---|---|---|
| F1 | cancel while `queued` waiting on a prepare job | ticket `done` silent; job keeps `running`, `waiters` decremented; other waiter still served after success |
| F2 | cancel during `starting/kStartIssued` before `running` | `abort_request` follows `start_request` in command log; `ended{kAborted}` frees slot; no `on_started` |
| F3 | cancel races `output_done` (both in one step, either order) | cancel first ⇒ silent, `output_done` ignored (stale); `output_done` first ⇒ success terminal, cancel is no-op |
| F4 | double / triple cancel with different causes | first cause kept; exactly one `abort_request` in the command log; `kOk` returned each time |
| F5 | abort never confirmed (`never_confirm_abort`) | at `+cancel_timeout`: `invalidate_plan(kAbortNotConfirmed)`; slot `quarantined`; queued ticket for same profile not started; after `gone`: slot freed, profile not `ready`, next ticket needs prepare (rejected `model_not_ready` with wait 0) |
| F6 | F5 and `gone` arrives late, then a late `ended` for the ticket | late `ended` ignored (`stale_events==1`), no double free |
| F7 | invalidation never completes (`never_gone`) | slot never freed at any clock value; `stuck_invalidations==1`; no prepare runnable; status shows reason |
| F8 | Worker lost during prepare (running job) | `cancel_prepare` issued; job `failed` with Worker reason; waiters `model_not_ready`; no plan created |
| F9 | Worker lost during `running` | terminal `worker_lost` immediately; `invalidate_plan(kWorkerLost)`; slot freed only at `gone`; queued tickets for the profile rejected `workers_unavailable` at dequeue (snapshot shows Worker offline) |
| F10 | local user returns while plan idle with retained session | `invalidate_plan(kLocalUserReturned)` without waiting for a running prepare of another profile; retained state gone |
| F11 | swap: A running, head needs B | release(A) job queued but not runnable until A's ticket `done`; then release → `gone` → prepare(B); A never aborted; P4 holds |
| F12 | swap release fails | prepare(B) `failed` "previous model could not be released"; B waiters `model_not_ready` |
| F13 | prepare idempotency: 5 tickets + 2 `POST :prepare` same profile | one job id everywhere; larger context while queued raises the job's ctx; while running adds a follow-up job |
| F14 | retained session reuse: same key+hint | `start_request.reuse` set; other key next ⇒ `drop_session` then start; same key other hint ⇒ drop; key revoked ⇒ drop |
| F15 | `drop_session` never confirmed | invalidation at `+cancel_timeout`; waiting ticket stays `starting/kDroppingRetained` until `gone`, then T17 back to `queued` at its original position; next dispatch needs a prepare (wait-budget rules apply) |
| F16 | shutdown with 3 queued + 1 running + running prepare | queued `shutting_down` at once; job `cancelled`; running finishes within grace or is cancelled at grace; all plans released; `detach` called last; outbox drained (4 terminals) |
| F17 | shutdown where abort is never confirmed and `gone` never comes | `shutdown` returns after its bound with problems listed; no thread leak |
| F18 | clock jump +1 h with queued tickets, cancelling ticket and job history | timers fire in `(deadline, seq)` order: cleanup-timeout first ⇒ invalidate; queue timeouts; idle release; history GC to ≤ 64 & ≤ 1 h |
| F19 | clock goes backwards 5 min | no timer fires, no deadline extended, demotion window not inflated |
| F20 | fairness: client A 8 tickets, B 1, C 1 enqueued after A's | dispatch order A1 B1 C1 A2 A3…; demote A after `max_slot_seconds` ⇒ B/C go first in each round while backlogged |
| F21 | equal enqueue timestamps | order by `TicketId`; identical traces over 100 runs |
| F22 | profile edited (fingerprint changed) while queued | dequeue reject `model_not_ready` "model changed"; rename-only edit passes |
| F23 | `sync_completion` mode (events inside commands) | same final states as async; no deadlock (L1 not held during commands) |
| F24 | backend violates contract (`ended` twice, event after `gone`) | ignored, counted; P1–P3 still hold |
| F25 | forced release during `running` | ticket `request_cancelled`; non-forced ⇒ `conflict`; retained-only plan ⇒ no conflict |
| F26 | observer calls `cancel` and `enqueue` from inside `on_terminal`; observer calls `shutdown` | first two work; `shutdown` from delivery thread ⇒ `kFailedPrecondition` |

### 8.4 Property and stress tests

* **Model-based property test** (manual mode, 10 000 seeds × 300 steps, seed printed on failure): random ops from
  {enqueue (k clients, 2 profiles, 2 context classes, hints), cancel (any ticket incl. done/unknown), complete any pending
  fake command (in random order, incl. failures), worker lost/back, key revoked, explicit prepare/cancel_job/release(force),
  clock advance (0–700 s, occasionally backwards), update_world (profile edit)}. Check P1–P9 after each step; at the end run
  "drain" (fake answers everything) and check P9. Shrink by step deletion.
* **TSAN stress** (`threaded` mode + `ThreadedFakeBackend`, build preset `tsan`: `-fsanitize=thread -O1 -g`, separate from
  the existing ASan/UBSan fuzz preset): 8 caller threads × 2 000 random API ops, 4 backend threads, 2 s wall budget, then
  shutdown. Assert no TSAN report, P1–P7 on the final state and on periodic `debug_snapshot()` (taken under L1).
  Targeted TSAN cases: cancel spam on one ticket from 8 threads while the fake streams tokens; `shutdown` concurrent with
  `enqueue`/`cancel`; observer callbacks re-entering.
* **Lock-rule checks**: debug builds wrap L1 in `CheckedMutex` that records the holder; `ExecutionBackend` calls and observer
  calls assert `!mu_.held_by_this_thread()`; `GatedSink::close` asserts the same.
* **Adapter conformance suite** (runs against `FakeBackend` now and `CoordinatorBackend` later, with in-process
  NodeWorkers as in `test_father_service.cpp`): E1–E6, incl. abort during prefill and decode, Worker killed mid-generate,
  invalidate during prepare.

## 9. Defects and ambiguities in scheduler-admission-v1 (proposed amendments / ADR notes)

| # | Where | Defect / ambiguity | Proposed resolution |
|---|---|---|---|
| D1 | §4 Cancellation, §1 | Contract assumes the scheduler can make "the Coordinator advance the epoch, release leases, destroy domains" at `cancel_timeout` while a request may still be inside `Coordinator::generate`. The Coordinator is not thread-safe, has no public `abort_window/abort_session` (only the cooperative `cancel` atomic) and `generate` can block up to `window_timeout` (30 s) > `cancel_timeout` (10 s) | Cross-module request: `Coordinator::interrupt(reason)` — thread-safe; advances the epoch, shuts down Node channels so blocked calls fail fast, makes `generate` return `kAborted`; then `release()` on the runner thread. Until it exists the adapter's `on_plan_gone` is late (safe), never early |
| D2 | §2, §4 | `finishing` is listed but undefined; cancel in `finishing` unspecified | Define as §2.3 T8/T9: response complete, session close/retain pending; cancel is a no-op; cleanup bounded by `cancel_timeout` → invalidation. Also allow `starting → queued` (T17) when the plan vanishes before any session of the ticket opened |
| D3 | §4 | "slot freed when the Host domains are gone" has no bound if a Host domain is stuck | State explicitly: no bound; slot stays quarantined, profile at most `compatible` with a reason, no new prepare until it clears (§7.2) |
| D4 | §3 table | `shutting_down` is check #8, so during shutdown clients get `queue_full`/`model_not_ready`/`rate_limit` with `Retry-After` instead of `shutting_down` | Evaluate server state immediately after check 1 (auth first, so nothing leaks to unauthenticated callers) |
| D5 | §4 | `queue_timeout` vs `prepare_wait_ms` for a ticket waiting on a job: which bound and which code | Deadline = earlier of both; prepare-wait expiry ⇒ `model_not_ready` + `job_id` + `Retry-After`; queue expiry ⇒ `queue_timeout`; tie ⇒ `model_not_ready` |
| D6 | §4 Warm retention / §2 Jobs | "local user returns ⇒ immediate release" conflicts with "release is a job, FIFO, one running" (would wait behind a long prepare) | Emergency release is plan invalidation, outside the job queue; jobs are only orderly prepare/release (idle, swap, management) |
| D7 | §4, §8 | `:prepare` "idempotent per profile" ignores the context class | Same fingerprint and ctx ≥ needed ⇒ same job; a queued job is raised to the larger ctx; a running smaller one gets a follow-up job; `:prepare` returns the job that will satisfy the request |
| D8 | §3 dequeue re-check | No reject code for "pinned revision's model identity/backend changed" | `model_not_ready` with reason "the model behind this name was changed"; deleted/exposure off ⇒ `model_not_found` |
| D9 | §4 Cancellation, §7 | Server-ended causes have no code mapping (revocation, management cancel) | Table §2.5: revocation ⇒ `invalid_api_key`, management/forced release ⇒ `request_cancelled`, shutdown ⇒ `shutting_down`, Worker loss/invalidated ⇒ `worker_lost`; `cancel_timeout` is internal (client already got its terminal) |
| D10 | §5 | Waiters of an explicitly cancelled job: outcome unspecified | `model_not_ready` with reason "preparation was cancelled" (+ `job_id`) |
| D11 | §4 `max_slot_seconds` | What counts as "holding the slot" | From dispatch (T4/T5) to ticket `done`; retained/dropping time is not counted (it never blocks another client beyond one drop) |
| D12 | §4 + readiness `busy` | Whether `cancelling`, `dropping` or `quarantined` slots count as active; whether retained does | Busy ⇔ slots `reserved|dropping|quarantined` == limit; `retained` is idle (preemptible) |
| D13 | §4 Warm retention | "Retained session state never delays another client" is not literally achievable: the `abort_session` round-trip must complete first | Reword: "delays another client by at most one confirmed `abort_session`, bounded by `cancel_timeout` (then invalidation)" |
| D14 | §4 Warm retention, §1 | Exact-extension check needs token ids, but the scheduler may hold none; the hint (`user`) may be personal data | The backend adapter performs the check and reports counts; the scheduler stores only a keyed digest of the hint |
| D15 | §2 Client | Client kinds omit `loopback-anon` (auth-scopes-v1 §5) | Add it to §2 |
| D16 | §5 Shutdown | Total bound not stated; starting tickets not mentioned | Starting = running for grace; bound = grace + cancel_timeout + invalidate/release timeouts (§1.4) |
| D17 | §4 Model swap | Swap when the release fails is unspecified | Prepare fails (allocations must not overlap); profile keeps reasons; operator action or invalidation clears |
| D18 | §5 Worker lost while queued | "re-run admission at dequeue" — only the re-check subset (§3) or all eight checks? Rate limits must not be charged twice | Dequeue runs the §3 re-check list plus readiness/resource checks 5–6; checks 2–4, 7 are not repeated and nothing is charged again |
