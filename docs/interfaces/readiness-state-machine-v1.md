# Readiness state machine v1

Frozen v1.1: ADR 0404, revised by ADR 0407. Replaces the four-state `catalog::TierState` (Unavailable / Available / Preparing /
Ready) with the seven states the spec lists, keeping its core rule: **`ready` is only ever the output of a pure function over
observed inputs and is never inferred, remembered or defaulted.** Existing machine-level `MachineState` (Busy / Available /
Preparing / Ready / Inferencing / Releasing / CleanupPending / Offline) is unchanged.

## 1. States (per profile, per context class)

Readiness is evaluated for a context class: the smallest `offered_profiles` entry ≥ the needed context (a request's prompt +
completion budget), or `context.default_tokens` when no context is given (lists, `/ready`, `/v1/models`).

The state is the **highest** row whose evidence holds, with one split: a *blocker* (something that needs a user or environment
action: backend not built, runtime missing, model missing/ambiguous/hash mismatch/unconfirmed pin, incompatible, invalid
document, unassigned binding, Experimental without opt-in) yields `unavailable`; a *transient* condition (a Worker busy,
offline, on battery, paused, stale observation, no feasible placement with the memory free right now) leaves the profile at
`compatible`. v1.0 drew busy/offline Workers both as `compatible` (table) and as `unavailable` (diagram); v1.1 fixes it to
`compatible`, so the UI can say "compatible but Worker A is in use".

| State | Meaning | Evidence required |
|---|---|---|
| `unavailable` | Cannot be used until something is fixed. `reasons[]` ≥ 1 (first is the headline); `blockers[]` machine codes; `reached` | any blocker |
| `installed` | Profile valid (L1); its backend is built and its runtime present; model verification not finished yet (scan/hash in progress) | backend probe ok |
| `compatible` | Model record resolved and verified, pin satisfied or user-confirmed, `check_model` accepts model/topology, validation levels 1–2 pass; but the profile is not loadable right now for a transient reason (listed in `reasons[]`) | `check_model` ok, pin ok |
| `loadable` | Compatible **and** every required slot is satisfiable by a currently usable machine (fresh observation, matching backend `build_hash`) **and** a feasible placement exists for the context class. Can be prepared (old `Available`) | machines usable, `plan.feasible_for_context` |
| `preparing` | A prepare job for this profile and context class is `queued` or `running`, with progress (`phase`, percent, ETA flagged estimate) | job state |
| `ready` | A prepared plan for this profile's current model/backend/topology revision and a context ≥ the class **and** every required domain Ready under that plan (old `Ready`) | `plan.plan_ready` + all domains Ready |
| `busy` | `ready` and at its concurrency limit with active generations (ADR 0405) | active request count == limit |

`unavailable` carries the *furthest level reached* (`reached: none|installed|compatible`) so the UI can say how far the profile
got (e.g. "model found but its hash does not match the pinned one").

"Prepared" in the API means `ready` or `busy`.

## 2. Transitions

The state is recomputed from inputs on every input change; the arrows below are the consequences, not separate rules.

```
              backend/runtime fixed               model verified + pinned
 unavailable ◄───────────────────► installed ───────────────────────► compatible
     ▲                                                                 │ ▲
     │ blocker appears (any state)              slots usable +         │ │ slot not usable (busy, offline, battery,
     │                                          placement feasible     ▼ │ paused, stale) / placement infeasible
     │                                                                loadable ──prepare job queued──► preparing
     │                                                                  ▲   ▲                           │ │
     │                                                                  │   └── cancel / failure ───────┘ │ success
     │                         plan invalidated (worker lost,           │                                 ▼
     │                         local user returned, fault, stale) ──────┴─────────────────────────────── ready
     │                         → whatever the inputs now give                                       request │ ▲ ends
     │                                                                                              starts  ▼ │
     │                                                                                                    busy
     └── idle timeout / explicit release / lease expiry / swap: ready → loadable (or lower; leases released, Worker memory returned)
```

Rules:

* Every regression is immediate and observed (Node state events, power events, lease expiry), not polled slowly; a
  missing observation ages out: inputs carry `observed_at`; older than `staleness_ttl` (default 15 s for machine state) ⇒
  treated as unknown ⇒ at best `compatible` with reason "G14: no recent status" — stale data never yields `loadable` or `ready`.
* `preparing → loadable` (or lower) on cancel or failure, with `last_error` kept (shown, never hidden). A blocker appearing
  mid-job gives `unavailable` and the job is cancelled.
* `ready`/`busy` → lower when the plan is invalidated; in-flight requests end with a structured error (scheduler doc §5).
* `ready → loadable` on idle release or swap; the API's `/ready` reports the model as not prepared from that instant.
* A profile edit that changes model identity, backend, topology or options makes an existing plan stale: the profile is no
  longer `ready` for the new revision (scheduler-admission-v1 §1).
* A model change/swap is never a transition of an active request: the scheduler queues the swap behind it (ADR 0405).

## 3. Inputs (extends `catalog::ReadinessInputs`; same purity, same reason strings where they exist)

`ModelAvailability` (library resolution kind incl. `ambiguous`, manifest present, hashes verified, root, user-confirmed root,
Experimental opt-in), `BackendAvailability` (+ `built`, `runtime_present`, `build_hash`), per-slot `MachineInputs` (selector
result, `MachineState`, paired, power incl. battery policy, Host-side pause, advertised capability descriptor incl. backend
`build_hash`, `observed_at`), `ProvisioningProgress`, `PlanReadiness` (+ plan `profile_revision`, plan context), `context_tokens`
(the class), `tokenizer_problem`, `active_requests`, `concurrency_limit`, `job: {id, state, phase?, last_error?}`,
`document_problem` (quarantined profile). Reasons keep the user-readable catalog sentences ("G14 is in use", "Qwen… is not
downloaded on Host") with machine names replaced by the user's Worker display names (and, for API keys without `status:read`,
by generic sentences; auth-scopes-v1 §5).

## 4. External surface

| Surface | Reports |
|---|---|
| IPC `profiles.list` / MCP `clusterlm_list_profiles` | `state`, `reached`, `reasons[]`, `notes[]`, `progress?` (percent + ETA flagged `estimate`; no rate ⇒ no ETA), `suggested_alternatives[]` (profile ids that are ready/loadable) |
| `GET /health` | process liveness only: `{"status":"ok"}`; no model information; no key needed |
| `GET /ready` (`inference`) | per model visible to the key: `{"model":"<api_model_id>","loadable":bool,"prepared":bool,"state":"…"}` where `loadable` = state ∈ {loadable, preparing, ready, busy} and `prepared` = state ∈ {ready, busy}; HTTP 200 if the server is up, body carries per-model truth; `?model=` form returns 200 if prepared, 503 if visible but not prepared, 404 if not visible or nonexistent. **loadable ≠ prepared.** |
| `GET /v1/models` | exposed models (profiles + aliases visible to the key) with `x_clusterlm.state`; listing a model is not a promise that it is ready |
| Chat completions | `ready`/`busy` ⇒ admit/queue; `loadable` ⇒ bounded wait + prepare per lifecycle policy, else structured `model_not_ready` carrying the job id; `preparing` ⇒ bounded wait on the existing job, else `model_not_ready` + job id; anything lower ⇒ structured error with reasons |

## 5. Mapping from today's tiers

`Unavailable` → `unavailable` when the reason is a blocker, `compatible` when it is transient (Worker busy/offline/battery,
placement infeasible), `installed` while verification is pending; `Available` → `loadable`; `Preparing` → `preparing`;
`Ready` → `ready`; `busy` is new. Existing tests of `evaluate()` keep their expectations through this mapping (the reason
strings are unchanged; only the state name for transient reasons moves from `Unavailable` to `compatible`); A extends the
function (new states, `reached`, staleness, build hash, revision) without changing any existing reason text for the same
condition.

## Amendments

* v1.1 (2026-10-10, ADR 0407): blocker vs transient rule (busy Worker ⇒ `compatible`), readiness per context class,
  `preparing` includes a queued job, plan revision and build-hash inputs, exact `/ready` booleans and 404 for invisible
  models, `installed` defined as verification pending, tier mapping made precise.
