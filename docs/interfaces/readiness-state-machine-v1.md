# Readiness state machine v1

Frozen by ADR 0404. Replaces the four-state `catalog::TierState` (Unavailable / Available / Preparing / Ready) with the seven
states the spec lists, keeping its core rule: **`ready` is only ever the output of a pure function over observed inputs and is
never inferred, remembered or defaulted.** Existing machine-level `MachineState` (Busy / Available / Preparing / Ready /
Inferencing / Releasing / CleanupPending / Offline) is unchanged.

## 1. States (per profile)

| State | Meaning | Evidence required |
|---|---|---|
| `unavailable` | Cannot be used now. `reasons[]` ≥ 1 (first is the headline); `blockers[]` machine codes | any blocker below |
| `installed` | Profile exists; its backend is built and its runtime present; model not yet verified against the library | backend probe ok |
| `compatible` | Model record verified (hashes), pin satisfied or user-confirmed, descriptor accepts model/topology, validation levels 1–2 pass; but required machines are not all usable now | `check_model` ok, pin ok |
| `loadable` | Compatible **and** every required slot is satisfiable by a currently usable machine **and** a feasible placement exists for the requested context. Can be prepared (old `Available`) | machines usable, `plan.feasible_for_context` |
| `preparing` | A prepare job for this profile is running (provisioning / node-preparing) with progress | job state |
| `ready` | Prepared plan current **and** every required domain Ready under that plan (old `Ready`) | `plan.plan_ready` + all domains Ready |
| `busy` | `ready` and at its concurrency limit with an active generation (ADR 0405) | active request count == limit |

`unavailable` carries the *furthest level reached* (`reached: installed|compatible|none`) so the UI can say "compatible but
Worker A is in use" rather than a bare "unavailable".

## 2. Transitions

```
              backend/runtime fixed               model verified+pinned
 unavailable ◄───────────────────► installed ───────────────────────► compatible
     ▲  ▲                                                                │ ▲
     │  │  slot unsatisfiable / worker busy,offline,battery,             │ │ slot lost / placement infeasible
     │  └─────────────── placement infeasible ──────────────────────────┐ ▼ │
     │                                                                  loadable ──prepare job──► preparing
     │   plan invalidated (worker lost, local user returned, fault)        ▲   ▲                    │ │
     ├──────────────────────────────────────────────── ready ◄────────────┘   └── cancel / failure ──┘ │
     │                                                  │ ▲ request starts / ends                       │ success
     │                                                  ▼ │                                          ▼
     │                                                busy                                          ready
     └── idle timeout / explicit release / lease expiry: ready → loadable (leases released, Worker memory returned)
```

Rules:

* Every regression is immediate and observed (Node state events, power events, lease expiry), not polled slowly; a
  missing observation ages out: inputs carry `observed_at`; older than `staleness_ttl` (default 15 s for machine state) ⇒
  treated as unknown ⇒ at best `compatible` with reason "G14: no recent status" — stale data never yields `ready`.
* `preparing → loadable` on cancel or failure, with `last_error` kept (shown, never hidden). `preparing → unavailable` if a
  blocker appears mid-job.
* `ready`/`busy → unavailable` when the plan is invalidated; in-flight requests end with a structured error (scheduler doc §5).
* `ready → loadable` on idle release; the API's `/ready` reports the model as not prepared from that instant.
* A model change/swap is never a transition of an active request: the scheduler queues the swap behind it (ADR 0405).

## 3. Inputs (extends `catalog::ReadinessInputs`; same purity, same reason strings where they exist)

`ModelAvailability` (manifest present, hashes verified, root, user-confirmed root), `BackendAvailability` (+ `built`,
`runtime_present`), per-slot `MachineInputs` (selector result, `MachineState`, paired, power incl. battery policy, advertised
capability descriptor, `observed_at`), `ProvisioningProgress`, `PlanReadiness`, `tokenizer_problem`, `active_requests`,
`concurrency_limit`, `job: {id, state, last_error?}`. Reasons keep the user-readable catalog sentences ("G14 is in use",
"Qwen… is not downloaded on Host") with machine names replaced by the user's Worker display names.

## 4. External surface

| Surface | Reports |
|---|---|
| IPC `profiles.list` / MCP `clusterlm_list_profiles` | `state`, `reached`, `reasons[]`, `notes[]`, `progress?` (percent + ETA flagged `estimate`; no rate ⇒ no ETA), `suggested_alternatives[]` (profile ids that are ready/loadable) |
| `GET /health` | process liveness only: `{"status":"ok"}`; no model information |
| `GET /ready` | per visible model: `{"model":"<api_model_id>","loadable":bool,"prepared":bool,"state":"…"}`; HTTP 200 if the server is up, body carries per-model truth; `?model=` form returns 503 unless that model is `prepared`. **loadable ≠ prepared.** |
| `GET /v1/models` | exposed models (profiles + aliases visible to the key) with `x_clusterlm.state`; listing a model is not a promise that it is ready |
| Chat completions | `ready`/`busy` ⇒ admit/queue; `loadable` ⇒ bounded wait + prepare per lifecycle policy, else structured `model_not_ready` carrying the job id; anything lower ⇒ structured error with reasons |

## 5. Mapping from today's tiers

`Unavailable` → `unavailable` (or `installed`/`compatible` by `reached`); `Available` → `loadable`; `Preparing` → `preparing`;
`Ready` → `ready`; `busy` is new. Existing tests of `evaluate()` keep their expectations through this mapping; A extends the
function (new states, `reached`, staleness) without changing any existing reason text for the same condition.
