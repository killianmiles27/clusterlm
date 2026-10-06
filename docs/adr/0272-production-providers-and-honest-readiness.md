# 0272 — Production providers, JSON IPC, and readiness that cannot lie

## Context

FatherService takes a DeploymentProvider and a ReadinessSource. The product needs real implementations driven by
persisted settings, a UI protocol over the existing opaque `kFatherRequest` payloads, and a guarantee that a build
without a real inference backend never shows a tier as Ready.

## Decision

* `ConfigDeploymentProvider` / `LiveReadinessSource` live in `orchestrator/father-service` next to the service.
  Assignments are global role -> fingerprint (catalog `TierAssignment`); the service is rebuilt when they change.
* Backend availability is a build fact: without Strata/llama the readiness input says "backend not available in
  this build" and the catalog rules keep every tier Unavailable. The only override is `--dev-fixture-model`, which
  swaps in the reference backend and a byte tokenizer, auto-confirms the unpinned model, and is stated in the tier
  details and the `hello` reply.
* Placement uses Measured profiles from the bench results directory when present, else the Synthetic fixtures; the
  provenance is stated in the tier details. Nothing is promoted beyond what the files say.
* The IPC payloads are versioned JSON (`docs/father-ipc.md`): self-describing, easy for a UI in any language, and
  bounded by the 1 MiB frame cap. Operation failures are normal replies with `ok:false`; only malformed requests
  become an `Ack`. Model output is serialized with replacement of invalid UTF-8.
* Session state seen by readiness is tracked from the service's own events, because `diagnostics()` itself
  observes readiness (calling it from an observation recursed and deadlocked in the first version).

## Consequences

Readiness probes open a short control connection to each idle Node (offer = Available); they are skipped while a
session is active. Provisioning progress from the Coordinator is not available (no callback) and is an injection
point. Real-hardware behaviour is pending HQ-PAIR-01 and the tier experiments.
