# 0401: Backend capability descriptors beside the BackendAdapter

## Context
`BackendAdapter`/`BackendInfo` describe how to create domains, not what a backend can honestly do. Compatibility decisions
(library, profile validation, UI, API) need declarative, versioned, evidence-linked capabilities. Strata and llama.cpp differ
sharply (cross-machine layer partitioning and MTP vs Host-only with recompute rollback).

## Decision
- Add `BackendDescriptor` (JSON-schema'd, `docs/interfaces/backend-capability-v1.md`), a read-only `BackendRegistry` and
  `check_model()`; keep `BackendAdapter` unchanged. Descriptors are embedded in the binary so a missing file can never
  advertise a backend.
- Ids: `llama-local` (factory `llama`), `strata-hybrid` (`strata`), `reference` (`reference`; development only). The factory
  rule stays: unknown or unbuilt backends never fall back.
- Capabilities are explicit booleans/enums; unknown is not a value. `validated_max_workers` is the largest topology with an
  automated test behind it (currently 2 for `strata-hybrid`, 0 for `llama-local`); profile validation enforces it.
- Labels are exactly the four spec labels, computed (never typed): ceiling = backend label; GGUF parse alone ≤ `Experimental`;
  `Supported and qualified` requires recorded passing hardware experiments listed in the descriptor.
- Tool calling is a template/parser concern of the serving layer, not a backend capability; backends expose only
  prerequisites (full logits, constrained decoding).

## Consequences
F owns real descriptors; A ships provisional ones for `reference`, `llama-local`, `strata-hybrid` flagged for F's review and
implements the type/registry (small exception to F's ownership of `runtime/`, CMR-0001). Distributed llama.cpp stays an
investigation: no `cross_machine: true` without a validated mode and test.
