# Backend capability contract v1

Frozen by ADR 0401. Schema: [backend-descriptor-v1](schemas/backend-descriptor-v1.schema.json). Examples (provisional,
for F to confirm): [llama-local](examples/backend-llama-local.example.json),
[strata-hybrid](examples/backend-strata-hybrid.example.json), [reference](examples/backend-reference.example.json).

The existing `domain::BackendAdapter` (`create_domain`) and `BackendInfo` stay exactly as they are: they are the **runtime**
half. This contract adds the **declarative** half, so profiles, the model library, the UI, the API and the scheduler can ask
"can this backend run this model on this topology, and how sure are we?" without instantiating anything.

## 1. Shape

```cpp
// runtime/domain (type + registry only; descriptors for real backends live with each backend, owned by F)
struct BackendDescriptor;                       // exact mirror of the JSON schema; parse/serialize + validate
struct BackendRuntimeStatus {                   // dynamic, probed on this machine
  bool built;                                   // backend_built(name)
  bool runtime_present;                         // CUDA runtime/driver, DLLs, etc.
  bool hardware_available;                      // == BackendInfo::hardware_available
  std::string build_hash;                       // == BackendInfo::build_hash; Host and Workers must agree
  std::vector<std::string> reasons;             // human sentences when any bool is false
};
class BackendRegistry {                         // process-wide, read-only after startup
 public:
  std::vector<const BackendDescriptor*> list() const;
  const BackendDescriptor* find(std::string_view id) const;
  BackendRuntimeStatus probe(std::string_view id) const;     // cheap, no allocation of model state
};
struct CompatReport {                           // result of check_model(); also what the UI shows
  std::string label;                            // one of the four labels (below)
  std::vector<Finding> findings;                // {severity: blocker|warning|note, code, message}
  bool hosts_alone, distributable;              // derived, see §3
  std::uint32_t max_workers;                    // descriptor limit applied to this model
};
CompatReport check_model(const BackendDescriptor&, const ModelRecord&, const BackendRuntimeStatus&);
```

`BackendAdapter::info()` keeps returning `BackendInfo`; the registry's `probe` is implemented by calling it. Adding a backend =
(1) a descriptor file/const, (2) a `BackendAdapter`, (3) a `backend_factory` entry, (4) an ADR extending the `id` enum.
The descriptor `id` ↔ factory name mapping is fixed by the schema (`llama-local`↔`llama`, `strata-hybrid`↔`strata`,
`reference`↔`reference`). The factory's rule is unchanged: an unknown or unbuilt backend never silently falls back.

## 2. What a descriptor must declare (all required unless marked)

| Field group | Meaning | Consumer |
|---|---|---|
| `model_support` | containers, executable tensor formats, per-family status + evidence, `unlisted_family_status` | library compat check, profile validation L2 |
| `devices` | CPU, GPU vendors, min compute capability, runtime prerequisites | probe; never offered if absent |
| `execution` | single-host / cross-machine, **validated** max workers + tested topologies, layer partitioning, CPU/GPU hybrid, token-free middle stages, manual ranges, which roles a Worker plays | profile validation L2, placement, UI |
| `state` | state kinds, rollback mode (`native`/`recompute`/`none`), per-client isolation, max sessions per domain | scheduler admission (concurrency, session isolation) |
| `speculation` | window commit/abort, MTP form, max q | profile `speculation`, scheduler |
| `serving` (optional) | full logits, constrained decoding, logprobs | serving layer refuses unsupported params instead of emulating |
| `limitations[]` | honest list shown in the UI compatibility report | UI, docs |
| `qualification` | label + HQ experiment ids required for the next label | status ledger, UI |

Tool calling is **not** a backend capability: it is the model's chat template + parser, resolved by the serving layer
(workstream B). The backend contributes only prerequisites (`full_logits_for_sampling`, `constrained_decoding`). Structured
output (`response_format`) is accepted only when `constrained_decoding` is true; otherwise the API returns
`unsupported_parameter` rather than prompting the model to "please output JSON".

## 3. Labels and how they are computed

Exactly the four user-facing labels. For model M on backend B:

1. If M's container/tensor format is not executable by B → `Unsupported` (blocker finding naming the tensor type).
2. Else if M's family matches a descriptor `families[]` entry → that entry's `status` (listed evidence shown).
3. Else the descriptor's `unlisted_family_status`. `Experimental` means: Host-only load is attempted only after an explicit
   user opt-in per model, it is never distributable, and the UI says so.
4. Parsing a GGUF successfully is never sufficient for anything above `Experimental`.
5. The backend-level `qualification.label` is a ceiling: no model gets a higher label than its backend.
6. `distributable = execution.cross_machine ∧ status ∈ {Supported and qualified, Supported, awaiting hardware qualification}`;
   `Experimental` and `Unsupported` models are never distributable. `max_workers = descriptor.validated_max_workers`.

Raising a label to `Supported and qualified` requires the listed `hardware_experiments` to have recorded passing results
(format owned by F, enforced by a G check): the descriptor change and the evidence land in the same commit. Fixture,
fake-engine or reference-backend results never raise a label (they appear only in `evidence` as mechanics proof).

## 4. Runtime invariants every backend must honour

* All allocation is reported by `describe_requirements()` before `prepare()` and is bounded by `DomainSpec`.
* Sequence state and speculative windows follow `execution_domain.hpp` verbatim (single outstanding window, commit 1..q,
  idempotent commit, abort_window, abort_session, stale epoch rejection).
* `release()` frees everything; Worker-side nothing survives a lease (no persistent weight caches).
* No silent requantization or model substitution: provisioning moves exact manifest byte ranges.
* `build_hash` identifies the exact build; Host and Workers must match for a plan to be admitted.

## 5. Future-backend rule

A backend that cannot truthfully fill every required field is not added. No "unknown"/"maybe" values: fields are explicit
booleans/enums so a missing capability is `false`, not omitted. AMD/Intel/CPU-only Workers are new descriptors (`gpu_vendors`,
`cpu_only_ok` in worker requirements) plus a `BackendAdapter`; no schema change is needed beyond the id enum.

## 6. Open items handed to F (not blockers for A/B/C)

* Confirm/replace the provisional example descriptors (family lists, rollback mode, isolation, tensor formats).
* Define the evidence/qualification-results record and the descriptor loader (embedded const vs shipped JSON; recommendation:
  embedded, so a missing file can never advertise a backend).
* Decide whether `llama-local` gains a distributed mode (investigation only; the descriptor stays `cross_machine: false`
  until a validated mode and test exist).
