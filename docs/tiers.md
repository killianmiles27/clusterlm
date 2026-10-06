# Tiers: catalog, readiness and the Father service

Fast, Strong and Ultra are data in `fixtures/catalog/clusterlm-catalog.json`, evaluated by
`clusterlm::catalog` and used through `clusterlm::father::FatherService`. See ADR 0170 and 0171.

## Catalog format

| Field | Meaning |
|---|---|
| `tiers[].id` | Exactly `fast`, `strong`, `ultra`. |
| `model` | `family`, `display_name` (the name every user-facing event uses), `artifact_id` (null until known), `quant`, `expected_files` (role, name, approx_bytes; null = recorded at first inspection), `expected_root_hash` (null = **unpinned**) and `pin_status`. |
| `backend` | `llama-local` (Fast, Father only) or `strata-hybrid` (Strong, Ultra). |
| `roles` | Pipeline-ordered roles: `father`, `node:laptop-class`, `node:designated-3060`. Never machine names. |
| `contexts` | 4K/8K/16K/32K/64K/128K profiles: `offered`, `qualified` (always false for now), `requirements`. |
| `performance_targets` | Goals with `status: pending_qualification` (Ultra: `decode_tok_s_median` 20). Never a promise. |
| `on_session_loss` | `max_retries` of the same tier, then `downgrade` or `stop`. |
| `fallback_order` | `ultra`, `strong`, `fast`. |

| Tier | Model | Roles |
|---|---|---|
| Fast | Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S | Father |
| Strong | Swift-1.5-Qwen3.8-Flash-Next GSQ-RCO IQ2_XS | Father + laptop-class node (G14) |
| Ultra | Qwen3.8-Flash-Next GSQ-RCO IQ3_S (ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF; transformer shard about 54.8 GB, lookup shard about 28.8 GB) | Father + laptop-class + designated 3060 |

Validation rejects unknown roles, missing fields, a third node in Ultra (Node 3 is excluded), wrong node order,
`llama-local` across nodes, a `pin_status` that disagrees with the hash, and nesting deeper than 16 or input over
1 MiB. A `TierAssignment` binds roles to the user's paired machines; every role must be bound and a machine may fill
only one role per tier.

## Readiness

`evaluate(tier, inputs)` is a pure function. States:

| State | Meaning |
|---|---|
| Unavailable | A prerequisite is missing; `reasons` says which. |
| Available | Everything required is present; the tier can be prepared. |
| Preparing | A node is Preparing or provisioning is in flight. Progress percent and an ETA labelled "estimate"; no measured/estimated rate means no ETA. |
| Ready | Plan current **and** every required domain Ready. Never reported otherwise. |

Inputs and the messages they produce:

| Input | Examples |
|---|---|
| Model: manifest present, hashes verified, pinned hash match or user-confirmed manifest | "Qwen3.8-... is not downloaded on Father", "The Ultra model is unpinned; confirm its inspected manifest to enable this tier" |
| Backend `hardware_available` | "The strata-hybrid backend cannot run on this machine" |
| Machine state per role | "G14 is in use" (Busy), "3060 needs cleanup", "G14 is offline", "G14 is releasing its previous lease", "G14 is serving another session" |
| Pairing, power | "3060 is not paired", "G14 is on battery power", "G14 is in battery saver mode" |
| Context profile | "128K context is not offered for Fast"; unqualified contexts add a non-blocking note |
| Placement | "No feasible placement for 32K context on Ultra" |
| Progress | "Preparing Ultra — 41% (about 3 min, estimate)" |

`evaluate_with_fallback` adds `suggested_fallback`: the lower tiers (catalog order) that are Ready or Available.

## Father service and fallback policy

`FatherService` offers list/select/prepare/chat/cancel/release/diagnostics with a typed event stream (tier selected,
prepare progress, tier ready, tokens, fallback, finished, error, released). Chat roles, text and tokenization exist
only in this Father-local layer; the Coordinator receives token arrays. Diagnostics hold counts and states and omit
the conversation unless explicitly requested; logs carry ids and counts only.

When a distributed session is invalidated (node lost, local activity, fault) the conversation is kept and the
catalog policy applies: retry the same tier up to `max_retries` while it is still observed viable, then downgrade to
the first viable lower tier, else stop with an error. The FallbackEvent is readable, for example "G14 is in use —
switching from Ultra (Qwen3.8-Flash-Next GSQ-RCO IQ3_S) to Fast (Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S); your
conversation is kept and the answer continues from where it stopped." Each token event and the final stats say which
model produced them; a model is never substituted silently.

## Pending qualification

Everything below needs the real machines and is tracked in HARDWARE-QUALIFICATION.md:

- Manifest root hashes (all tiers are unpinned): HQ-TIER-01.
- Every context profile's `qualified` flag and whether 64K/128K stay offered: HQ-TIER-01, HQ-PERF-02.
- Ultra's 20 tok/s median target: HQ-PERF-01; Strong HQ-STRONG-01; Fast HQ-FAST-01.
- Preparation ETAs: rates come from the readiness source; real transfer and upload rates are HQ-PROV-01 / HQ-PERF-03.
- The `llama-local` and `strata-hybrid` backends are skeletons here; the service is exercised on the fixture model
  through the reference backend, so a model name in the tests is catalog identity, not the executed weights.
