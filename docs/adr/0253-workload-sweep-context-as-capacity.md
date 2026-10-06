# ADR 0253: Workload sweep: context profile is capacity, prompt is the prefill length

**Status:** Accepted.

## Context

Plans must be evaluated per context profile (4K ... 128K) because sequence state grows with the context and
displaces GPU expert residency, which can flip feasibility. The old request used `context_tokens` both as the
state capacity and as the number of tokens to prefill, which would make every large-profile plan look like a
128K-token prefill.

## Decision

`PlacementRequest` gets `prompt_tokens` (0 = `context_tokens`). State, admission and residency use
`context_tokens`; prefill time uses `prompt_tokens`. `search_workload` runs one search per (context profile, q),
sets `prompt_tokens = min(expected_prompt, context)`, `output_tokens` and the lease token share from the
described workload, merges the results of all q at a context into one 3-D Pareto frontier, and recommends the
lowest objective over every q. The workload's recommendation comes from the smallest profile covering
`prompt + output`; if none does, there is no recommendation and a note says so.

Acceptance (mean emitted tokens per round) is an input per q, never defaulted for q > 1; q = 1 may use the
request's acceptance. The workload provenance is the weakest of every acceptance used.

## Consequences

A profile whose state cannot be held is reported infeasible with the most common rejection reasons instead of
being dropped. Cost is linear in contexts x q values.
