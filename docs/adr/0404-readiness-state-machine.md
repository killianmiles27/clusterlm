# 0404: Seven-state readiness machine over a pure evaluator

## Context
Readiness has four tier states today. The API needs "loadable" vs "prepared" and "busy"; the UI needs to explain how far a
profile got.

## Decision
States `unavailable (reached) | installed | compatible | loadable | preparing | ready | busy` computed by a pure function over
observed inputs with timestamps; stale inputs cannot produce `ready`; regressions are immediate; `/ready` reports
`loadable` and `prepared` separately; `busy` is `ready` at its concurrency limit. Existing tier reason strings are kept.
Mapping from the old states is in `docs/interfaces/readiness-state-machine-v1.md` §5.

## Consequences
A extends `catalog::evaluate` (moved under profiles) without changing existing expectations; C feeds live job and activity
inputs; B/MCP/UI only display the output.
