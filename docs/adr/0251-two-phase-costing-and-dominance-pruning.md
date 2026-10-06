# ADR 0251: Two-phase costing, dominance pruning and a 3-D Pareto frontier

**Status:** Accepted.

## Context

Sweeping prefix sizes, tail sizes, every ordered subset of up to three nodes and several context/q points
produces thousands of stage assignments per search. Materialising a full `PlacementPlan` (ledgers, expert
lists per layer, provenance strings, hash) for each of them costs far more than costing it, and most of them
can never be recommended.

## Decision

1. `Evaluator::cost` computes metrics from memoised per-(domain, ranges) admission and residency
   (`DomainEval`) and returns a light `Costed`; `materialize` builds the `PlacementPlan` only for survivors.
   The plan hash is `SHA-256(header, stages, per-domain digests)`; each domain digest is computed once per
   (domain, ranges) and covers its ledger, local batch and expert residency.
2. With `prune_dominated`, a candidate is dropped when another is no worse on ALL of `prepare_s`,
   `effective_prepare_s`, `lease_overrun_s`, `prefill_s`, `decode_tok_s` and strictly better on one. Those are
   exactly the quantities the objective and the Pareto frontier read (the objective is monotone in the last
   four, with non-negative weights), so pruning cannot change the recommended plan, the best-throughput or
   best-preparation plan, or the frontier. It is off by default so callers that look up plans by key still see
   every feasible plan; workload searches turn it on.
3. The frontier is 3-D: minimise `prepare_s`, maximise `decode_tok_s`, minimise `prefill_s`; equal-metric
   plans keep the earliest in the deterministic order.
4. Candidate order is objective, then layer boundaries, then plan hash: ties never depend on domain names.

## Consequences

Search for 48 layers and 3 remote nodes enumerates a few thousand assignments in a fraction of a second in
RelWithDebInfo. `PlacementResult::pareto` changed meaning from 2-D to 3-D (a superset of the old frontier).
