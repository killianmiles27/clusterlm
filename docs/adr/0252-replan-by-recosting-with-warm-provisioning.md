# ADR 0252: Replan decision = re-cost the deployed plan, compare against a warm search

**Status:** Accepted.

## Context

Inputs drift: a node's safe RAM shrinks, a CPU slows, a link degrades, the context requirement grows, a node
leaves, a lease estimate shortens. Replanning has a cost (moving bytes to nodes) and churn is itself harmful,
so "the best plan changed" is not a sufficient trigger.

## Decision

`planning::should_replan(current, inputs)`:

1. Re-evaluates the current stage assignment under the new inputs (`evaluate_plan`). Failing admission, or
   losing a node, forces a replan.
2. Searches the new world with `already_provisioned` set to what the nodes hold for the current plan
   (`current_plan_provisioned`, default true). The candidate therefore pays amortised provisioning only for
   bytes it would newly move, while keeping the current plan costs none: switching cost is inside the
   comparison, which is also the hysteresis.
3. Replans for quality only if `candidate.objective < current.objective * (1 - min_objective_gain)`
   (default 10%).
4. Reasons name what moved by comparing the current plan's own earlier prediction with its re-costed one:
   the stage (domain and layer range) or link whose time grew, prefill and amortised-preparation increases,
   new lease-overrun flags, context growth. Reprovisioning is reported in bytes and seconds of the candidate
   when replanning, otherwise of the current plan (non-zero only when it is not provisioned).

## Consequences

Decisions are direction-correct under the stated cost model, not optimal. A running deployment ignores a
slower provisioning link, a cold one does not. Thresholds are policy (`ReplanPolicy`), not hardware claims.
