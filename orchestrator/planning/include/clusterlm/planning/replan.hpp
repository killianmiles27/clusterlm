#pragma once
// Replan triggers: should the cluster keep its current plan when the inputs change?
//
// `current` is the plan that is deployed. `inputs.request` is the up-to-date view of the world: profiles
// (a node's safe RAM shrank, its CPU got slower), the network (a link degraded), the context requirement,
// lease expectations. The current stage assignment is re-costed under that view and compared with the best
// plan the search finds under it. Switching cost is part of the comparison, not an afterthought: nodes are
// assumed to hold the layers of the current plan (already_provisioned), so a candidate that needs new bytes
// pays their amortised provisioning in its objective while the current plan pays none.
//
// The decision is a rational-direction heuristic under the stated cost model, not a claim of optimality.
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/placement/placement.hpp"

namespace clusterlm::planning {

struct ReplanPolicy {
  // Replan for a better plan only if its objective is lower by at least this fraction (hysteresis against
  // churn; 0 = replan for any strictly better plan).
  double min_objective_gain = 0.10;
  // Fractional worsening of the current plan's own prediction that is reported as a reason.
  double decode_regression = 0.05;
  double prefill_regression = 0.10;
  double preparation_regression = 0.10;
  // Fractional per-stage / per-link slowdown that is named in a regression reason.
  double component_regression = 0.05;
};

struct ReplanInputs {
  placement::PlacementRequest request;     // the new world
  std::vector<std::string> unavailable_nodes;  // nodes that disappeared (removed from the search)
  // True: nodes already hold the current plan's layers (a running deployment). False: the current plan was
  // never provisioned, so its own preparation is charged.
  bool current_plan_provisioned = true;
  ReplanPolicy policy;
};

enum class ReplanReasonKind : std::uint8_t {
  kNodeUnavailable,       // a node of the current plan disappeared
  kPlanInadmissible,      // the current assignment no longer passes admission (RAM/VRAM/links)
  kContextGrowth,         // the context requirement grew beyond what the plan was costed for
  kDecodeRegression,      // predicted decode throughput of the current plan fell
  kPrefillRegression,     // predicted prefill time rose
  kPreparationRegression, // amortised preparation cost rose, or a node now overruns its lease
  kBetterPlanAvailable,   // a candidate beats the current plan by min_objective_gain
  kNoFeasiblePlan,        // nothing is feasible under the new inputs
};
std::string_view to_string(ReplanReasonKind k) noexcept;

struct ReplanReason {
  ReplanReasonKind kind = ReplanReasonKind::kBetterPlanAvailable;
  std::string detail;   // human-readable, names the domain/link; never contains activation or token data
  double magnitude = 0; // fractional change (regressions/gain) or bytes (reprovision); 0 if not applicable
};

struct ReplanDecision {
  bool replan = false;
  bool reprovision_needed = false;  // bytes must move to a node (for the candidate if replanning, else the current plan)
  double reprovision_bytes = 0;
  double reprovision_s = 0;
  std::vector<ReplanReason> reasons;
  std::optional<placement::PlacementPlan> current_under_new_inputs;  // unset if inadmissible
  std::optional<placement::PlacementPlan> candidate;                 // best plan under the new inputs
  std::vector<std::string> inadmissible_reasons;                     // admission rejections of the current plan
};

Result<ReplanDecision> should_replan(const placement::PlacementPlan& current, const ReplanInputs& inputs);

}  // namespace clusterlm::planning
