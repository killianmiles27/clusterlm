#pragma once
// Placement reports for tools (ClusterLM Bench calls these; Bench itself is not edited by the planner).
//
// Both reports state provenance at the top level and on every plan, list the provenance of each input class,
// and never carry anything but aggregate numbers and layer/domain identifiers (no tokens, prompts, logits or
// routing sequences). Placement tools label results Synthetic or Measured; they never emit Qualified.
#include <string>

#include "clusterlm/placement/placement.hpp"
#include "clusterlm/placement/workload.hpp"

namespace clusterlm::planning {

// Single-search report: the placement report (placement::report_to_json) extended with the inputs' provenance
// and the search statistics. Schema "clusterlm.placement_report.v2".
std::string placement_report_json(const placement::PlacementRequest& request, const placement::PlacementResult& result,
                                  bool include_expert_lists = false, int indent = 2);

// Workload report: per-context Pareto frontiers, per-q summaries, the recommended plan for the described
// workload, and notes about infeasible contexts. Schema "clusterlm.placement_workload_report.v1".
std::string placement_report_json(const placement::PlacementRequest& request, const placement::WorkloadResult& result,
                                  bool include_expert_lists = false, int indent = 2);

}  // namespace clusterlm::planning
