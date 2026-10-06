#pragma once
// Workload-level placement: one search per (context profile, verify width q), merged into per-context Pareto
// frontiers and one recommendation for a described workload.
//
// A context profile is the sequence capacity a plan must hold state for (4K ... 128K). State grows with it and
// displaces GPU expert residency, so feasibility and the best cut points can change between profiles; the
// prompt that is actually prefilled is the workload's expected prompt (clamped to the profile). Each verify
// width q needs an acceptance Quantity (mean emitted tokens per round): supplied by the caller, Synthetic
// unless a benchmark measured it. The search never invents acceptance.
//
// Cost: contexts x q_values independent searches (each is search_placements with prune_dominated), so
// wall time is linear in both; searches share nothing but the immutable request.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/placement/placement.hpp"

namespace clusterlm::placement {

struct WorkloadDescription {
  std::uint32_t expected_prompt_tokens = 1024;
  std::uint32_t expected_output_tokens = 256;
  // Requests one node lease is expected to serve; sets LeaseExpectation::expected_tokens (requests *
  // output tokens) unless the caller disabled derivation.
  std::uint32_t requests_per_lease = 100;
};

struct WorkloadSpec {
  std::vector<std::uint32_t> context_profiles{4096, 8192, 16384, 32768, 65536, 131072};
  std::vector<std::uint32_t> q_values{1};
  // Acceptance per q. q = 1 may be omitted (the base request's acceptance is used); any other q must be present.
  std::map<std::uint32_t, Quantity> acceptance_by_q;
  WorkloadDescription workload;
  bool derive_lease_tokens = true;
};

struct QSummary {
  std::uint32_t q = 1;
  std::uint64_t feasible = 0, rejected = 0, pruned_dominated = 0;
  double best_decode_tok_s = 0;
  double best_objective = 0;
  std::string best_key;
};

struct ContextPlacement {
  std::uint32_t context_tokens = 0, prompt_tokens = 0;
  bool feasible = false;
  // 3-D Pareto frontier over (prepare_s, decode_tok_s, prefill_s) across every q, ordered as
  // PlacementResult::pareto. Each plan states its own q.
  std::vector<PlacementPlan> frontier;
  std::optional<PlacementPlan> recommended;  // lowest objective over every q (need not be on the frontier)
  std::vector<QSummary> per_q;
  std::vector<std::string> infeasible_reasons;  // when !feasible: the most common rejection reasons
};

struct WorkloadResult {
  WorkloadDescription workload;
  std::vector<ContextPlacement> contexts;  // one per context profile, ascending
  // Smallest profile that covers expected prompt + output; unset if none does.
  std::optional<std::size_t> selected;
  Provenance provenance = Provenance::kSynthetic;  // weakest over the base request's inputs and every acceptance used
  std::vector<std::string> notes;

  // The recommended plan of the selected context, if feasible.
  const PlacementPlan* recommended() const;
};

Result<WorkloadResult> search_workload(const PlacementRequest& base, const WorkloadSpec& spec);

}  // namespace clusterlm::placement
