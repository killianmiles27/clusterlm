#pragma once
// Planning bridge: connects the measurement-driven placement search to the executable runtime.
//
//   ModelManifest --cost_inputs_from_manifest--> ModelCostInputs --search_placements--> PlacementPlan
//   PlacementPlan --to_cluster_plan--> ClusterPlan (stage ranges, Node indices, per-object GPU/CPU targets)
//
// Byte sizes come from the manifest (structural truth for the selected artifact). Routing frequencies and
// Father-side draft time are not knowable from the manifest: unless measured aggregates are supplied, they
// are filled with explicitly Synthetic placeholders, so any plan built on them stays Synthetic.
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/objects/manifest.hpp"
#include "clusterlm/placement/model_inputs.hpp"
#include "clusterlm/placement/placement.hpp"

namespace clusterlm::planning {

struct CostInputOptions {
  // Measured aggregate routing frequencies (layer x expert, rows summing to n_active). Empty = uniform
  // synthetic placeholder. Normally filled by apply_routing_aggregates.
  std::vector<std::vector<double>> routing_freq;
  placement::Provenance routing_provenance = placement::Provenance::kSynthetic;
  std::string routing_source;  // recorded in ModelCostInputs::source
  std::optional<placement::Quantity> draft_ms;  // measured Father per-round overhead; default synthetic
  std::uint64_t lookup_working_set_bytes = 0;   // Father SSD-backed lookup cache budget counted in RAM
};

// Aggregate routing statistics for placement: how often each (layer, expert) is selected per position. These
// are the ONLY routing information any planner input may carry: no token ids, text, prompts, per-position or
// per-sequence routing, and no per-request data. Loading rejects any field outside the schema below, so a
// file holding sequences cannot be accepted by accident.
//
//   { "schema": "clusterlm.routing_aggregates.v1",
//     "provenance": "synthetic" | "measured",       // "qualified" is refused: only mark_qualified qualifies
//     "source": "bench:run-id",
//     "n_layers": L, "n_experts": E, "n_active": K,
//     "positions": N,                                // positions aggregated (required with "counts")
//     "frequencies": [[...E...] x L]  |  "counts": [[...E...] x L] }   // counts / positions = frequency
//
// Rows must sum to n_active (frequencies: within 1e-3 relative, then renormalised exactly; counts: within
// rounding of positions * n_active) and each entry lies in [0, 1].
struct RoutingAggregates {
  std::vector<std::vector<double>> frequencies;
  placement::Provenance provenance = placement::Provenance::kSynthetic;
  std::string source;
  std::uint64_t positions = 0;  // 0 when the file gave frequencies only
  std::uint32_t n_layers = 0, n_experts = 0, n_active = 0;
};
Result<RoutingAggregates> routing_aggregates_from_json(std::string_view json);
Result<RoutingAggregates> load_routing_aggregates(const std::string& path);
std::string to_json(const RoutingAggregates& a, int indent = 2);
// Feeds the aggregates into cost_inputs_from_manifest (frequencies, provenance, source).
void apply_routing_aggregates(CostInputOptions& options, const RoutingAggregates& aggregates);

Result<placement::ModelCostInputs> cost_inputs_from_manifest(const objects::ModelManifest& manifest,
                                                             const CostInputOptions& options = {});

// Converts a placement plan into an executable ClusterPlan. `node_index` maps placement domain ids to the
// Coordinator's Node indices; the Father domain id maps to Father. GPU-resident experts chosen by the
// residency fill become kGpuResident objects; every other assigned object is kCpuResident.
Result<coordinator::ClusterPlan> to_cluster_plan(const placement::PlacementPlan& plan,
                                                 const objects::ModelManifest& manifest,
                                                 const std::string& father_id,
                                                 const std::map<std::string, int>& node_index,
                                                 std::uint32_t max_context, std::uint32_t max_window);

}  // namespace clusterlm::planning
