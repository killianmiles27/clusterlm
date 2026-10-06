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
#include <vector>

#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/objects/manifest.hpp"
#include "clusterlm/placement/model_inputs.hpp"
#include "clusterlm/placement/placement.hpp"

namespace clusterlm::planning {

struct CostInputOptions {
  // Measured aggregate routing frequencies (layer x expert, rows summing to n_active). Empty = uniform
  // synthetic placeholder.
  std::vector<std::vector<double>> routing_freq;
  placement::Provenance routing_provenance = placement::Provenance::kSynthetic;
  std::optional<placement::Quantity> draft_ms;  // measured Father per-round overhead; default synthetic
  std::uint64_t lookup_working_set_bytes = 0;   // Father SSD-backed lookup cache budget counted in RAM
};

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
