#include "clusterlm/planning/planning.hpp"

namespace clusterlm::planning {

using objects::ObjectKind;

Result<placement::ModelCostInputs> cost_inputs_from_manifest(const objects::ModelManifest& m,
                                                             const CostInputOptions& options) {
  CLM_RETURN_IF_ERROR(m.validate());
  const auto& g = m.geometry;
  placement::ModelCostInputs in;
  in.name = m.artifact_id;
  in.n_experts = g.n_experts;
  in.n_active = g.n_active_experts;
  in.layers.resize(g.n_layers);
  for (std::uint32_t l = 0; l < g.n_layers; ++l) {
    in.layers[l].kind = g.layer_kinds[l] == objects::LayerKind::kFullAttention ? placement::LayerKind::kAttention
                                                                               : placement::LayerKind::kRecurrent;
    // Recurrent state: one H-wide FP32 vector per layer in the reference runtime. The real backend reports
    // its own allocation (GDN conv + recurrent state) through describe_requirements(); replace when available.
    if (in.layers[l].kind == placement::LayerKind::kRecurrent) in.layers[l].fixed_state_bytes = std::uint64_t{g.hidden_size} * 4;
  }
  for (const auto& o : m.objects) {
    switch (o.kind) {
      case ObjectKind::kLayerDense:
      case ObjectKind::kSharedExpert:
        if (o.layer && *o.layer < g.n_layers) in.layers[*o.layer].dense_bytes += o.byte_size;
        break;
      case ObjectKind::kRoutedExpert:
        if (o.layer && *o.layer < g.n_layers) {
          auto& lc = in.layers[*o.layer];
          // Placement models one size per layer; use the largest so admission is conservative.
          lc.expert_bytes = std::max(lc.expert_bytes, o.byte_size);
          lc.quant = o.representation.quant_type;
        }
        break;
      case ObjectKind::kEmbedding: in.father_only.embedding += o.byte_size; break;
      case ObjectKind::kOutputHead: in.father_only.head += o.byte_size; break;
      case ObjectKind::kMtp: in.father_only.mtp += o.byte_size; break;
      case ObjectKind::kPleLookup: break;  // SSD-backed on Father; only the working set counts in RAM
    }
  }
  in.father_only.lookup_working_set = options.lookup_working_set_bytes;
  // FP16-equivalent K+V per attention layer per token in the reference runtime uses FP32: 2·kv·hd·4 bytes.
  in.state_bytes_per_context_token[static_cast<std::size_t>(placement::LayerKind::kAttention)] =
      std::uint64_t{2} * g.n_kv_heads * g.head_dim * 4;
  in.boundary_bytes_per_position =
      (std::uint64_t{g.residual_streams} * g.hidden_size + g.hidden_size + g.residual_streams) * 4;
  in.ple_layer = static_cast<std::int32_t>(g.ple_layer);
  if (!options.routing_freq.empty()) {
    if (options.routing_freq.size() != g.n_layers)
      return make_error(ErrorCode::kInvalidArgument, "routing aggregates cover " + std::to_string(options.routing_freq.size()) +
                                                         " layers, the manifest has " + std::to_string(g.n_layers));
    for (const auto& row : options.routing_freq)
      if (row.size() != g.n_experts)
        return make_error(ErrorCode::kInvalidArgument, "routing aggregates have " + std::to_string(row.size()) +
                                                           " experts per layer, the manifest has " + std::to_string(g.n_experts));
    in.routing_freq = options.routing_freq;
  } else {
    in.routing_freq.assign(g.n_layers, std::vector<double>(g.n_experts, static_cast<double>(g.n_active_experts) /
                                                                         static_cast<double>(g.n_experts)));
  }
  in.draft_ms = options.draft_ms.value_or(
      placement::Quantity::synthetic(1.0, "synthetic: Father draft/head/sampling time not yet measured"));
  // Bytes are structural truth from the manifest, but the routing distribution decides miss costs; the inputs
  // are only as strong as the weakest of the two.
  in.provenance = options.routing_freq.empty() ? placement::Provenance::kSynthetic : options.routing_provenance;
  in.source = "manifest " + m.root_hash().hex().substr(0, 16) +
              (options.routing_freq.empty() ? " + synthetic uniform routing"
                                              : " + routing aggregates" + (options.routing_source.empty() ? std::string() : " (" + options.routing_source + ")"));
  CLM_RETURN_IF_ERROR(placement::validate(in));
  return in;
}

Result<coordinator::ClusterPlan> to_cluster_plan(const placement::PlacementPlan& plan,
                                                 const objects::ModelManifest& m, const std::string& father_id,
                                                 const std::map<std::string, int>& node_index,
                                                 std::uint32_t max_context, std::uint32_t max_window) {
  coordinator::ClusterPlan out;
  out.max_context = max_context;
  out.max_window = max_window;
  const auto n_layers = m.geometry.n_layers;
  auto add = [&](domain::StageRole role, objects::LayerRange layers, int dom) {
    coordinator::StagePlan s;
    s.stage = StageId{static_cast<std::uint32_t>(out.stages.size())};
    s.role = role;
    s.layers = layers;
    s.domain = dom;
    out.stages.push_back(s);
  };
  for (const auto& st : plan.stages) {
    const objects::LayerRange range{st.layers.begin, st.layers.end};
    switch (st.role) {
      case placement::StageRole::kFull:
        // Father-only: one prefix over every layer plus a head-only tail.
        add(domain::StageRole::kPrefix, range, coordinator::kFatherDomain);
        add(domain::StageRole::kTail, {n_layers, n_layers}, coordinator::kFatherDomain);
        break;
      case placement::StageRole::kPrefix:
        add(domain::StageRole::kPrefix, range, coordinator::kFatherDomain);
        break;
      case placement::StageRole::kTail:
        add(domain::StageRole::kTail, range, coordinator::kFatherDomain);
        break;
      case placement::StageRole::kMiddle: {
        auto it = node_index.find(st.domain_id);
        if (it == node_index.end())
          return make_error(ErrorCode::kNotFound, "placement names unknown domain " + st.domain_id);
        add(domain::StageRole::kMiddle, range, it->second);
        break;
      }
    }
  }
  // A plan whose last layers run on a Node returns to Father for the head only.
  if (out.stages.back().role != domain::StageRole::kTail)
    add(domain::StageRole::kTail, {n_layers, n_layers}, coordinator::kFatherDomain);

  // GPU residency -> per-object targets.
  for (const auto& res : plan.residency) {
    if (res.domain_id == father_id) continue;  // Father's local domains bind the canonical store directly
    for (std::size_t layer = 0; layer < res.gpu_experts_by_layer.size(); ++layer)
      for (auto expert : res.gpu_experts_by_layer[layer])
        out.targets.emplace_back(objects::expert_object_name(static_cast<std::uint32_t>(layer), expert),
                                 objects::AllocationTarget::kGpuResident);
  }
  CLM_RETURN_IF_ERROR(out.validate(m.geometry, node_index.size()));
  return out;
}

}  // namespace clusterlm::planning
