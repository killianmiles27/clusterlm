#include "clusterlm/planning/report.hpp"

#include <nlohmann/json.hpp>

namespace clusterlm::planning {

using nlohmann::json;
using placement::Provenance;

namespace {

std::string banner(Provenance p) {
  if (p == Provenance::kQualified) return "QUALIFIED: every input quantity is qualified";
  if (p == Provenance::kMeasured) return "MEASURED, NOT QUALIFIED: at least one input is an unqualified measurement";
  return "SYNTHETIC: development estimate, not a measurement; do not use as a qualified placement decision";
}

json inputs_json(const placement::PlacementRequest& r) {
  json in;
  in["father"] = {{"id", r.father.id}, {"provenance", std::string(to_string(weakest_provenance(r.father)))}};
  in["nodes"] = json::array();
  for (const auto& n : r.nodes)
    in["nodes"].push_back({{"id", n.id}, {"provenance", std::string(to_string(weakest_provenance(n)))}});
  in["network"] = {{"provenance", std::string(to_string(weakest_provenance(r.network)))}};
  in["model"] = {{"name", r.model.name},
                 {"structure_provenance", std::string(to_string(r.model.provenance))},
                 {"draft_ms_provenance", std::string(to_string(r.model.draft_ms.provenance))},
                 {"source", r.model.source}};
  in["acceptance"] = {{"value", r.acceptance.value}, {"provenance", std::string(to_string(r.acceptance.provenance))}};
  in["search"] = {{"granularity", r.granularity},         {"prefill_chunk", r.prefill_chunk},
                  {"max_remote_nodes", r.max_remote_nodes}, {"sweep_prefix", r.sweep_prefix},
                  {"prune_dominated", r.prune_dominated},  {"overlap_factor", r.overlap_factor}};
  return in;
}

json frontier_entry(const placement::PlacementPlan& p) {
  return {{"key", p.key},
          {"q", p.q},
          {"provenance", std::string(to_string(p.provenance))},
          {"prepare_s", p.metrics.prepare_s},
          {"decode_tok_s", p.metrics.decode_tok_s},
          {"prefill_s", p.metrics.prefill_s},
          {"objective", p.metrics.objective}};
}

}  // namespace

std::string placement_report_json(const placement::PlacementRequest& request, const placement::PlacementResult& result,
                                  bool include_expert_lists, int indent) {
  json j = json::parse(placement::report_to_json(result, include_expert_lists, 0));
  j["schema"] = "clusterlm.placement_report.v2";
  j["inputs"] = inputs_json(request);
  j["pareto_axes"] = {"prepare_s:min", "decode_tok_s:max", "prefill_s:min"};
  return j.dump(indent);
}

std::string placement_report_json(const placement::PlacementRequest& request, const placement::WorkloadResult& result,
                                  bool include_expert_lists, int indent) {
  json j;
  j["schema"] = "clusterlm.placement_workload_report.v1";
  j["provenance"] = std::string(to_string(result.provenance));
  j["provenance_banner"] = banner(result.provenance);
  j["inputs"] = inputs_json(request);
  j["workload"] = {{"expected_prompt_tokens", result.workload.expected_prompt_tokens},
                   {"expected_output_tokens", result.workload.expected_output_tokens},
                   {"requests_per_lease", result.workload.requests_per_lease}};
  j["pareto_axes"] = {"prepare_s:min", "decode_tok_s:max", "prefill_s:min"};
  j["notes"] = result.notes;
  j["selected_context"] = result.selected ? json(result.contexts[*result.selected].context_tokens) : json(nullptr);
  if (const placement::PlacementPlan* rec = result.recommended())
    j["recommended_plan"] = json::parse(placement::plan_to_json(*rec, include_expert_lists, 0));
  else
    j["recommended_plan"] = nullptr;
  j["contexts"] = json::array();
  for (const auto& c : result.contexts) {
    json cj = {{"context_tokens", c.context_tokens}, {"prompt_tokens", c.prompt_tokens}, {"feasible", c.feasible}};
    cj["per_q"] = json::array();
    for (const auto& q : c.per_q)
      cj["per_q"].push_back({{"q", q.q},
                             {"feasible", q.feasible},
                             {"rejected", q.rejected},
                             {"pruned_dominated", q.pruned_dominated},
                             {"best_decode_tok_s", q.best_decode_tok_s},
                             {"best_objective", q.best_objective},
                             {"best_key", q.best_key}});
    cj["frontier"] = json::array();
    for (const auto& p : c.frontier) cj["frontier"].push_back(frontier_entry(p));
    cj["recommended"] = c.recommended ? frontier_entry(*c.recommended) : json(nullptr);
    cj["infeasible_reasons"] = c.infeasible_reasons;
    j["contexts"].push_back(std::move(cj));
  }
  return j.dump(indent);
}

}  // namespace clusterlm::planning
