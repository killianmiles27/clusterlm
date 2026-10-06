#include "clusterlm/planning/replan.hpp"

#include <algorithm>
#include <cstdio>

namespace clusterlm::planning {

using placement::PlacementPlan;
using placement::PlacementRequest;
using placement::StageRole;

std::string_view to_string(ReplanReasonKind k) noexcept {
  switch (k) {
    case ReplanReasonKind::kNodeUnavailable: return "node_unavailable";
    case ReplanReasonKind::kPlanInadmissible: return "plan_inadmissible";
    case ReplanReasonKind::kContextGrowth: return "context_growth";
    case ReplanReasonKind::kDecodeRegression: return "decode_regression";
    case ReplanReasonKind::kPrefillRegression: return "prefill_regression";
    case ReplanReasonKind::kPreparationRegression: return "preparation_regression";
    case ReplanReasonKind::kBetterPlanAvailable: return "better_plan_available";
    case ReplanReasonKind::kNoFeasiblePlan: return "no_feasible_plan";
  }
  return "?";
}

namespace {

std::string fmt(const char* f, double a, double b = 0, double c = 0) {
  char buf[160];
  std::snprintf(buf, sizeof buf, f, a, b, c);
  return buf;
}

double rel(double now, double before) { return before > 0 ? (now - before) / before : 0.0; }

}  // namespace

Result<ReplanDecision> should_replan(const PlacementPlan& current, const ReplanInputs& in) {
  const ReplanPolicy& pol = in.policy;
  ReplanDecision d;
  PlacementRequest req = in.request;

  // Nodes that disappeared are removed from the world before anything is costed.
  std::vector<std::string> gone;
  for (const auto& id : in.unavailable_nodes)
    if (std::find(gone.begin(), gone.end(), id) == gone.end()) gone.push_back(id);
  std::erase_if(req.nodes, [&](const placement::HardwareProfile& n) {
    return std::find(gone.begin(), gone.end(), n.id) != gone.end();
  });
  for (const auto& id : gone) req.already_provisioned.erase(id);

  // Nodes the current plan relies on that the new world no longer offers.
  std::vector<std::string> missing;
  bool father_ok = false;
  for (const auto& s : current.stages) {
    if (s.domain_id == req.father.id) {
      father_ok = true;
      continue;
    }
    const bool present = std::any_of(req.nodes.begin(), req.nodes.end(), [&](const auto& n) { return n.id == s.domain_id; });
    if (!present && std::find(missing.begin(), missing.end(), s.domain_id) == missing.end()) missing.push_back(s.domain_id);
  }
  if (!father_ok) return make_error(ErrorCode::kInvalidArgument, "replan: the current plan does not run on the request's Father");

  // A running deployment: nodes hold the layers of the current plan, so keeping it costs no provisioning and
  // a candidate pays only for bytes not already there (switching cost inside the comparison).
  if (in.current_plan_provisioned)
    for (const auto& s : current.stages)
      if (s.role == StageRole::kMiddle) req.already_provisioned[s.domain_id].push_back(s.layers);

  PlacementRequest search_req = req;
  search_req.prune_dominated = true;  // the recommended plan is unaffected by pruning
  auto searched = placement::search_placements(search_req);
  if (!searched.is_ok()) return searched.status();
  if (searched->recommended) d.candidate = searched->candidates[*searched->recommended];

  for (const auto& id : missing)
    d.reasons.push_back({ReplanReasonKind::kNodeUnavailable, "node '" + id + "' of the current plan is no longer available", 0});

  if (current.context_tokens > 0 && req.context_tokens > current.context_tokens)
    d.reasons.push_back({ReplanReasonKind::kContextGrowth,
                         fmt("context requirement grew from %.0f to %.0f tokens", current.context_tokens, req.context_tokens),
                         rel(req.context_tokens, current.context_tokens)});

  bool admissible = missing.empty();
  if (admissible) {
    placement::PlanEvaluation ev = placement::evaluate_plan(req, current.stages);
    if (ev.plan) {
      d.current_under_new_inputs = std::move(*ev.plan);
    } else {
      admissible = false;
      d.inadmissible_reasons = ev.reasons;
      std::string why;
      for (std::size_t i = 0; i < ev.reasons.size() && i < 3; ++i) why += (i ? "; " : "") + ev.reasons[i];
      d.reasons.push_back({ReplanReasonKind::kPlanInadmissible, "current plan no longer passes admission: " + why, 0});
    }
  }

  if (d.current_under_new_inputs) {
    const auto& now = d.current_under_new_inputs->metrics;
    const auto& was = current.metrics;
    const double drop = was.decode_tok_s > 0 ? (was.decode_tok_s - now.decode_tok_s) / was.decode_tok_s : 0.0;
    if (drop >= pol.decode_regression) {
      std::string cause;
      for (std::size_t i = 0; i < now.stages.size() && i < was.stages.size(); ++i) {
        const double r = rel(now.stages[i].stage_ms, was.stages[i].stage_ms);
        if (r >= pol.component_regression)
          cause += (cause.empty() ? "" : "; ") + now.stages[i].domain_id + "[" + std::to_string(now.stages[i].layers.begin) + "," +
                   std::to_string(now.stages[i].layers.end) + ") stage " + fmt("%.2f -> %.2f ms", was.stages[i].stage_ms, now.stages[i].stage_ms);
      }
      for (std::size_t i = 0; i < now.transports.size() && i < was.transports.size(); ++i) {
        const double r = rel(now.transports[i].ms, was.transports[i].ms);
        if (r >= pol.component_regression)
          cause += (cause.empty() ? "" : "; ") + std::string("link ") + now.transports[i].from + "->" + now.transports[i].to +
                   fmt(" %.3f -> %.3f ms", was.transports[i].ms, now.transports[i].ms);
      }
      if (now.commit_ms > was.commit_ms * (1 + pol.component_regression))
        cause += (cause.empty() ? "" : "; ") + std::string("commit round trip ") + fmt("%.3f -> %.3f ms", was.commit_ms, now.commit_ms);
      d.reasons.push_back({ReplanReasonKind::kDecodeRegression,
                           fmt("predicted decode fell %.1f -> %.1f tok/s", was.decode_tok_s, now.decode_tok_s) +
                               (cause.empty() ? "" : " (" + cause + ")"),
                           drop});
    }
    const double pre = rel(now.prefill_s, was.prefill_s);
    if (pre >= pol.prefill_regression)
      d.reasons.push_back({ReplanReasonKind::kPrefillRegression, fmt("predicted prefill rose %.2f -> %.2f s", was.prefill_s, now.prefill_s), pre});
    const double prep = rel(now.effective_prepare_s, was.effective_prepare_s);
    std::string overrun;
    for (const auto& f : now.flags)
      if (std::find(was.flags.begin(), was.flags.end(), f) == was.flags.end()) overrun += (overrun.empty() ? "" : ", ") + f;
    if (prep >= pol.preparation_regression && now.effective_prepare_s - was.effective_prepare_s > 1e-9)
      d.reasons.push_back({ReplanReasonKind::kPreparationRegression,
                           fmt("amortised preparation rose %.2f -> %.2f s", was.effective_prepare_s, now.effective_prepare_s) +
                               (overrun.empty() ? "" : " (" + overrun + ")"),
                           prep});
    else if (!overrun.empty())
      d.reasons.push_back({ReplanReasonKind::kPreparationRegression, "new lease overrun: " + overrun, 0});
  }

  // ---- decision ----
  if (!admissible) {
    d.replan = true;
    if (!d.candidate) d.reasons.push_back({ReplanReasonKind::kNoFeasiblePlan, "no placement is feasible under the new inputs", 0});
  } else if (d.candidate && d.candidate->stages != current.stages) {
    const double cur_obj = d.current_under_new_inputs->metrics.objective;
    const double new_obj = d.candidate->metrics.objective;
    if (new_obj < cur_obj * (1.0 - pol.min_objective_gain) && new_obj < cur_obj) {
      d.replan = true;
      d.reasons.push_back({ReplanReasonKind::kBetterPlanAvailable,
                           "candidate " + d.candidate->key + fmt(" scores %.2f vs %.2f for the current plan", new_obj, cur_obj),
                           cur_obj > 0 ? (cur_obj - new_obj) / cur_obj : 0.0});
    }
  }

  // ---- reprovisioning ----
  if (d.replan && d.candidate) {
    d.reprovision_bytes = d.candidate->metrics.provisioning_bytes;
    d.reprovision_s = d.candidate->metrics.provisioning_s;
  } else if (!d.replan && d.current_under_new_inputs) {
    d.reprovision_bytes = d.current_under_new_inputs->metrics.provisioning_bytes;
    d.reprovision_s = d.current_under_new_inputs->metrics.provisioning_s;
  }
  d.reprovision_needed = d.reprovision_bytes > 0;
  return d;
}

}  // namespace clusterlm::planning
