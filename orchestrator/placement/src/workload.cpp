#include "clusterlm/placement/workload.hpp"

#include <algorithm>

namespace clusterlm::placement {

const PlacementPlan* WorkloadResult::recommended() const {
  if (!selected || *selected >= contexts.size()) return nullptr;
  const auto& c = contexts[*selected];
  return c.recommended ? &*c.recommended : nullptr;
}

Result<WorkloadResult> search_workload(const PlacementRequest& base, const WorkloadSpec& spec) {
  auto err = [](std::string m) { return make_error(ErrorCode::kInvalidArgument, "workload: " + std::move(m)); };
  if (spec.context_profiles.empty()) return err("no context profiles");
  if (spec.q_values.empty()) return err("no q values");
  if (spec.workload.expected_prompt_tokens < 1 || spec.workload.expected_output_tokens < 1)
    return err("expected prompt/output tokens must be >= 1");
  if (spec.workload.requests_per_lease < 1) return err("requests_per_lease must be >= 1");

  std::vector<std::uint32_t> contexts = spec.context_profiles;
  std::sort(contexts.begin(), contexts.end());
  contexts.erase(std::unique(contexts.begin(), contexts.end()), contexts.end());
  std::vector<std::uint32_t> qs = spec.q_values;
  std::sort(qs.begin(), qs.end());
  qs.erase(std::unique(qs.begin(), qs.end()), qs.end());

  WorkloadResult out;
  out.workload = spec.workload;
  out.provenance = base.model.provenance;
  out.provenance = weakest(out.provenance, base.model.draft_ms.provenance);
  out.provenance = weakest(out.provenance, weakest_provenance(base.father));
  out.provenance = weakest(out.provenance, weakest_provenance(base.network));
  for (const auto& n : base.nodes) out.provenance = weakest(out.provenance, weakest_provenance(n));

  std::map<std::uint32_t, Quantity> acceptance;
  for (std::uint32_t q : qs) {
    if (q < 1) return err("q must be >= 1");
    auto it = spec.acceptance_by_q.find(q);
    if (it != spec.acceptance_by_q.end())
      acceptance[q] = it->second;
    else if (q == 1)
      acceptance[q] = base.acceptance;
    else
      return err("no acceptance supplied for q=" + std::to_string(q) + " (acceptance is an input, never invented)");
    out.provenance = weakest(out.provenance, acceptance[q].provenance);
  }

  for (std::uint32_t ctx : contexts) {
    ContextPlacement cp;
    cp.context_tokens = ctx;
    cp.prompt_tokens = std::min(spec.workload.expected_prompt_tokens, ctx);
    std::vector<PlacementPlan> pool;            // every surviving plan of every q at this context
    std::map<std::string, std::size_t> reason_counts;
    for (std::uint32_t q : qs) {
      PlacementRequest r = base;
      r.context_tokens = ctx;
      r.prompt_tokens = cp.prompt_tokens;
      r.q = q;
      r.acceptance = acceptance[q];
      r.output_tokens = spec.workload.expected_output_tokens;
      r.prune_dominated = true;
      if (spec.derive_lease_tokens)
        r.default_lease.expected_tokens =
            static_cast<double>(spec.workload.requests_per_lease) * static_cast<double>(spec.workload.expected_output_tokens);
      auto res = search_placements(r);
      if (!res.is_ok()) return err("context " + std::to_string(ctx) + " q " + std::to_string(q) + ": " + res.status().message());
      QSummary qs_sum;
      qs_sum.q = q;
      qs_sum.feasible = res->candidates.size() + res->pruned_dominated;
      qs_sum.rejected = res->rejected.size();
      qs_sum.pruned_dominated = res->pruned_dominated;
      if (res->recommended) {
        const auto& best = res->candidates[*res->recommended];
        qs_sum.best_objective = best.metrics.objective;
        qs_sum.best_key = best.key;
        qs_sum.best_decode_tok_s = res->candidates[*res->best_throughput].metrics.decode_tok_s;
      }
      cp.per_q.push_back(std::move(qs_sum));
      for (const auto& x : res->rejected)
        for (const auto& why : x.reasons) ++reason_counts[why];
      for (auto& p : res->candidates) pool.push_back(std::move(p));
    }
    cp.feasible = !pool.empty();
    if (cp.feasible) {
      // Recommended: lowest objective; ties by q, then boundaries, then hash (all independent of names).
      std::size_t best = 0;
      for (std::size_t i = 1; i < pool.size(); ++i) {
        const auto& a = pool[i];
        const auto& b = pool[best];
        if (a.metrics.objective != b.metrics.objective ? a.metrics.objective < b.metrics.objective
                                                       : (a.q != b.q ? a.q < b.q : a.plan_hash < b.plan_hash))
          best = i;
      }
      cp.recommended = pool[best];
      std::vector<const PlacementPlan*> ptrs;
      for (const auto& p : pool) ptrs.push_back(&p);
      for (std::size_t i : pareto_frontier(ptrs)) cp.frontier.push_back(pool[i]);
    } else {
      std::vector<std::pair<std::size_t, std::string>> ranked;
      for (const auto& [why, n] : reason_counts) ranked.emplace_back(n, why);
      std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
      for (std::size_t i = 0; i < ranked.size() && i < 5; ++i) cp.infeasible_reasons.push_back(ranked[i].second);
      out.notes.push_back("context " + std::to_string(ctx) + " has no feasible placement");
    }
    out.contexts.push_back(std::move(cp));
  }

  const std::uint64_t need = std::uint64_t{spec.workload.expected_prompt_tokens} + spec.workload.expected_output_tokens;
  for (std::size_t i = 0; i < out.contexts.size(); ++i)
    if (out.contexts[i].context_tokens >= need) {
      out.selected = i;
      break;
    }
  if (!out.selected) out.notes.push_back("no context profile covers the expected prompt + output tokens");
  else if (!out.contexts[*out.selected].feasible)
    out.notes.push_back("the context profile covering the workload has no feasible placement");
  return out;
}

}  // namespace clusterlm::placement
