#include "clusterlm/catalog/readiness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace clusterlm::catalog {

std::string_view to_string(MachineState s) noexcept {
  switch (s) {
    case MachineState::kBusy: return "Busy";
    case MachineState::kAvailable: return "Available";
    case MachineState::kPreparing: return "Preparing";
    case MachineState::kReady: return "Ready";
    case MachineState::kInferencing: return "Inferencing";
    case MachineState::kReleasing: return "Releasing";
    case MachineState::kCleanupPending: return "CleanupPending";
    case MachineState::kOffline: return "Offline";
  }
  return "Offline";
}

std::string_view to_string(TierState s) noexcept {
  switch (s) {
    case TierState::kUnavailable: return "Unavailable";
    case TierState::kAvailable: return "Available";
    case TierState::kPreparing: return "Preparing";
    case TierState::kReady: return "Ready";
  }
  return "Unavailable";
}

std::string format_duration(double seconds) {
  if (!(seconds >= 0) || !std::isfinite(seconds)) return "unknown";
  if (seconds < 60) return "under a minute";
  const auto minutes = static_cast<std::uint64_t>(std::llround(seconds / 60.0));
  if (minutes < 60) return "about " + std::to_string(minutes) + " min";
  const auto h = minutes / 60, m = minutes % 60;
  return "about " + std::to_string(h) + " h" + (m ? " " + std::to_string(m) + " min" : "");
}

namespace {

std::string context_label(std::uint32_t tokens) {
  if (tokens >= 1024 && tokens % 1024 == 0) return std::to_string(tokens / 1024) + "K";
  return std::to_string(tokens);
}

const MachineInputs* find_machine(const ReadinessInputs& in, const std::string& role) {
  for (const auto& m : in.machines)
    if (m.role == role) return &m;
  return nullptr;
}

}  // namespace

TierReadiness evaluate(const TierEntry& tier, const ReadinessInputs& in) {
  TierReadiness out;
  out.tier_id = tier.id;
  out.model_name = tier.model.display_name;
  auto block = [&](std::string why) { out.reasons.push_back(std::move(why)); };

  // Context profile.
  const ContextProfile* ctx = tier.find_context(in.context_tokens);
  if (!ctx || !ctx->offered)
    block(context_label(in.context_tokens) + " context is not offered for " + tier.display_name);
  else if (!ctx->qualified)
    out.notes.push_back(context_label(in.context_tokens) + " context on " + tier.display_name +
                        " is not yet qualified on the target hardware");

  // Model: present, hash-verified, and either pinned (and equal) or user-confirmed.
  if (!in.model.manifest_present) {
    block(tier.model.display_name + " is not downloaded on Father");
  } else if (!in.model.hashes_verified) {
    block(tier.model.display_name + " has not been verified against its manifest hashes");
  } else if (tier.model.expected_root_hash) {
    if (in.model.manifest_root_hex != *tier.model.expected_root_hash)
      block("The local " + tier.display_name + " model does not match the pinned catalog hash");
  } else if (in.model.user_confirmed_root_hex.empty() || in.model.user_confirmed_root_hex != in.model.manifest_root_hex) {
    block("The " + tier.display_name + " model is unpinned; confirm its inspected manifest to enable this tier");
  }

  // Tokenizer (built on Father from the model's GGUF metadata).
  if (!in.tokenizer_problem.empty()) block("The " + tier.model.display_name + " tokenizer is unavailable: " + in.tokenizer_problem);

  // Backend.
  if (!in.backend.hardware_available)
    block("The " + std::string(to_string(tier.backend)) + " backend cannot run on this machine" +
          (in.backend.name.empty() ? std::string() : " (" + in.backend.name + ")"));

  // Machines per role.
  bool any_preparing = false;
  bool all_ready = true;
  for (const auto& role : tier.roles) {
    const MachineInputs* m = find_machine(in, role);
    if (!m) {
      block("No machine is assigned to " + role);
      all_ready = false;
      continue;
    }
    const std::string name = m->machine_id.empty() ? role : m->machine_id;
    bool usable = true;
    if (!m->paired) {
      block(name + " is not paired");
      usable = false;
    } else {
      switch (m->state) {
        case MachineState::kOffline: block(name + " is offline"); usable = false; break;
        case MachineState::kBusy: block(name + " is in use"); usable = false; break;
        case MachineState::kCleanupPending: block(name + " needs cleanup"); usable = false; break;
        case MachineState::kReleasing: block(name + " is releasing its previous lease"); usable = false; break;
        case MachineState::kInferencing:
          if (!in.plan.plan_ready) { block(name + " is serving another session"); usable = false; }
          break;
        case MachineState::kPreparing: any_preparing = true; break;
        case MachineState::kAvailable:
        case MachineState::kReady: break;
      }
    }
    if (usable && !m->power.on_ac) { block(name + " is on battery power"); usable = false; }
    if (usable && m->power.battery_saver) { block(name + " is in battery saver mode"); usable = false; }
    if (!usable) { all_ready = false; continue; }
    if (m->state != MachineState::kReady && m->state != MachineState::kInferencing) all_ready = false;
  }

  // Placement feasibility for the requested context.
  if (ctx && ctx->offered && ctx->has_requirement(kReqFeasiblePlacement) && !in.plan.feasible_for_context)
    block("No feasible placement for " + context_label(in.context_tokens) + " context on " + tier.display_name);

  if (!out.reasons.empty()) {
    out.state = TierState::kUnavailable;
    return out;
  }

  // Everything required is present. Honest Ready: plan current AND every domain Ready.
  if (in.plan.plan_ready && all_ready) {
    out.state = TierState::kReady;
    out.reasons.push_back(tier.display_name + " is ready (" + tier.model.display_name + ")");
    return out;
  }

  const bool provisioning_active = in.provisioning && in.provisioning->bytes_total > 0 &&
                                   in.provisioning->bytes_done < in.provisioning->bytes_total;
  if (any_preparing || provisioning_active) {
    out.state = TierState::kPreparing;
    std::string msg = "Preparing " + tier.display_name;
    if (in.provisioning && in.provisioning->bytes_total > 0) {
      PrepareProgress p;
      const auto& pr = *in.provisioning;
      p.percent = 100.0 * static_cast<double>(std::min(pr.bytes_done, pr.bytes_total)) / static_cast<double>(pr.bytes_total);
      if (pr.rate_bytes_per_s && *pr.rate_bytes_per_s > 0)
        p.eta_seconds = static_cast<double>(pr.bytes_total - std::min(pr.bytes_done, pr.bytes_total)) / *pr.rate_bytes_per_s;
      char pct[16];
      std::snprintf(pct, sizeof pct, "%d%%", static_cast<int>(p.percent));
      msg += std::string(" \xE2\x80\x94 ") + pct;
      if (p.eta_seconds) msg += " (" + format_duration(*p.eta_seconds) + ", estimate)";
      out.progress = p;
    }
    out.reasons.push_back(std::move(msg));
    return out;
  }

  out.state = TierState::kAvailable;
  if (in.plan.plan_ready || std::any_of(in.machines.begin(), in.machines.end(), [](const MachineInputs& m) {
        return m.state == MachineState::kReady || m.state == MachineState::kInferencing;
      }))
    out.reasons.push_back(tier.display_name + " is not prepared under a current plan; it must be prepared again");
  else
    out.reasons.push_back(tier.display_name + " can be prepared");
  return out;
}

TierReadiness evaluate_with_fallback(const Catalog& catalog, const TierEntry& tier,
                                     const std::vector<std::pair<std::string, ReadinessInputs>>& inputs) {
  auto find_in = [&](const std::string& id) -> const ReadinessInputs* {
    for (const auto& [k, v] : inputs)
      if (k == id) return &v;
    return nullptr;
  };
  TierReadiness r;
  if (const auto* in = find_in(tier.id)) {
    r = evaluate(tier, *in);
  } else {
    r.tier_id = tier.id;
    r.model_name = tier.model.display_name;
    r.reasons.push_back("No readiness information for " + tier.display_name);
  }
  for (const auto* lower : catalog.fallbacks_after(tier.id)) {
    const auto* in = find_in(lower->id);
    if (!in) continue;
    const auto lr = evaluate(*lower, *in);
    if (lr.state == TierState::kReady || lr.state == TierState::kAvailable) r.suggested_fallback.push_back(lower->id);
  }
  return r;
}

}  // namespace clusterlm::catalog
