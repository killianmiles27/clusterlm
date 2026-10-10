#include <algorithm>

#include "clusterlm/migration/migration.hpp"
#include "clusterlm/profiles/validate.hpp"

namespace clusterlm::migration {

using profiles::Profile;

std::string example_profile_id(std::string_view tier_id) { return std::string(kExampleProfilePrefix) + std::string(tier_id); }

std::string tier_id_of_example(std::string_view profile_id) {
  if (profile_id.substr(0, kExampleProfilePrefix.size()) != kExampleProfilePrefix) return {};
  const auto tier = profile_id.substr(kExampleProfilePrefix.size());
  return (tier == "fast" || tier == "strong" || tier == "ultra") ? std::string(tier) : std::string();
}

namespace {

std::string worker_label(const std::string& role) {
  if (role == std::string(catalog::kRoleLaptop)) return "Laptop-class Worker";
  if (role == std::string(catalog::kRole3060)) return "Designated 3060 Worker";
  return "Worker " + role;
}

std::string describe(const catalog::TierEntry& t) {
  const auto nodes = t.node_role_count();
  if (nodes == 0) return "Whole model on the Host (migrated from tier '" + t.id + "').";
  if (t.id == "strong") return "Host plus one laptop-class Worker (migrated from tier 'strong').";
  if (t.id == "ultra") return "Host plus laptop-class and designated-3060 Workers (migrated from tier 'ultra').";
  return "Host plus " + std::to_string(nodes) + " Worker(s) (migrated from tier '" + t.id + "').";
}

}  // namespace

void apply_keep_ready(Profile& p, const KeepReady& k) {
  if (k.enabled) {
    p.lifecycle.preparation = profiles::Lifecycle::Preparation::kKeepReady;
    p.lifecycle.release_after_idle_seconds = k.release_after_idle_minutes * 60;
  } else {
    p.lifecycle.preparation = profiles::Lifecycle::Preparation::kOnDemand;
    p.lifecycle.release_after_idle_seconds = 60;  // v1's fixed idle release (father_service_api watchdog)
  }
}

Result<Profile> profile_from_tier(const catalog::TierEntry& tier, const catalog::Catalog& cat, const domain::BackendRegistry& backends,
                                  const KeepReady* keep_ready) {
  Profile p;
  p.id = example_profile_id(tier.id);
  p.name = tier.display_name;
  p.description = describe(tier);
  p.revision = 1;
  p.example_of = "tier:" + tier.id;
  p.qualification_experiments = tier.qualification_experiments;

  auto& id = p.model.identity;
  id.family = tier.model.family;
  id.display_name = tier.model.display_name;
  id.quant = tier.model.quant;
  id.artifact_id = tier.model.artifact_id;
  id.expected_root_hash = tier.model.expected_root_hash;  // null stays null: never auto-pinned
  for (const auto& f : tier.model.expected_files) id.expected_files.push_back({f.role, f.name, f.approx_bytes});

  p.backend.id = std::string(catalog::to_string(tier.backend));
  const auto* desc = backends.find(p.backend.id);
  if (desc == nullptr) return make_error(ErrorCode::kNotFound, "no backend descriptor for tier '" + tier.id + "'");

  // Contexts: offered sizes only; the rest are derivable (absent from offered_profiles).
  for (const auto& c : tier.contexts)
    if (c.offered) p.context.offered_profiles.push_back(c.tokens);
  std::sort(p.context.offered_profiles.begin(), p.context.offered_profiles.end());
  if (p.context.offered_profiles.empty()) return make_error(ErrorCode::kInvalidArgument, "tier '" + tier.id + "' offers no context size");
  p.context.max_tokens = p.context.offered_profiles.back();
  p.context.default_tokens = std::min<std::uint32_t>(4096, p.context.max_tokens);

  // Topology: roles[0] is the Host; every other role becomes a worker slot bound by the OLD role string.
  profiles::Slot host;
  host.kind = profiles::Slot::Kind::kHost;
  host.slot = "host";
  host.label = "Host";
  p.topology.slots.push_back(std::move(host));
  std::uint32_t n = 0;
  for (const auto& role : tier.roles) {
    if (role == catalog::kRoleFather) continue;
    profiles::Slot s;
    s.kind = profiles::Slot::Kind::kWorker;
    s.slot = "w" + std::to_string(++n);
    s.label = worker_label(role);
    s.select = profiles::Selector{profiles::Selector::Mode::kBinding, role, "", std::nullopt};
    p.topology.slots.push_back(std::move(s));
  }
  p.topology.min_workers = n;
  p.topology.max_workers = n;

  // Speculation: v1 tiers speculate only when distributed (the Fast tier is a plain local model). The width is the ceiling
  // the backend validates, never more than the profile schema allows.
  const bool speculates = n > 0 && desc->speculation.window_commit_abort && desc->speculation.max_q > 1;
  p.speculation.enabled = speculates;
  p.speculation.max_q = speculates ? std::min<std::uint32_t>(desc->speculation.max_q, 16) : 1;

  // Session loss -> Worker loss. `downgrade` follows the catalog fallback order to the next lower tier.
  p.on_worker_loss.max_retries = tier.on_session_loss.max_retries;
  if (tier.on_session_loss.then == catalog::LossAction::kDowngrade) {
    auto lower = cat.fallbacks_after(tier.id);
    if (!lower.empty()) {
      p.on_worker_loss.then = profiles::OnWorkerLoss::Then::kFallbackProfile;
      p.on_worker_loss.fallback_profile_id = example_profile_id(lower.front()->id);
    }
  }

  for (const auto& t : tier.targets) {
    profiles::Goal g;
    g.metric = t.metric;
    g.value = t.value;
    if (!t.experiment.empty()) g.experiment = t.experiment;
    p.goals.push_back(std::move(g));
  }

  p.exposure.api = false;
  p.exposure.api_model_id = "example-" + tier.id;
  if (keep_ready != nullptr) apply_keep_ready(p, *keep_ready);
  CLM_RETURN_IF_ERROR(profiles::validate(p));
  return p;
}

Result<std::vector<Profile>> profiles_from_catalog(const catalog::Catalog& cat, const domain::BackendRegistry& backends,
                                                   const KeepReady* keep_ready) {
  std::vector<Profile> out;
  // Fast, Strong, Ultra order (the catalog lists them as in the file; sort by ascending capability = reverse fallback order).
  std::vector<std::string> order(cat.fallback_order().rbegin(), cat.fallback_order().rend());
  for (const auto& id : order) {
    const auto* t = cat.find(id);
    if (t == nullptr) return make_error(ErrorCode::kInternal, "catalog fallback order names an unknown tier");
    CLM_ASSIGN_OR_RETURN(auto p, profile_from_tier(*t, cat, backends, keep_ready));
    out.push_back(std::move(p));
  }
  CLM_RETURN_IF_ERROR(profiles::validate_set(out, {}));
  return out;
}

}  // namespace clusterlm::migration
