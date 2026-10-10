#include "clusterlm/profiles/exchange.hpp"

#include <algorithm>
#include <set>

#include "clusterlm/profiles/validate.hpp"

namespace clusterlm::profiles {

namespace {

std::string slug(std::string_view s, std::size_t max) {
  std::string out;
  bool dash = false;
  for (unsigned char c : s) {
    const char l = static_cast<char>(std::tolower(c));
    if ((l >= 'a' && l <= 'z') || (l >= '0' && l <= '9')) {
      if (dash && !out.empty()) out += '-';
      dash = false;
      out += l;
    } else {
      dash = true;
    }
  }
  if (out.size() > max) out.resize(max);
  while (!out.empty() && out.back() == '-') out.pop_back();
  return out;
}

}  // namespace

std::string worker_binding_slug(std::string_view display_name) {
  const std::string s = slug(display_name, 32);
  return "worker:" + (s.empty() ? std::string("worker") : s);
}

Profile export_profile(const Profile& in, const std::function<std::string(const std::string&)>& worker_name) {
  Profile p = in;
  p.model.library_id.reset();
  std::set<std::string> used;
  for (const auto& s : p.topology.slots)
    if (s.select && s.select->mode == Selector::Mode::kBinding) used.insert(s.select->binding);
  for (auto& s : p.topology.slots) {
    if (!s.select || s.select->mode != Selector::Mode::kMachine) continue;
    const std::string base = worker_binding_slug(worker_name ? worker_name(s.select->machine) : std::string());
    std::string name = base;
    for (int n = 2; used.count(name) != 0; ++n) {
      const std::string suffix = "-" + std::to_string(n);
      name = base.substr(0, std::min<std::size_t>(base.size(), 7 + 32 - suffix.size())) + suffix;
    }
    used.insert(name);
    std::optional<Requirements> extra = s.select->requirements;
    s.select = Selector{Selector::Mode::kBinding, name, "", std::move(extra)};  // the fingerprint is dropped
  }
  return p;
}

std::string fresh_profile_id(std::string_view name, const std::vector<Profile>& existing) {
  std::string base = slug(name, 30);
  std::replace(base.begin(), base.end(), '-', '_');
  if (base.empty()) base = "profile";
  auto taken = [&](const std::string& id) {
    return std::any_of(existing.begin(), existing.end(), [&](const Profile& p) { return p.id == id; });
  };
  std::string id = "prof_" + base;
  if (id.size() < 8) id += "_new";
  for (int n = 2; taken(id); ++n) id = "prof_" + base + "_" + std::to_string(n);
  return id;
}

Result<ImportResult> import_profile(std::string_view text, ImportMode mode, const ImportContext& ctx) {
  CLM_ASSIGN_OR_RETURN(Profile p, profile_from_json(text));
  static const std::vector<Profile> kNone;
  const std::vector<Profile>& existing = ctx.existing != nullptr ? *ctx.existing : kNone;
  ImportResult out;

  for (const auto& s : p.topology.slots)
    if (s.select && s.select->mode == Selector::Mode::kMachine)
      return make_error(ErrorCode::kInvalidArgument, "an exported profile never names a machine; this document pins a device and cannot be imported");

  // Imports never turn on exposure and never carry a local model pointer.
  p.model.library_id.reset();
  p.exposure.api = false;
  p.exposure.allow_lan = false;

  const Profile* same_id = nullptr;
  for (const auto& e : existing)
    if (e.id == p.id) same_id = &e;
  if (same_id != nullptr) {
    Profile a = *same_id, b = p;
    a.model.library_id.reset();
    a.exposure.api = b.exposure.api = false;
    a.exposure.allow_lan = b.exposure.allow_lan = false;
    if (a == b) {
      out.action = ImportResult::Action::kUnchanged;
      out.profile = *same_id;
      return out;
    }
    if (mode == ImportMode::kReject)
      return make_error(ErrorCode::kAlreadyExists, "a different profile with this id already exists; replace it or import a copy");
    if (mode == ImportMode::kReplace) {
      p.revision = same_id->revision + 1;
      p.exposure = same_id->exposure;  // replacing content never changes what the user chose to expose
      p.model.library_id = same_id->model.library_id;
      out.action = ImportResult::Action::kReplaced;
    }
  }
  if (same_id == nullptr || mode == ImportMode::kDuplicate) {
    if (same_id != nullptr) {
      p.id = fresh_profile_id(p.name, existing);
      p.example_of.reset();
      p.revision = 1;
      out.action = ImportResult::Action::kDuplicated;
      // A copy never inherits the original's fallback target pointing back at itself.
      if (p.on_worker_loss.fallback_profile_id && *p.on_worker_loss.fallback_profile_id == p.id) p.on_worker_loss.fallback_profile_id.reset();
    }
  }

  // api_model_id must stay unique: the document's id is dropped unless it is free, and the user is told.
  auto api_taken = [&](const std::string& id) {
    for (const auto& e : existing)
      if (e.exposure.api_model_id == id && e.id != p.id) return true;
    if (ctx.aliases != nullptr)
      for (const auto& a : *ctx.aliases)
        if (a.api_model_id == id) return true;
    return false;
  };
  if (!ctx.new_api_model_id.empty()) {
    if (!is_api_model_id(ctx.new_api_model_id) || ctx.new_api_model_id == "auto" || ctx.new_api_model_id == "default" ||
        ctx.new_api_model_id.rfind("clusterlm-", 0) == 0 || api_taken(ctx.new_api_model_id))
      return make_error(ErrorCode::kInvalidArgument, "the chosen API model name is not allowed or already in use");
    p.exposure.api_model_id = ctx.new_api_model_id;
  } else if (p.exposure.api_model_id && api_taken(*p.exposure.api_model_id)) {
    return make_error(ErrorCode::kAlreadyExists, "the API model name '" + *p.exposure.api_model_id + "' is already used here; choose another");
  }

  // Full level-1 validation of the stored form, including the set rules.
  std::vector<Profile> merged = existing;
  merged.erase(std::remove_if(merged.begin(), merged.end(), [&](const Profile& e) { return e.id == p.id; }), merged.end());
  merged.push_back(p);
  const std::vector<RoutingAlias> no_aliases;
  for (const auto& pr : find_set_problems(merged, ctx.aliases != nullptr ? *ctx.aliases : no_aliases))
    if (!pr.dangling && pr.kind == SetProblem::Kind::kProfile && merged[pr.index].id == p.id)
      return make_error(ErrorCode::kInvalidArgument, pr.message);

  // What the user must still do.
  if (ctx.library != nullptr) {
    library::ModelIdentityQuery q{p.model.identity.family, p.model.identity.quant, p.model.identity.expected_root_hash};
    const auto r = ctx.library->resolve(q);
    if (r.kind == library::Resolution::Kind::kMissing) out.todo.push_back("Add the model '" + p.model.identity.family + "' (" + p.model.identity.quant + ") to your library.");
    if (r.kind == library::Resolution::Kind::kAmbiguous) out.todo.push_back("Several library models match; confirm which one this profile should use.");
  }
  if (ctx.bindings != nullptr)
    for (const auto& s : p.topology.slots)
      if (s.select && s.select->mode == Selector::Mode::kBinding && ctx.bindings->count(s.select->binding) == 0)
        out.todo.push_back("Assign a Worker to '" + (s.label.empty() ? s.slot : s.label) + "' (" + s.select->binding + ").");
  if (p.on_worker_loss.fallback_profile_id) {
    const auto& f = *p.on_worker_loss.fallback_profile_id;
    if (f != p.id && std::none_of(existing.begin(), existing.end(), [&](const Profile& e) { return e.id == f; }))
      out.todo.push_back("The fallback profile '" + f + "' is not present here.");
  }
  out.profile = std::move(p);
  return out;
}

}  // namespace clusterlm::profiles
