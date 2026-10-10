#include <set>

#include <nlohmann/json.hpp>

#include "clusterlm/common/strict_json.hpp"
#include "clusterlm/migration/migration.hpp"

namespace clusterlm::migration {

using nlohmann::json;
using strict::bad;
using strict::Obj;

namespace {

constexpr std::size_t kMaxDocument = 1u << 20;

const std::set<std::string> kKnownV1 = {"version", "kind", "paired_nodes", "assignments", "selected_tier", "context_tokens",
                                       "keep_ready", "model_dirs", "confirmed_model_roots", "advanced"};

Status check_string_map(const json& doc, const char* key) {
  auto it = doc.find(key);
  if (it == doc.end()) return Status::ok();
  if (!it->is_object()) return bad(std::string("field '") + key + "' must be an object");
  for (auto e = it->begin(); e != it->end(); ++e)
    if (!e.value().is_string()) return bad(std::string("values of '") + key + "' must be strings");
  return Status::ok();
}

}  // namespace

Result<std::string> migrate_father_settings_v1_to_v2(std::string_view text) {
  if (text.size() > kMaxDocument) return bad("settings document too large");
  if (!strict::depth_within(text, 16)) return bad("settings document nested too deeply");
  json doc = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object()) return bad("not a JSON object");
  if (!doc.contains("version") || doc["version"] != 1) return bad("not a version 1 settings document");

  CLM_RETURN_IF_ERROR(check_string_map(doc, "assignments"));
  CLM_RETURN_IF_ERROR(check_string_map(doc, "model_dirs"));
  CLM_RETURN_IF_ERROR(check_string_map(doc, "confirmed_model_roots"));
  if (auto it = doc.find("paired_nodes"); it != doc.end() && !it->is_array()) return bad("field 'paired_nodes' must be an array");
  if (auto it = doc.find("selected_tier"); it != doc.end() && !it->is_string()) return bad("field 'selected_tier' must be a string");
  if (auto it = doc.find("context_tokens"); it != doc.end() && !it->is_number_unsigned()) return bad("field 'context_tokens' must be a non-negative integer");
  if (auto it = doc.find("advanced"); it != doc.end() && !it->is_object()) return bad("field 'advanced' must be an object");

  KeepReady keep;
  json keep_json = json{{"enabled", false}, {"release_after_idle_minutes", 30}};
  if (auto it = doc.find("keep_ready"); it != doc.end()) {
    if (!it->is_object()) return bad("field 'keep_ready' must be an object");
    CLM_ASSIGN_OR_RETURN(Obj ko, Obj::of(*it, "keep_ready"));
    CLM_RETURN_IF_ERROR(ko.opt_bool("enabled", keep.enabled));
    CLM_RETURN_IF_ERROR(ko.opt_u32("release_after_idle_minutes", keep.release_after_idle_minutes, 0, 7 * 24 * 60));
    keep_json = {{"enabled", keep.enabled}, {"release_after_idle_minutes", keep.release_after_idle_minutes}};
  }

  json out = json::object();
  out["version"] = 2;
  out["kind"] = "clusterlm-father-settings";
  out["paired_nodes"] = doc.value("paired_nodes", json::array());
  out["machine_bindings"] = doc.value("assignments", json::object());
  if (doc.contains("context_tokens")) out["context_tokens"] = doc["context_tokens"];
  if (doc.contains("advanced")) out["advanced"] = doc["advanced"];

  const std::string tier = doc.value("selected_tier", std::string("fast"));
  const std::string selected = tier_id_of_example(example_profile_id(tier)).empty() ? example_profile_id("fast") : example_profile_id(tier);
  out["selected_profile_id"] = selected;
  out["examples_seeded"] = true;

  json profiles_json = json::array();
  for (auto& p : example_profiles()) {
    apply_keep_ready(p, keep);
    profiles_json.push_back(json::parse(profiles::to_json(p)));
  }
  if (profiles_json.size() != 3) return make_error(ErrorCode::kInternal, "the embedded example profile set is damaged");
  out["profiles"] = std::move(profiles_json);
  out["aliases"] = json::array();

  // Existing model directories become library scan roots (scanned on first v2 load, by the caller: this function is pure).
  json roots = json::array();
  std::set<std::string> seen;
  if (auto it = doc.find("model_dirs"); it != doc.end())
    for (auto e = it->begin(); e != it->end(); ++e)
      if (seen.insert(e.value().get<std::string>()).second) roots.push_back(e.value());
  out["library"] = {{"schema", "clusterlm-model-library"}, {"schema_version", 1}, {"scan_roots", std::move(roots)}, {"models", json::array()}};

  // Nothing is dropped: the tier-era values stay verbatim until the profile layer has resolved them.
  out["legacy_v1"] = {{"selected_tier", tier},
                      {"keep_ready", std::move(keep_json)},
                      {"model_dirs", doc.value("model_dirs", json::object())},
                      {"confirmed_model_roots", doc.value("confirmed_model_roots", json::object())}};

  for (auto it = doc.begin(); it != doc.end(); ++it)
    if (kKnownV1.count(it.key()) == 0) out[it.key()] = it.value();  // v1 ignored unknown fields; v2 preserves them
  return out.dump(2) + "\n";
}

}  // namespace clusterlm::migration
