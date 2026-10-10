// Fast/Strong/Ultra -> profiles, and settings v1 -> v2 text migration.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "clusterlm/migration/migration.hpp"
#include "clusterlm/profiles/validate.hpp"

using namespace clusterlm;
using namespace clusterlm::migration;
using nlohmann::json;

namespace {
catalog::Catalog shipped_catalog() {
  auto c = catalog::Catalog::load(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json");
  REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
  return std::move(c).value();
}
}  // namespace

TEST_CASE("the shipped Fast/Strong/Ultra catalog migrates to exactly the embedded example set") {
  const auto cat = shipped_catalog();
  auto derived = profiles_from_catalog(cat, domain::BackendRegistry::builtin());
  REQUIRE_MESSAGE(derived.is_ok(), derived.status().to_string());
  const auto examples = example_profiles();
  REQUIRE(examples.size() == 3);
  REQUIRE(derived->size() == 3);
  for (std::size_t i = 0; i < 3; ++i) CHECK_MESSAGE((*derived)[i] == examples[i], examples[i].id);
  CHECK(profiles::validate_set(*derived, {}).is_ok());
  CHECK(profiles::fallback_chain(*derived, "prof_example_ultra") ==
        std::vector<std::string>{"prof_example_ultra", "prof_example_strong", "prof_example_fast"});
}

TEST_CASE("nothing of the tier catalog is lost or promoted by the migration") {
  const auto cat = shipped_catalog();
  auto derived = profiles_from_catalog(cat, domain::BackendRegistry::builtin());
  REQUIRE(derived.is_ok());
  for (const auto& p : *derived) {
    const auto* tier = cat.find(tier_id_of_example(p.id));
    REQUIRE(tier != nullptr);
    CHECK(p.model.identity.family == tier->model.family);
    CHECK(p.model.identity.quant == tier->model.quant);
    CHECK(p.model.identity.artifact_id == tier->model.artifact_id);
    CHECK_FALSE(p.model.identity.expected_root_hash);      // unpinned stays unpinned
    CHECK(p.model.identity.expected_files.size() == tier->model.expected_files.size());
    CHECK(p.qualification_experiments == tier->qualification_experiments);
    CHECK(p.goals.size() == tier->targets.size());
    for (const auto& g : p.goals) CHECK_FALSE(g.experiment.value_or("").empty());
    CHECK(profiles::worker_slot_count(p.topology) == tier->node_role_count());
    // Roles survive as binding names, in pipeline order.
    std::size_t k = 0;
    for (const auto& role : tier->roles) {
      if (role == catalog::kRoleFather) continue;
      const auto& slot = p.topology.slots[1 + k++];
      REQUIRE(slot.select);
      CHECK(slot.select->binding == role);
    }
    // Only offered contexts are listed.
    std::size_t offered = 0;
    for (const auto& c : tier->contexts) offered += c.offered ? 1 : 0;
    CHECK(p.context.offered_profiles.size() == offered);
    // No exposure by default.
    CHECK_FALSE(p.exposure.api);
    CHECK_FALSE(p.exposure.allow_lan);
  }
  CHECK_FALSE(derived->front().speculation.enabled);  // llama-local cannot commit/abort windows
}

TEST_CASE("keep_ready from settings v1 applies to the lifecycle of every migrated profile") {
  const auto cat = shipped_catalog();
  KeepReady on{true, 45};
  auto kept = profiles_from_catalog(cat, domain::BackendRegistry::builtin(), &on);
  REQUIRE(kept.is_ok());
  for (const auto& p : *kept) {
    CHECK(p.lifecycle.preparation == profiles::Lifecycle::Preparation::kKeepReady);
    CHECK(p.lifecycle.release_after_idle_seconds == 45u * 60u);
  }
  KeepReady zero{true, 0};  // v1: 0 = never release
  auto never = profiles_from_catalog(cat, domain::BackendRegistry::builtin(), &zero);
  CHECK(never->front().lifecycle.release_after_idle_seconds == 0);
  KeepReady off{false, 30};
  auto def = profiles_from_catalog(cat, domain::BackendRegistry::builtin(), &off);
  CHECK(def->front().lifecycle.release_after_idle_seconds == 60);  // v1's fixed 60 s idle release
}

TEST_CASE("settings v1 text migrates to v2 keeping every value") {
  const std::string fp1(64, 'a'), fp2(64, 'b'), root(64, 'c');
  json v1 = {{"version", 1},
             {"kind", "clusterlm-father-settings"},
             {"paired_nodes", json::array({{{"fingerprint", fp1}, {"name", "G14"}, {"address", "10.0.0.2:47600"}, {"role", "node"}, {"paired_at_unix", 1700000000}}})},
             {"assignments", {{"node:laptop-class", fp1}, {"node:designated-3060", fp2}}},
             {"selected_tier", "ultra"},
             {"context_tokens", 16384},
             {"keep_ready", {{"enabled", true}, {"release_after_idle_minutes", 45}}},
             {"model_dirs", {{"ultra", "D:\\models\\ultra"}, {"strong", "D:\\models\\strong"}, {"fast", "D:\\models\\ultra"}}},
             {"confirmed_model_roots", {{"ultra", root}}},
             {"advanced", {{"log_level", "debug"}, {"default_q", 3}}},
             {"future_thing", {{"x", 1}}}};
  auto out = migrate_father_settings_v1_to_v2(v1.dump());
  REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());
  const json v2 = json::parse(*out);
  CHECK(v2["version"] == 2);
  CHECK(v2["paired_nodes"] == v1["paired_nodes"]);
  CHECK(v2["machine_bindings"] == v1["assignments"]);
  CHECK(v2["selected_profile_id"] == "prof_example_ultra");
  CHECK(v2["context_tokens"] == 16384);
  CHECK(v2["advanced"] == v1["advanced"]);
  CHECK(v2["future_thing"] == v1["future_thing"]);
  CHECK(v2["profiles"].size() == 3);
  for (const auto& p : v2["profiles"]) {
    CHECK(p["lifecycle"]["preparation"] == "keep-ready");
    CHECK(p["lifecycle"]["release_after_idle_seconds"] == 45 * 60);
    CHECK(p["model"]["identity"]["expected_root_hash"].is_null());  // never auto-pinned
  }
  CHECK(v2["aliases"].empty());
  CHECK(v2["library"]["scan_roots"] == json::array({"D:\\models\\ultra", "D:\\models\\strong"}));  // deduplicated, in key order of first sight
  CHECK(v2["legacy_v1"]["model_dirs"] == v1["model_dirs"]);
  CHECK(v2["legacy_v1"]["confirmed_model_roots"] == v1["confirmed_model_roots"]);
  CHECK(v2["legacy_v1"]["selected_tier"] == "ultra");
  CHECK(v2["legacy_v1"]["keep_ready"] == v1["keep_ready"]);
}

TEST_CASE("migration refuses what a v1 reader would also have refused") {
  CHECK_FALSE(migrate_father_settings_v1_to_v2("{").is_ok());
  CHECK_FALSE(migrate_father_settings_v1_to_v2(R"({"version":2})").is_ok());
  CHECK_FALSE(migrate_father_settings_v1_to_v2(R"({"version":1,"assignments":{"a":3}})").is_ok());
  CHECK_FALSE(migrate_father_settings_v1_to_v2(R"({"version":1,"context_tokens":"big"})").is_ok());
  CHECK_FALSE(migrate_father_settings_v1_to_v2(R"({"version":1,"keep_ready":{"enabled":"yes"}})").is_ok());
  // Defaults: an empty v1 document selects Fast with on-demand profiles.
  auto out = migrate_father_settings_v1_to_v2(R"({"version":1})");
  REQUIRE(out.is_ok());
  const json v2 = json::parse(*out);
  CHECK(v2["selected_profile_id"] == "prof_example_fast");
  CHECK(v2["legacy_v1"]["selected_tier"] == "fast");
  // An unknown tier is kept verbatim in legacy_v1 and falls back to Fast in the profile layer.
  out = migrate_father_settings_v1_to_v2(R"({"version":1,"selected_tier":"turbo"})");
  REQUIRE(out.is_ok());
  CHECK(json::parse(*out)["selected_profile_id"] == "prof_example_fast");
  CHECK(json::parse(*out)["legacy_v1"]["selected_tier"] == "turbo");
}
