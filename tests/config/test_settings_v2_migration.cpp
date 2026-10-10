// "Existing configs survive": a REAL version-1 father-settings.json (written by the pre-profile code at 9b76ef6, stored in
// tests/config/fixtures) opens through the store, is migrated to version 2 with the original kept byte-for-byte, keeps every
// value, and still drives the old tier machinery; the Fast/Strong/Ultra example set is present and importable.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/config/store.hpp"
#include "clusterlm/migration/migration.hpp"
#include "clusterlm/profiles/topology.hpp"
#include "clusterlm/profiles/validate.hpp"

using namespace clusterlm;
using namespace clusterlm::config;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct Dir {
  fs::path path;
  explicit Dir(const char* name)
      : path(fs::temp_directory_path() / (std::string("clm-v2-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    fs::create_directories(path);
  }
  ~Dir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

const std::string kSrc = CLUSTERLM_SOURCE_DIR;
const std::string kFpA(64, 'a'), kFpB(64, 'b'), kRoot(64, 'c');

fs::path install_v1(const Dir& d) {
  const auto file = d.path / "father-settings.json";
  fs::copy_file(kSrc + "/tests/config/fixtures/father-settings-v1.json", file);
  return file;
}

}  // namespace

TEST_CASE("a real v1 settings file is migrated on first open: original kept, every value intact") {
  Dir d("survive");
  const auto file = install_v1(d);
  const std::string original = slurp(file);

  auto store = FatherSettingsStore::open(file);  // no hook passed: upgrade needs no caller wiring
  REQUIRE_MESSAGE(store.is_ok(), store.status().to_string());
  CHECK(store.value()->report().outcome == LoadReport::Outcome::kMigrated);
  REQUIRE_FALSE(store.value()->report().backup.empty());
  CHECK(slurp(store.value()->report().backup) == original);  // v1 file kept byte for byte
  CHECK(json::parse(slurp(file))["version"] == 2);

  const FatherSettings s = store.value()->get();
  REQUIRE(s.paired_nodes.size() == 2);
  CHECK(s.paired_nodes[0].name == "G14");
  CHECK(s.paired_nodes[0].address == "10.0.0.2:47600");
  CHECK(s.paired_nodes[1].fingerprint == kFpB);
  CHECK(s.assignments.at("node:laptop-class") == kFpA);
  CHECK(s.assignments.at("node:designated-3060") == kFpB);
  CHECK(s.selected_tier == "ultra");
  CHECK(s.selected_profile_id == "prof_example_ultra");
  CHECK(s.context_tokens == 16384);
  CHECK(s.keep_ready.enabled);
  CHECK(s.keep_ready.release_after_idle_minutes == 45);
  CHECK(s.model_dirs.at("fast") == "D:\\models\\fast");
  CHECK(s.model_dirs.at("strong") == "D:\\models\\strong");
  CHECK(s.model_dirs.at("ultra") == "D:\\models\\ultra");
  CHECK(s.confirmed_model_roots.at("ultra") == kRoot);
  CHECK(s.advanced.bench_results_dir == "D:\\bench");
  CHECK(s.advanced.log_level == "debug");
  CHECK_FALSE(s.advanced.direct_peer);
  CHECK(s.advanced.default_q == 3);
  CHECK(s.advanced.prefill_chunk == 256);
  CHECK(s.examples_seeded);
  CHECK(s.quarantine.empty());
  CHECK(s.library.scan_roots().size() == 3);
  CHECK(s.library.records().empty());  // scanning is I/O: done by the app, never by a pure migration

  // The second open loads v2 as is and agrees exactly.
  auto again = FatherSettingsStore::open(file);
  REQUIRE(again.is_ok());
  CHECK(again.value()->report().outcome == LoadReport::Outcome::kLoaded);
  CHECK(again.value()->get() == s);
}

TEST_CASE("the three migrated profiles are the Fast/Strong/Ultra example set, validate as a set, and carry keep-ready") {
  Dir d("profiles");
  auto store = FatherSettingsStore::open(install_v1(d));
  REQUIRE(store.is_ok());
  const FatherSettings s = store.value()->get();
  REQUIRE(s.profiles.size() == 3);
  CHECK(profiles::validate_set(s.profiles, s.aliases).is_ok());
  for (const auto& p : s.profiles) {
    CHECK(p.lifecycle.preparation == profiles::Lifecycle::Preparation::kKeepReady);
    CHECK(p.lifecycle.release_after_idle_seconds == 45u * 60u);
    CHECK_FALSE(p.exposure.api);                         // migrated installs expose nothing
    CHECK_FALSE(p.model.identity.expected_root_hash);    // pins are never auto-promoted
  }
  // Same content as the embedded set apart from the keep-ready policy.
  const auto examples = migration::example_profiles();
  for (std::size_t i = 0; i < 3; ++i) {
    auto expected = examples[i];
    migration::apply_keep_ready(expected, {true, 45});
    CHECK(s.profiles[i] == expected);
  }
}

TEST_CASE("the old tier machinery keeps working on the migrated settings") {
  Dir d("tiers");
  auto store = FatherSettingsStore::open(install_v1(d));
  REQUIRE(store.is_ok());
  const FatherSettings s = store.value()->get();

  auto cat = catalog::Catalog::load(kSrc + "/fixtures/catalog/clusterlm-catalog.json");
  REQUIRE(cat.is_ok());
  catalog::TierAssignment assignment;
  assignment.bind(std::string(catalog::kRoleFather), "father-pc");
  for (const auto& [role, fp] : s.assignments) assignment.bind(role, fp);  // exactly what father_service_api does
  for (const auto& tier : cat->tiers()) CHECK_MESSAGE(assignment.validate_for(tier).is_ok(), tier.id);

  // The same bindings drive the new topology resolution to the same machines, in pipeline order.
  std::vector<profiles::WorkerCapability> workers;
  for (const auto& n : s.paired_nodes) {
    profiles::WorkerCapability w;
    w.fingerprint = n.fingerprint;
    w.name = n.name;
    w.backends = {"strata-hybrid"};
    workers.push_back(std::move(w));
  }
  const auto* ultra = s.find_profile("prof_example_ultra");
  REQUIRE(ultra != nullptr);
  profiles::ResolveOptions opt;
  opt.backend_id = ultra->backend.id;
  auto r = profiles::resolve_topology(ultra->topology, s.assignments, workers, opt);
  CHECK(r.satisfiable);
  CHECK(r.fingerprints() == std::vector<std::string>{kFpA, kFpB});
  const auto* strong = s.find_profile("prof_example_strong");
  REQUIRE(strong != nullptr);
  CHECK(profiles::resolve_topology(strong->topology, s.assignments, workers, opt).fingerprints() == std::vector<std::string>{kFpA});
  const auto* fast = s.find_profile("prof_example_fast");
  REQUIRE(fast != nullptr);
  CHECK(profiles::resolve_topology(fast->topology, s.assignments, {}, {}).satisfiable);  // Host only
}

TEST_CASE("select_tier keeps the legacy and profile views in step and survives a restart") {
  Dir d("select");
  auto store = FatherSettingsStore::open(install_v1(d));
  REQUIRE(store.is_ok());
  REQUIRE(store.value()->update([](FatherSettings& s) { s.select_tier("strong"); return Status::ok(); }).is_ok());
  auto again = FatherSettingsStore::open(d.path / "father-settings.json");
  REQUIRE(again.is_ok());
  CHECK(again.value()->get().selected_tier == "strong");
  CHECK(again.value()->get().selected_profile_id == "prof_example_strong");
  // A selected profile that does not exist is refused.
  CHECK_FALSE(again.value()->update([](FatherSettings& s) { s.selected_profile_id = "prof_nope"; return Status::ok(); }).is_ok());
}

TEST_CASE("a damaged profile is quarantined on its own; pairing records and the other profiles survive") {
  Dir d("quarantine");
  const auto file = install_v1(d);
  { auto store = FatherSettingsStore::open(file); REQUIRE(store.is_ok()); }
  json doc = json::parse(slurp(file));
  doc["profiles"][1]["exposur"] = json::object();        // typo'd field: strict reader rejects the document
  doc["profiles"].push_back(doc["profiles"][2]);         // duplicate id of an existing profile
  {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << doc.dump();
  }
  auto store = FatherSettingsStore::open(file);
  REQUIRE(store.is_ok());
  CHECK(store.value()->report().outcome == LoadReport::Outcome::kLoaded);  // NOT recovered-corrupt
  const FatherSettings s = store.value()->get();
  CHECK(s.paired_nodes.size() == 2);
  CHECK(s.assignments.size() == 2);
  CHECK(s.profiles.size() == 2);
  REQUIRE(s.quarantine.size() == 2);
  CHECK(s.quarantine[0].kind == "profile");
  CHECK(s.quarantine[0].document.find("exposur") != std::string::npos);  // kept verbatim
  // The next save keeps the quarantined documents.
  REQUIRE(store.value()->update([](FatherSettings& x) { x.context_tokens = 8192; return Status::ok(); }).is_ok());
  auto reopened = FatherSettingsStore::open(file);
  REQUIRE(reopened.is_ok());
  CHECK(reopened.value()->get().quarantine.size() == 2);
}

TEST_CASE("unknown top-level fields are preserved across a rewrite") {
  Dir d("preserve");
  const auto file = install_v1(d);
  { auto store = FatherSettingsStore::open(file); REQUIRE(store.is_ok()); }
  json doc = json::parse(slurp(file));
  doc["server"] = {{"enabled", true}, {"port", 11434}};
  { std::ofstream out(file, std::ios::binary | std::ios::trunc); out << doc.dump(); }
  auto store = FatherSettingsStore::open(file);
  REQUIRE(store.is_ok());
  REQUIRE(store.value()->update([](FatherSettings& x) { x.context_tokens = 4096; return Status::ok(); }).is_ok());
  CHECK(json::parse(slurp(file))["server"] == doc["server"]);
}

TEST_CASE("a fresh install is seeded with the same example set a migrated one has") {
  FatherSettings fresh;
  CHECK(fresh.profiles.empty());
  ensure_example_set(fresh);
  CHECK(fresh.examples_seeded);
  CHECK(fresh.profiles.size() == 3);
  CHECK(fresh.selected_profile_id == "prof_example_fast");
  CHECK(validate(fresh).is_ok());
  auto before = fresh;
  ensure_example_set(fresh);  // idempotent
  CHECK(fresh == before);
  // Deleting an example never brings it back.
  fresh.profiles.pop_back();
  fresh.profiles.pop_back();  // ultra and strong gone; fast's fallback chain is none
  ensure_example_set(fresh);
  CHECK(fresh.profiles.size() == 1);
  // Round trip through JSON.
  auto back = father_settings_from_json(to_json(fresh));
  REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
  CHECK(*back == fresh);
}

TEST_CASE("the shipped importable example files parse, validate as a set and match the embedded set") {
  std::vector<profiles::Profile> set;
  for (const char* n : {"fast", "strong", "ultra"}) {
    auto p = profiles::profile_from_json(slurp(kSrc + "/fixtures/profiles-example/" + n + ".profile.json"));
    REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
    set.push_back(std::move(p).value());
  }
  auto alias = profiles::alias_from_json(slurp(kSrc + "/fixtures/profiles-example/auto.alias.json"));
  REQUIRE_MESSAGE(alias.is_ok(), alias.status().to_string());
  CHECK(profiles::validate_set(set, {*alias}).is_ok());
  CHECK(set == migration::example_profiles());
}
