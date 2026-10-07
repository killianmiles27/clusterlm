// Persistent configuration: round trip, defaults, corrupt-file recovery with a preserved backup, owner-only
// permissions, schema validation and migration.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "clusterlm/config/store.hpp"
#include "clusterlm/platform/paths.hpp"

using namespace clusterlm;
using namespace clusterlm::config;
namespace fs = std::filesystem;

namespace {

struct TempDir {
  fs::path path;
  explicit TempDir(const char* name)
      : path(fs::temp_directory_path() /
             (std::string("clm-config-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    { std::error_code ec_rm; fs::remove_all(path, ec_rm); }
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void spit(const fs::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out << text;
}

const std::string kFpA(64, 'a');
const std::string kFpB(64, 'b');

PairedDevice node(const std::string& fp, const char* name, const char* addr) {
  PairedDevice d;
  d.fingerprint = fp;
  d.name = name;
  d.address = addr;
  d.role = "node";
  d.paired_at_unix = 1700000000;
  return d;
}

}  // namespace

TEST_CASE("first run yields defaults without creating a document; updates round trip through disk") {
  TempDir dir("roundtrip");
  const auto file = dir.path / "cfg" / "father-settings.json";
  {
    auto store = FatherSettingsStore::open(file);
    REQUIRE(store.is_ok());
    CHECK(store.value()->report().outcome == LoadReport::Outcome::kDefaultsNoFile);
    CHECK_FALSE(fs::exists(file));
    CHECK(store.value()->get() == FatherSettings{});

    REQUIRE(store.value()
                ->update([&](FatherSettings& s) {
                  s.upsert_node(node(kFpA, "G14", "192.168.1.20:47600"));
                  s.upsert_node(node(kFpB, "3060", "192.168.1.21:47600"));
                  s.assignments["node:laptop-class"] = kFpA;
                  s.assignments["node:designated-3060"] = kFpB;
                  s.selected_tier = "ultra";
                  s.context_tokens = 16384;
                  s.keep_ready = {true, 45};
                  s.model_dirs["ultra"] = "D:\\models\\ultra";
                  s.advanced.bench_results_dir = "C:\\bench";
                  s.advanced.log_level = "debug";
                  s.advanced.direct_peer = false;
                  s.advanced.default_q = 3;
                  return Status::ok();
                })
                .is_ok());
    CHECK(fs::exists(file));
    CHECK_FALSE(fs::exists(fs::path(file).concat(".tmp")));  // atomic write leaves no temporary behind
  }
  auto again = FatherSettingsStore::open(file);
  REQUIRE(again.is_ok());
  CHECK(again.value()->report().outcome == LoadReport::Outcome::kLoaded);
  const auto s = again.value()->get();
  CHECK(s.paired_nodes.size() == 2);
  CHECK(s.assignments.at("node:laptop-class") == kFpA);
  CHECK(s.selected_tier == "ultra");
  CHECK(s.context_tokens == 16384);
  CHECK(s.keep_ready.enabled);
  CHECK(s.keep_ready.release_after_idle_minutes == 45);
  CHECK(s.model_dirs.at("ultra") == "D:\\models\\ultra");
  CHECK_FALSE(s.advanced.direct_peer);
  CHECK(s.advanced.default_q == 3);
}

TEST_CASE("node settings round trip including the paired Father") {
  TempDir dir("node");
  const auto file = dir.path / "node-settings.json";
  auto store = NodeSettingsStore::open(file);
  REQUIRE(store.is_ok());
  PairedDevice father;
  father.fingerprint = kFpA;
  father.name = "Father";
  father.role = "father";
  REQUIRE(store.value()
              ->update([&](NodeSettings& s) {
                s.name = "G14";
                s.paired_father = father;
                s.allow_when_idle = false;
                s.idle_seconds = 120;
                s.ac_only = false;
                s.temp_storage_limit_gib = 24;
                s.paused = true;
                s.start_with_system = false;
                s.caps = {8, 6, 12};
                return Status::ok();
              })
              .is_ok());
  auto again = NodeSettingsStore::open(file);
  REQUIRE(again.is_ok());
  CHECK(again.value()->get() == store.value()->get());
  CHECK(again.value()->get().paired_father->fingerprint == kFpA);
  CHECK(again.value()->get().caps.vram_gib == 6);
}

TEST_CASE("a corrupt file yields defaults and the original is preserved, never deleted") {
  TempDir dir("corrupt");
  const auto file = dir.path / "father-settings.json";
  const std::string garbage = "{ this is not json \x01\x02";
  spit(file, garbage);
  auto store = FatherSettingsStore::open(file);
  REQUIRE(store.is_ok());
  CHECK(store.value()->report().outcome == LoadReport::Outcome::kRecoveredCorrupt);
  CHECK(store.value()->get() == FatherSettings{});
  const auto backup = store.value()->report().backup;
  REQUIRE_FALSE(backup.empty());
  CHECK(fs::exists(backup));
  CHECK(slurp(backup) == garbage);
  CHECK_FALSE(fs::exists(file));
  // The product keeps working: a later update creates a fresh valid document next to the backup.
  REQUIRE(store.value()->update([](FatherSettings& s) { s.selected_tier = "strong"; return Status::ok(); }).is_ok());
  auto reopened = FatherSettingsStore::open(file);
  REQUIRE(reopened.is_ok());
  CHECK(reopened.value()->get().selected_tier == "strong");
  CHECK(fs::exists(backup));
}

TEST_CASE("schema violations are treated as corruption; invalid updates are refused and change nothing") {
  TempDir dir("schema");
  const auto file = dir.path / "father-settings.json";
  spit(file, R"({"version":1,"paired_nodes":[{"fingerprint":"nothex","name":"x","address":"1.2.3.4:5","role":"node"}]})");
  auto store = FatherSettingsStore::open(file);
  REQUIRE(store.is_ok());
  CHECK(store.value()->report().outcome == LoadReport::Outcome::kRecoveredCorrupt);

  spit(file, R"({"version":1,"context_tokens":"big"})");
  auto wrong_type = FatherSettingsStore::open(file);
  REQUIRE(wrong_type.is_ok());
  CHECK(wrong_type.value()->report().outcome == LoadReport::Outcome::kRecoveredCorrupt);

  // Refused update: the in-memory value and the disk are untouched.
  REQUIRE(store.value()->update([](FatherSettings& s) { s.selected_tier = "strong"; return Status::ok(); }).is_ok());
  const std::string before = slurp(file);
  auto st = store.value()->update([](FatherSettings& s) {
    s.assignments["node:laptop-class"] = std::string(64, 'c');  // not a paired device
    return Status::ok();
  });
  CHECK_FALSE(st.is_ok());
  CHECK(slurp(file) == before);
  CHECK(store.value()->get().assignments.empty());
  st = store.value()->update([](FatherSettings& s) {
    s.context_tokens = 3;
    return Status::ok();
  });
  CHECK_FALSE(st.is_ok());
}

TEST_CASE("documents and their directory are owner-only, and a too-open file is tightened on load") {
  TempDir dir("perm");
  const auto file = dir.path / "sub" / "node-settings.json";
  auto store = NodeSettingsStore::open(file);
  REQUIRE(store.is_ok());
  REQUIRE(store.value()->update([](NodeSettings& s) { s.name = "n"; return Status::ok(); }).is_ok());
  auto file_ok = platform::is_owner_only(file);
  REQUIRE(file_ok.is_ok());
  CHECK(file_ok.value());
  auto dir_ok = platform::is_owner_only(file.parent_path());
  REQUIRE(dir_ok.is_ok());
  CHECK(dir_ok.value());

#ifndef _WIN32
  fs::permissions(file, fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read | fs::perms::others_read,
                  fs::perm_options::replace);
  CHECK_FALSE(platform::is_owner_only(file).value());
  auto reopened = NodeSettingsStore::open(file);
  REQUIRE(reopened.is_ok());
  CHECK(reopened.value()->report().permissions_tightened);
  CHECK(reopened.value()->report().outcome == LoadReport::Outcome::kLoaded);
  CHECK(platform::is_owner_only(file).value());
#endif
}

TEST_CASE("an older version is migrated through the hook with the original kept; a newer one is refused untouched") {
  TempDir dir("migrate");
  const auto file = dir.path / "node-settings.json";
  // Imagined version 0: the display name lived under "label" and idle time was in minutes.
  const std::string v0 = R"({"version":0,"label":"old-g14","idle_minutes":7})";
  spit(file, v0);
  int calls = 0;
  MigrationHook hook = [&](std::string_view doc, int from) -> Result<std::string> {
    ++calls;
    if (from != 0) return make_error(ErrorCode::kInternal, "unexpected version");
    (void)doc;
    return std::string(R"({"version":1,"name":"old-g14","idle_seconds":420})");
  };
  auto store = NodeSettingsStore::open(file, hook);
  REQUIRE(store.is_ok());
  CHECK(calls == 1);
  CHECK(store.value()->report().outcome == LoadReport::Outcome::kMigrated);
  CHECK(store.value()->get().name == "old-g14");
  CHECK(store.value()->get().idle_seconds == 420);
  REQUIRE_FALSE(store.value()->report().backup.empty());
  CHECK(slurp(store.value()->report().backup) == v0);
  // The migrated document is on disk and loads without the hook.
  auto again = NodeSettingsStore::open(file);
  REQUIRE(again.is_ok());
  CHECK(again.value()->report().outcome == LoadReport::Outcome::kLoaded);
  CHECK(again.value()->get().idle_seconds == 420);

  // Without a hook an old document cannot be understood: defaults, original preserved.
  spit(file, v0);
  auto no_hook = NodeSettingsStore::open(file);
  REQUIRE(no_hook.is_ok());
  CHECK(no_hook.value()->report().outcome == LoadReport::Outcome::kRecoveredCorrupt);

  // A document from the future is refused and left exactly as it is.
  const std::string future = R"({"version":99,"name":"x"})";
  spit(file, future);
  auto newer = NodeSettingsStore::open(file);
  REQUIRE_FALSE(newer.is_ok());
  CHECK(newer.status().code() == ErrorCode::kVersionMismatch);
  CHECK(slurp(file) == future);
}

TEST_CASE("removing a paired node also removes its assignments; duplicate and mislabelled devices are rejected") {
  FatherSettings s;
  s.upsert_node(node(kFpA, "G14", "10.0.0.2:47600"));
  s.upsert_node(node(kFpB, "3060", "10.0.0.3:47600"));
  s.assignments["node:laptop-class"] = kFpA;
  s.assignments["node:designated-3060"] = kFpB;
  CHECK(validate(s).is_ok());
  CHECK(s.remove_node(kFpA));
  CHECK_FALSE(s.remove_node(kFpA));
  CHECK(s.assignments.count("node:laptop-class") == 0);
  CHECK(s.assignments.count("node:designated-3060") == 1);

  FatherSettings dup;
  dup.paired_nodes = {node(kFpA, "a", "10.0.0.2:1"), node(kFpA, "b", "10.0.0.3:1")};
  CHECK_FALSE(validate(dup).is_ok());
  FatherSettings bad_addr;
  bad_addr.paired_nodes = {node(kFpA, "a", "no-port")};
  CHECK_FALSE(validate(bad_addr).is_ok());
  FatherSettings wrong_role;
  auto d = node(kFpA, "a", "10.0.0.2:1");
  d.role = "father";
  wrong_role.paired_nodes = {d};
  CHECK_FALSE(validate(wrong_role).is_ok());
}

TEST_CASE("default settings paths follow the platform layout") {
  platform::PathRoots roots;
  roots.local_app_data = "/data";
  roots.program_data = "/state";
  roots.runtime_dir = "/run";
  const auto p = platform::default_paths_from(roots, platform::PathStyle::kPosix);
  CHECK(father_settings_path(p).filename() == "father-settings.json");
  CHECK(node_settings_path(p).filename() == "node-settings.json");
  CHECK(father_settings_path(p).parent_path() == p.father_root);
  CHECK(node_settings_path(p).parent_path() == p.node_root);
}
