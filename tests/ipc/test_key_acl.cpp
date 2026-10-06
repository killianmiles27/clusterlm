// The device private key is written owner-only (POSIX 0600 / Windows protected DACL: owner + SYSTEM).
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "clusterlm/platform/paths.hpp"
#include "clusterlm/transport/security.hpp"

#ifndef _WIN32
#include <sys/stat.h>
#endif

using namespace clusterlm;
namespace fs = std::filesystem;

namespace {
struct TempDir {
  fs::path path;
  explicit TempDir(const char* name) : path(fs::temp_directory_path() / (std::string("clm-key-") + name)) {
    fs::remove_all(path);
  }
  ~TempDir() { fs::remove_all(path); }
};
}  // namespace

TEST_CASE("DeviceIdentity::save writes the private key owner-only and it still loads") {
  TempDir t("save");
  auto id = transport::DeviceIdentity::generate("acl-test");
  REQUIRE(id.is_ok());
  REQUIRE(id->save(t.path).is_ok());
  const fs::path key = t.path / "device_key.pem";
  auto only = platform::is_owner_only(key);
  REQUIRE(only.is_ok());
  CHECK(only.value());
#ifndef _WIN32
  struct stat st {};
  REQUIRE(::stat(key.c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
#endif
  auto loaded = transport::DeviceIdentity::load(t.path);
  REQUIRE(loaded.is_ok());
  CHECK(loaded->fingerprint() == id->fingerprint());
}

TEST_CASE("saving over an existing permissive key tightens it before writing") {
  TempDir t("tighten");
  fs::create_directories(t.path);
  const fs::path key = t.path / "device_key.pem";
  std::ofstream(key) << "stale";
#ifndef _WIN32
  ::chmod(key.c_str(), 0644);
  CHECK_FALSE(platform::is_owner_only(key).value());
#endif
  auto id = transport::DeviceIdentity::generate("acl-test");
  REQUIRE(id.is_ok());
  REQUIRE(id->save(t.path).is_ok());
  CHECK(platform::is_owner_only(key).value());
  CHECK(transport::DeviceIdentity::load(t.path).is_ok());
}

TEST_CASE("load_or_generate creates an owner-only key on first run") {
  TempDir t("first-run");
  auto id = transport::DeviceIdentity::load_or_generate(t.path, "node");
  REQUIRE(id.is_ok());
  CHECK(platform::is_owner_only(t.path / "device_key.pem").value());
}
