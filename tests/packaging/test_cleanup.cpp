// `clusterlm-node-service --cleanup`: what the Node uninstaller runs. Real binary, real lease store files.
// POSIX only (popen/setenv); on Windows the same behaviour is covered by packaging/smoke-test.ps1.
#include <doctest/doctest.h>

#ifndef _WIN32

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "clusterlm/platform/paths.hpp"

namespace fs = std::filesystem;
using namespace clusterlm;

namespace {

struct Run {
  int exit_code = -1;
  std::string output;
};

Run run(const std::string& command) {
  Run r;
  FILE* p = ::popen((command + " 2>&1").c_str(), "r");
  if (p == nullptr) return r;
  char buf[512];
  while (std::fgets(buf, sizeof buf, p) != nullptr) r.output += buf;
  const int status = ::pclose(p);
  r.exit_code = status == -1 ? -1 : (status >> 8) & 0xff;
  return r;
}

std::string quote(const fs::path& p) { return "\"" + p.string() + "\""; }

struct Dir {
  fs::path path;
  explicit Dir(const char* name) : path(fs::temp_directory_path() / (std::string("clm-packaging-") + name)) {
    { std::error_code ec_rm; fs::remove_all(path, ec_rm); }
    fs::create_directories(path);
  }
  ~Dir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

std::size_t count_files(const fs::path& root) {
  std::size_t n = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec))
    if (it->is_regular_file(ec)) ++n;
  return n;
}

const std::string kService = CLUSTERLM_NODE_SERVICE_BINARY;

void point_default_layout_at(const Dir& d) {
  ::setenv("XDG_STATE_HOME", (d.path / "state").c_str(), 1);
  ::setenv("XDG_DATA_HOME", (d.path / "data").c_str(), 1);
}

}  // namespace

TEST_CASE("cleanup deletes the staging root of a crashed lease, including its journal") {
  Dir d("crashed");
  const fs::path staging = d.path / "staging";
  // A lease that was mid-flight when the process died (the crash helper exits at the "inference" phase).
  const auto crash = run(quote(CLUSTERLM_CRASH_HELPER_BINARY) + " " + quote(staging) + " inference");
  REQUIRE(crash.exit_code == 3);
  REQUIRE(count_files(staging) > 1);  // objects + journal are on disk

  const auto r = run(quote(kService) + " --cleanup --staging " + quote(staging));
  INFO(r.output);
  CHECK(r.exit_code == 0);
  CHECK(r.output.find("CLUSTERLM_NODE_CLEANUP leases_found=1") != std::string::npos);
  CHECK(r.output.find("clean=1") != std::string::npos);
  CHECK_FALSE(fs::exists(staging));
}

TEST_CASE("cleanup removes entries that no journal record accounts for") {
  Dir d("orphan");
  const fs::path staging = d.path / "staging";
  fs::create_directories(staging / "leases" / "9");
  std::ofstream(staging / "leases" / "9" / "obj-0.part") << std::string(4096, 'x');
  const auto r = run(quote(kService) + " --cleanup --staging " + quote(staging));
  INFO(r.output);
  CHECK(r.exit_code == 0);
  CHECK_FALSE(fs::exists(staging));
}

TEST_CASE("cleanup of a Node that never ran is a success") {
  Dir d("missing");
  point_default_layout_at(d);
  const auto r = run(quote(kService) + " --cleanup --staging " + quote(d.path / "nothing"));
  INFO(r.output);
  CHECK(r.exit_code == 0);
}

TEST_CASE("default layout: cleanup keeps a pairing identity, --purge removes it") {
  Dir d("layout");
  point_default_layout_at(d);
  auto paths = platform::default_paths();
  REQUIRE(paths.is_ok());
  auto populate = [&] {
    fs::create_directories(paths->node_staging / "leases" / "1");
    std::ofstream(paths->node_staging / "leases" / "1" / "obj-0.part") << "fragment";
    fs::create_directories(paths->node_identity);
    std::ofstream(paths->node_identity / "device_key.pem") << "key";
    fs::create_directories(paths->node_logs);
    std::ofstream(paths->node_logs / "node.log") << "log";
    std::ofstream(paths->node_root / "node-settings.json") << "{}";  // the paired Father and caps
    std::ofstream(paths->node_root / "node-settings.json.corrupt-1") << "x";
  };
  populate();

  auto r = run(quote(kService) + " --cleanup");
  INFO(r.output);
  CHECK(r.exit_code == 0);
  CHECK_FALSE(fs::exists(paths->node_staging));                // model fragments never survive
  CHECK(fs::exists(paths->node_identity / "device_key.pem"));  // the pairing identity does, by default
  CHECK(fs::exists(paths->node_logs / "node.log"));
  CHECK(fs::exists(paths->node_root / "node-settings.json"));

  populate();
  r = run(quote(kService) + " --cleanup --purge");
  INFO(r.output);
  CHECK(r.exit_code == 0);
  CHECK_FALSE(fs::exists(paths->node_staging));
  CHECK_FALSE(fs::exists(paths->node_identity));
  CHECK_FALSE(fs::exists(paths->node_logs));
  CHECK_FALSE(fs::exists(paths->node_root));
}

TEST_CASE("--purge never deletes files it does not know") {
  Dir d("unknown");
  point_default_layout_at(d);
  auto paths = platform::default_paths();
  REQUIRE(paths.is_ok());
  fs::create_directories(paths->node_root);
  std::ofstream(paths->node_root / "settings.json") << "{}";
  const auto r = run(quote(kService) + " --cleanup --purge");
  INFO(r.output);
  CHECK(r.exit_code == 0);
  CHECK(fs::exists(paths->node_root / "settings.json"));
}

#else

TEST_CASE("cleanup tests are POSIX-only") { MESSAGE("covered on Windows by packaging/smoke-test.ps1"); }

#endif
