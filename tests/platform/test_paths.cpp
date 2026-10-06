#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "clusterlm/platform/paths.hpp"

#ifndef _WIN32
#include <sys/stat.h>
#endif

using namespace clusterlm;
using namespace clusterlm::platform;
namespace fs = std::filesystem;

namespace {
struct TempDir {
  fs::path path;
  explicit TempDir(const char* name) : path(fs::temp_directory_path() / (std::string("clm-paths-") + name)) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
};
}  // namespace

TEST_CASE("Windows layout: Node under ProgramData, Father under LocalAppData") {
  PathRoots roots;
  roots.program_data = "C:/ProgramData";
  roots.local_app_data = "C:/Users/alice/AppData/Local";
  const auto p = default_paths_from(roots, PathStyle::kWindows);
  CHECK(p.node_root == fs::path("C:/ProgramData/ClusterLM/Node"));
  CHECK(p.node_staging == fs::path("C:/ProgramData/ClusterLM/Node/staging"));
  CHECK(p.node_identity == fs::path("C:/ProgramData/ClusterLM/Node/identity"));
  CHECK(p.node_logs == fs::path("C:/ProgramData/ClusterLM/Node/logs"));
  CHECK(p.father_root == fs::path("C:/Users/alice/AppData/Local/ClusterLM"));
  CHECK(p.father_models == fs::path("C:/Users/alice/AppData/Local/ClusterLM/models"));
  CHECK(p.father_identity == fs::path("C:/Users/alice/AppData/Local/ClusterLM/identity"));
}

TEST_CASE("POSIX layout follows XDG roots; the Node staging root is never under the Father model store") {
  PathRoots roots;
  roots.program_data = "/home/u/.local/state";
  roots.local_app_data = "/home/u/.local/share";
  roots.runtime_dir = "/run/user/1000";
  const auto p = default_paths_from(roots, PathStyle::kPosix);
  CHECK(p.node_staging == fs::path("/home/u/.local/state/clusterlm/node/staging"));
  CHECK(p.father_models == fs::path("/home/u/.local/share/clusterlm/father/models"));
  CHECK(p.ipc_dir == fs::path("/run/user/1000/clusterlm"));
  CHECK(p.node_staging.string().find(p.father_models.string()) == std::string::npos);
}

TEST_CASE("default_paths resolves on this host") {
  auto p = default_paths();
  // HOME is set in CI; an environment without HOME and XDG vars legitimately reports kFailedPrecondition.
  if (p.is_ok()) {
    CHECK(p->node_staging.is_absolute());
    CHECK(p->father_models.is_absolute());
  } else {
    CHECK(p.status().code() == ErrorCode::kFailedPrecondition);
  }
}

TEST_CASE("write_owner_only_file creates, tightens and verifies owner-only files") {
  TempDir t("file");
  const fs::path f = t.path / "secret.bin";
  const Bytes data{1, 2, 3, 4};
  REQUIRE(write_owner_only_file(f, data).is_ok());
  auto only = is_owner_only(f);
  REQUIRE(only.is_ok());
  CHECK(only.value());
  CHECK(fs::file_size(f) == 4);

#ifndef _WIN32
  struct stat st {};
  REQUIRE(::stat(f.c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
  // A pre-existing world-readable file is tightened, not left open.
  ::chmod(f.c_str(), 0644);
  CHECK_FALSE(is_owner_only(f).value());
  REQUIRE(write_owner_only_file(f, data).is_ok());
  CHECK(is_owner_only(f).value());
#endif
}

TEST_CASE("restrict_to_owner fixes a permissive file and is_owner_only notices") {
  TempDir t("restrict");
  const fs::path f = t.path / "open.txt";
  std::ofstream(f) << "x";
#ifndef _WIN32
  ::chmod(f.c_str(), 0666);
  CHECK_FALSE(is_owner_only(f).value());
#endif
  REQUIRE(restrict_to_owner(f).is_ok());
  CHECK(is_owner_only(f).value());
  CHECK_FALSE(is_owner_only(t.path / "missing").is_ok());
}

TEST_CASE("create_owner_only_directory makes the whole leaf owner-only, including an existing directory") {
  TempDir t("dir");
  const fs::path d = t.path / "a" / "staging";
  REQUIRE(create_owner_only_directory(d).is_ok());
  CHECK(fs::is_directory(d));
  CHECK(is_owner_only(d).value());
#ifndef _WIN32
  ::chmod(d.c_str(), 0755);
  CHECK_FALSE(is_owner_only(d).value());
  REQUIRE(create_owner_only_directory(d).is_ok());
  CHECK(is_owner_only(d).value());
#endif
}
