#include <doctest/doctest.h>

#include <fstream>

#include "../lease_store/test_support.hpp"
#include "clusterlm/platform/fs_safety.hpp"

using namespace clusterlm;
using namespace clusterlm::platform;
namespace fs = std::filesystem;

static void touch(const fs::path& p, std::size_t n) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << std::string(n, 'x');
}

TEST_CASE("resolve_under_root accepts plain relative paths") {
  testing::TempDir t;
  auto r = resolve_under_root(t.path(), "leases/7/obj-3.part");
  REQUIRE(r.is_ok());
  CHECK(*r == t.path() / "leases" / "7" / "obj-3.part");
}

TEST_CASE("resolve_under_root rejects traversal, absolute paths and device names") {
  testing::TempDir t;
  for (const char* bad : {"", "..", "../x", "a/../../x", "a/./b", "/etc/passwd", "a//b", "a/", "a\\b", "c:x",
                          "CON", "nul.txt", "a/COM1", "lpt9.part", "trailing.", "sp ", "a/b?c"}) {
    INFO("path: " << std::string(bad));
    CHECK_FALSE(resolve_under_root(t.path(), bad).is_ok());
  }
  // COM10 is not a reserved name and ordinary names containing a device name as a substring are fine.
  CHECK(resolve_under_root(t.path(), "com10").is_ok());
  CHECK(resolve_under_root(t.path(), "console").is_ok());
}

TEST_CASE("remove_tree_no_follow refuses targets outside the root") {
  testing::TempDir t;
  testing::TempDir other;
  touch(other.path() / "keep.bin", 10);
  CHECK_FALSE(remove_tree_no_follow(t.path(), other.path()).is_ok());
  CHECK_FALSE(remove_tree_no_follow(t.path(), t.path()).is_ok());  // root itself
  CHECK_FALSE(remove_tree_no_follow(t.path(), t.path() / ".." / "x").is_ok());
  CHECK(fs::exists(other.path() / "keep.bin"));
}

TEST_CASE("remove_tree_no_follow removes a tree and tolerates a missing path") {
  testing::TempDir t;
  touch(t.path() / "a" / "b" / "c.bin", 100);
  touch(t.path() / "a" / "d.bin", 50);
  CHECK(remove_tree_no_follow(t.path(), t.path() / "a").is_ok());
  CHECK_FALSE(fs::exists(t.path() / "a"));
  CHECK(remove_tree_no_follow(t.path(), t.path() / "a").is_ok());
}

#ifndef _WIN32
TEST_CASE("remove_tree_no_follow removes symlinks as links and never touches targets") {
  testing::TempDir t;
  testing::TempDir outside;
  touch(outside.path() / "secret" / "data.bin", 64);
  touch(t.path() / "dir" / "f.bin", 8);
  fs::create_directory_symlink(outside.path() / "secret", t.path() / "dir" / "dlink");
  fs::create_symlink(outside.path() / "secret" / "data.bin", t.path() / "dir" / "flink");
  REQUIRE(is_link_or_reparse(t.path() / "dir" / "dlink"));

  auto census = census_under(t.path());
  REQUIRE(census.is_ok());
  CHECK(census->links == 2);
  CHECK(census->bytes == 8);  // links are not followed or sized

  CHECK(remove_tree_no_follow(t.path(), t.path() / "dir").is_ok());
  CHECK_FALSE(fs::exists(t.path() / "dir"));
  CHECK(fs::exists(outside.path() / "secret" / "data.bin"));
  CHECK(fs::file_size(outside.path() / "secret" / "data.bin") == 64);
}

TEST_CASE("paths through a link ancestor are refused") {
  testing::TempDir t;
  testing::TempDir outside;
  touch(outside.path() / "victim" / "x.bin", 4);
  fs::create_directory_symlink(outside.path(), t.path() / "escape");
  CHECK_FALSE(resolve_under_root(t.path(), "escape/victim").is_ok());
  CHECK_FALSE(remove_tree_no_follow(t.path(), t.path() / "escape" / "victim").is_ok());
  CHECK(fs::exists(outside.path() / "victim" / "x.bin"));
}
#endif

TEST_CASE("census counts partial files and nothing for a missing root") {
  testing::TempDir t;
  touch(t.path() / "p" / "one.part", 1000);
  touch(t.path() / "p" / "two.part", 24);
  auto bytes = allocated_bytes_under(t.path());
  REQUIRE(bytes.is_ok());
  CHECK(*bytes == 1024);
  auto none = census_under(t.path() / "missing");
  REQUIRE(none.is_ok());
  CHECK(none->empty());
  CHECK(none->bytes == 0);
}
