#include <doctest/doctest.h>

#include <fstream>
#include <string>
#include <vector>

#include "clusterlm/lease/lease_store.hpp"
#include "clusterlm/platform/fs_safety.hpp"
#include "test_support.hpp"

#ifndef _WIN32
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

using namespace clusterlm;
using namespace clusterlm::lease;
namespace fs = std::filesystem;

namespace {

std::uint64_t census_bytes(const fs::path& root) {
  std::uint64_t total = 0;
  for (const auto& e : fs::directory_iterator(root)) {
    if (e.path().filename() == "journal.log") continue;
    total += *platform::allocated_bytes_under(e.path());
  }
  return total;
}

#ifndef _WIN32
// Runs the helper; returns its exit status or -1.
int run_helper(const fs::path& root, const std::string& phase) {
  const std::string exe = LEASE_CRASH_HELPER_PATH;
  const std::string root_s = root.string();
  std::vector<char*> argv = {const_cast<char*>(exe.c_str()), const_cast<char*>(root_s.c_str()),
                             const_cast<char*>(phase.c_str()), nullptr};
  pid_t pid = 0;
  if (::posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ) != 0) return -1;
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

}  // namespace

#ifdef _WIN32
TEST_CASE("crash recovery for every phase" * doctest::skip()) {
  MESSAGE("skipped: the crash harness uses posix_spawn; a CreateProcessW variant is future work");
}
#else

TEST_CASE("orphan recovery after a crash at every phase") {
  const std::vector<std::string> phases = {"transfer", "hashing", "packing",   "mapping",
                                           "allocation", "ready",   "inference", "cleanup"};
  for (const auto& phase : phases) {
    CAPTURE(phase);
    testing::TempDir dir;
    const fs::path root = dir.path() / "stage";

    REQUIRE(run_helper(root, phase) == 3);  // died abruptly at that phase

    // The helper stages the RAM object first, then two disk objects; "mapping" is the first phase at which
    // a disk file (object 1) exists, so from there on there is something physical to recover.
    const std::uint64_t before = census_bytes(root);
    const bool disk_exists = phase != "transfer" && phase != "hashing" && phase != "packing";
    if (disk_exists) CHECK(before > 0);

    auto s = LeaseStore::open(root);
    REQUIRE(s.is_ok());
    const RecoveryReport& rep = (*s)->recovery_report();
    CHECK(rep.leases_found == 1);
    CHECK(rep.clean());
    CHECK(rep.bytes_reclaimed == before);
    if (before > 0) CHECK(rep.files_removed > 0);
    CHECK(census_bytes(root) == 0);
    CHECK((*s)->state() == LeaseState::kNone);

    // The restarted service accepts a new lease only now, and old generations stay stale.
    CHECK((*s)->begin_lease(LeaseGeneration(1), LeaseBudget{1, 1}).code() == ErrorCode::kStaleEpoch);
    CHECK((*s)->begin_lease(LeaseGeneration((*s)->last_generation() + 1), LeaseBudget{1 << 20, 1 << 20}).is_ok());
    CHECK((*s)->release().storage_cleaned);
    CHECK(census_bytes(root) == 0);
  }
}

TEST_CASE("the crash helper runs to completion and cleans up when no phase matches") {
  testing::TempDir dir;
  const fs::path root = dir.path() / "stage";
  CHECK(run_helper(root, "never") == 0);
  CHECK(census_bytes(root) == 0);
  auto s = LeaseStore::open(root);
  REQUIRE(s.is_ok());
  CHECK((*s)->recovery_report().leases_found == 0);
}

TEST_CASE("repeated crashes on one root: each restart recovers before the next lease") {
  testing::TempDir dir;
  const fs::path root = dir.path() / "stage";
  REQUIRE(run_helper(root, "inference") == 3);
  // The helper itself opens the store, which recovers the previous orphan before beginning a new lease.
  REQUIRE(run_helper(root, "cleanup") == 3);
  REQUIRE(run_helper(root, "ready") == 3);
  auto s = LeaseStore::open(root);
  REQUIRE(s.is_ok());
  CHECK((*s)->recovery_report().leases_found == 1);
  CHECK(census_bytes(root) == 0);
}

TEST_CASE("recovery after a crash with a torn journal tail") {
  testing::TempDir dir;
  const fs::path root = dir.path() / "stage";
  REQUIRE(run_helper(root, "ready") == 3);
  {
    std::ofstream f(root / "journal.log", std::ios::binary | std::ios::app);
    f << "B 99\nF 99 obj-";  // simulate a crash mid-append after the real records
  }
  auto s = LeaseStore::open(root);
  REQUIRE(s.is_ok());
  CHECK((*s)->recovery_report().journal_torn_tail);
  CHECK((*s)->recovery_report().clean());
  CHECK(census_bytes(root) == 0);
  CHECK((*s)->last_generation() == 99);
}

#endif
