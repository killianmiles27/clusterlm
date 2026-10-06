#include <doctest/doctest.h>

#include <fstream>

#include "clusterlm/lease/journal.hpp"
#include "clusterlm/lease/lease_store.hpp"
#include "clusterlm/platform/durable_file.hpp"
#include "test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::lease;
namespace fs = std::filesystem;

static std::string slurp(const fs::path& p) {
  auto b = platform::read_file_bytes(p);
  REQUIRE(b.is_ok());
  return std::string(b->begin(), b->end());
}

TEST_CASE("journal records are content-free and replay reconstructs leases") {
  testing::TempDir dir;
  const fs::path p = dir.path() / "journal.log";
  {
    auto j = Journal::open(p);
    REQUIRE(j.is_ok());
    CHECK(j->append_begin(LeaseGeneration(1)).is_ok());
    CHECK(j->append_file(LeaseGeneration(1), "obj-0.part").is_ok());
    CHECK(j->append_released(LeaseGeneration(1)).is_ok());
    CHECK(j->append_begin(LeaseGeneration(2)).is_ok());
    CHECK(j->append_file(LeaseGeneration(2), "obj-7.part").is_ok());
    // Names from a manifest (or anything not store-generated) are refused outright.
    CHECK(j->append_file(LeaseGeneration(2), "../etc/passwd").code() == ErrorCode::kInvalidArgument);
    CHECK(j->append_file(LeaseGeneration(2), "llama-70b-q4.gguf").code() == ErrorCode::kInvalidArgument);
  }
  CHECK(slurp(p) == "B 1\nF 1 obj-0.part\nR 1\nB 2\nF 2 obj-7.part\n");
  auto r = Journal::replay(p);
  REQUIRE(r.is_ok());
  REQUIRE(r->leases.size() == 2);
  CHECK(r->leases[0].released);
  CHECK_FALSE(r->leases[1].released);
  CHECK(r->leases[1].files == std::vector<std::string>{"obj-7.part"});
  CHECK(r->max_generation == 2);
  CHECK_FALSE(r->torn_tail);
  CHECK(r->corrupt_lines == 0);
}

TEST_CASE("journal tolerates a torn tail and garbage lines") {
  testing::TempDir dir;
  const fs::path p = dir.path() / "journal.log";
  {
    std::ofstream f(p, std::ios::binary);
    f << "B 1\nF 1 obj-0.part\nthis is not a record\nF 9 obj-1.part\nB 2\nF 2 obj-";  // torn final line
  }
  auto r = Journal::replay(p);
  REQUIRE(r.is_ok());
  CHECK(r->torn_tail);
  CHECK(r->corrupt_lines == 2);  // garbage + file record for unknown generation
  REQUIRE(r->leases.size() == 2);
  CHECK(r->leases[1].generation == 2);
  CHECK(r->leases[1].files.empty());

  // Reopening for append terminates the torn line so a new record never glues onto it.
  auto j = Journal::open(p);
  REQUIRE(j.is_ok());
  CHECK(j->append_begin(LeaseGeneration(3)).is_ok());
  auto r2 = Journal::replay(p);
  REQUIRE(r2.is_ok());
  CHECK_FALSE(r2->torn_tail);
  CHECK(r2->max_generation == 3);
}

TEST_CASE("journal compaction keeps only the highest generation") {
  testing::TempDir dir;
  const fs::path p = dir.path() / "journal.log";
  auto j = Journal::open(p);
  REQUIRE(j.is_ok());
  REQUIRE(j->append_begin(LeaseGeneration(4)).is_ok());
  REQUIRE(j->append_released(LeaseGeneration(4)).is_ok());
  CHECK(j->compact(4).is_ok());
  CHECK(slurp(p) == "G 4\n");
  CHECK(j->append_begin(LeaseGeneration(5)).is_ok());  // still appendable after compaction
  auto r = Journal::replay(p);
  REQUIRE(r.is_ok());
  CHECK(r->max_generation == 5);
  CHECK(r->leases.size() == 1);
}

TEST_CASE("a missing journal replays as empty") {
  testing::TempDir dir;
  auto r = Journal::replay(dir.path() / "none.log");
  REQUIRE(r.is_ok());
  CHECK(r->leases.empty());
  CHECK(r->max_generation == 0);
}

TEST_CASE("LeaseStore compacts the journal when no lease is active and the journal never names objects") {
  testing::TempDir dir;
  const fs::path root = dir.path() / "stage";
  auto s = LeaseStore::open(root);
  REQUIRE(s.is_ok());
  CHECK(slurp(root / "journal.log") == "G 0\n");
  REQUIRE((*s)->begin_lease(LeaseGeneration(3), LeaseBudget{1000, 1000}).is_ok());
  const Bytes data = testing::pattern_bytes(100);
  auto w = (*s)->create_object(42, data.size(), Sha256::of(ByteSpan(data)), Placement::kDisk);
  REQUIRE(w.is_ok());
  const std::string active = slurp(root / "journal.log");
  CHECK(active == "G 0\nB 3\nF 3 obj-42.part\n");
  CHECK((*s)->release().storage_cleaned);
  CHECK(slurp(root / "journal.log") == "G 3\n");
}

TEST_CASE("LeaseStore recovers from a torn journal tail and ignores generations it never began") {
  testing::TempDir dir;
  const fs::path root = dir.path() / "stage";
  fs::create_directories(root / "leases" / "5");
  std::ofstream(root / "leases" / "5" / "obj-0.part") << std::string(777, 'o');
  {
    std::ofstream f(root / "journal.log", std::ios::binary);
    f << "B 5\nF 5 obj-0.part\nB 6\nF 6 obj-";  // generation 6's begin may be durable; the tail is torn
  }
  auto s = LeaseStore::open(root);
  REQUIRE(s.is_ok());
  const RecoveryReport& rep = (*s)->recovery_report();
  CHECK(rep.journal_torn_tail);
  CHECK(rep.leases_found == 2);  // 5 and 6 (6 had no directory: nothing to delete)
  CHECK(rep.files_removed == 1);
  CHECK(rep.bytes_reclaimed == 777);
  CHECK(rep.clean());
  CHECK_FALSE(fs::exists(root / "leases" / "5"));
  CHECK((*s)->last_generation() == 6);
  CHECK((*s)->begin_lease(LeaseGeneration(6), LeaseBudget{1, 1}).code() == ErrorCode::kStaleEpoch);
  CHECK((*s)->begin_lease(LeaseGeneration(7), LeaseBudget{1, 1}).is_ok());
}

TEST_CASE("recovery also removes unrecorded entries under leases/ but leaves nothing counted as clean wrongly") {
  testing::TempDir dir;
  const fs::path root = dir.path() / "stage";
  fs::create_directories(root / "leases" / "99");
  std::ofstream(root / "leases" / "99" / "obj-1.part") << std::string(10, 'u');
  std::ofstream(root / "leases" / "stray.bin") << std::string(20, 's');
  auto s = LeaseStore::open(root);
  REQUIRE(s.is_ok());
  const RecoveryReport& rep = (*s)->recovery_report();
  CHECK(rep.leases_found == 0);
  CHECK(rep.unrecorded_entries_removed == 2);
  CHECK(rep.bytes_reclaimed == 30);
  CHECK(rep.clean());
  CHECK(fs::is_empty(root / "leases"));
}
