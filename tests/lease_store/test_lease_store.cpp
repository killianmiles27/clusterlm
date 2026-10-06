#include <doctest/doctest.h>

#include <cstring>
#include <fstream>

#include "clusterlm/lease/lease_store.hpp"
#include "clusterlm/platform/fs_safety.hpp"
#include "test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::lease;
namespace fs = std::filesystem;

namespace {

struct Fixture {
  testing::TempDir dir;
  std::unique_ptr<LeaseStore> store;
  Fixture() {
    auto s = LeaseStore::open(dir.path() / "stage");
    REQUIRE(s.is_ok());
    store = std::move(s).value();
  }
  fs::path root() const { return dir.path() / "stage"; }
};

LeaseBudget budget(std::uint64_t ram, std::uint64_t disk) { return LeaseBudget{ram, disk}; }

Status write_all(ObjectWriter& w, const Bytes& data, std::size_t chunk) {
  for (std::size_t off = 0; off < data.size(); off += chunk) {
    const std::size_t n = std::min(chunk, data.size() - off);
    ByteSpan span(data.data() + off, n);
    CLM_RETURN_IF_ERROR(w.write_chunk(off, span, Sha256::of(span)));
  }
  return Status::ok();
}

Result<ObjectWriter*> make_sealed(LeaseStore& s, std::uint32_t idx, const Bytes& data, Placement p) {
  auto w = s.create_object(idx, data.size(), Sha256::of(ByteSpan(data)), p);
  if (!w.is_ok()) return w;
  CLM_RETURN_IF_ERROR(write_all(**w, data, 1000));
  CLM_RETURN_IF_ERROR((*w)->seal());
  return w;
}

std::uint64_t census_bytes(const fs::path& root) {
  std::uint64_t total = 0;
  for (const auto& e : fs::directory_iterator(root)) {
    if (e.path().filename() == "journal.log") continue;
    total += *platform::allocated_bytes_under(e.path());
  }
  return total;
}

}  // namespace

TEST_CASE("chunk validation: bad digest, bounds, overlap, duplicates") {
  Fixture f;
  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  const Bytes data = testing::pattern_bytes(4096);
  auto w = f.store->create_object(0, data.size(), Sha256::of(ByteSpan(data)), Placement::kRam);
  REQUIRE(w.is_ok());
  ObjectWriter& ow = **w;

  ByteSpan first(data.data(), 1024);
  CHECK(ow.write_chunk(0, first, Sha256::of(ByteSpan(data.data() + 1, 1024))).code() == ErrorCode::kDataLoss);
  CHECK(ow.bytes_received() == 0);
  CHECK(ow.write_chunk(4000, ByteSpan(data.data(), 200), Sha256::of(ByteSpan(data.data(), 200))).code() ==
        ErrorCode::kOutOfRange);
  CHECK(ow.write_chunk(5000, first, Sha256::of(first)).code() == ErrorCode::kOutOfRange);
  CHECK(ow.write_chunk(0, ByteSpan{}, Sha256::of(ByteSpan{})).code() == ErrorCode::kInvalidArgument);

  CHECK(ow.write_chunk(0, first, Sha256::of(first)).is_ok());
  CHECK(ow.write_chunk(0, first, Sha256::of(first)).code() == ErrorCode::kAlreadyExists);  // duplicate
  ByteSpan overlap(data.data() + 512, 1024);
  CHECK(ow.write_chunk(512, overlap, Sha256::of(overlap)).code() == ErrorCode::kAlreadyExists);
  ByteSpan tail_overlap(data.data() + 1000, 100);  // starts before the end of [0,1024)
  CHECK(ow.write_chunk(1000, tail_overlap, Sha256::of(tail_overlap)).code() == ErrorCode::kAlreadyExists);
  // A later range that would straddle an existing one from the left.
  ByteSpan mid(data.data() + 2048, 512);
  CHECK(ow.write_chunk(2048, mid, Sha256::of(mid)).is_ok());
  ByteSpan straddle(data.data() + 1024, 1100);  // [1024, 2124) overlaps [2048,2560)
  CHECK(ow.write_chunk(1024, straddle, Sha256::of(straddle)).code() == ErrorCode::kAlreadyExists);
  CHECK(ow.bytes_received() == 1024 + 512);
}

TEST_CASE("seal requires completeness and a matching whole-object digest") {
  Fixture f;
  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  const Bytes data = testing::pattern_bytes(3000);

  SUBCASE("incomplete") {
    auto w = f.store->create_object(0, data.size(), Sha256::of(ByteSpan(data)), Placement::kRam);
    REQUIRE(w.is_ok());
    ByteSpan part(data.data(), 1000);
    REQUIRE((*w)->write_chunk(0, part, Sha256::of(part)).is_ok());
    CHECK((*w)->seal().code() == ErrorCode::kDataLoss);
    CHECK_FALSE((*w)->is_sealed());
    CHECK_FALSE(f.store->sealed_bytes(0).is_ok());
  }
  SUBCASE("wrong whole-object digest (RAM and disk)") {
    for (auto placement : {Placement::kRam, Placement::kDisk}) {
      const auto idx = static_cast<std::uint32_t>(placement == Placement::kRam ? 0 : 1);
      Bytes other = data;
      other[17] ^= 0xFF;
      auto w = f.store->create_object(idx, data.size(), Sha256::of(ByteSpan(other)), placement);
      REQUIRE(w.is_ok());
      REQUIRE(write_all(**w, data, 700).is_ok());  // chunk digests are right, whole digest is not
      CHECK((*w)->seal().code() == ErrorCode::kDataLoss);
      CHECK_FALSE((*w)->is_sealed());
    }
  }
  SUBCASE("reset allows a retransfer") {
    auto w = f.store->create_object(0, data.size(), Sha256::of(ByteSpan(data)), Placement::kRam);
    REQUIRE(w.is_ok());
    ByteSpan part(data.data(), 1000);
    REQUIRE((*w)->write_chunk(0, part, Sha256::of(part)).is_ok());
    CHECK((*w)->reset().is_ok());
    CHECK(write_all(**w, data, 3000).is_ok());
    CHECK((*w)->seal().is_ok());
    CHECK((*w)->write_chunk(0, part, Sha256::of(part)).code() == ErrorCode::kFailedPrecondition);
  }
}

TEST_CASE("budgets are enforced at create time") {
  Fixture f;
  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1000, 2000)).is_ok());
  const Digest256 d{};
  CHECK(f.store->create_object(0, 600, d, Placement::kRam).is_ok());
  CHECK(f.store->ram_used() == 600);
  CHECK(f.store->create_object(1, 401, d, Placement::kRam).status().code() == ErrorCode::kResourceExhausted);
  CHECK(f.store->create_object(1, 400, d, Placement::kRam).is_ok());
  CHECK(f.store->create_object(2, 2001, d, Placement::kDisk).status().code() == ErrorCode::kResourceExhausted);
  CHECK(f.store->create_object(2, 2000, d, Placement::kDisk).is_ok());
  CHECK(f.store->disk_used() == 2000);
  CHECK(f.store->create_object(2, 1, d, Placement::kRam).status().code() == ErrorCode::kAlreadyExists);
  CHECK(f.store->create_object(3, UINT64_MAX, d, Placement::kDisk).status().code() == ErrorCode::kResourceExhausted);
}

TEST_CASE("lease state machine: mark_ready needs every object sealed") {
  Fixture f;
  CHECK(f.store->state() == LeaseState::kNone);
  CHECK(f.store->create_object(0, 1, Digest256{}, Placement::kRam).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(f.store->mark_ready().code() == ErrorCode::kFailedPrecondition);

  REQUIRE(f.store->begin_lease(LeaseGeneration(5), budget(1 << 20, 1 << 20)).is_ok());
  CHECK(f.store->state() == LeaseState::kPreparing);
  CHECK(f.store->begin_lease(LeaseGeneration(6), budget(1, 1)).code() == ErrorCode::kFailedPrecondition);

  const Bytes a = testing::pattern_bytes(2000, 1), b = testing::pattern_bytes(2500, 2);
  REQUIRE(make_sealed(*f.store, 0, a, Placement::kRam).is_ok());
  auto wb = f.store->create_object(1, b.size(), Sha256::of(ByteSpan(b)), Placement::kDisk);
  REQUIRE(wb.is_ok());
  CHECK(f.store->mark_ready().code() == ErrorCode::kFailedPrecondition);  // object 1 not sealed
  REQUIRE(write_all(**wb, b, 512).is_ok());
  REQUIRE((*wb)->seal().is_ok());
  CHECK(f.store->is_sealed(0));
  CHECK(f.store->is_sealed(1));
  CHECK_FALSE(f.store->is_sealed(9));

  CHECK(f.store->mark_ready().is_ok());
  CHECK(f.store->state() == LeaseState::kReady);
  CHECK(f.store->create_object(2, 1, Digest256{}, Placement::kRam).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(f.store->mark_ready().code() == ErrorCode::kFailedPrecondition);  // already ready
}

TEST_CASE("sealed RAM and disk objects expose their bytes; release leaves zero census") {
  Fixture f;
  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  const Bytes ram = testing::pattern_bytes(5000, 3), disk = testing::pattern_bytes(70000, 4);
  REQUIRE(make_sealed(*f.store, 0, ram, Placement::kRam).is_ok());
  REQUIRE(make_sealed(*f.store, 1, disk, Placement::kDisk).is_ok());
  REQUIRE(f.store->mark_ready().is_ok());

  auto r = f.store->sealed_bytes(0);
  REQUIRE(r.is_ok());
  REQUIRE(r->size() == ram.size());
  CHECK(std::memcmp(r->data(), ram.data(), ram.size()) == 0);
  CHECK(reinterpret_cast<std::uintptr_t>(r->data()) % 64 == 0);  // 64-byte aligned RAM objects
  auto d = f.store->sealed_bytes(1);
  REQUIRE(d.is_ok());
  REQUIRE(d->size() == disk.size());
  CHECK(std::memcmp(d->data(), disk.data(), disk.size()) == 0);
  CHECK(census_bytes(f.root()) == disk.size());  // the disk object is physically present
  CHECK(f.store->sealed_bytes(7).status().code() == ErrorCode::kNotFound);

  ReleaseReport rep = f.store->release();
  CHECK(rep.errors.empty());
  CHECK(rep.resources_released);
  CHECK(rep.storage_cleaned);
  CHECK(rep.residual_bytes == 0);
  CHECK(rep.residual_files == 0);
  CHECK(census_bytes(f.root()) == 0);
  CHECK_FALSE(fs::exists(f.root() / "leases" / "1"));
  CHECK(f.store->state() == LeaseState::kNone);
  CHECK(f.store->ram_used() == 0);
  CHECK(f.store->disk_used() == 0);

  // Next lease: generation must advance; the old one is stale. Nothing is reused.
  CHECK(f.store->begin_lease(LeaseGeneration(1), budget(1, 1)).code() == ErrorCode::kStaleEpoch);
  CHECK(f.store->begin_lease(LeaseGeneration(2), budget(1, 1)).is_ok());
  CHECK_FALSE(f.store->is_sealed(0));
}

TEST_CASE("release before ready and a release with no lease are both fine") {
  Fixture f;
  CHECK(f.store->release().storage_cleaned);  // nothing to do
  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  auto w = f.store->create_object(0, 5000, Digest256{}, Placement::kDisk);  // never written: partial file
  REQUIRE(w.is_ok());
  CHECK(census_bytes(f.root()) == 5000);
  auto rep = f.store->release();
  CHECK(rep.storage_cleaned);
  CHECK(census_bytes(f.root()) == 0);
}

TEST_CASE("release removes an unexpected extra file planted in the lease dir") {
  Fixture f;
  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  REQUIRE(make_sealed(*f.store, 0, testing::pattern_bytes(1000), Placement::kDisk).is_ok());
  const fs::path lease_dir = f.root() / "leases" / "1";
  fs::create_directories(lease_dir / "nested" / "deeper");
  std::ofstream(lease_dir / "nested" / "deeper" / "stray.bin") << std::string(333, 'z');
  std::ofstream(lease_dir / "stray.tmp") << "hello";
  CHECK(census_bytes(f.root()) == 1000 + 333 + 5);

  auto rep = f.store->release();
  CHECK(rep.resources_released);
  CHECK(rep.storage_cleaned);
  CHECK(census_bytes(f.root()) == 0);
  CHECK_FALSE(fs::exists(lease_dir));
}

#ifndef _WIN32
TEST_CASE("release removes a planted symlink without touching its target outside the root") {
  Fixture f;
  testing::TempDir outside;
  std::ofstream(outside.path() / "precious.bin") << std::string(128, 'p');
  fs::create_directory(outside.path() / "precious_dir");
  std::ofstream(outside.path() / "precious_dir" / "inner.bin") << "inner";

  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  REQUIRE(make_sealed(*f.store, 0, testing::pattern_bytes(500), Placement::kDisk).is_ok());
  const fs::path lease_dir = f.root() / "leases" / "1";
  fs::create_symlink(outside.path() / "precious.bin", lease_dir / "evil-file");
  fs::create_directory_symlink(outside.path() / "precious_dir", lease_dir / "evil-dir");

  auto rep = f.store->release();
  CHECK(rep.storage_cleaned);
  CHECK(rep.errors.empty());
  CHECK_FALSE(fs::exists(lease_dir));
  CHECK(fs::file_size(outside.path() / "precious.bin") == 128);
  CHECK(fs::exists(outside.path() / "precious_dir" / "inner.bin"));
}

TEST_CASE("a failed deletion is reported honestly and retry_cleanup completes it") {
  if (::geteuid() == 0) {
    MESSAGE("running as root: directory permissions cannot force a deletion failure; skipped");
    return;
  }
  Fixture f;
  REQUIRE(f.store->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  REQUIRE(make_sealed(*f.store, 0, testing::pattern_bytes(1000), Placement::kDisk).is_ok());
  const fs::path lease_dir = f.root() / "leases" / "1";
  fs::create_directory(lease_dir / "locked");
  std::ofstream(lease_dir / "locked" / "x.bin") << "x";
  fs::permissions(lease_dir / "locked", fs::perms::owner_read | fs::perms::owner_exec);  // no write: unlink fails

  auto rep = f.store->release();
  CHECK(rep.resources_released);
  CHECK_FALSE(rep.storage_cleaned);
  CHECK(rep.residual_files > 0);
  CHECK_FALSE(rep.errors.empty());
  CHECK(f.store->state() == LeaseState::kCleanupPending);
  CHECK(f.store->begin_lease(LeaseGeneration(2), budget(1, 1)).code() == ErrorCode::kFailedPrecondition);

  fs::permissions(lease_dir / "locked", fs::perms::owner_all);
  auto retry = f.store->retry_cleanup();
  CHECK(retry.storage_cleaned);
  CHECK(f.store->state() == LeaseState::kNone);
  CHECK(census_bytes(f.root()) == 0);
}
#endif

TEST_CASE("destroying the store releases an active lease") {
  testing::TempDir dir;
  const fs::path root = dir.path() / "stage";
  {
    auto s = LeaseStore::open(root);
    REQUIRE(s.is_ok());
    REQUIRE((*s)->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
    REQUIRE(make_sealed(**s, 0, testing::pattern_bytes(4096), Placement::kDisk).is_ok());
  }
  CHECK(census_bytes(root) == 0);
  CHECK_FALSE(fs::exists(root / "leases" / "1"));
}

TEST_CASE("crash hook sees the documented phases in order") {
  testing::TempDir dir;
  std::vector<std::string> phases;
  LeaseStoreOptions opts;
  opts.crash_hook = [&](std::string_view p) {
    if (phases.empty() || phases.back() != p) phases.emplace_back(p);
  };
  auto s = LeaseStore::open(dir.path() / "stage", opts);
  REQUIRE(s.is_ok());
  REQUIRE((*s)->begin_lease(LeaseGeneration(1), budget(1 << 20, 1 << 20)).is_ok());
  REQUIRE(make_sealed(**s, 0, testing::pattern_bytes(2048), Placement::kDisk).is_ok());
  (*s)->notify_phase("allocation");
  REQUIRE((*s)->mark_ready().is_ok());
  (*s)->notify_phase("inference");
  (void)(*s)->release();
  const std::vector<std::string> expected = {"transfer", "hashing", "packing", "mapping",
                                             "allocation", "ready", "inference", "cleanup"};
  CHECK(phases == expected);
}
