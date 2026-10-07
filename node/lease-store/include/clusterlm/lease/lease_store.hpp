#pragma once
// Ephemeral lease object store (spec §10-§11).
//
// Invariant: once a Node stops participating, no application-owned model weights, converted packs, tensor
// fragments or state files remain on its persistent storage, and nothing is cached across leases. Objects
// live in process RAM or in files under `root/leases/<gen>/`; a tiny content-free journal lets a restarted
// service delete crash orphans BEFORE it accepts a new lease.
//
// Objects are identified by a numeric index only; file names are store-generated ("obj-<index>.part"), never
// taken from a manifest. An adapter to the objects-module ObjectResolver is built on sealed_bytes().
//
// Threading: LeaseStore methods are serialised by an internal mutex. The crash hook runs with internal mutexes held
// and must not call back into the store or its writers. ObjectWriter pointers are valid until release() and, like the span
// from sealed_bytes(), must not be used concurrently with or after release().
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/digest.hpp"
#include "clusterlm/common/ids.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/lease/journal.hpp"
#include "clusterlm/platform/mapped_file.hpp"

namespace clusterlm::lease {

enum class Placement : std::uint8_t { kRam, kDisk };
enum class LeaseState : std::uint8_t { kNone, kPreparing, kReady, kReleasing, kCleanupPending };
std::string_view to_string(LeaseState s) noexcept;

struct LeaseBudget {
  std::uint64_t ram_bytes = 0;
  std::uint64_t disk_bytes = 0;
};

// Crash-test hook. Phases: "transfer" (each accepted chunk), "hashing" (seal start), "packing" (digest
// verified, before residency), "mapping" (disk object about to be remapped read-only), "allocation" and
// "inference" (raised by the worker via LeaseStore::notify_phase), "ready" (mark_ready done), "cleanup"
// (midway through deletion). Production leaves the hook empty.
using CrashHook = std::function<void(std::string_view phase)>;

struct LeaseStoreOptions {
  CrashHook crash_hook;
};

struct RecoveryReport {
  std::uint64_t leases_found = 0;               // journal leases begun but never released
  std::uint64_t files_removed = 0;              // files/links physically deleted
  std::uint64_t bytes_reclaimed = 0;
  std::uint64_t unrecorded_entries_removed = 0; // entries under leases/ with no journal record
  std::uint64_t journal_corrupt_lines = 0;
  bool journal_torn_tail = false;
  std::vector<std::string> errors;
  bool clean() const noexcept { return errors.empty(); }
};

// Resource release and storage cleanup are reported separately: storage_cleaned is true only if every
// resource was released AND a census of the root (journal excluded) finds nothing left.
struct ReleaseReport {
  bool resources_released = false;  // RAM freed, views unmapped, mapping + file handles closed
  bool storage_cleaned = false;
  std::uint64_t residual_bytes = 0;
  std::uint64_t residual_files = 0;
  std::uint64_t duration_ns = 0;
  std::string errors;
};

// Receives one object's bytes. Obtained from LeaseStore::create_object; owned by the store.
class ObjectWriter {
 public:
  ~ObjectWriter();
  ObjectWriter(const ObjectWriter&) = delete;
  ObjectWriter& operator=(const ObjectWriter&) = delete;

  std::uint32_t index() const noexcept { return index_; }
  std::uint64_t size() const noexcept { return size_; }
  Placement placement() const noexcept { return placement_; }
  bool is_sealed() const;
  std::uint64_t bytes_received() const;

  // Validates SHA-256 of `data` against `chunk_digest` (kDataLoss), bounds (kOutOfRange), and that the
  // range does not overlap or duplicate an already accepted one (kAlreadyExists), then stores it.
  Status write_chunk(std::uint64_t offset, ByteSpan data, const Digest256& chunk_digest);
  // Requires every byte present (kDataLoss otherwise) and whole-object SHA-256 == expected (kDataLoss).
  // Disk objects are remapped read-only. After a failed seal the object stays unsealed; reset() allows a
  // re-transfer, otherwise release the lease.
  Status seal();
  // Discards received data and ranges of an unsealed object.
  Status reset();
  // Empty unless sealed.
  ByteSpan sealed_bytes() const;

 private:
  friend class LeaseStore;
  ObjectWriter(std::uint32_t index, std::uint64_t size, Digest256 expected, Placement placement,
               const CrashHook* hook)
      : index_(index), size_(size), expected_(expected), placement_(placement), hook_(hook) {}

  struct AlignedFree {
    void operator()(std::uint8_t* p) const noexcept;
  };
  Status init_ram();
  Status init_disk(const std::filesystem::path& path);
  // Frees RAM / unmaps / closes handles. Returns failure if a view or handle could not be released.
  Status release_resources();
  std::uint8_t* writable_data();
  ByteSpan data_view() const;
  void fire(std::string_view phase) const {
    if (hook_ != nullptr && *hook_) (*hook_)(phase);
  }

  const std::uint32_t index_;
  const std::uint64_t size_;
  const Digest256 expected_;
  const Placement placement_;
  const CrashHook* hook_;

  mutable std::mutex mu_;
  std::map<std::uint64_t, std::uint64_t> ranges_;  // disjoint [start,end) intervals, touching ones merged
  bool sealed_ = false;
  std::unique_ptr<std::uint8_t, AlignedFree> ram_;  // 64-byte aligned
  platform::MappedFile file_;
  std::filesystem::path file_path_;
};

class LeaseStore {
 public:
  ~LeaseStore();  // best-effort release of an active lease
  LeaseStore(const LeaseStore&) = delete;
  LeaseStore& operator=(const LeaseStore&) = delete;

  // Creates the staging root if needed, then performs orphan recovery before the store is usable. If
  // recovery cannot fully clean, the store comes up in kCleanupPending: begin_lease is refused until
  // retry_cleanup() succeeds. Layout: root/journal.log, root/leases/<gen>/obj-<index>.part.
  static Result<std::unique_ptr<LeaseStore>> open(const std::filesystem::path& root, LeaseStoreOptions options = {});

  const RecoveryReport& recovery_report() const noexcept { return recovery_; }

  // The "begin" journal record is durable before the lease directory exists. One active lease at a time;
  // generations must strictly increase (kStaleEpoch otherwise), including across restarts.
  Status begin_lease(LeaseGeneration generation, LeaseBudget budget);

  // Budgets are enforced here (kResourceExhausted). Disk objects: the journal "file" record is durable
  // before the file is created.
  Result<ObjectWriter*> create_object(std::uint32_t index, std::uint64_t size, const Digest256& expected_digest,
                                      Placement placement);
  // Creates every object of a plan at once. Budgets and indices are checked for all of them first; the file
  // records of all disk objects are then made durable in one journal append (one sync) before any file exists.
  // Equivalent to create_object per spec, but plan admission stays fast with thousands of disk-backed objects.
  struct ObjectSpec {
    std::uint32_t index = 0;
    std::uint64_t size = 0;
    Digest256 expected_digest;
    Placement placement = Placement::kRam;
  };
  Status create_objects(const std::vector<ObjectSpec>& specs);
  ObjectWriter* find_object(std::uint32_t index);

  bool is_sealed(std::uint32_t index) const;
  Result<ByteSpan> sealed_bytes(std::uint32_t index) const;

  // kPreparing -> kReady; every created object must be sealed.
  Status mark_ready();

  // Worker-raised phase marker ("allocation", "inference") routed to the crash hook.
  void notify_phase(std::string_view phase);

  // Frees RAM, unmaps, closes handles, deletes the lease directory without following links, censuses the
  // root, journals "released" and compacts. On failure the state is kCleanupPending.
  ReleaseReport release();
  // Re-attempts a pending release (or orphan recovery after open). Same report semantics.
  ReleaseReport retry_cleanup();

  LeaseState state() const;
  LeaseGeneration generation() const;       // current/last lease generation
  std::uint64_t last_generation() const;    // highest ever begun
  std::uint64_t ram_used() const;
  std::uint64_t disk_used() const;
  const std::filesystem::path& root() const noexcept { return root_; }

 private:
  LeaseStore(std::filesystem::path root, LeaseStoreOptions options)
      : root_(std::move(root)), options_(std::move(options)) {}

  void fire(std::string_view phase) const {
    if (options_.crash_hook) options_.crash_hook(phase);
  }
  // Both require mu_ held.
  void run_recovery_locked(const JournalReplay& replay);
  ReleaseReport cleanup_locked(ReleaseReport report);
  Status census_clean_locked(std::uint64_t& residual_bytes, std::uint64_t& residual_files) const;
  std::filesystem::path journal_path() const { return root_ / "journal.log"; }
  std::filesystem::path leases_dir() const { return root_ / "leases"; }

  const std::filesystem::path root_;
  LeaseStoreOptions options_;
  mutable std::mutex mu_;
  Journal journal_;
  RecoveryReport recovery_;
  LeaseState state_ = LeaseState::kNone;
  bool orphans_pending_ = false;
  LeaseGeneration generation_{};
  std::uint64_t last_generation_ = 0;
  LeaseBudget budget_{};
  std::uint64_t ram_used_ = 0;
  std::uint64_t disk_used_ = 0;
  std::filesystem::path lease_dir_;
  std::map<std::uint32_t, std::unique_ptr<ObjectWriter>> objects_;
};

}  // namespace clusterlm::lease
