#include "clusterlm/lease/lease_store.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <set>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/platform/fs_safety.hpp"

namespace clusterlm::lease {

namespace fs = std::filesystem;
namespace plat = clusterlm::platform;

namespace {

constexpr std::align_val_t kRamAlign{64};

void append_error(std::string& errors, const std::string& what) {
  if (!errors.empty()) errors += "; ";
  errors += what;
}

}  // namespace

std::string_view to_string(LeaseState s) noexcept {
  switch (s) {
    case LeaseState::kNone: return "none";
    case LeaseState::kPreparing: return "preparing";
    case LeaseState::kReady: return "ready";
    case LeaseState::kReleasing: return "releasing";
    case LeaseState::kCleanupPending: return "cleanup_pending";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------------------------------------
// ObjectWriter
// ---------------------------------------------------------------------------------------------------------

void ObjectWriter::AlignedFree::operator()(std::uint8_t* p) const noexcept { ::operator delete(p, kRamAlign); }

ObjectWriter::~ObjectWriter() { (void)release_resources(); }

Status ObjectWriter::init_ram() {
  if (size_ == 0) return Status::ok();
  if (size_ > static_cast<std::uint64_t>(PTRDIFF_MAX))
    return make_error(ErrorCode::kResourceExhausted, "object too large for address space");
  auto* p = static_cast<std::uint8_t*>(::operator new(static_cast<std::size_t>(size_), kRamAlign, std::nothrow));
  if (p == nullptr) return make_error(ErrorCode::kResourceExhausted, "RAM allocation failed");
  ram_.reset(p);
  return Status::ok();
}

Status ObjectWriter::init_disk(const fs::path& path) {
  auto mf = plat::MappedFile::create(path, size_);
  if (!mf.is_ok()) return mf.status();
  file_ = std::move(mf).value();
  file_path_ = path;
  return Status::ok();
}

Status ObjectWriter::release_resources() {
  std::lock_guard lock(mu_);
  ram_.reset();
  Status st = file_.close();
  sealed_ = false;
  ranges_.clear();
  return st;
}

std::uint8_t* ObjectWriter::writable_data() {
  if (placement_ == Placement::kRam) return ram_.get();
  return file_.writable_span().data();
}

ByteSpan ObjectWriter::data_view() const {
  if (size_ == 0) return {};
  if (placement_ == Placement::kRam) return {ram_.get(), static_cast<std::size_t>(size_)};
  return file_.span();
}

bool ObjectWriter::is_sealed() const {
  std::lock_guard lock(mu_);
  return sealed_;
}

std::uint64_t ObjectWriter::bytes_received() const {
  std::lock_guard lock(mu_);
  std::uint64_t n = 0;
  for (const auto& [b, e] : ranges_) n += e - b;
  return n;
}

Status ObjectWriter::write_chunk(std::uint64_t offset, ByteSpan data, const Digest256& chunk_digest) {
  {
    std::lock_guard lock(mu_);
    if (sealed_) return make_error(ErrorCode::kFailedPrecondition, "object already sealed");
    if (data.empty()) return make_error(ErrorCode::kInvalidArgument, "empty chunk");
    if (offset > size_ || data.size() > size_ - offset)
      return make_error(ErrorCode::kOutOfRange, "chunk exceeds object bounds");
    if (Sha256::of(data) != chunk_digest)
      return make_error(ErrorCode::kDataLoss, "chunk digest mismatch");
    const std::uint64_t end = offset + data.size();
    // Overlap with the previous interval (starts at/before offset) or any starting inside [offset,end).
    auto next = ranges_.lower_bound(offset);
    if (next != ranges_.end() && next->first < end)
      return make_error(ErrorCode::kAlreadyExists, "chunk overlaps a received range");
    if (next != ranges_.begin() && std::prev(next)->second > offset)
      return make_error(ErrorCode::kAlreadyExists, "chunk overlaps a received range");
    std::uint8_t* dst = writable_data();
    if (dst == nullptr) return make_error(ErrorCode::kFailedPrecondition, "object storage is not available");
    std::memcpy(dst + offset, data.data(), data.size());
    // Record and merge touching neighbours so completeness is a single-interval check.
    std::uint64_t b = offset, e = end;
    if (next != ranges_.begin() && std::prev(next)->second == offset) {
      auto prev = std::prev(next);
      b = prev->first;
      ranges_.erase(prev);
    }
    if (next != ranges_.end() && next->first == end) {
      e = next->second;
      ranges_.erase(next);
    }
    ranges_[b] = e;
  }
  fire("transfer");
  return Status::ok();
}

Status ObjectWriter::seal() {
  std::lock_guard lock(mu_);
  if (sealed_) return make_error(ErrorCode::kFailedPrecondition, "object already sealed");
  const bool complete = size_ == 0 ? ranges_.empty()
                                   : (ranges_.size() == 1 && ranges_.begin()->first == 0 &&
                                      ranges_.begin()->second == size_);
  if (!complete) return make_error(ErrorCode::kDataLoss, "object incomplete: not every byte was received");
  fire("hashing");
  if (Sha256::of(data_view()) != expected_) return make_error(ErrorCode::kDataLoss, "object digest mismatch");
  fire("packing");
  if (placement_ == Placement::kDisk) {
    fire("mapping");
    CLM_RETURN_IF_ERROR(file_.flush());
    CLM_RETURN_IF_ERROR(file_.close());
    auto ro = plat::MappedFile::open(file_path_, plat::MapMode::kReadOnly);
    if (!ro.is_ok()) return ro.status();
    file_ = std::move(ro).value();
    if (file_.size() != size_) return make_error(ErrorCode::kDataLoss, "sealed file size changed");
  }
  sealed_ = true;
  return Status::ok();
}

Status ObjectWriter::reset() {
  std::lock_guard lock(mu_);
  if (sealed_) return make_error(ErrorCode::kFailedPrecondition, "cannot reset a sealed object");
  ranges_.clear();
  return Status::ok();
}

ByteSpan ObjectWriter::sealed_bytes() const {
  std::lock_guard lock(mu_);
  return sealed_ ? data_view() : ByteSpan{};
}

// ---------------------------------------------------------------------------------------------------------
// LeaseStore
// ---------------------------------------------------------------------------------------------------------

LeaseStore::~LeaseStore() {
  // A normal shutdown must not leave staging behind; if this fails the journal lets the next start finish.
  if (state_ == LeaseState::kPreparing || state_ == LeaseState::kReady || state_ == LeaseState::kCleanupPending)
    (void)release();
}

Result<std::unique_ptr<LeaseStore>> LeaseStore::open(const fs::path& root_in, LeaseStoreOptions options) {
  if (root_in.empty()) return make_error(ErrorCode::kInvalidArgument, "empty staging root");
  std::error_code ec;
  fs::path root = fs::absolute(root_in, ec).lexically_normal();
  if (ec) return make_error(ErrorCode::kInvalidArgument, "cannot resolve staging root: " + ec.message());
  if (plat::is_link_or_reparse(root))
    return make_error(ErrorCode::kPermissionDenied, "staging root must not be a link");
  fs::create_directories(root, ec);
  if (ec) return make_error(ErrorCode::kInternal, "cannot create staging root: " + ec.message());
  fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace, ec);  // best effort (no-op semantics on NTFS)

  std::unique_ptr<LeaseStore> store(new LeaseStore(std::move(root), std::move(options)));
  std::lock_guard lock(store->mu_);
  if (plat::is_link_or_reparse(store->leases_dir()))
    return make_error(ErrorCode::kPermissionDenied, "leases directory must not be a link");

  // Replay BEFORE reopening for append: opening terminates a torn tail, which would hide it from the report.
  CLM_ASSIGN_OR_RETURN(JournalReplay replay, Journal::replay(store->journal_path()));
  CLM_ASSIGN_OR_RETURN(store->journal_, Journal::open(store->journal_path()));
  fs::create_directories(store->leases_dir(), ec);
  if (ec) return make_error(ErrorCode::kInternal, "cannot create leases directory: " + ec.message());
  store->run_recovery_locked(replay);
  return store;
}

void LeaseStore::run_recovery_locked(const JournalReplay& replay) {
  recovery_ = RecoveryReport{};
  recovery_.journal_corrupt_lines = replay.corrupt_lines;
  recovery_.journal_torn_tail = replay.torn_tail;
  last_generation_ = std::max(last_generation_, replay.max_generation);

  std::set<std::string> recorded, handled;
  auto remove_counted = [&](const fs::path& dir) {
    auto before = plat::census_under(dir);
    Status st = plat::remove_tree_no_follow(root_, dir);
    if (!st.is_ok()) {
      recovery_.errors.push_back(st.message());
      return;
    }
    if (before.is_ok()) {
      recovery_.files_removed += before->files + before->links;
      recovery_.bytes_reclaimed += before->bytes;
    }
  };

  for (const auto& l : replay.leases) {
    const std::string name = std::to_string(l.generation);
    recorded.insert(name);
    if (l.released) continue;
    handled.insert(name);
    ++recovery_.leases_found;
    auto dir = plat::resolve_under_root(root_, fs::path("leases") / name);
    if (!dir.is_ok()) {
      recovery_.errors.push_back(dir.status().message());
      continue;
    }
    remove_counted(*dir);
  }

  // leases/ is wholly app-owned. Anything in it that no journal record names (e.g. left by a lost journal)
  // is still staging by construction and is removed, but counted separately.
  std::error_code ec;
  std::vector<fs::path> strays;
  for (fs::directory_iterator it(leases_dir(), ec), end; !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (handled.count(name) != 0) continue;  // unreleased lease: already removed (or its error recorded)
    if (recorded.count(name) == 0) ++recovery_.unrecorded_entries_removed;
    strays.push_back(it->path());
  }
  if (ec) recovery_.errors.push_back("cannot list leases directory: " + ec.message());
  for (const auto& p : strays) remove_counted(p);

  std::uint64_t residual_bytes = 0, residual_entries = 0;
  Status census = census_clean_locked(residual_bytes, residual_entries);
  if (!census.is_ok()) recovery_.errors.push_back(census.message());
  else if (residual_entries != 0)
    recovery_.errors.push_back("storage not clean after recovery: " + std::to_string(residual_entries) +
                               " entries, " + std::to_string(residual_bytes) + " bytes remain");

  if (recovery_.errors.empty()) {
    orphans_pending_ = false;
    state_ = LeaseState::kNone;
    Status st = journal_.compact(last_generation_);
    if (!st.is_ok()) log::warn("lease.journal_compact_failed", {{"error", st.message()}});
    if (recovery_.leases_found != 0 || recovery_.unrecorded_entries_removed != 0)
      log::info("lease.recovery", {{"leases", std::to_string(recovery_.leases_found)},
                                   {"files_removed", std::to_string(recovery_.files_removed)},
                                   {"bytes_reclaimed", std::to_string(recovery_.bytes_reclaimed)}});
  } else {
    orphans_pending_ = true;
    state_ = LeaseState::kCleanupPending;
    log::warn("lease.recovery_incomplete", {{"errors", std::to_string(recovery_.errors.size())}});
  }
}

Status LeaseStore::census_clean_locked(std::uint64_t& residual_bytes, std::uint64_t& residual_entries) const {
  residual_bytes = 0;
  residual_entries = 0;
  std::error_code ec;
  for (fs::directory_iterator it(root_, ec), end; !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (name == "journal.log" || name == "journal.log.tmp") continue;  // content-free by design
    auto c = plat::census_under(it->path());
    if (!c.is_ok()) return c.status();
    residual_bytes += c->bytes;
    // Entries below the top-level child; a stray top-level file/link is counted by census_under itself.
    residual_entries += c->files + c->links + c->dirs;
  }
  if (ec) return make_error(ErrorCode::kInternal, "cannot list staging root: " + ec.message());
  return Status::ok();
}

Status LeaseStore::begin_lease(LeaseGeneration generation, LeaseBudget budget) {
  std::lock_guard lock(mu_);
  if (state_ == LeaseState::kCleanupPending)
    return make_error(ErrorCode::kFailedPrecondition, "storage cleanup pending; retry_cleanup() first");
  if (state_ != LeaseState::kNone) return make_error(ErrorCode::kFailedPrecondition, "a lease is already active");
  if (generation.value <= last_generation_)
    return make_error(ErrorCode::kStaleEpoch, "lease generation not newer than last begun generation");

  auto dir = plat::resolve_under_root(root_, fs::path("leases") / generation.str());
  if (!dir.is_ok()) return dir.status();

  // Durable intent first: if we crash between here and the mkdir, recovery simply finds nothing to delete.
  CLM_RETURN_IF_ERROR(journal_.append_begin(generation));
  last_generation_ = generation.value;

  std::error_code ec;
  fs::create_directories(leases_dir(), ec);
  if (!ec) fs::create_directory(*dir, ec);
  if (ec) {
    (void)journal_.append_released(generation);  // nothing was created, so nothing is outstanding
    return make_error(ErrorCode::kInternal, "cannot create lease directory: " + ec.message());
  }
  fs::permissions(*dir, fs::perms::owner_all, fs::perm_options::replace, ec);

  generation_ = generation;
  budget_ = budget;
  ram_used_ = disk_used_ = 0;
  lease_dir_ = *dir;
  state_ = LeaseState::kPreparing;
  return Status::ok();
}

Result<ObjectWriter*> LeaseStore::create_object(std::uint32_t index, std::uint64_t size,
                                                const Digest256& expected_digest, Placement placement) {
  std::lock_guard lock(mu_);
  if (state_ != LeaseState::kPreparing)
    return make_error(ErrorCode::kFailedPrecondition, "objects can only be created while preparing a lease");
  if (objects_.count(index) != 0) return make_error(ErrorCode::kAlreadyExists, "object index already created");

  std::uint64_t& used = placement == Placement::kRam ? ram_used_ : disk_used_;
  const std::uint64_t cap = placement == Placement::kRam ? budget_.ram_bytes : budget_.disk_bytes;
  if (size > cap || used > cap - size)
    return make_error(ErrorCode::kResourceExhausted,
                      std::string(placement == Placement::kRam ? "RAM" : "disk") + " lease budget exceeded");

  std::unique_ptr<ObjectWriter> w(new ObjectWriter(index, size, expected_digest, placement, &options_.crash_hook));
  if (placement == Placement::kRam) {
    CLM_RETURN_IF_ERROR(w->init_ram());
  } else {
    const std::string name = lease_file_name(index);
    auto path = plat::resolve_under_root(root_, fs::path("leases") / generation_.str() / name);
    if (!path.is_ok()) return path.status();
    CLM_RETURN_IF_ERROR(journal_.append_file(generation_, name));  // durable BEFORE the file exists
    CLM_RETURN_IF_ERROR(w->init_disk(*path));
  }
  used += size;
  ObjectWriter* raw = w.get();
  objects_.emplace(index, std::move(w));
  return raw;
}

ObjectWriter* LeaseStore::find_object(std::uint32_t index) {
  std::lock_guard lock(mu_);
  auto it = objects_.find(index);
  return it == objects_.end() ? nullptr : it->second.get();
}

bool LeaseStore::is_sealed(std::uint32_t index) const {
  std::lock_guard lock(mu_);
  auto it = objects_.find(index);
  return it != objects_.end() && it->second->is_sealed();
}

Result<ByteSpan> LeaseStore::sealed_bytes(std::uint32_t index) const {
  std::lock_guard lock(mu_);
  auto it = objects_.find(index);
  if (it == objects_.end()) return make_error(ErrorCode::kNotFound, "no such object in lease");
  if (!it->second->is_sealed()) return make_error(ErrorCode::kFailedPrecondition, "object not sealed");
  return it->second->sealed_bytes();
}

Status LeaseStore::mark_ready() {
  std::lock_guard lock(mu_);
  if (state_ != LeaseState::kPreparing)
    return make_error(ErrorCode::kFailedPrecondition, "mark_ready requires a preparing lease");
  for (const auto& [idx, w] : objects_)
    if (!w->is_sealed()) return make_error(ErrorCode::kFailedPrecondition, "an object is not sealed");
  state_ = LeaseState::kReady;
  fire("ready");
  return Status::ok();
}

void LeaseStore::notify_phase(std::string_view phase) {
  std::lock_guard lock(mu_);
  fire(phase);
}

ReleaseReport LeaseStore::cleanup_locked(ReleaseReport report) {
  std::string errors;

  // 1. Resources: views unmapped and mapping + file handles closed, RAM freed. Only then can the OS
  //    reclaim file space (Windows refuses to delete a mapped file at all).
  for (auto it = objects_.begin(); it != objects_.end();) {
    Status st = it->second->release_resources();
    if (st.is_ok()) {
      it = objects_.erase(it);
    } else {
      append_error(errors, "object " + std::to_string(it->first) + ": " + st.message());
      ++it;
    }
  }
  report.resources_released = objects_.empty();
  if (report.resources_released) ram_used_ = disk_used_ = 0;

  // 2. Storage: delete children of the lease directory one by one (so a crash can land midway), then the
  //    directory itself. Everything goes through validated no-follow removal.
  bool deletion_ok = true;
  if (!lease_dir_.empty()) {
    if (!plat::is_link_or_reparse(lease_dir_)) {
      std::error_code ec;
      std::vector<fs::path> kids;
      for (fs::directory_iterator it(lease_dir_, ec), end; !ec && it != end; it.increment(ec))
        kids.push_back(it->path());
      if (ec && ec != std::errc::no_such_file_or_directory) {
        deletion_ok = false;
        append_error(errors, "cannot list lease directory: " + ec.message());
      }
      bool fired = false;
      for (const auto& kid : kids) {
        Status st = plat::remove_tree_no_follow(root_, kid);
        if (!st.is_ok()) {
          deletion_ok = false;
          append_error(errors, st.message());
        }
        if (!fired) {
          fire("cleanup");
          fired = true;
        }
      }
      if (!fired) fire("cleanup");
    }
    Status st = plat::remove_tree_no_follow(root_, lease_dir_);
    if (!st.is_ok()) {
      deletion_ok = false;
      append_error(errors, st.message());
    }
  }

  // 3. Verify by census; never trust that deletion calls succeeding means space is free.
  std::uint64_t residual_bytes = 0, residual_entries = 0;
  Status census = census_clean_locked(residual_bytes, residual_entries);
  report.residual_bytes = residual_bytes;
  report.residual_files = residual_entries;
  if (!census.is_ok()) {
    deletion_ok = false;
    append_error(errors, census.message());
  }
  report.storage_cleaned = report.resources_released && deletion_ok && census.is_ok() && residual_entries == 0;
  if (!report.storage_cleaned && errors.empty()) append_error(errors, "storage not verified clean");

  if (report.storage_cleaned) {
    Status st = journal_.append_released(generation_);
    if (st.is_ok()) st = journal_.compact(last_generation_);
    // The directory is already gone, so a journal failure cannot strand data: recovery would find nothing.
    if (!st.is_ok()) append_error(errors, "journal: " + st.message());
    lease_dir_.clear();
    state_ = LeaseState::kNone;
  } else {
    state_ = LeaseState::kCleanupPending;
  }
  report.errors = std::move(errors);
  return report;
}

ReleaseReport LeaseStore::release() {
  std::lock_guard lock(mu_);
  Stopwatch sw;
  ReleaseReport report;
  if (orphans_pending_) {
    // The store never became usable: finish orphan recovery instead of releasing a lease.
    auto replay = Journal::replay(journal_path());
    if (!replay.is_ok()) {
      report.errors = replay.status().message();
      report.resources_released = true;
      report.duration_ns = sw.elapsed_ns();
      return report;
    }
    run_recovery_locked(*replay);
    report.resources_released = true;
    report.storage_cleaned = recovery_.clean();
    for (const auto& e : recovery_.errors) append_error(report.errors, e);
    std::uint64_t b = 0, n = 0;
    if (census_clean_locked(b, n).is_ok()) {
      report.residual_bytes = b;
      report.residual_files = n;
    }
    report.duration_ns = sw.elapsed_ns();
    return report;
  }
  if (state_ == LeaseState::kNone) {
    report.resources_released = true;
    report.storage_cleaned = true;
    report.duration_ns = sw.elapsed_ns();
    return report;
  }
  state_ = LeaseState::kReleasing;
  report = cleanup_locked(std::move(report));
  report.duration_ns = sw.elapsed_ns();
  return report;
}

ReleaseReport LeaseStore::retry_cleanup() { return release(); }

LeaseState LeaseStore::state() const {
  std::lock_guard lock(mu_);
  return state_;
}

LeaseGeneration LeaseStore::generation() const {
  std::lock_guard lock(mu_);
  return generation_;
}

std::uint64_t LeaseStore::last_generation() const {
  std::lock_guard lock(mu_);
  return last_generation_;
}

std::uint64_t LeaseStore::ram_used() const {
  std::lock_guard lock(mu_);
  return ram_used_;
}

std::uint64_t LeaseStore::disk_used() const {
  std::lock_guard lock(mu_);
  return disk_used_;
}

}  // namespace clusterlm::lease
