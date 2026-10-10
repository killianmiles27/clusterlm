#pragma once
// ModelLibrary: the in-memory collection of ModelRecords plus the scan roots, with identity resolution for profiles.
//
// Identity resolution never guesses (no silent model or quantization substitution):
//   * a pinned identity (expected_root_hash) matches only a record whose verified root_hash or pinned_root equals it;
//   * an unpinned identity matches by family + quant (case-insensitive) and is reported kUnpinnedMatch: the caller
//     (readiness) must treat it as "needs user confirmation", never as verified;
//   * more than one candidate is kAmbiguous, never "first wins".
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/library/model_record.hpp"
#include "clusterlm/library/scanner.hpp"

namespace clusterlm::library {

inline constexpr std::size_t kMaxLibraryRecords = 4096;
inline constexpr std::size_t kMaxScanRoots = 64;

struct ModelIdentityQuery {
  std::string family;
  std::string quant;
  std::optional<std::string> expected_root_hash;  // null = unpinned
};

struct Resolution {
  enum class Kind : std::uint8_t { kMissing, kVerified, kUnpinnedMatch, kAmbiguous } kind = Kind::kMissing;
  const ModelRecord* record = nullptr;           // set for kVerified / kUnpinnedMatch
  std::vector<std::string> candidates;           // record ids, for kAmbiguous
  std::string detail;                            // human sentence
};

class ModelLibrary {
 public:
  const std::vector<ModelRecord>& records() const { return records_; }
  const std::vector<std::string>& scan_roots() const { return scan_roots_; }

  const ModelRecord* find(std::string_view id) const;
  // Adds a record or refreshes an existing one with the same id (keeps its name/family/root_hash/pinned_root labels).
  Status upsert(ModelRecord record);
  bool remove(std::string_view id);
  Status add_scan_root(std::string dir);
  bool remove_scan_root(std::string_view dir);
  // Scans every root (and nothing else) and upserts what it finds.
  Result<ScanResult> rescan(const ScanOptions& opt = {});
  // Explicit user confirmation of an inspected root; the record must exist and root must be 64 hex.
  Status pin_root(std::string_view id, std::string root_hash);
  // Records the verified manifest root (verification). A different root than an existing verified one is refused.
  Status set_verified_root(std::string_view id, std::string root_hash);

  Resolution resolve(const ModelIdentityQuery& q) const;

  Status validate() const;
  std::string to_json() const;
  static Result<ModelLibrary> from_json(std::string_view json);

  friend bool operator==(const ModelLibrary&, const ModelLibrary&) = default;

 private:
  std::vector<ModelRecord> records_;
  std::vector<std::string> scan_roots_;
};

}  // namespace clusterlm::library
