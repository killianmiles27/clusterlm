#pragma once
// SettingsStore: crash-safe persistence for a settings document.
//
//   * Writes are atomic and owner-only: content goes to "<file>.tmp" created owner-only from the first instant
//     (platform::write_owner_only_file), is flushed, then renamed over the target and the directory is synced.
//     A reader sees the old or the new document, never a mixture. The parent directory is created owner-only.
//   * Loading never trusts the file: size-capped, schema-validated (settings.hpp). A document that cannot be
//     parsed or fails validation is MOVED ASIDE to "<file>.corrupt-<stamp>" (never deleted) and defaults are
//     returned, so a damaged file can neither crash the product nor be silently destroyed.
//   * A file readable by other accounts is tightened to owner-only on load and the load report says so.
//   * Versioning: a document newer than this build is refused (kVersionMismatch, the file is untouched); an
//     older one goes through the migration hook (document text in, document text out, one version at a time),
//     the original is kept as "<file>.v<old>.bak", and the migrated document is saved.
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "clusterlm/common/status.hpp"
#include "clusterlm/config/settings.hpp"
#include "clusterlm/platform/paths.hpp"

namespace clusterlm::config {

// Receives the document text and the version it carries; returns the text of version `from_version + 1`.
using MigrationHook = std::function<Result<std::string>(std::string_view document, int from_version)>;

struct LoadReport {
  enum class Outcome : std::uint8_t {
    kLoaded,
    kDefaultsNoFile,     // first run: nothing on disk yet
    kRecoveredCorrupt,   // damaged/invalid file moved to `backup`; defaults in use
    kMigrated,           // older version migrated; original kept at `backup`
  };
  Outcome outcome = Outcome::kLoaded;
  std::filesystem::path backup;   // set for kRecoveredCorrupt and kMigrated
  bool permissions_tightened = false;
  std::string note;               // short human-readable detail (never document content)
};

template <typename T>
class SettingsStore {
 public:
  // Loads (or defaults) according to the rules above. Fails only for I/O errors on the directory and for a
  // document newer than this build.
  static Result<std::unique_ptr<SettingsStore>> open(std::filesystem::path file, MigrationHook migrate = {});

  T get() const;
  // Read-modify-write under the store's lock: `mutate` edits a copy, the result is validated and persisted
  // atomically, then becomes current. If `mutate` or validation fails nothing changes on disk or in memory.
  Status update(const std::function<Status(T&)>& mutate);
  Status replace(T value);

  const LoadReport& report() const { return report_; }
  const std::filesystem::path& file() const { return file_; }

 private:
  explicit SettingsStore(std::filesystem::path file) : file_(std::move(file)) {}
  Status persist_locked(const T& value);

  std::filesystem::path file_;
  LoadReport report_;
  mutable std::mutex mu_;
  T value_{};
};

extern template class SettingsStore<FatherSettings>;
extern template class SettingsStore<NodeSettings>;
using FatherSettingsStore = SettingsStore<FatherSettings>;
using NodeSettingsStore = SettingsStore<NodeSettings>;

// Default file locations: <father_root>/father-settings.json and <node_root>/node-settings.json.
std::filesystem::path father_settings_path(const platform::DefaultPaths& paths);
std::filesystem::path node_settings_path(const platform::DefaultPaths& paths);

}  // namespace clusterlm::config
