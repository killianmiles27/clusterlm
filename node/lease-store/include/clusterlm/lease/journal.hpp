#pragma once
// Content-free lease journal (spec §10-§11).
//
// An append-only text file that exists only so a restarted service can find and delete staging orphaned by a
// crash BEFORE it accepts a new lease. It records generation numbers, store-generated relative file names and
// state words only: never object names from manifests, token IDs, model paths or digests.
//
//   B <gen>            lease generation began (durable BEFORE its directory exists)
//   F <gen> <name>     file <name> is about to be created in the lease directory (durable BEFORE creation)
//   R <gen>            lease fully released: resources freed and storage verified clean
//   G <gen>            compaction marker: highest generation ever begun (kept so stale generations stay stale)
//
// Reading tolerates a torn final line (no trailing newline) and skips unparsable lines, counting them.
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/ids.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/platform/durable_file.hpp"

namespace clusterlm::lease {

// "obj-<decimal index>.part": the only file name the store ever creates inside a lease directory.
std::string lease_file_name(std::uint32_t object_index);
bool is_valid_lease_file_name(std::string_view name);

struct JournalLease {
  std::uint64_t generation = 0;
  std::vector<std::string> files;  // store-generated names recorded for this lease
  bool released = false;
};

struct JournalReplay {
  std::vector<JournalLease> leases;  // in order of first appearance
  std::uint64_t max_generation = 0;
  std::uint64_t corrupt_lines = 0;   // complete lines that did not parse (ignored)
  bool torn_tail = false;            // final line lacked a newline (crash mid-append) and was ignored
};

class Journal {
 public:
  Journal() = default;

  // Reads and parses `path`; a missing file is an empty replay.
  static Result<JournalReplay> replay(const std::filesystem::path& path);
  // Opens for durable append, creating the file if needed. If the existing tail is torn, a newline is
  // appended first so new records never glue onto garbage.
  static Result<Journal> open(const std::filesystem::path& path);

  Status append_begin(LeaseGeneration gen);
  Status append_file(LeaseGeneration gen, std::string_view name);
  Status append_released(LeaseGeneration gen);

  // Atomically rewrites the journal to a single "G <max_generation>" record. Only legal when no lease is
  // active (the caller guarantees it); drops all history.
  Status compact(std::uint64_t max_generation);

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  Status append_line(const std::string& line);

  std::filesystem::path path_;
  platform::DurableAppendFile file_;
};

}  // namespace clusterlm::lease
