#pragma once
// Cache / temp census (HQ-STORE-01): before/after snapshots of the places a run could leave files behind that
// ClusterLM's own staging census does not see, and a diff that reports what is new or changed. Names, sizes and
// modification times only; file CONTENTS are never read.
//
// Known locations (each is a "root"; absent roots are recorded as absent, not as empty):
//   CUDA compute cache        Windows %APPDATA%\NVIDIA\ComputeCache      Linux ~/.nv/ComputeCache  (CUDA_CACHE_PATH wins)
//   NVIDIA shader caches      Windows %LOCALAPPDATA%\NVIDIA\{DXCache,GLCache}   Linux ~/.cache/nvidia
//   OS temp                   %TEMP% / TMPDIR (/tmp)                       (shallow scan: busy and shared)
//   ClusterLM per-user data   Windows %LOCALAPPDATA%\ClusterLM             Linux ~/.local/share/clusterlm
//   Node staging roots        passed by the caller (the bench knows each LocalCluster Node's)
//
// A diff is attributed to a run window when the file is new, or when its modification time falls inside the
// window. A file in OS temp that another program wrote during the run is indistinguishable here; the report says
// "new or changed during the window", which is what can be known without kernel tracing (ETW/procmon, see
// HARDWARE-QUALIFICATION.md HQ-STORE-01).
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "clusterlm/common/status.hpp"
#include "result.hpp"

namespace clusterlm::bench {

struct CensusRoot {
  std::string label;
  std::filesystem::path path;
  int max_depth = 8;               // directory levels below the root that are descended into (1 = direct children only)
  std::size_t max_entries = 50000;  // scan stops (and the snapshot is flagged truncated) beyond this many files
  std::vector<std::filesystem::path> exclude;  // subtrees skipped entirely (e.g. the bench's own work directory)
};

struct CensusFile {
  std::uint64_t size = 0;
  std::int64_t mtime_ns = 0;  // since the file-clock epoch; only compared with other snapshots of the same host
};

struct RootSnapshot {
  std::string label;
  std::string path;
  bool exists = false;
  bool truncated = false;
  std::string error;                        // directory that could not be read
  std::map<std::string, CensusFile> files;  // path relative to the root, '/'-separated
};

struct CensusSnapshot {
  std::int64_t taken_ns = 0;  // file-clock time of the snapshot
  std::vector<RootSnapshot> roots;
};

// The roots above for this host/user. `extra` are appended (staging roots etc.); `exclude` applies to every root.
std::vector<CensusRoot> default_census_roots(const std::vector<CensusRoot>& extra = {},
                                             const std::vector<std::filesystem::path>& exclude = {});
// Environment lookups are injectable for tests.
struct CensusEnvironment {
  std::map<std::string, std::string> vars;
  bool windows = false;
  static CensusEnvironment from_process();
};
std::vector<CensusRoot> census_roots_for(const CensusEnvironment& env);

CensusSnapshot take_census(const std::vector<CensusRoot>& roots);

struct CensusChange {
  std::string root, name;
  std::uint64_t size_before = 0, size_after = 0;  // before = 0 for new files
  bool in_window = true;  // new, or modified inside the run window
};

struct CensusDiff {
  std::vector<CensusChange> added, changed, removed;
  std::vector<std::string> truncated_roots;
  std::vector<std::string> absent_roots;  // absent in both snapshots
  std::uint64_t added_bytes = 0, changed_bytes_delta_abs = 0, removed_bytes = 0;
  // root label -> bytes added+changed in that root
  std::map<std::string, std::uint64_t> bytes_by_root;
};

// `window_start_ns` .. `window_end_ns` bound attribution (file-clock ns; 0 = unbounded) .
CensusDiff diff_census(const CensusSnapshot& before, const CensusSnapshot& after, std::int64_t window_start_ns = 0,
                       std::int64_t window_end_ns = 0);

nlohmann::json census_to_json(const CensusSnapshot& s);
Result<CensusSnapshot> census_from_json(const nlohmann::json& j);
// Lists are capped at `max_listed` names per kind; totals are always complete.
nlohmann::json census_diff_to_json(const CensusDiff& d, std::size_t max_listed = 200);

// metrics: <prefix>census.{added_files,added_bytes,changed_files,removed_files,...}, <prefix>census.diff (JSON), and a
// check that nothing new was left under a Node staging root.
void emit_census_diff(BenchmarkResult& r, const CensusDiff& d, const std::string& prefix = "");

}  // namespace clusterlm::bench
