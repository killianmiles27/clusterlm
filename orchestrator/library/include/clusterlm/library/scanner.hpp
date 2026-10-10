#pragma once
// Identify model artifacts on disk. Reads GGUF headers and tensor directories only (objects::open_gguf_model); tensor
// data is never read, so scanning a directory of 80 GB models takes milliseconds per file.
#include <filesystem>
#include <vector>

#include "clusterlm/library/model_record.hpp"
#include "clusterlm/objects/gguf.hpp"

namespace clusterlm::library {

struct ScanIssue {
  std::string file;     // file name only (never a full path)
  std::string message;  // why it was skipped
};

struct ScanResult {
  std::vector<ModelRecord> records;  // one per model (a split set is one record), ordered by directory order
  std::vector<ScanIssue> issues;     // unreadable / inconsistent files; never fatal for the scan
};

struct ScanOptions {
  bool recursive = false;
  std::size_t max_files = 4096;
  objects::GgufLimits limits;
  std::int64_t now_unix = 0;  // stamped into discovered_at_unix (injected for tests)
};

// Identifies the model made of `paths` (a single file, or any/all shards of a split set; shards are expanded and
// cross-validated). `name`/`family` default to general.name / general.basename.
Result<ModelRecord> identify_model(const std::vector<std::filesystem::path>& paths, const ScanOptions& opt = {});

// Scans `dir` for *.gguf. Split sets ("-00001-of-00003.gguf") collapse into one record; a set with a missing shard
// is reported as an issue, not recorded.
Result<ScanResult> scan_directory(const std::filesystem::path& dir, const ScanOptions& opt = {});

}  // namespace clusterlm::library
