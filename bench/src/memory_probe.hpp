#pragma once
// Host memory measurements: totals, commit limit, bounded allocation probing with page touching, peak
// tracking and multi-threaded read bandwidth.
//
// Safety policy. A probe never takes the machine below `reserve_bytes` of available physical memory and never
// exceeds `max_bytes` or the hard cap (kHardCapBytes), whichever is smallest. The allocation probe reports
// when it stopped because of policy rather than because an allocation failed.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "result.hpp"

namespace clusterlm::bench {

inline constexpr std::uint64_t kHardCapBytes = 64ull << 30;

struct MemoryInfo {
  std::uint64_t total_physical = 0;
  std::uint64_t available_physical = 0;
  std::optional<std::uint64_t> commit_limit;      // Windows ullTotalPageFile; Linux /proc/meminfo CommitLimit
  std::optional<std::uint64_t> commit_available;  // Windows ullAvailPageFile; Linux CommitLimit - Committed_AS
  std::uint64_t page_size = 4096;
};

MemoryInfo probe_memory();
// Parses the text of /proc/meminfo (exposed for tests). Fields absent from the text stay empty/zero.
MemoryInfo parse_meminfo(const std::string& text);

// Peak physical memory held by this process so far (Windows PeakWorkingSetSize, Linux VmHWM). 0 if unknown.
std::uint64_t process_peak_memory_bytes();

struct AllocProbeOptions {
  std::uint64_t max_bytes = 4ull << 30;
  std::uint64_t step_bytes = 256ull << 20;
  std::uint64_t reserve_bytes = 2ull << 30;  // physical memory that must stay available
  bool touch_pages = true;
};

struct AllocStep {
  std::uint64_t bytes = 0;
  bool ok = false;
  double touch_seconds = 0;
};

struct AllocProbeResult {
  std::uint64_t largest_ok_bytes = 0;
  std::uint64_t ceiling_bytes = 0;        // the policy limit actually applied
  bool stopped_by_policy = false;         // true: reached the ceiling without a failed allocation
  double touch_bytes_per_s = 0;           // first-touch (page fault + write) rate at the largest size
  std::uint64_t peak_before = 0, peak_after = 0;
  std::vector<AllocStep> steps;
};

// The ceiling a probe would use now: min(options.max_bytes, hard cap, available - reserve), 0 if none.
std::uint64_t allocation_ceiling(const MemoryInfo& info, const AllocProbeOptions& options);

// Largest single contiguous allocation (ascending steps) that succeeds AND can be fully touched.
Result<AllocProbeResult> probe_largest_allocation(const AllocProbeOptions& options);

struct ReadBandwidth {
  unsigned threads = 1;
  Distribution bytes_per_s;
};

// Read bandwidth over a `bytes` buffer split evenly across `threads` (each thread reads its own slice).
// Repetitions are `reps` measured passes after one warm-up pass.
Result<ReadBandwidth> measure_read_bandwidth_mt(std::uint64_t bytes, unsigned threads, unsigned reps);

}  // namespace clusterlm::bench
