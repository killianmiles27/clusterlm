#include "memory_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>

#include "clusterlm/common/clock.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace clusterlm::bench {

namespace {

// "Name:   12345 kB" -> bytes.
bool meminfo_kb(const std::string& text, const char* name, std::uint64_t& out) {
  std::istringstream in(text);
  std::string line;
  const std::string key = std::string(name) + ":";
  while (std::getline(in, line)) {
    if (line.rfind(key, 0) != 0) continue;
    std::istringstream ls(line.substr(key.size()));
    std::uint64_t kb = 0;
    if (!(ls >> kb)) return false;
    out = kb * 1024;
    return true;
  }
  return false;
}

}  // namespace

MemoryInfo parse_meminfo(const std::string& text) {
  MemoryInfo m;
  meminfo_kb(text, "MemTotal", m.total_physical);
  if (!meminfo_kb(text, "MemAvailable", m.available_physical)) {
    std::uint64_t free_b = 0, cached = 0;
    if (meminfo_kb(text, "MemFree", free_b)) {
      meminfo_kb(text, "Cached", cached);
      m.available_physical = free_b + cached;
    }
  }
  std::uint64_t limit = 0, committed = 0;
  if (meminfo_kb(text, "CommitLimit", limit)) {
    m.commit_limit = limit;
    if (meminfo_kb(text, "Committed_AS", committed)) m.commit_available = limit > committed ? limit - committed : 0;
  }
  return m;
}

MemoryInfo probe_memory() {
#if defined(_WIN32)
  MemoryInfo m;
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof ms;
  if (GlobalMemoryStatusEx(&ms)) {
    m.total_physical = ms.ullTotalPhys;
    m.available_physical = ms.ullAvailPhys;
    m.commit_limit = ms.ullTotalPageFile;
    m.commit_available = ms.ullAvailPageFile;
  }
  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  m.page_size = si.dwPageSize;
  return m;
#else
  std::ifstream f("/proc/meminfo");
  std::stringstream ss;
  ss << f.rdbuf();
  MemoryInfo m = parse_meminfo(ss.str());
  const long page = sysconf(_SC_PAGESIZE);
  if (page > 0) m.page_size = static_cast<std::uint64_t>(page);
  if (m.total_physical == 0) {
    const long pages = sysconf(_SC_PHYS_PAGES), avail = sysconf(_SC_AVPHYS_PAGES);
    if (pages > 0) m.total_physical = static_cast<std::uint64_t>(pages) * m.page_size;
    if (avail > 0) m.available_physical = static_cast<std::uint64_t>(avail) * m.page_size;
  }
  return m;
#endif
}

std::uint64_t process_peak_memory_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return pmc.PeakWorkingSetSize;
  return 0;
#else
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmHWM:", 0) != 0) continue;
    std::istringstream ls(line.substr(6));
    std::uint64_t kb = 0;
    if (ls >> kb) return kb * 1024;
  }
  return 0;
#endif
}

std::uint64_t allocation_ceiling(const MemoryInfo& info, const AllocProbeOptions& o) {
  std::uint64_t ceiling = std::min(o.max_bytes, kHardCapBytes);
  const std::uint64_t usable = info.available_physical > o.reserve_bytes ? info.available_physical - o.reserve_bytes : 0;
  return std::min(ceiling, usable);
}

Result<AllocProbeResult> probe_largest_allocation(const AllocProbeOptions& o) {
  if (o.step_bytes == 0) return make_error(ErrorCode::kInvalidArgument, "step must be positive");
  const MemoryInfo info = probe_memory();
  AllocProbeResult r;
  r.ceiling_bytes = allocation_ceiling(info, o);
  r.peak_before = process_peak_memory_bytes();
  bool failed = false;
  for (std::uint64_t size = o.step_bytes; size <= r.ceiling_bytes; size += o.step_bytes) {
    AllocStep step;
    step.bytes = size;
    void* p = std::malloc(static_cast<std::size_t>(size));
    if (p) {
      Stopwatch sw;
      if (o.touch_pages) {
        // One write per page forces the OS to back every page now (overcommit would otherwise hide failure).
        auto* c = static_cast<volatile unsigned char*>(p);
        for (std::uint64_t off = 0; off < size; off += info.page_size) c[off] = 1;
      }
      step.touch_seconds = sw.elapsed_ms() / 1000.0;
      step.ok = true;
      std::free(p);
    }
    r.steps.push_back(step);
    if (!step.ok) {
      failed = true;
      break;
    }
    r.largest_ok_bytes = size;
    if (step.touch_seconds > 0) r.touch_bytes_per_s = static_cast<double>(size) / step.touch_seconds;
  }
  r.stopped_by_policy = !failed;
  r.peak_after = process_peak_memory_bytes();
  return r;
}

Result<ReadBandwidth> measure_read_bandwidth_mt(std::uint64_t bytes, unsigned threads, unsigned reps) {
  threads = std::max(1u, threads);
  const std::size_t words = static_cast<std::size_t>(bytes / sizeof(std::uint64_t)) / 8 * 8;
  if (words < std::size_t{8} * threads) return make_error(ErrorCode::kInvalidArgument, "buffer too small for the thread count");
  std::unique_ptr<std::uint64_t[]> buf(new (std::nothrow) std::uint64_t[words]);
  if (!buf) return make_error(ErrorCode::kResourceExhausted, "cannot allocate the bandwidth buffer");
  std::iota(buf.get(), buf.get() + words, std::uint64_t{1});  // touches every page
  const std::size_t slice = words / threads / 8 * 8;

  auto pass = [&]() {
    std::vector<std::thread> pool;
    std::vector<std::uint64_t> sinks(threads, 0);
    for (unsigned t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        const std::uint64_t* p = buf.get() + t * slice;
        std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (std::size_t i = 0; i < slice; i += 8) {
          a0 += p[i];
          a1 += p[i + 1];
          a2 += p[i + 2];
          a3 += p[i + 3];
          a4 += p[i + 4];
          a5 += p[i + 5];
          a6 += p[i + 6];
          a7 += p[i + 7];
        }
        sinks[t] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
      });
    for (auto& th : pool) th.join();
    volatile std::uint64_t keep = std::accumulate(sinks.begin(), sinks.end(), std::uint64_t{0});
    (void)keep;
  };

  ReadBandwidth out;
  out.threads = threads;
  pass();  // warm-up
  for (unsigned i = 0; i < reps; ++i) {
    Stopwatch sw;
    pass();
    out.bytes_per_s.add(static_cast<double>(slice * threads * sizeof(std::uint64_t)) / (sw.elapsed_ms() / 1000.0));
  }
  return out;
}

}  // namespace clusterlm::bench
