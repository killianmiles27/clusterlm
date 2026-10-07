#pragma once
// OS-level observations that surround a bench run: per-process memory (RSS and commit charge), NIC byte counters,
// the power plan / CPU governor, and a sampler that records the former as series. Everything here READS operating
// system counters; nothing is simulated. A source the OS cannot provide is reported as unavailable with a reason,
// never as zero.
//
//   Linux:   /proc/<pid>/status, /proc/net/dev, /proc/net/route, /sys/devices/system/cpu/cpu*/cpufreq
//   Windows: GetProcessMemoryInfo (PROCESS_MEMORY_COUNTERS_EX), GetIfTable2 / GetBestInterfaceEx,
//            PowerGetActiveScheme + PowerReadFriendlyName
//
// The text parsers are exposed so tests can feed them captured text on any host.
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "result.hpp"

namespace clusterlm::bench {

// ---- process memory ---------------------------------------------------------------------------------------

struct ProcessMemory {
  std::uint64_t rss_bytes = 0;     // resident set (Windows WorkingSetSize, Linux VmRSS)
  // Commit charge: Windows PrivateUsage. Linux has no per-process commit figure; VmData (private writable
  // virtual memory: heap, anonymous mappings, stack) is the closest equivalent and is reported as such.
  std::uint64_t commit_bytes = 0;
};

// Parses the text of /proc/<pid>/status. kDataLoss when VmRSS is absent (kernel threads, zombies).
Result<ProcessMemory> parse_proc_status_memory(const std::string& text);
// pid <= 0 samples this process.
Result<ProcessMemory> sample_process_memory(std::int64_t pid);

// ---- NIC byte counters ------------------------------------------------------------------------------------

struct NicCounters {
  std::string name;
  std::uint64_t rx_bytes = 0, tx_bytes = 0;
  std::uint64_t rx_packets = 0, tx_packets = 0;
  std::uint64_t rx_errors = 0, tx_errors = 0;
  std::uint64_t rx_dropped = 0, tx_dropped = 0;
};

// Parses /proc/net/dev. Skips the two header lines.
Result<std::vector<NicCounters>> parse_proc_net_dev(const std::string& text);
// Interface of the best IPv4 route to `ipv4` (dotted quad) from the text of /proc/net/route: longest matching
// prefix among routes that are up, lowest metric on a tie. 127.0.0.0/8 is "lo" (loopback is not in the table).
Result<std::string> parse_route_interface(const std::string& proc_net_route_text, const std::string& ipv4);

std::vector<std::string> list_nics();
Result<NicCounters> read_nic_counters(const std::string& name);
// The interface the OS would use to reach `host` (an IPv4 literal; names are not resolved here).
Result<std::string> default_nic_for(const std::string& host);

struct NicDelta {
  std::string name;
  std::uint64_t rx_bytes = 0, tx_bytes = 0, rx_packets = 0, tx_packets = 0, rx_errors = 0, tx_errors = 0, rx_dropped = 0,
                tx_dropped = 0;
};
// after - before, counter by counter; a counter that went backwards (reset, wrap) reads 0 and sets *wrapped.
NicDelta diff_nic(const NicCounters& before, const NicCounters& after, bool* wrapped = nullptr);

// Reads `nic` before a run and again after it, and reports the delta beside the boundary-payload counters.
class NicWindow {
 public:
  // Empty `nic`: the interface of the route to `first_peer_host`. A failure (no such interface, no counters on this
  // OS) is recorded as the unavailable reason rather than returned.
  NicWindow(std::string nic, const std::string& first_peer_host);
  void begin();
  void end();
  bool available() const { return unavailable_.empty(); }
  const std::string& nic() const { return nic_; }
  const std::string& unavailable_reason() const { return unavailable_; }
  const NicDelta& delta() const { return delta_; }
  // metrics: <prefix>nic.name, .rx_bytes, .tx_bytes, ... or <prefix>nic.unavailable
  void emit(BenchmarkResult& r, const std::string& prefix = "") const;

 private:
  std::string nic_, unavailable_;
  std::optional<NicCounters> before_;
  NicDelta delta_;
  bool wrapped_ = false;
};

// ---- power plan / governor --------------------------------------------------------------------------------

struct PowerEnvironment {
  std::optional<std::string> plan_guid;  // Windows: PowerGetActiveScheme
  std::optional<std::string> plan_name;  // Windows: PowerReadFriendlyName
  std::optional<std::string> governor;   // Linux cpufreq: one value, or "mixed:a,b"
  std::string unavailable;               // why a field is missing ("" when everything this OS offers was read)
  std::string summary() const;           // "plan 'Balanced' {guid}" / "governor 'performance'" / "unavailable: ..."
};
PowerEnvironment probe_power_environment();
// Reduces the per-CPU governor strings to one value (exposed for tests).
std::string summarize_governors(const std::vector<std::string>& per_cpu);
// metrics: power.plan_guid, power.plan_name, power.governor, power.environment_unavailable
void emit_power_environment(BenchmarkResult& r, const PowerEnvironment& p);

// ---- resource sampler -------------------------------------------------------------------------------------

class NvmlTelemetry;  // nvml_probe.hpp

// Records process memory (and, when NVML is present, VRAM used) at points the caller chooses: once per cycle or
// per round. The series go into the result as arrays (decimated beyond kMaxSeriesPoints) with a distribution and
// the first-to-last growth beside each.
class ResourceSampler {
 public:
  static constexpr std::size_t kMaxSeriesPoints = 2000;

  // `pid` is called at every sample (a restarted Node has a new pid); <= 0 means this process.
  void add_process(const std::string& label, std::function<std::int64_t()> pid);
  void set_gpu(NvmlTelemetry* nvml) { nvml_ = nvml; }
  // Always records. `phase` is a short tag ("cycle", "round", "generation"); counts only, never content.
  void sample(const std::string& phase);
  // Records only when at least `min_interval_ms` passed since the last recorded sample.
  void sample_throttled(const std::string& phase, double min_interval_ms);
  std::size_t samples() const;
  void emit(BenchmarkResult& r, const std::string& prefix = "resources.") const;

  struct Point {
    double t_s = 0;
    std::string phase;
    std::vector<std::optional<ProcessMemory>> memory;     // per process, nullopt when the read failed
    std::vector<std::optional<std::uint64_t>> vram_used;  // per GPU
  };
  // For tests.
  std::vector<Point> points() const;

 private:
  struct Proc {
    std::string label;
    std::function<std::int64_t()> pid;
    std::string last_error;
  };
  void record_locked(const std::string& phase);

  mutable std::mutex mu_;
  std::vector<Proc> procs_;
  NvmlTelemetry* nvml_ = nullptr;
  std::vector<Point> points_;
  std::vector<std::string> gpu_names_;
  double last_t_ms_ = -1e18;
  std::uint64_t start_ns_ = 0;
  std::string gpu_error_;
};

}  // namespace clusterlm::bench
