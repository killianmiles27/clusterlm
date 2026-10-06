#pragma once
// CPU measurements for placement: routed-expert throughput by representation, q scaling with expert-union
// selection patterns, thread scaling and sustained (thermal) behaviour. Works on any ExpertKernelProvider.
//
// Workload model. One "round" is the routed-expert work of one layer for a verification window of q
// positions: each position selects `active` experts out of the bank; the round executes the UNION of the
// selected experts once, each over the positions that selected it. A round's experts are spread over the
// worker threads (separate experts per thread, caller thread included); the round ends when all are done.
// Effective bytes/s = (stored bytes of the union's experts) / round wall time, which is the quantity
// HardwareProfile.cpu.expert_bytes_per_s documents (dequant + GEMV included).
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "clusterlm/platform/adapters.hpp"
#include "expert_kernels.hpp"
#include "result.hpp"

namespace clusterlm::bench {

// How the q positions of a window pick experts.
enum class SelectionPattern : std::uint8_t {
  kSame,     // every position selects the same experts: union = active (the q_scaling basis)
  kOverlap,  // each position reuses about half of the previous position's experts
  kDisjoint  // independent draws: union approaches q * active (bank size permitting)
};
const char* to_string(SelectionPattern p);

struct CpuBenchOptions {
  std::string provider = "reference";
  std::vector<std::string> representations;  // empty = every representation the provider reports
  ExpertShape shape;
  std::uint32_t active = 10;                 // routed experts per position per layer
  std::vector<std::uint32_t> qs = {1, 2, 4};
  std::vector<unsigned> threads;             // empty = sweep 1..max
  unsigned max_threads = 0;                  // 0 = logical CPUs
  unsigned reps = 5;                         // measured repetitions per configuration (>= 5 for qualification runs)
  unsigned warmup_reps = 1;
  double target_rep_seconds = 0.5;           // each repetition runs enough rounds to last about this long
  std::uint64_t bank_bytes = 1ull << 30;     // expert working set; must exceed the last-level cache to be meaningful
  std::uint64_t seed = 0x434C4D42;
  bool quick = false;
};

// A named expert-bank working set small enough for smoke tests: hidden 256, ff 64.
CpuBenchOptions quick_cpu_options();

struct QPoint {
  std::uint32_t q = 1;
  SelectionPattern pattern = SelectionPattern::kSame;
  double union_mean = 0;  // mean experts executed per round
  Distribution round_ms;
  Distribution bytes_per_s;
};

struct RepresentationReport {
  std::string representation;
  std::uint64_t bytes_per_expert = 0;
  std::size_t bank_experts = 0;
  std::map<unsigned, Distribution> thread_bytes_per_s;  // q=1, kSame, by thread count
  unsigned best_threads = 1;
  double best_bytes_per_s = 0;      // median at best_threads
  double single_thread_bytes_per_s = 0;
  unsigned usable_threads = 1;      // smallest count reaching >= 95% of the best, minus a service-responsiveness reserve
  std::vector<QPoint> q_points;
  std::map<std::uint32_t, std::string> kernel_paths;  // q -> kernel path the provider reports (empty: single path)
  double q_scaling = 0;             // mean over q>1 of (t_q/t_1 - 1)/(q-1) on the kSame pattern
  bool q_scaling_valid = false;
  double dequant_fraction = 0;      // of expert execution time at best_threads, q=1
  double dequant_bytes_per_s = 0;   // stored bytes/s through dequantization alone (0 when none)
  double gemv_bytes_per_s = 0;      // stored bytes/s through the GEMV part alone
};

struct CpuBenchReport {
  std::string provider_id, provider_description, isa;
  ExpertShape shape;
  std::uint32_t active = 0;
  unsigned max_threads = 0;
  std::vector<RepresentationReport> representations;
};

Result<CpuBenchReport> run_cpu_bench(const CpuBenchOptions& options, const std::function<void(const std::string&)>& progress = {});

// One sample of a sustained run.
struct SustainedSample {
  double t_s = 0;
  double bytes_per_s = 0;
  std::optional<bool> on_ac_power;  // Windows power adapter; nullopt where unavailable
};

struct SustainedReport {
  std::string representation;
  unsigned threads = 1;
  double minutes = 0;
  double sample_interval_s = 0;
  std::vector<SustainedSample> samples;
  double initial_bytes_per_s = 0;    // median of the first quarter of samples (at most the first 5)
  double final_bytes_per_s = 0;      // median of the last quarter
  double sustained_factor = 1;       // final/initial, capped to 1 (a speed-up is noise, not headroom)
  double slope_pct_per_min = 0;      // least-squares trend relative to the initial rate
  double time_to_equilibrium_s = 0;  // first time the rate stays within 3% of the final rate; 0 if never reliably
  bool power_source_changed = false;
};

// Runs the q=1 same-pattern workload on `threads` for `minutes`, sampling every `sample_interval_s`.
// `power` may be null. Stops early when `should_stop` returns true.
Result<SustainedReport> run_cpu_sustained(const CpuBenchOptions& options, const std::string& representation, unsigned threads,
                                          double minutes, double sample_interval_s, platform::PowerMonitor* power,
                                          const std::function<bool()>& should_stop = {});

// Summarises samples into the SustainedReport fields (exposed for tests with synthetic series).
void summarize_sustained(SustainedReport& r);

// The thread counts swept on a host with `max_threads` logical CPUs: 1,2,3,4,6,8,12,16,... plus max_threads.
std::vector<unsigned> thread_sweep(unsigned max_threads);

// Real power monitor on Windows, nullptr elsewhere.
std::unique_ptr<platform::PowerMonitor> make_host_power_monitor();

// Emit the report as result metrics (keys documented in docs/benchmark-methodology.md).
void emit_cpu_metrics(BenchmarkResult& r, const CpuBenchReport& report);
void emit_sustained_metrics(BenchmarkResult& r, const SustainedReport& s);

}  // namespace clusterlm::bench
