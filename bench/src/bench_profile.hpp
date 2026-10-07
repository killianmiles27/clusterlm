#pragma once
// From measurements to placement inputs: HardwareProfile / NetworkProfile construction with provenance and the
// merge rules for updating a profile file in place.
//
// Provenance rules (docs/benchmark-methodology.md, ADR 0151):
//   * a field the bench measured is Measured with source "bench:<run-id>@<machine>";
//   * a field it did not measure is a Synthetic placeholder (or keeps whatever the base/existing file had);
//   * nothing is ever Qualified by the bench and nothing is promoted beyond Measured;
//   * merge: Synthetic is replaced by Measured; Measured is replaced only by a NEWER Measured run; Qualified is
//     never touched; a Synthetic incoming value never replaces anything.
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/placement/profile.hpp"
#include "cpu_bench.hpp"
#include "gpu_probe.hpp"
#include "host_probe.hpp"
#include "memory_probe.hpp"
#include "system_probe.hpp"
#include "transport_bench.hpp"

namespace clusterlm::bench {

// "bench:<run-id>@<machine>"
std::string bench_source(const std::string& run_id, const std::string& machine_id);
// Splits a bench source; nullopt when `source` was not produced by this tool.
struct ParsedSource {
  std::string run_id, machine_id;
};
std::optional<ParsedSource> parse_bench_source(const std::string& source);
// "run-<UTC yyyymmddThhmmssZ>-<4 hex>"; lexicographic order is chronological order.
std::string make_run_id();

// A sustained run shorter than this is recorded in the result but does not populate cpu.sustained_factor
// (HQ-CPU-02 specifies 30 minutes at thermal equilibrium).
inline constexpr double kMinSustainedMinutes = 30.0;

struct MemoryMeasurement {
  MemoryInfo info;
  std::optional<AllocProbeResult> alloc;
  std::vector<ReadBandwidth> bandwidth;  // by thread count
};

struct HardwareMeasurements {
  std::string machine_id;
  placement::DomainRole role = placement::DomainRole::kNode;
  std::string run_id;
  HostInfo host;
  std::optional<CpuBenchReport> cpu;
  std::optional<SustainedReport> sustained;
  std::optional<MemoryMeasurement> memory;
  std::optional<GpuBenchReport> gpu;
  // OS-reported usable VRAM for this process (DXGI budget on Windows) per CUDA device order, when known.
  std::vector<std::uint64_t> dxgi_budget_bytes;
  // Windows active power plan / Linux cpufreq governor in force while measuring (HQ-PROF-01); recorded in the profile note.
  std::optional<PowerEnvironment> power;
};

// A profile holding only what `m` measured (Measured) plus Synthetic placeholders for everything else.
placement::HardwareProfile build_measured_profile(const HardwareMeasurements& m);

// Dotted paths of the quantities in `p` that are Measured/Qualified (for reporting).
std::vector<std::string> measured_paths(placement::HardwareProfile& p);

struct MergeReport {
  std::vector<std::string> added, replaced, kept_newer, kept_qualified, kept_unordered;
};

// Merges `incoming` into `base` per the rules above. `base` is modified in place; identity, role and the
// informational fields (arch, features, gpu name) are taken from `incoming` when it measured them.
MergeReport merge_hardware_profile(placement::HardwareProfile& base, const placement::HardwareProfile& incoming);
MergeReport merge_network_profile(placement::NetworkProfile& base, const placement::NetworkProfile& incoming);

struct PeerLink {
  std::string peer_id;  // name used in the profile (e.g. "node-g14")
  LinkMeasure measure;
  bool real_link = false;  // false: loopback or impaired -> Synthetic quantities
};

// Link bandwidth is the SLOWER of the two directions at the largest message size; rtt_ms is the p50 of the
// smallest message; jitter_ms is its sample standard deviation. father_egress is the concurrent egress when two
// or more peers were loaded at once, else the single-link tx rate recorded as Synthetic (concurrency unmeasured).
placement::NetworkProfile build_network_profile(const std::string& machine_id, const std::string& run_id,
                                                const std::vector<PeerLink>& peers, const ConcurrentMeasure* concurrent);

}  // namespace clusterlm::bench
