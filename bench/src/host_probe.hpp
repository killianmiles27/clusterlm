#pragma once
// Host discovery for ClusterLM Bench. Detects capabilities from the running machine (CPUID plus OS-enabled
// register state, never a marketing name) and measures what can be measured here. Whatever is reported is a
// measurement of THIS host only.
#include <cstdint>
#include <string>
#include <vector>

namespace clusterlm::bench {

struct HostInfo {
  std::string os;
  std::string cpu_brand;
  std::string arch;
  std::vector<std::string> cpu_features;  // e.g. avx2, fma, f16c, avx512f, avx512bw, avx512vl, avx512vnni
  unsigned logical_cpus = 0;
  std::uint64_t ram_bytes = 0;
  std::uint64_t ram_available_bytes = 0;
  bool has_cuda_device = false;  // this build does not link CUDA; always false unless a backend reports one
};

HostInfo probe_host();
bool has_feature(const HostInfo& h, const std::string& f);

// Single-threaded sequential read bandwidth over `bytes` (bytes/s), best of `repeats`.
double measure_read_bandwidth(std::uint64_t bytes, int repeats);

}  // namespace clusterlm::bench
