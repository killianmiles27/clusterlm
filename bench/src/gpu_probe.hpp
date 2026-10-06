#pragma once
// GpuProbe: GPU and PCIe measurements for placement. The CUDA implementation is compiled only when
// `find_package(CUDAToolkit)` succeeds and CLUSTERLM_BENCH_CUDA=ON (see bench/CMakeLists.txt); otherwise
// make_gpu_probe() returns a probe that reports exactly why no measurement is possible. Everything above the
// probe (the gpu command, calibrate, profile mapping) is plain C++ and is tested with a fake probe.
//
// Backend-specific kernels (Strata registers later) plug in through GpuKernelRegistry: a case supplies a
// launch function enqueued on the probe's stream; the probe times it with device events.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/platform/adapters.hpp"
#include "result.hpp"

namespace clusterlm::bench {

struct GpuDeviceInfo {
  int index = 0;
  std::string name;
  std::uint64_t total_bytes = 0;
  std::uint64_t free_bytes = 0;
  int compute_major = 0, compute_minor = 0;
  int multiprocessors = 0;
  int driver_version = 0;   // CUDA driver API version, e.g. 12040
  int runtime_version = 0;  // CUDA runtime version
};

struct AllocOverheadPoint {
  std::uint64_t bytes = 0;
  Distribution malloc_us, free_us;
};

enum class HostMemoryKind : std::uint8_t { kPageable, kPinnedAlloc, kPinnedRegister };
const char* to_string(HostMemoryKind k);

struct TransferPoint {
  std::uint64_t bytes = 0;
  HostMemoryKind kind = HostMemoryKind::kPageable;
  Distribution h2d_bytes_per_s, d2h_bytes_per_s;
};

struct PinnedProbeResult {
  std::uint64_t largest_ok_bytes = 0;
  std::uint64_t cap_bytes = 0;
  bool stopped_by_cap = false;  // true: reached the cap without a failed pin
  std::vector<std::pair<std::uint64_t, bool>> steps;
};

enum class GemmPrecision : std::uint8_t { kF32, kF16 };
const char* to_string(GemmPrecision p);

struct GemmShape {
  std::string name;       // e.g. "expert_gate_up"
  std::uint32_t rows = 0; // weight rows (m)
  std::uint32_t cols = 0; // weight cols (k)
  std::uint32_t batch = 1;  // positions (n): 1 = GEMV
};

struct GemmPoint {
  GemmShape shape;
  GemmPrecision precision = GemmPrecision::kF32;
  Distribution ms;
  double weight_bytes_per_s = 0;  // weight bytes / median time
  double tflops = 0;
};

// A backend-specific kernel to be timed. `launch` enqueues work on `stream` (an opaque cudaStream_t) and
// returns without synchronizing.
struct GpuKernelCase {
  std::string name;
  std::function<Status()> setup;
  std::function<Status(void* stream)> launch;
  std::function<void()> teardown;
};

class GpuKernelRegistry {
 public:
  static GpuKernelRegistry& instance();
  void register_case(GpuKernelCase c);
  const std::vector<GpuKernelCase>& cases() const { return cases_; }
  void clear() { cases_.clear(); }

 private:
  std::vector<GpuKernelCase> cases_;
};

struct KernelTimePoint {
  std::string name;
  Distribution ms;
};

class GpuProbe {
 public:
  virtual ~GpuProbe() = default;
  // False when this build or host cannot measure; reason says why (no CUDA build, no device, driver error).
  virtual bool available() const = 0;
  virtual std::string unavailable_reason() const = 0;
  virtual Result<std::vector<GpuDeviceInfo>> enumerate() = 0;
  virtual Result<std::vector<AllocOverheadPoint>> allocation_overhead(int device, const std::vector<std::uint64_t>& sizes,
                                                                      unsigned reps) = 0;
  // H2D and D2H bandwidth for every (kind, ring size) pair.
  virtual Result<std::vector<TransferPoint>> transfers(int device, const std::vector<std::uint64_t>& ring_bytes,
                                                       const std::vector<HostMemoryKind>& kinds, unsigned reps) = 0;
  // Pins host memory in `step` increments until `cap` or failure, then releases everything.
  virtual Result<PinnedProbeResult> pinned_limit(int device, std::uint64_t cap_bytes, std::uint64_t step_bytes) = 0;
  virtual Result<std::vector<GemmPoint>> gemm(int device, const std::vector<GemmShape>& shapes,
                                              const std::vector<GemmPrecision>& precisions, unsigned reps) = 0;
  virtual Result<std::vector<KernelTimePoint>> time_kernels(int device, const std::vector<GpuKernelCase>& cases,
                                                            unsigned reps) = 0;
};

// CUDA-backed probe when compiled in, otherwise an always-unavailable probe.
std::unique_ptr<GpuProbe> make_gpu_probe();
bool gpu_probe_built_with_cuda();
// A probe whose every call fails with kHardwareUnavailable(reason).
std::unique_ptr<GpuProbe> make_unavailable_gpu_probe(std::string reason);

// Real DXGI budget probe on Windows, nullptr elsewhere.
std::unique_ptr<platform::GpuBudgetProbe> make_host_gpu_budget_probe();

// Model-relevant shapes: Flash-Next expert (2560x640, 640x2560), square hidden (2560x2560), attention-like
// projections (q: 2560x6144, kv: 2560x512, o: 6144x2560), at positions 1 (GEMV), 2, 4 and 512 (prefill chunk).
std::vector<GemmShape> model_gemm_shapes(bool quick);

struct GpuBenchOptions {
  std::vector<std::uint64_t> ring_bytes = {64ull << 20, 256ull << 20, 1ull << 30};
  std::vector<std::uint64_t> alloc_sizes = {1ull << 20, 16ull << 20, 256ull << 20, 1ull << 30};
  std::uint64_t pinned_cap_bytes = 4ull << 30;
  std::uint64_t pinned_step_bytes = 256ull << 20;
  unsigned reps = 5;
  bool quick = false;
  bool include_registered_kernels = true;
};

struct GpuDeviceReport {
  GpuDeviceInfo info;
  std::vector<AllocOverheadPoint> alloc;
  std::vector<TransferPoint> transfers;
  PinnedProbeResult pinned;
  std::vector<GemmPoint> gemm;
  std::vector<KernelTimePoint> kernels;
  std::vector<std::string> notes;  // sub-measurements that failed (never a silent omission)
};

struct GpuBenchReport {
  bool available = false;
  std::string unavailable_reason;
  std::vector<GpuDeviceReport> devices;
};

GpuBenchReport run_gpu_bench(GpuProbe& probe, const GpuBenchOptions& options);

void emit_gpu_metrics(BenchmarkResult& r, const GpuBenchReport& report);

}  // namespace clusterlm::bench
