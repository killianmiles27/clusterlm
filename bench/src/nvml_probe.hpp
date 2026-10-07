#pragma once
// NVML telemetry by dynamic loading (HQ-GPU-03, HQ-CPU-02): clocks, power, temperature, memory used and clock
// throttle reasons per NVIDIA GPU, sampled on an interval during sustained runs.
//
// NVML is loaded at run time (libnvidia-ml.so.1 on Linux, nvml.dll on Windows) through the small function table
// below. There is no link-time dependency and no CUDA toolkit is needed; the library ships with the driver. When
// it cannot be loaded the telemetry is "unavailable: <reason>" and a result says exactly that instead of zeros.
// Tests inject a fake function table (NvmlTelemetry::open_with) to exercise the available path without a GPU.
//
// Samples are numbers from the driver. Nothing is simulated and no model, token or activation data is involved.
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "result.hpp"

namespace clusterlm::bench {

struct NvmlMemory {
  std::uint64_t total = 0, free = 0, used = 0;
};

// The subset of the NVML C API used. Every function returns an nvmlReturn_t (0 = success). A null entry means the
// symbol was not exported by this driver; the metric that needs it is then reported as unsupported.
struct NvmlApi {
  using Device = void*;
  std::function<int()> init;
  std::function<int()> shutdown;
  std::function<int(unsigned*)> device_count;
  std::function<int(unsigned, Device*)> device_by_index;
  std::function<int(Device, char*, unsigned)> device_name;
  std::function<int(Device, int clock_type, unsigned*)> clock_mhz;   // 1 = SM, 2 = memory
  std::function<int(Device, unsigned*)> power_mw;
  std::function<int(Device, unsigned*)> temperature_c;               // GPU sensor
  std::function<int(Device, NvmlMemory*)> memory;
  std::function<int(Device, unsigned long long*)> throttle_reasons;
  std::function<int(char*, unsigned)> driver_version;
  std::function<std::string(int)> error_string;
  std::function<void()> unload;  // closes the library
};

// Loads the NVML library and resolves the function table. `library_path` empty: the platform default name
// (override with the CLUSTERLM_NVML_LIBRARY environment variable). kHardwareUnavailable with the reason otherwise.
Result<NvmlApi> load_nvml_api(const std::string& library_path = "");

struct GpuTelemetrySample {
  double t_s = 0;  // since the telemetry object was opened
  unsigned device = 0;
  std::optional<double> sm_clock_mhz, mem_clock_mhz, power_w, temperature_c;
  std::optional<std::uint64_t> mem_used_bytes, mem_total_bytes;
  std::optional<std::uint64_t> throttle_reasons;  // NVML clocksThrottleReason bit mask
};

// "gpu_idle", "sw_power_cap", "hw_slowdown", ... for each set bit (unknown bits as "0x...").
std::vector<std::string> throttle_reason_names(std::uint64_t mask);
// True when the mask holds a reason that lowers clocks involuntarily (power cap, thermal or hardware slowdown),
// as opposed to idle / application clock settings.
bool throttle_is_slowdown(std::uint64_t mask);

class NvmlTelemetry {
 public:
  // Dynamic load of the real library. kHardwareUnavailable when NVML or a device is absent.
  static Result<std::unique_ptr<NvmlTelemetry>> open();
  // Uses the given function table (tests).
  static Result<std::unique_ptr<NvmlTelemetry>> open_with(NvmlApi api);
  ~NvmlTelemetry();
  NvmlTelemetry(const NvmlTelemetry&) = delete;
  NvmlTelemetry& operator=(const NvmlTelemetry&) = delete;

  const std::string& driver_version() const { return driver_; }
  const std::vector<std::string>& device_names() const { return names_; }
  // One sample per device. A single failing counter becomes nullopt; failing every counter of a device is an error.
  Result<std::vector<GpuTelemetrySample>> sample();

 private:
  NvmlTelemetry() = default;
  NvmlApi api_;
  std::vector<NvmlApi::Device> devices_;
  std::vector<std::string> names_;
  std::string driver_;
  std::uint64_t start_ns_ = 0;
  std::mutex mu_;
};

// Samples a NvmlTelemetry on its own thread every `interval_s` between start() and stop().
class TelemetryRecorder {
 public:
  TelemetryRecorder(NvmlTelemetry& nvml, double interval_s);
  ~TelemetryRecorder();
  void start();
  std::vector<GpuTelemetrySample> stop();  // joins the thread; takes one final sample
  const std::string& last_error() const { return error_; }

 private:
  NvmlTelemetry& nvml_;
  double interval_s_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::mutex mu_;
  std::vector<GpuTelemetrySample> samples_;
  std::string error_;
};

// metrics per device d: gpu.<d>.{sm_clock_mhz,mem_clock_mhz,power_w,temperature_c,mem_used_bytes} as distributions
// (+ ".series" arrays), gpu.<d>.throttle_reasons_seen, gpu.<d>.slowdown_sample_fraction, gpu.<d>.name; gpu.driver.
void emit_gpu_telemetry(BenchmarkResult& r, const NvmlTelemetry& nvml, const std::vector<GpuTelemetrySample>& samples,
                        const std::string& prefix = "nvml.");
// gpu telemetry marker for a host without NVML: <prefix>unavailable = "unavailable: <reason>".
void emit_gpu_telemetry_unavailable(BenchmarkResult& r, const std::string& reason, const std::string& prefix = "nvml.");

}  // namespace clusterlm::bench
