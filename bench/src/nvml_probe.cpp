#include "nvml_probe.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include "clusterlm/common/clock.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace clusterlm::bench {

namespace {

constexpr int kNvmlSuccess = 0;
constexpr int kClockSm = 1;
constexpr int kClockMem = 2;

// NVML clocksThrottleReason bits (nvml.h: nvmlClocksThrottleReason*).
struct ThrottleBit {
  std::uint64_t bit;
  const char* name;
  bool slowdown;
};
constexpr ThrottleBit kThrottleBits[] = {
    {0x1, "gpu_idle", false},
    {0x2, "applications_clocks_setting", false},
    {0x4, "sw_power_cap", true},
    {0x8, "hw_slowdown", true},
    {0x10, "sync_boost", false},
    {0x20, "sw_thermal_slowdown", true},
    {0x40, "hw_thermal_slowdown", true},
    {0x80, "hw_power_brake_slowdown", true},
    {0x100, "display_clock_setting", false},
};

#if defined(_WIN32)
using LibHandle = HMODULE;
LibHandle open_library(const std::string& path) { return LoadLibraryA(path.c_str()); }
void* find_symbol(LibHandle h, const char* name) {
  FARPROC p = GetProcAddress(h, name);
  static_assert(sizeof(void*) == sizeof p);
  return std::bit_cast<void*>(p);  // FARPROC -> void* without a function-pointer cast diagnostic
}
void close_library(LibHandle h) { FreeLibrary(h); }
std::string library_error() { return "error " + std::to_string(GetLastError()); }
std::vector<std::string> default_library_paths() {
  std::vector<std::string> out = {"nvml.dll"};
  if (const char* pf = std::getenv("ProgramW6432")) out.push_back(std::string(pf) + "\\NVIDIA Corporation\\NVSMI\\nvml.dll");
  return out;
}
#else
using LibHandle = void*;
LibHandle open_library(const std::string& path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }
void* find_symbol(LibHandle h, const char* name) { return dlsym(h, name); }
void close_library(LibHandle h) { dlclose(h); }
std::string library_error() {
  const char* e = dlerror();
  return e ? e : "unknown error";
}
std::vector<std::string> default_library_paths() { return {"libnvidia-ml.so.1"}; }
#endif

template <typename F>
F symbol(LibHandle h, const char* name) {
  void* p = find_symbol(h, name);
  static_assert(sizeof(F) == sizeof p);
  return std::bit_cast<F>(p);  // object pointer -> function pointer without a reinterpret_cast diagnostic
}

std::string error_text(const NvmlApi& api, int code) {
  if (api.error_string) {
    std::string s = api.error_string(code);
    if (!s.empty()) return s;
  }
  return "nvmlReturn " + std::to_string(code);
}

}  // namespace

Result<NvmlApi> load_nvml_api(const std::string& library_path) {
  std::vector<std::string> candidates;
  if (!library_path.empty()) {
    candidates.push_back(library_path);
  } else if (const char* env = std::getenv("CLUSTERLM_NVML_LIBRARY"); env != nullptr && *env != '\0') {
    candidates.push_back(env);
  } else {
    candidates = default_library_paths();
  }
  LibHandle lib{};
  std::string tried;
  for (const auto& c : candidates) {
    lib = open_library(c);
    if (lib) break;
    tried += (tried.empty() ? "" : "; ") + c + ": " + library_error();
  }
  if (!lib) return make_error(ErrorCode::kHardwareUnavailable, "NVML library not found (" + tried + ")");

  using InitFn = int (*)();
  using CountFn = int (*)(unsigned*);
  using ByIndexFn = int (*)(unsigned, void**);
  using NameFn = int (*)(void*, char*, unsigned);
  using ClockFn = int (*)(void*, int, unsigned*);
  using UIntFn = int (*)(void*, unsigned*);
  using TempFn = int (*)(void*, int, unsigned*);
  using MemFn = int (*)(void*, NvmlMemory*);
  using ThrottleFn = int (*)(void*, unsigned long long*);
  using DriverFn = int (*)(char*, unsigned);
  using ErrFn = const char* (*)(int);

  auto init = symbol<InitFn>(lib, "nvmlInit_v2");
  auto shutdown = symbol<InitFn>(lib, "nvmlShutdown");
  auto count = symbol<CountFn>(lib, "nvmlDeviceGetCount_v2");
  auto by_index = symbol<ByIndexFn>(lib, "nvmlDeviceGetHandleByIndex_v2");
  if (!init || !shutdown || !count || !by_index) {
    close_library(lib);
    return make_error(ErrorCode::kHardwareUnavailable, "NVML library lacks the core entry points (nvmlInit_v2, nvmlDeviceGet*)");
  }
  auto name = symbol<NameFn>(lib, "nvmlDeviceGetName");
  auto clock = symbol<ClockFn>(lib, "nvmlDeviceGetClockInfo");
  auto power = symbol<UIntFn>(lib, "nvmlDeviceGetPowerUsage");
  auto temp = symbol<TempFn>(lib, "nvmlDeviceGetTemperature");
  auto mem = symbol<MemFn>(lib, "nvmlDeviceGetMemoryInfo");
  auto throttle = symbol<ThrottleFn>(lib, "nvmlDeviceGetCurrentClocksThrottleReasons");
  if (!throttle) throttle = symbol<ThrottleFn>(lib, "nvmlDeviceGetCurrentClocksEventReasons");
  auto driver = symbol<DriverFn>(lib, "nvmlSystemGetDriverVersion");
  auto err = symbol<ErrFn>(lib, "nvmlErrorString");

  NvmlApi api;
  api.init = init;
  api.shutdown = shutdown;
  api.device_count = count;
  api.device_by_index = by_index;
  if (name) api.device_name = name;
  if (clock) api.clock_mhz = clock;
  if (power) api.power_mw = power;
  if (temp) api.temperature_c = [temp](NvmlApi::Device d, unsigned* v) { return temp(d, 0, v); };  // NVML_TEMPERATURE_GPU
  if (mem) api.memory = mem;
  if (throttle) api.throttle_reasons = throttle;
  if (driver) api.driver_version = driver;
  if (err) api.error_string = [err](int code) { const char* s = err(code); return std::string(s ? s : ""); };
  api.unload = [lib] { close_library(lib); };
  return api;
}

std::vector<std::string> throttle_reason_names(std::uint64_t mask) {
  std::vector<std::string> out;
  std::uint64_t known = 0;
  for (const auto& b : kThrottleBits) {
    known |= b.bit;
    if (mask & b.bit) out.emplace_back(b.name);
  }
  if (const std::uint64_t rest = mask & ~known; rest != 0) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "0x%llx", static_cast<unsigned long long>(rest));
    out.emplace_back(buf);
  }
  return out;
}

bool throttle_is_slowdown(std::uint64_t mask) {
  for (const auto& b : kThrottleBits)
    if (b.slowdown && (mask & b.bit)) return true;
  return false;
}

// ---- NvmlTelemetry ----------------------------------------------------------------------------------------

Result<std::unique_ptr<NvmlTelemetry>> NvmlTelemetry::open() {
  CLM_ASSIGN_OR_RETURN(auto api, load_nvml_api());
  return open_with(std::move(api));
}

Result<std::unique_ptr<NvmlTelemetry>> NvmlTelemetry::open_with(NvmlApi api) {
  auto unload = [&api] {
    if (api.unload) api.unload();
  };
  if (!api.init || !api.device_count || !api.device_by_index) {
    unload();
    return make_error(ErrorCode::kHardwareUnavailable, "NVML function table is incomplete");
  }
  if (int rc = api.init(); rc != kNvmlSuccess) {
    const std::string why = error_text(api, rc);
    unload();
    return make_error(ErrorCode::kHardwareUnavailable, "nvmlInit failed: " + why);
  }
  std::unique_ptr<NvmlTelemetry> t(new NvmlTelemetry());
  t->api_ = std::move(api);
  unsigned n = 0;
  if (int rc = t->api_.device_count(&n); rc != kNvmlSuccess)
    return make_error(ErrorCode::kHardwareUnavailable, "nvmlDeviceGetCount failed: " + error_text(t->api_, rc));
  if (n == 0) return make_error(ErrorCode::kHardwareUnavailable, "NVML reports no NVIDIA devices");
  for (unsigned i = 0; i < n; ++i) {
    NvmlApi::Device d = nullptr;
    if (int rc = t->api_.device_by_index(i, &d); rc != kNvmlSuccess)
      return make_error(ErrorCode::kHardwareUnavailable,
                        "nvmlDeviceGetHandleByIndex(" + std::to_string(i) + ") failed: " + error_text(t->api_, rc));
    t->devices_.push_back(d);
    char buf[128] = {};
    if (t->api_.device_name && t->api_.device_name(d, buf, sizeof buf) == kNvmlSuccess) t->names_.emplace_back(buf);
    else t->names_.emplace_back("unknown");
  }
  char drv[96] = {};
  if (t->api_.driver_version && t->api_.driver_version(drv, sizeof drv) == kNvmlSuccess) t->driver_ = drv;
  t->start_ns_ = monotonic_ns();
  return t;
}

NvmlTelemetry::~NvmlTelemetry() {
  if (api_.shutdown) (void)api_.shutdown();
  if (api_.unload) api_.unload();
}

Result<std::vector<GpuTelemetrySample>> NvmlTelemetry::sample() {
  std::lock_guard lock(mu_);
  std::vector<GpuTelemetrySample> out;
  const double t_s = static_cast<double>(monotonic_ns() - start_ns_) * 1e-9;
  for (std::size_t i = 0; i < devices_.size(); ++i) {
    GpuTelemetrySample s;
    s.t_s = t_s;
    s.device = static_cast<unsigned>(i);
    auto* d = devices_[i];
    unsigned v = 0;
    bool any = false;
    std::string first_error;
    auto note = [&](int rc) {
      if (rc == kNvmlSuccess) {
        any = true;
        return true;
      }
      if (first_error.empty()) first_error = error_text(api_, rc);
      return false;
    };
    if (api_.clock_mhz && note(api_.clock_mhz(d, kClockSm, &v))) s.sm_clock_mhz = v;
    if (api_.clock_mhz && note(api_.clock_mhz(d, kClockMem, &v))) s.mem_clock_mhz = v;
    if (api_.power_mw && note(api_.power_mw(d, &v))) s.power_w = static_cast<double>(v) / 1000.0;
    if (api_.temperature_c && note(api_.temperature_c(d, &v))) s.temperature_c = v;
    if (NvmlMemory m; api_.memory && note(api_.memory(d, &m))) {
      s.mem_used_bytes = m.used;
      s.mem_total_bytes = m.total;
    }
    if (unsigned long long mask = 0; api_.throttle_reasons && note(api_.throttle_reasons(d, &mask))) s.throttle_reasons = mask;
    if (!any)
      return make_error(ErrorCode::kHardwareUnavailable,
                        "no NVML counter could be read for device " + std::to_string(i) +
                            (first_error.empty() ? "" : ": " + first_error));
    out.push_back(s);
  }
  return out;
}

// ---- TelemetryRecorder ------------------------------------------------------------------------------------

TelemetryRecorder::TelemetryRecorder(NvmlTelemetry& nvml, double interval_s)
    : nvml_(nvml), interval_s_(std::max(0.01, interval_s)) {}

TelemetryRecorder::~TelemetryRecorder() {
  if (thread_.joinable()) (void)stop();
}

void TelemetryRecorder::start() {
  stop_ = false;
  thread_ = std::thread([this] {
    auto next = std::chrono::steady_clock::now();
    while (!stop_.load()) {
      auto s = nvml_.sample();
      {
        std::lock_guard lock(mu_);
        if (s.is_ok()) samples_.insert(samples_.end(), s->begin(), s->end());
        else error_ = s.status().to_string();
      }
      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(interval_s_));
      while (!stop_.load() && std::chrono::steady_clock::now() < next)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });
}

std::vector<GpuTelemetrySample> TelemetryRecorder::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
  auto last = nvml_.sample();  // the end state of the run
  std::lock_guard lock(mu_);
  if (last.is_ok()) samples_.insert(samples_.end(), last->begin(), last->end());
  else error_ = last.status().to_string();
  return samples_;
}

// ---- result emission --------------------------------------------------------------------------------------

void emit_gpu_telemetry(BenchmarkResult& r, const NvmlTelemetry& nvml, const std::vector<GpuTelemetrySample>& samples,
                        const std::string& prefix) {
  if (!nvml.driver_version().empty()) r.metric(prefix + "driver_version", nvml.driver_version());
  for (std::size_t d = 0; d < nvml.device_names().size(); ++d) {
    const std::string k = prefix + "gpu." + std::to_string(d) + ".";
    r.metric(k + "name", nvml.device_names()[d]);
    Distribution sm, mem, pw, temp, used, t_s;
    std::uint64_t seen = 0, slow = 0, count = 0, total = 0;
    nlohmann::json sm_series = nlohmann::json::array(), pw_series = nlohmann::json::array(),
                   temp_series = nlohmann::json::array(), used_series = nlohmann::json::array(),
                   t_series = nlohmann::json::array();
    for (const auto& s : samples) {
      if (s.device != d) continue;
      ++count;
      t_series.push_back(s.t_s);
      if (s.sm_clock_mhz) sm.add(*s.sm_clock_mhz), sm_series.push_back(*s.sm_clock_mhz);
      if (s.mem_clock_mhz) mem.add(*s.mem_clock_mhz);
      if (s.power_w) pw.add(*s.power_w), pw_series.push_back(*s.power_w);
      if (s.temperature_c) temp.add(*s.temperature_c), temp_series.push_back(*s.temperature_c);
      if (s.mem_used_bytes) used.add(static_cast<double>(*s.mem_used_bytes)), used_series.push_back(*s.mem_used_bytes);
      if (s.mem_total_bytes) total = *s.mem_total_bytes;
      if (s.throttle_reasons) {
        seen |= *s.throttle_reasons;
        if (throttle_is_slowdown(*s.throttle_reasons)) ++slow;
      }
    }
    r.metric(k + "samples", count);
    r.metric(k + "sample_t_s.series", t_series);
    if (!sm.empty()) r.metric(k + "sm_clock_mhz", sm, "MHz"), r.metric(k + "sm_clock_mhz.series", sm_series);
    else r.metric(k + "sm_clock_mhz.unsupported", "unavailable: clock query not supported by this driver");
    if (!mem.empty()) r.metric(k + "mem_clock_mhz", mem, "MHz");
    if (!pw.empty()) r.metric(k + "power_w", pw, "W"), r.metric(k + "power_w.series", pw_series);
    if (!temp.empty()) r.metric(k + "temperature_c", temp, "C"), r.metric(k + "temperature_c.series", temp_series);
    if (!used.empty()) {
      r.metric(k + "mem_used_bytes", used, "bytes");
      r.metric(k + "mem_used_bytes.series", used_series);
      r.metric(k + "mem_total_bytes", total);
    }
    r.metric(k + "throttle_reasons_seen", nlohmann::json(throttle_reason_names(seen)));
    r.metric(k + "slowdown_sample_fraction", count > 0 ? static_cast<double>(slow) / static_cast<double>(count) : 0.0);
  }
}

void emit_gpu_telemetry_unavailable(BenchmarkResult& r, const std::string& reason, const std::string& prefix) {
  r.metric(prefix + "unavailable", "unavailable: " + reason);
}

}  // namespace clusterlm::bench
