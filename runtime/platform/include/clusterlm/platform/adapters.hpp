#pragma once
// Abstract Windows platform adapters. Surrounding policy logic (idle gating, power gating, worker
// containment, VRAM budgeting) depends only on these interfaces so it can be tested on Linux with the Mock*
// implementations in mock_adapters.hpp. Real implementations: src/windows/adapters_win.cpp (WIN32 only).
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

// ---- Local user activity ------------------------------------------------------------------------------
struct ActivitySample {
  std::uint32_t idle_seconds = 0;  // time since last local keyboard/mouse input
  bool session_locked = false;     // interactive session locked or on the secure desktop
};

class ActivityMonitor {
 public:
  virtual ~ActivityMonitor() = default;
  virtual Result<ActivitySample> sample() = 0;
};

// ---- Power --------------------------------------------------------------------------------------------
struct PowerSample {
  bool on_ac_power = true;
  bool battery_saver = false;                     // Windows power saver / battery saver active
  std::optional<std::uint8_t> battery_percent;    // nullopt: no battery or unknown
};

class PowerMonitor {
 public:
  virtual ~PowerMonitor() = default;
  virtual Result<PowerSample> sample() = 0;
};

// ---- Worker containment (Windows Job Object) ----------------------------------------------------------
struct JobLimits {
  std::uint64_t process_memory_limit_bytes = 0;  // 0 = no per-process commit limit
  bool kill_on_close = true;                     // closing the job kills every process in it
};

class ProcessJob {
 public:
  virtual ~ProcessJob() = default;
  virtual Status set_limits(const JobLimits& limits) = 0;
  virtual Status assign_process(std::uint32_t pid) = 0;
  virtual Result<std::uint32_t> active_process_count() = 0;
  virtual Status terminate_all(std::uint32_t exit_code) = 0;
  // Closes the job handle; with kill_on_close the whole worker tree dies. Idempotent.
  virtual Status close() = 0;
};

// ---- GPU memory budget (DXGI QueryVideoMemoryInfo) ----------------------------------------------------
struct GpuAdapterBudget {
  std::uint32_t adapter_index = 0;
  std::string name;
  std::uint64_t budget_bytes = 0;         // what the OS currently lets this process use (local segment)
  std::uint64_t current_usage_bytes = 0;  // this process' current usage
  std::uint64_t available_for_reservation_bytes = 0;
  std::uint64_t dedicated_video_memory_bytes = 0;
};

class GpuBudgetProbe {
 public:
  virtual ~GpuBudgetProbe() = default;
  virtual Result<std::vector<GpuAdapterBudget>> query() = 0;
};

#ifdef _WIN32
std::unique_ptr<ActivityMonitor> make_windows_activity_monitor();
std::unique_ptr<PowerMonitor> make_windows_power_monitor();
std::unique_ptr<ProcessJob> make_windows_process_job(const std::string& name = {});
std::unique_ptr<GpuBudgetProbe> make_windows_gpu_budget_probe();
#endif

}  // namespace clusterlm::platform
