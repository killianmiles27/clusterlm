// Real Windows implementations of the platform adapters. Compiled only when WIN32 (see CMakeLists.txt); it
// cannot be built on the Linux CI, so it is deliberately conservative: every Win32/COM call is checked and
// mapped to a Status, no exceptions, no global state.
#ifdef _WIN32

#include <dxgi1_4.h>
#include <windows.h>
#include <wrl/client.h>

#include <memory>
#include <string>
#include <vector>

#include "clusterlm/platform/adapters.hpp"
#include "../internal_errors.hpp"

namespace clusterlm::platform {

namespace {

using detail::win_status;

std::string narrow(const wchar_t* w) {
  if (w == nullptr || *w == L'\0') return {};
  const int n = ::WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (n <= 1) return {};
  std::string out(static_cast<std::size_t>(n - 1), '\0');
  ::WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), n, nullptr, nullptr);
  return out;
}

// ---- Activity -----------------------------------------------------------------------------------------
// Idle time comes from GetLastInputInfo (session-wide input clock). Lock detection here is the conservative
// OpenInputDesktop heuristic: while the workstation is locked the input desktop is Winlogon and cannot be
// opened with DESKTOP_SWITCHDESKTOP. A service should additionally subscribe to WTS_SESSION_LOCK/UNLOCK
// (WTSRegisterSessionNotification) and feed that into policy; this sampler is the polling fallback.
class WindowsActivityMonitor final : public ActivityMonitor {
 public:
  Result<ActivitySample> sample() override {
    ActivitySample s;
    LASTINPUTINFO lii{};
    lii.cbSize = sizeof(lii);
    if (!::GetLastInputInfo(&lii)) return win_status("GetLastInputInfo");
    const DWORD now = ::GetTickCount();  // same 32-bit tick clock as dwTime; unsigned subtraction wraps correctly
    s.idle_seconds = static_cast<std::uint32_t>((now - lii.dwTime) / 1000u);
    HDESK desk = ::OpenInputDesktop(0, FALSE, DESKTOP_SWITCHDESKTOP);
    if (desk == nullptr) {
      s.session_locked = true;
    } else {
      ::CloseDesktop(desk);
    }
    return s;
  }
};

// ---- Power --------------------------------------------------------------------------------------------
class WindowsPowerMonitor final : public PowerMonitor {
 public:
  Result<PowerSample> sample() override {
    SYSTEM_POWER_STATUS sps{};
    if (!::GetSystemPowerStatus(&sps)) return win_status("GetSystemPowerStatus");
    PowerSample p;
    p.on_ac_power = sps.ACLineStatus != 0;  // 0 = offline (battery), 1 = online, 255 = unknown (treated as AC)
    if (sps.ACLineStatus == 0) p.on_ac_power = false;
    p.battery_saver = (sps.SystemStatusFlag & 1) != 0;  // battery saver on
    if (sps.BatteryLifePercent != 255) p.battery_percent = sps.BatteryLifePercent;
    return p;
  }
};

// ---- Job object ---------------------------------------------------------------------------------------
class WindowsProcessJob final : public ProcessJob {
 public:
  explicit WindowsProcessJob(const std::string& name) {
    std::wstring wname;
    if (!name.empty()) wname.assign(name.begin(), name.end());  // names are ASCII identifiers chosen by us
    job_ = ::CreateJobObjectW(nullptr, wname.empty() ? nullptr : wname.c_str());
    create_error_ = job_ == nullptr ? ::GetLastError() : 0;
  }
  ~WindowsProcessJob() override { (void)close(); }

  Status set_limits(const JobLimits& l) override {
    if (job_ == nullptr) return job_unavailable();
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    if (l.kill_on_close) info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (l.process_memory_limit_bytes != 0) {
      info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
      info.ProcessMemoryLimit = static_cast<SIZE_T>(l.process_memory_limit_bytes);
    }
    if (!::SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &info, sizeof(info)))
      return win_status("SetInformationJobObject");
    return Status::ok();
  }

  Status assign_process(std::uint32_t pid) override {
    if (job_ == nullptr) return job_unavailable();
    HANDLE proc = ::OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, pid);
    if (proc == nullptr) return win_status("OpenProcess");
    const BOOL ok = ::AssignProcessToJobObject(job_, proc);
    const DWORD err = ok ? 0 : ::GetLastError();
    ::CloseHandle(proc);
    if (!ok) return win_status("AssignProcessToJobObject", err);
    return Status::ok();
  }

  Result<std::uint32_t> active_process_count() override {
    if (job_ == nullptr) return job_unavailable();
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION acct{};
    if (!::QueryInformationJobObject(job_, JobObjectBasicAccountingInformation, &acct, sizeof(acct), nullptr))
      return win_status("QueryInformationJobObject");
    return static_cast<std::uint32_t>(acct.ActiveProcesses);
  }

  Status terminate_all(std::uint32_t exit_code) override {
    if (job_ == nullptr) return job_unavailable();
    if (!::TerminateJobObject(job_, exit_code)) return win_status("TerminateJobObject");
    return Status::ok();
  }

  Status close() override {
    if (job_ == nullptr) return Status::ok();
    const BOOL ok = ::CloseHandle(job_);  // KILL_ON_JOB_CLOSE fires when the last handle closes
    job_ = nullptr;
    return ok ? Status::ok() : win_status("CloseHandle(job)");
  }

 private:
  Status job_unavailable() const {
    return create_error_ != 0 ? win_status("CreateJobObjectW", create_error_)
                              : make_error(ErrorCode::kFailedPrecondition, "job closed");
  }
  HANDLE job_ = nullptr;
  DWORD create_error_ = 0;
};

// ---- GPU budget ---------------------------------------------------------------------------------------
class WindowsGpuBudgetProbe final : public GpuBudgetProbe {
 public:
  Result<std::vector<GpuAdapterBudget>> query() override {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    HRESULT hr = ::CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(factory.GetAddressOf()));
    if (FAILED(hr)) return make_error(ErrorCode::kHardwareUnavailable, "CreateDXGIFactory1 failed");
    std::vector<GpuAdapterBudget> out;
    for (UINT i = 0;; ++i) {
      Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
      hr = factory->EnumAdapters1(i, adapter1.GetAddressOf());
      if (hr == DXGI_ERROR_NOT_FOUND) break;
      if (FAILED(hr)) return make_error(ErrorCode::kInternal, "EnumAdapters1 failed");
      DXGI_ADAPTER_DESC1 desc{};
      if (FAILED(adapter1->GetDesc1(&desc))) continue;
      if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) continue;  // WARP / basic render driver
      Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
      if (FAILED(adapter1.As(&adapter3))) continue;  // pre-Windows 10: no QueryVideoMemoryInfo
      DXGI_QUERY_VIDEO_MEMORY_INFO mem{};
      if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mem))) continue;
      GpuAdapterBudget b;
      b.adapter_index = i;
      b.name = narrow(desc.Description);
      b.budget_bytes = mem.Budget;
      b.current_usage_bytes = mem.CurrentUsage;
      b.available_for_reservation_bytes = mem.AvailableForReservation;
      b.dedicated_video_memory_bytes = static_cast<std::uint64_t>(desc.DedicatedVideoMemory);
      out.push_back(std::move(b));
    }
    return out;
  }
};

}  // namespace

std::unique_ptr<ActivityMonitor> make_windows_activity_monitor() { return std::make_unique<WindowsActivityMonitor>(); }
std::unique_ptr<PowerMonitor> make_windows_power_monitor() { return std::make_unique<WindowsPowerMonitor>(); }
std::unique_ptr<ProcessJob> make_windows_process_job(const std::string& name) {
  return std::make_unique<WindowsProcessJob>(name);
}
std::unique_ptr<GpuBudgetProbe> make_windows_gpu_budget_probe() { return std::make_unique<WindowsGpuBudgetProbe>(); }

}  // namespace clusterlm::platform

#endif  // _WIN32
