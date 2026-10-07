// Windows Service Control Manager integration (ADR 0131). Compiled only on Windows.
#ifdef _WIN32

#include <windows.h>
#include <wtsapi32.h>

#include <atomic>
#include <string>

#include "../internal_errors.hpp"
#include "clusterlm/platform/service_host.hpp"
#include "win_events.hpp"

namespace clusterlm::platform {

namespace {

using detail::win_status;

// A process hosts exactly one service, and ServiceMain/HandlerEx carry no usable user pointer for the former,
// so the state is file-static.
struct HostState {
  ServiceApp* app = nullptr;
  ServiceHostOptions options;
  std::wstring name;
  SERVICE_STATUS_HANDLE handle = nullptr;
  std::atomic<DWORD> current_state{SERVICE_STOPPED};
  DWORD checkpoint = 0;
  int exit_code = 0;
  Status start_error;
} g;

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<std::size_t>(n), L'\0');
  ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}

void report(DWORD state, DWORD win32_exit = NO_ERROR, DWORD service_exit = 0, DWORD wait_hint_ms = 0) {
  SERVICE_STATUS st{};
  st.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
  st.dwCurrentState = state;
  if (state == SERVICE_RUNNING) {
    st.dwControlsAccepted = SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN | SERVICE_ACCEPT_PRESHUTDOWN |
                            SERVICE_ACCEPT_POWEREVENT | SERVICE_ACCEPT_SESSIONCHANGE;
  }
  if (win32_exit == ERROR_SERVICE_SPECIFIC_ERROR) st.dwServiceSpecificExitCode = service_exit;
  st.dwWin32ExitCode = win32_exit;
  st.dwWaitHint = wait_hint_ms;
  st.dwCheckPoint = (state == SERVICE_START_PENDING || state == SERVICE_STOP_PENDING) ? ++g.checkpoint : 0;
  g.current_state.store(state);
  if (g.handle != nullptr) ::SetServiceStatus(g.handle, &st);
}

DWORD WINAPI control_handler(DWORD control, DWORD event_type, LPVOID event_data, LPVOID) {
  switch (control) {
    case SERVICE_CONTROL_INTERROGATE: return NO_ERROR;
    case SERVICE_CONTROL_STOP:
    case SERVICE_CONTROL_SHUTDOWN:
    case SERVICE_CONTROL_PRESHUTDOWN: {
      const StopReason why = control == SERVICE_CONTROL_STOP       ? StopReason::kStop
                             : control == SERVICE_CONTROL_SHUTDOWN ? StopReason::kShutdown
                                                                   : StopReason::kPreshutdown;
      report(SERVICE_STOP_PENDING, NO_ERROR, 0, static_cast<DWORD>(g.options.stop_wait_hint.count()));
      g.app->on_stop(why);
      return NO_ERROR;
    }
    case SERVICE_CONTROL_POWEREVENT: {
      // PBT_APMSUSPEND must be answered quickly: the app revokes synchronously inside on_power_event.
      if (auto ev = detail::map_power_broadcast(event_type, event_data)) g.app->on_power_event(*ev);
      return NO_ERROR;
    }
    case SERVICE_CONTROL_SESSIONCHANGE: {
      const auto* n = static_cast<const WTSSESSION_NOTIFICATION*>(event_data);
      if (n != nullptr)
        if (auto ev = detail::map_session_change(event_type, n->dwSessionId)) g.app->on_session_event(*ev);
      return NO_ERROR;
    }
    default: return ERROR_CALL_NOT_IMPLEMENTED;
  }
}

void WINAPI service_main(DWORD, LPWSTR*) {
  g.handle = ::RegisterServiceCtrlHandlerExW(g.name.c_str(), control_handler, nullptr);
  if (g.handle == nullptr) return;  // nothing can be reported without a status handle
  report(SERVICE_START_PENDING, NO_ERROR, 0, static_cast<DWORD>(g.options.start_wait_hint.count()));
  if (Status st = g.app->on_start(); !st.is_ok()) {
    g.start_error = st;
    report(SERVICE_STOPPED, ERROR_SERVICE_SPECIFIC_ERROR, 1);
    return;
  }
  report(SERVICE_RUNNING);
  g.exit_code = g.app->run();
  if (g.exit_code != 0) {
    report(SERVICE_STOPPED, ERROR_SERVICE_SPECIFIC_ERROR, static_cast<DWORD>(g.exit_code));
  } else {
    report(SERVICE_STOPPED);
  }
}

}  // namespace

Result<int> run_as_service(ServiceApp& app, const ServiceHostOptions& options) {
  g.app = &app;
  g.options = options;
  g.name = widen(options.service_name);
  SERVICE_TABLE_ENTRYW table[] = {{g.name.data(), service_main}, {nullptr, nullptr}};
  if (!::StartServiceCtrlDispatcherW(table)) {
    const DWORD err = ::GetLastError();
    if (err == ERROR_FAILED_SERVICE_CONTROLLER_CONNECT)
      return make_error(ErrorCode::kFailedPrecondition, "not started by the Service Control Manager (use --console)");
    return win_status("StartServiceCtrlDispatcherW", err);
  }
  if (!g.start_error.is_ok()) return g.start_error;
  return g.exit_code;
}

// ---- Installation (ADR 0131) -------------------------------------------------------------------------------
namespace {
struct ScHandle {
  SC_HANDLE h = nullptr;
  ~ScHandle() {
    if (h != nullptr) ::CloseServiceHandle(h);
  }
};
}  // namespace

Status install_service(const ServiceInstallSpec& spec, const std::wstring& executable_path) {
  ScHandle scm{::OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CREATE_SERVICE)};
  if (scm.h == nullptr) return win_status("OpenSCManagerW");
  std::wstring cmd = L"\"" + executable_path + L"\"";
  for (const auto& a : spec.args) cmd += L" " + widen(a);
  const std::wstring name = widen(spec.name), display = widen(spec.display_name), account = widen(spec.account);
  const DWORD start = spec.start_type == ServiceStartType::kDemand     ? SERVICE_DEMAND_START
                      : spec.start_type == ServiceStartType::kDisabled ? SERVICE_DISABLED
                                                                        : SERVICE_AUTO_START;
  ScHandle svc{::CreateServiceW(scm.h, name.c_str(), display.c_str(), SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS,
                                start, SERVICE_ERROR_NORMAL, cmd.c_str(), nullptr, nullptr, nullptr, account.c_str(),
                                nullptr)};
  if (svc.h == nullptr) {
    if (::GetLastError() != ERROR_SERVICE_EXISTS) return win_status("CreateServiceW");
    svc.h = ::OpenServiceW(scm.h, name.c_str(), SERVICE_ALL_ACCESS);
    if (svc.h == nullptr) return win_status("OpenServiceW");
    if (!::ChangeServiceConfigW(svc.h, SERVICE_WIN32_OWN_PROCESS, start, SERVICE_ERROR_NORMAL, cmd.c_str(), nullptr,
                                nullptr, nullptr, account.c_str(), nullptr, display.c_str()))
      return win_status("ChangeServiceConfigW");
  }
  std::wstring description = widen(spec.description);
  SERVICE_DESCRIPTIONW desc{description.data()};
  if (!::ChangeServiceConfig2W(svc.h, SERVICE_CONFIG_DESCRIPTION, &desc)) return win_status("SERVICE_CONFIG_DESCRIPTION");

  SERVICE_DELAYED_AUTO_START_INFO delayed{spec.start_type == ServiceStartType::kAutoDelayed ? TRUE : FALSE};
  if (!::ChangeServiceConfig2W(svc.h, SERVICE_CONFIG_DELAYED_AUTO_START_INFO, &delayed))
    return win_status("SERVICE_CONFIG_DELAYED_AUTO_START_INFO");

  std::vector<SC_ACTION> actions;
  for (const auto& a : spec.recovery) {
    SC_ACTION act{};
    act.Type = a.kind == ServiceRecoveryAction::Kind::kRestart ? SC_ACTION_RESTART : SC_ACTION_NONE;
    act.Delay = static_cast<DWORD>(a.delay.count());
    actions.push_back(act);
  }
  SERVICE_FAILURE_ACTIONSW fa{};
  fa.dwResetPeriod = static_cast<DWORD>(spec.recovery_reset_period.count());
  fa.cActions = static_cast<DWORD>(actions.size());
  fa.lpsaActions = actions.empty() ? nullptr : actions.data();
  if (!::ChangeServiceConfig2W(svc.h, SERVICE_CONFIG_FAILURE_ACTIONS, &fa)) return win_status("SERVICE_CONFIG_FAILURE_ACTIONS");
  SERVICE_FAILURE_ACTIONS_FLAG flag{TRUE};  // also restart when the process exits with a nonzero code
  if (!::ChangeServiceConfig2W(svc.h, SERVICE_CONFIG_FAILURE_ACTIONS_FLAG, &flag))
    return win_status("SERVICE_CONFIG_FAILURE_ACTIONS_FLAG");

  SERVICE_PRESHUTDOWN_INFO pre{static_cast<DWORD>(spec.preshutdown_timeout.count())};
  if (!::ChangeServiceConfig2W(svc.h, SERVICE_CONFIG_PRESHUTDOWN_INFO, &pre)) return win_status("SERVICE_CONFIG_PRESHUTDOWN_INFO");

  SERVICE_SID_INFO sid{spec.unrestricted_service_sid ? static_cast<DWORD>(SERVICE_SID_TYPE_UNRESTRICTED) : static_cast<DWORD>(SERVICE_SID_TYPE_NONE)};
  if (!::ChangeServiceConfig2W(svc.h, SERVICE_CONFIG_SERVICE_SID_INFO, &sid)) return win_status("SERVICE_CONFIG_SERVICE_SID_INFO");
  return Status::ok();
}

Status set_service_start_type(const std::string& name, ServiceStartType type) {
  ScHandle scm{::OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)};
  if (scm.h == nullptr) return win_status("OpenSCManagerW");
  ScHandle svc{::OpenServiceW(scm.h, widen(name).c_str(), SERVICE_CHANGE_CONFIG)};
  if (svc.h == nullptr) return win_status("OpenServiceW");
  const DWORD start = type == ServiceStartType::kDemand     ? SERVICE_DEMAND_START
                      : type == ServiceStartType::kDisabled ? SERVICE_DISABLED
                                                            : SERVICE_AUTO_START;
  // Only the start type changes: binary path, account and everything else are left as installed.
  if (!::ChangeServiceConfigW(svc.h, SERVICE_NO_CHANGE, start, SERVICE_NO_CHANGE, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr))
    return win_status("ChangeServiceConfigW");
  SERVICE_DELAYED_AUTO_START_INFO delayed{type == ServiceStartType::kAutoDelayed ? TRUE : FALSE};
  if (start == SERVICE_AUTO_START && !::ChangeServiceConfig2W(svc.h, SERVICE_CONFIG_DELAYED_AUTO_START_INFO, &delayed))
    return win_status("SERVICE_CONFIG_DELAYED_AUTO_START_INFO");
  return Status::ok();
}

Status uninstall_service(const std::string& name) {
  ScHandle scm{::OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)};
  if (scm.h == nullptr) return win_status("OpenSCManagerW");
  ScHandle svc{::OpenServiceW(scm.h, widen(name).c_str(), SERVICE_STOP | DELETE | SERVICE_QUERY_STATUS)};
  if (svc.h == nullptr) {
    if (::GetLastError() == ERROR_SERVICE_DOES_NOT_EXIST) return Status::ok();
    return win_status("OpenServiceW");
  }
  SERVICE_STATUS st{};
  ::ControlService(svc.h, SERVICE_CONTROL_STOP, &st);  // best effort: it may not be running
  if (!::DeleteService(svc.h) && ::GetLastError() != ERROR_SERVICE_MARKED_FOR_DELETE) return win_status("DeleteService");
  return Status::ok();
}

}  // namespace clusterlm::platform

#endif  // _WIN32
