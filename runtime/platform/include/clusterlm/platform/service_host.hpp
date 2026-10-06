#pragma once
// Service host: runs a ServiceApp under the Windows Service Control Manager, or in a console for development.
//
// SCM mode (Windows): StartServiceCtrlDispatcherW -> ServiceMain -> RegisterServiceCtrlHandlerExW, with
// START_PENDING/RUNNING/STOP_PENDING/STOPPED reporting, STOP / SHUTDOWN / PRESHUTDOWN handling and forwarding of
// SERVICE_CONTROL_POWEREVENT and SERVICE_CONTROL_SESSIONCHANGE. Console mode (any OS): Ctrl-C / SIGTERM stop it.
//
// Threading: on_stop/on_power_event/on_session_event arrive on the SCM control thread (SCM mode) and must not
// block for long; run() executes on the service main thread.
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/platform/platform_events.hpp"

namespace clusterlm::platform {

enum class StopReason : std::uint8_t { kStop, kShutdown, kPreshutdown, kConsoleInterrupt };

class ServiceApp {
 public:
  virtual ~ServiceApp() = default;
  // Before RUNNING is reported. A failure fails service start (START_PENDING -> STOPPED).
  virtual Status on_start() = 0;
  // Blocks until a stop is requested and cleanup is done; returns the process/service exit code.
  virtual int run() = 0;
  // Thread-safe; must make run() return promptly.
  virtual void on_stop(StopReason reason) = 0;
  virtual void on_power_event(const PowerEvent&) {}
  virtual void on_session_event(const SessionEvent&) {}
};

struct ServiceHostOptions {
  std::string service_name = "ClusterLMNode";
  std::chrono::milliseconds start_wait_hint{10000};
  std::chrono::milliseconds stop_wait_hint{15000};
};

// Windows: connects to the SCM (fails with kFailedPrecondition when started from a console). Elsewhere:
// kUnimplemented. Returns the app's exit code.
Result<int> run_as_service(ServiceApp& app, const ServiceHostOptions& options);
// Any OS. Calls on_start, then run() with SIGINT/SIGTERM (Ctrl-C on Windows) mapped to on_stop(kConsoleInterrupt).
Result<int> run_in_console(ServiceApp& app, const ServiceHostOptions& options);

// ---- Declarative installation data (ADR 0131), shared with the installer workstream -----------------------
enum class ServiceStartType : std::uint8_t { kAutoDelayed, kAuto, kDemand, kDisabled };
struct ServiceRecoveryAction {
  enum class Kind : std::uint8_t { kRestart, kNone } kind = Kind::kRestart;
  std::chrono::milliseconds delay{5000};
  bool operator==(const ServiceRecoveryAction&) const = default;
};
struct ServiceInstallSpec {
  std::string name;
  std::string display_name;
  std::string description;
  std::string account;  // "NT AUTHORITY\\LocalService"
  ServiceStartType start_type = ServiceStartType::kAutoDelayed;
  std::vector<ServiceRecoveryAction> recovery;  // first, second, subsequent failures
  std::chrono::seconds recovery_reset_period{86400};
  std::chrono::milliseconds preshutdown_timeout{15000};
  bool unrestricted_service_sid = true;  // NT SERVICE\<name> usable in ACLs
  std::vector<std::string> args;
};
ServiceInstallSpec node_service_install_spec();

#ifdef _WIN32
// Create/update the service (SCM ChangeServiceConfig2: delayed auto start, failure actions, preshutdown timeout,
// service SID type) / remove it. Needs elevation.
Status install_service(const ServiceInstallSpec& spec, const std::wstring& executable_path);
Status uninstall_service(const std::string& name);
#endif

}  // namespace clusterlm::platform
