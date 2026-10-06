// Portable parts of the service host: console mode and the declarative install spec. The SCM integration is in
// windows/service_host_win.cpp.
#include "clusterlm/platform/service_host.hpp"

#include <atomic>
#include <csignal>
#include <thread>

namespace clusterlm::platform {

namespace {
std::atomic<bool> g_console_stop{false};
extern "C" void console_signal_handler(int) { g_console_stop.store(true); }
}  // namespace

Result<int> run_in_console(ServiceApp& app, const ServiceHostOptions&) {
  g_console_stop.store(false);
  auto prev_int = std::signal(SIGINT, console_signal_handler);
  auto prev_term = std::signal(SIGTERM, console_signal_handler);
  struct Restore {
    void (*i)(int);
    void (*t)(int);
    ~Restore() {
      std::signal(SIGINT, i);
      std::signal(SIGTERM, t);
    }
  } restore{prev_int, prev_term};

  CLM_RETURN_IF_ERROR(app.on_start());
  std::atomic<bool> done{false};
  // Signal handlers may only set a flag; this thread turns the flag into the on_stop call.
  std::thread watcher([&] {
    while (!done.load()) {
      if (g_console_stop.exchange(false)) {
        app.on_stop(StopReason::kConsoleInterrupt);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  const int code = app.run();
  done.store(true);
  watcher.join();
  return code;
}

ServiceInstallSpec node_service_install_spec() {
  ServiceInstallSpec s;
  s.name = "ClusterLMNode";
  s.display_name = "ClusterLM Node";
  s.description = "Offers this PC's idle compute to a paired ClusterLM Father. Releases everything the moment the PC is used.";
  s.account = "NT AUTHORITY\\LocalService";
  s.start_type = ServiceStartType::kAutoDelayed;
  s.recovery = {{ServiceRecoveryAction::Kind::kRestart, std::chrono::milliseconds(5000)},
                {ServiceRecoveryAction::Kind::kRestart, std::chrono::milliseconds(30000)},
                {ServiceRecoveryAction::Kind::kRestart, std::chrono::milliseconds(60000)}};
  s.recovery_reset_period = std::chrono::hours(24);
  s.preshutdown_timeout = std::chrono::milliseconds(15000);
  s.unrestricted_service_sid = true;
  return s;
}

#ifndef _WIN32
Result<int> run_as_service(ServiceApp&, const ServiceHostOptions&) {
  return make_error(ErrorCode::kUnimplemented, "the Service Control Manager exists only on Windows; use console mode");
}
#endif

}  // namespace clusterlm::platform
