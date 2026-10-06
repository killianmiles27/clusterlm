// clusterlm-node-helper: the per-user session helper of the ClusterLM Node.
//
// Runs in the interactive session (started at logon by the service or the machine Run key), shows no UI of its
// own, and tells the Node service two things over a local pipe: how long the user has been idle and whether the
// session is locked. On Windows it also listens for session lock/unlock and power notifications through a hidden
// window so those reach the service immediately instead of at the next poll.
//
// Development (any OS): `--stdin` reads commands instead of the real activity source:
//   activity | idle [SECONDS] | lock | unlock | pause [SECONDS] | resume | status | quit
// and the helper connects to the service's Unix socket in `--ipc-dir`.
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/platform/helper_client.hpp"
#include "clusterlm/platform/mock_adapters.hpp"
#include "clusterlm/platform/paths.hpp"
#include "clusterlm/platform/platform_events.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace clusterlm;

namespace {

// Sampled activity with the lock state overridden by WTS lock/unlock events when they have been seen (the
// polling heuristic can lag the event by up to one interval).
class LockAwareMonitor final : public platform::ActivityMonitor {
 public:
  explicit LockAwareMonitor(platform::ActivityMonitor& inner) : inner_(inner) {}
  Result<platform::ActivitySample> sample() override {
    auto s = inner_.sample();
    if (!s.is_ok()) return s;
    const int forced = forced_lock_.load();
    if (forced >= 0) s->session_locked = forced == 1;
    return s;
  }
  void set_locked(bool locked) { forced_lock_.store(locked ? 1 : 0); }

 private:
  platform::ActivityMonitor& inner_;
  std::atomic<int> forced_lock_{-1};
};

// Thread-safe settable activity source for --stdin mode.
class StdinMonitor final : public platform::ActivityMonitor {
 public:
  Result<platform::ActivitySample> sample() override {
    std::lock_guard lock(mu_);
    return current_;
  }
  void set(std::uint32_t idle, bool locked) {
    std::lock_guard lock(mu_);
    current_ = {idle, locked};
  }
  platform::ActivitySample get() {
    std::lock_guard lock(mu_);
    return current_;
  }

 private:
  std::mutex mu_;
  platform::ActivitySample current_;
};

std::atomic<bool> g_quit{false};
std::mutex g_wake_mu;
std::condition_variable g_wake;
void wake_up() {
  std::lock_guard lock(g_wake_mu);
  g_wake.notify_all();
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  log::set_component("node-helper");

  platform::HelperClient::Options opts;
  opts.session_id = static_cast<std::uint32_t>(args.integer("session-id", 0));
  auto paths = platform::default_paths();
  opts.endpoint.socket_dir = args.get("ipc-dir", paths.is_ok() ? paths->ipc_dir.string() : std::string());
  if (args.has("expect-service-user")) opts.client.expected_server_user_ids = args.all("expect-service-user");
  const auto interval = std::chrono::milliseconds(args.integer("interval-ms", 1000));

  std::unique_ptr<platform::ActivityMonitor> real;
  platform::PowerEventHub power_events;
  platform::SessionEventHub session_events;
  std::unique_ptr<platform::EventWindow> window;
  StdinMonitor stdin_monitor;
  std::unique_ptr<LockAwareMonitor> lock_aware;
  platform::ActivityMonitor* source = &stdin_monitor;

#ifdef _WIN32
  // One helper per session: a second instance exits quietly (the service and the Run key may both try to start it).
  HANDLE single = ::CreateMutexW(nullptr, TRUE, L"Local\\ClusterLM.NodeHelper");
  if (single != nullptr && ::GetLastError() == ERROR_ALREADY_EXISTS) return 0;
  if (!args.has("stdin")) {
    DWORD session = 0;
    if (::ProcessIdToSessionId(::GetCurrentProcessId(), &session)) opts.session_id = session;
    real = platform::make_windows_activity_monitor();
    lock_aware = std::make_unique<LockAwareMonitor>(*real);
    source = lock_aware.get();
    window = platform::make_windows_event_window(power_events, session_events);
    session_events.subscribe([&](const platform::SessionEvent& e) {
      if (e.session_id != opts.session_id) return;
      if (e.kind == platform::SessionEventKind::kLock) lock_aware->set_locked(true);
      if (e.kind == platform::SessionEventKind::kUnlock) lock_aware->set_locked(false);
      wake_up();  // report the change right now
    });
    power_events.subscribe([&](const platform::PowerEvent& e) {
      if (e.kind == platform::PowerEventKind::kResume) wake_up();
    });
    if (auto st = window->start(); !st.is_ok()) {
      std::fprintf(stderr, "event window: %s\n", st.to_string().c_str());
      return 1;
    }
  }
#else
  if (!args.has("stdin")) {
    std::fprintf(stderr, "real activity sampling is Windows-only; use --stdin\n");
    return 2;
  }
#endif
  (void)power_events;
  (void)session_events;
  (void)real;
  (void)lock_aware;
  (void)window;

  platform::HelperClient client(opts, *source);

  std::thread input;
  if (args.has("stdin")) {
    input = std::thread([&] {
      std::string line;
      while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        std::uint32_t n = 0;
        const bool has_n = static_cast<bool>(ss >> n);
        auto cur = stdin_monitor.get();
        if (cmd == "quit") break;
        if (cmd == "activity") stdin_monitor.set(0, cur.session_locked);
        if (cmd == "idle") stdin_monitor.set(has_n ? n : 3600, cur.session_locked);
        if (cmd == "lock") stdin_monitor.set(cur.idle_seconds, true);
        if (cmd == "unlock") stdin_monitor.set(cur.idle_seconds, false);
        if (cmd == "pause") (void)client.pause(std::chrono::seconds(has_n ? n : 0));
        if (cmd == "resume") (void)client.resume();
        if (cmd == "status") {
          auto s = client.status();
          if (s.is_ok())
            std::printf("CLUSTERLM_HELPER_STATUS state=%s storage_bytes=%llu fresh=%d\n",
                        std::string(ipc::to_string(s->state)).c_str(), static_cast<unsigned long long>(s->storage_bytes),
                        s->helper_reports_fresh ? 1 : 0);
          else
            std::printf("CLUSTERLM_HELPER_STATUS error=%s\n", s.status().to_string().c_str());
          std::fflush(stdout);
        }
        wake_up();
      }
      g_quit.store(true);
      wake_up();
    });
  }

  bool was_connected = false;
  while (!g_quit.load()) {
    const Status st = client.report_once();
    if (st.is_ok() != was_connected) {
      log::info(st.is_ok() ? "service_connected" : "service_unreachable", {{"status", st.is_ok() ? "ok" : st.to_string()}});
      was_connected = st.is_ok();
    }
    std::unique_lock lock(g_wake_mu);
    g_wake.wait_for(lock, interval);
  }
  if (input.joinable()) input.join();
#ifdef _WIN32
  if (window) window->stop();
#endif
  return 0;
}
