// clusterlm-node-service: the ClusterLM Node service process.
//
// Windows product mode (no flags, started by the Service Control Manager): runs as a service under LocalService,
// supervises `clusterlm-node` in a Job Object, takes local activity from the per-session helper over a named pipe
// and reacts to power/session notifications. Console mode (`--console`, the default on other OSes) runs the same
// ServiceCore in the foreground for development; there activity is helper-fed over a Unix socket, or simulated
// with `--simulate-activity` (stdin lines: activity | idle | lock | unlock | suspend | resume | quit).
//
//   clusterlm-node-service --console --worker PATH --idle-seconds 300 --simulate-activity -- <clusterlm-node args>
//   clusterlm-node-service --install | --uninstall            (Windows, elevated; --uninstall also removes the firewall rules)
//   clusterlm-node-service --cleanup [--purge] [--staging DIR] (any OS: lease-store recovery, then delete the staging root;
//                                                              --purge also deletes the Node identity, logs and root)
//   clusterlm-node-service --print-firewall-specs --port N    (any OS: declarative rule data for the installer)
//
// Without "-- <worker args>" the worker gets the default Node layout (platform::default_paths()).
#include <atomic>
#include <cstdio>
#include <iostream>
#include <thread>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/lease/lease_store.hpp"
#include "clusterlm/node/service_core.hpp"
#include "clusterlm/platform/firewall.hpp"
#include "clusterlm/platform/mock_adapters.hpp"
#include "clusterlm/platform/paths.hpp"
#include "clusterlm/platform/service_host.hpp"

#ifdef _WIN32
#include <windows.h>
#include <objbase.h>
#endif

using namespace clusterlm;

namespace {

constexpr std::uint64_t kDefaultNodePort = 47600;  // arbitrary default, not a registered port; --port overrides

int fail(const Status& s) {
  std::fprintf(stderr, "error: %s\n", s.to_string().c_str());
  return 1;
}

// Installer/uninstaller entry (--cleanup). Runs LeaseStore recovery first, so a crashed lease is journal-accounted and
// its files are deleted by the same code that deletes them at service start, then removes the staging root itself.
// Fails (non-zero) if anything of the lease store survives: the uninstaller must never leave model fragments behind.
// With the default layout, --purge additionally removes the identity and log directories and the Node root (only if they are then empty).
int cleanup_node_data(const cli::Args& args, const platform::DefaultPaths& paths) {
  namespace fs = std::filesystem;
  const fs::path staging = args.get("staging", paths.node_staging.string());
  std::error_code ec;
  if (fs::exists(staging, ec)) {
    {
      auto store = lease::LeaseStore::open(staging);
      if (!store.is_ok()) return fail(store.status());
      const auto& r = (*store)->recovery_report();
      std::printf("CLUSTERLM_NODE_CLEANUP leases_found=%llu files_removed=%llu bytes_reclaimed=%llu clean=%d\n",
                  static_cast<unsigned long long>(r.leases_found), static_cast<unsigned long long>(r.files_removed),
                  static_cast<unsigned long long>(r.bytes_reclaimed), r.clean() ? 1 : 0);
      if (!r.clean()) {
        std::fprintf(stderr, "error: lease recovery could not delete everything under the staging root\n");
        return 1;
      }
    }  // store closed before the directory is removed
    fs::remove_all(staging, ec);
    if (ec || fs::exists(staging)) {
      std::fprintf(stderr, "error: cannot remove the staging root: %s\n", ec.message().c_str());
      return 1;
    }
  }
  if (args.has("staging")) {
    // An explicit staging root (tests, non-default layouts): nothing else is touched.
  } else if (args.has("purge")) {
    for (const auto& dir : {paths.node_identity, paths.node_logs}) fs::remove_all(dir, ec);
    fs::remove(paths.node_root, ec);  // only succeeds when empty: unknown files are never deleted
  } else {
    fs::remove(paths.node_identity, ec);  // empty directory only: a paired identity is kept
    fs::remove(paths.node_root, ec);
  }
  std::printf("CLUSTERLM_NODE_CLEANUP staging_removed=1\n");
  return 0;
}

class NodeServiceApp final : public platform::ServiceApp {
 public:
  NodeServiceApp(node::ServiceCoreConfig cfg, std::unique_ptr<platform::PowerMonitor> power,
                 std::unique_ptr<platform::ProcessJob> job, bool simulate, std::uint32_t idle_required)
      : power_(std::move(power)),
        core_(std::move(cfg), *power_, std::move(job)),
        simulate_(simulate),
        idle_required_(idle_required) {}

  Status on_start() override {
    core_.set_event_sink([](const node::SupervisorEvent& e) {
      std::printf("CLUSTERLM_NODE_SERVICE_EVENT kind=%s latency_ms=%.1f residual_bytes=%llu\n",
                  std::string(node::to_string(e.kind)).c_str(), e.latency_ms,
                  static_cast<unsigned long long>(e.residual_bytes));
      std::fflush(stdout);
    });
    CLM_RETURN_IF_ERROR(core_.start());
    std::printf("CLUSTERLM_NODE_SERVICE worker_endpoint=%s device_id=%s\n", core_.worker_endpoint().c_str(),
                core_.worker_device_id().c_str());
    std::fflush(stdout);
    if (simulate_) {
      core_.helper_activity().submit({0, false, 0});  // starts in use; stdin drives the rest
      input_ = std::thread([this] { simulate_input(); });
    }
    return Status::ok();
  }

  int run() override {
    core_.run();
    if (input_.joinable()) input_.detach();  // blocked on stdin; the process exits right after
    return 0;
  }

  void on_stop(platform::StopReason) override { core_.request_stop(); }
  void on_power_event(const platform::PowerEvent& e) override { core_.handle_power_event(e); }
  void on_session_event(const platform::SessionEvent& e) override { core_.handle_session_event(e); }

 private:
  void simulate_input() {
    std::string line;
    std::uint32_t idle = 0;
    bool locked = false;
    while (std::getline(std::cin, line)) {
      if (line == "quit") break;
      if (line == "activity") idle = 0;
      if (line == "idle") idle = idle_required_;
      if (line == "lock") locked = true;
      if (line == "unlock") locked = false;
      if (line == "suspend") core_.handle_power_event({platform::PowerEventKind::kSuspend, true, false});
      if (line == "resume") core_.handle_power_event({platform::PowerEventKind::kResume, true, false});
      core_.helper_activity().submit({idle, locked, 0});
    }
    core_.request_stop();
  }

  std::unique_ptr<platform::PowerMonitor> power_;
  node::ServiceCore core_;
  bool simulate_;
  std::uint32_t idle_required_;
  std::thread input_;
};

}  // namespace

int main(int argc, char** argv) {
  // Split "service options -- worker arguments".
  int split = argc;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--") {
      split = i;
      break;
    }
  cli::Args args(split, argv);
  log::set_component("node-service");
  const std::uint16_t port = static_cast<std::uint16_t>(args.integer("port", kDefaultNodePort));

  if (args.has("print-firewall-specs")) {
    platform::FirewallSpecOptions fo;
    fo.node_program_path = args.get("worker", (platform::executable_dir() / "clusterlm-node").string());
    fo.node_data_port = port;
    fo.remote_addresses = args.all("remote");
    fo.allow_public_profile = args.has("allow-public");
    std::printf("%s", platform::firewall_specs_to_json(platform::firewall_rule_specs(fo)).c_str());
    return 0;
  }

  if (args.has("cleanup")) {
    auto p = platform::default_paths();
    if (!p.is_ok()) return fail(p.status());
    return cleanup_node_data(args, *p);
  }

#ifdef _WIN32
  if (args.has("install") || args.has("uninstall")) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    auto spec = platform::node_service_install_spec();
    if (args.has("uninstall")) {
      auto st = platform::uninstall_service(spec.name);
      if (!st.is_ok()) return fail(st);
      // The rules this installation created; a missing rule is success.
      auto fw = platform::make_windows_firewall_rules();
      for (const char* rule : {platform::kNodeRuleName, platform::kFatherRuleName}) {
        if (auto rs = fw->remove(rule); !rs.is_ok()) return fail(rs);
      }
      return 0;
    }
    wchar_t exe[MAX_PATH];
    const DWORD n = ::GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return fail(make_error(ErrorCode::kInternal, "GetModuleFileNameW failed"));
    spec.args = {"--port", std::to_string(port)};
    if (auto st = platform::install_service(spec, std::wstring(exe, n)); !st.is_ok()) return fail(st);
    // Firewall rule scoped to the worker program (the service itself opens no listening socket).
    platform::FirewallSpecOptions fo;
    fo.node_program_path = (platform::executable_dir() / "clusterlm-node.exe").string();
    fo.node_data_port = port;
    fo.remote_addresses = args.all("remote");
    auto fw = platform::make_windows_firewall_rules();
    if (auto st = platform::apply_firewall_specs(*fw, platform::firewall_rule_specs(fo)); !st.is_ok()) return fail(st);
    return 0;
  }
#endif

  auto paths = platform::default_paths();
  if (!paths.is_ok()) return fail(paths.status());

  node::ServiceCoreConfig cfg;
  cfg.supervisor.worker_binary = args.get("worker", (platform::executable_dir() / "clusterlm-node").string());
  const std::filesystem::path staging = args.get("staging", paths->node_staging.string());
  const std::filesystem::path identity = args.get("identity", paths->node_identity.string());
  if (split < argc) {
    for (int i = split + 1; i < argc; ++i) cfg.supervisor.worker_args.emplace_back(argv[i]);
  } else {
    // Product defaults. The Node data directories are owner-only (service account + SYSTEM on Windows).
    for (const auto& dir : {paths->node_root, staging, identity}) {
      if (auto st = platform::create_owner_only_directory(dir); !st.is_ok()) return fail(st);
    }
    cfg.supervisor.worker_args = {"--name",    args.get("name", "node"),
                                  "--listen",  "0.0.0.0:" + std::to_string(port),
                                  "--staging", staging.string(),
                                  "--identity", identity.string()};
  }
  cfg.staging_root = staging;
  cfg.supervisor.policy.idle_seconds_required = static_cast<std::uint32_t>(args.integer("idle-seconds", 300));
  cfg.supervisor.policy.require_ac_power = !args.has("allow-battery");
  cfg.supervisor.cooperative_deadline = std::chrono::milliseconds(args.integer("deadline-ms", 2000));
  cfg.helper_endpoint.socket_dir = args.get("ipc-dir", paths->ipc_dir.string());
  cfg.helper_auth.require_interactive_session = true;
  cfg.paired_father = args.get("paired-father");

  const bool simulate = args.has("simulate-activity");
  std::unique_ptr<platform::PowerMonitor> power;
  std::unique_ptr<platform::ProcessJob> job;
  bool console = args.has("console");
#ifdef _WIN32
  if (!simulate) {
    power = platform::make_windows_power_monitor();
    job = platform::make_windows_process_job("ClusterLM-Node-Worker");
    auto host = std::shared_ptr<platform::HelperHost>(platform::make_windows_helper_host());
    cfg.helper_host = host;
    cfg.helper_startup.helper_exe = platform::executable_dir() / "clusterlm-node-helper.exe";
    cfg.helper_startup.service_has_tcb_privilege = platform::current_process_has_tcb_privilege();
  }
#else
  console = true;  // no SCM here
  cfg.helper_auth.require_interactive_session = false;  // POSIX peers carry no session id (uid check is built in)
#endif
  if (!power) power = std::make_unique<platform::MockPowerMonitor>();  // dev: always on AC
  if (simulate) cfg.helper_report_stale_after = std::chrono::hours(24);

  NodeServiceApp app(std::move(cfg), std::move(power), std::move(job), simulate,
                     static_cast<std::uint32_t>(args.integer("idle-seconds", 300)));
  platform::ServiceHostOptions options;
  Result<int> rc = console ? platform::run_in_console(app, options) : platform::run_as_service(app, options);
  if (!rc.is_ok()) return fail(rc.status());
  return rc.value();
}
