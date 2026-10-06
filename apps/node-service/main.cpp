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
//   clusterlm-node-service --pair [--pair-port N] [--pair-window-seconds S]   enter pairing mode at startup
//   clusterlm-node-service --unpair                           forget the paired Father (service stopped)
// While running, the service also answers the helper pipe's settings and pairing-mode messages (the Node UI: docs/ui.md)
// and honours an UnpairNotice from its paired Father (docs/pairing.md): lease released, trust dropped, setting cleared.
//   [--settings FILE]   Node settings document (default: <node_root>/node-settings.json)
//   [--helper-pipe NAME] helper pipe name for side-by-side dev/test instances (default: the product name)
//
// Pairing (docs/pairing.md): pairing mode opens a time-boxed listener, prints one line
//   CLUSTERLM_NODE_PAIRING code=ABCD-EFGH endpoint=HOST:PORT fingerprint=xxxx-xxxx-xxxx
// and, once a Father proves knowledge of the code, stores it in the settings and restarts the worker trusting
// exactly that Father (CLUSTERLM_NODE_PAIRED ...). Console stdin (with --simulate-activity) also accepts
// `pair` and `unpair`.
//   clusterlm-node-service --print-firewall-specs --port N    (any OS: declarative rule data for the installer)
//
// Without "-- <worker args>" the worker gets the default Node layout (platform::default_paths()).
#include <atomic>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/lease/lease_store.hpp"
#include "clusterlm/config/store.hpp"
#include "clusterlm/node/service_core.hpp"
#include "clusterlm/pairing/pairing.hpp"
#include "clusterlm/platform/firewall.hpp"
#include "clusterlm/platform/mock_adapters.hpp"
#include "clusterlm/platform/paths.hpp"
#include "clusterlm/platform/service_host.hpp"
#include "clusterlm/transport/security.hpp"

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
    // The settings document (paired Father, caps) and any corrupt-file backups the settings store set aside.
    const fs::path settings = config::node_settings_path(paths);
    fs::remove(settings, ec);
    for (fs::directory_iterator it(paths.node_root, ec), end; !ec && it != end; it.increment(ec)) {
      const auto name = it->path().filename().string();
      if (name.rfind(settings.filename().string() + ".", 0) == 0 && it->is_regular_file(ec)) fs::remove(it->path(), ec);
    }
    fs::remove(paths.node_root, ec);  // only succeeds when empty: unknown files are never deleted
  } else {
    fs::remove(paths.node_identity, ec);  // empty directory only: a paired identity is kept
    fs::remove(paths.node_root, ec);
  }
  std::printf("CLUSTERLM_NODE_CLEANUP staging_removed=1\n");
  return 0;
}
// Firewall rule for the service's own pairing listener (created by --install, removed by --uninstall).
[[maybe_unused]] constexpr const char* kPairingRuleName = "ClusterLM Node pairing (TCP-In)";

// Everything the app needs to run pairing mode and persist its outcome.
struct PairingSetup {
  std::shared_ptr<const transport::DeviceIdentity> identity;
  config::NodeSettingsStore* settings = nullptr;
  std::string name;
  transport::Endpoint listen{"0.0.0.0", 0};
  std::chrono::seconds window{300};
  bool start_now = false;
};

class NodeServiceApp final : public platform::ServiceApp {
 public:
  NodeServiceApp(node::ServiceCoreConfig cfg, std::unique_ptr<platform::PowerMonitor> power,
                 std::unique_ptr<platform::ProcessJob> job, bool simulate, std::uint32_t idle_required,
                 PairingSetup pairing, bool start_paused)
      : power_(std::move(power)),
        core_(std::move(cfg), *power_, std::move(job)),
        simulate_(simulate),
        idle_required_(idle_required),
        pairing_(std::move(pairing)),
        start_paused_(start_paused) {}

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
    if (start_paused_) core_.helper_activity().pause(std::nullopt);  // settings: paused / not allowed when idle
    core_.set_pairing_starter([this] { return start_pairing(); });
    if (simulate_) {
      core_.helper_activity().submit({0, false, 0});  // starts in use; stdin drives the rest
      input_ = std::thread([this] { simulate_input(); });
    }
    if (pairing_.start_now) {
      if (auto st = start_pairing(); !st.is_ok()) std::fprintf(stderr, "pairing: %s\n", st.status().to_string().c_str());
    }
    return Status::ok();
  }

  // ---- pairing -----------------------------------------------------------------------------------------
  Result<ipc::PairingModeReply> start_pairing() {
    std::lock_guard lk(pair_mu_);
    stop_pairing_locked();
    pairing::ResponderOptions o;
    o.identity = pairing_.identity;
    o.listen = pairing_.listen;
    o.window = pairing_.window;
    auto data = transport::Endpoint::parse(core_.worker_endpoint());
    o.self = {pairing_.name, "node", data.is_ok() ? data->port : std::uint16_t{0}};
    CLM_ASSIGN_OR_RETURN(responder_, pairing::PairingResponder::start(std::move(o)));
    std::printf("CLUSTERLM_NODE_PAIRING code=%s endpoint=%s fingerprint=%s\n", responder_->code().c_str(),
                responder_->endpoint().str().c_str(), pairing::short_fingerprint(pairing_.identity->fingerprint()).c_str());
    std::fflush(stdout);
    auto* raw = responder_.get();
    pair_thread_ = std::thread([this, raw, window = pairing_.window] { await_pairing(raw, window); });
    ipc::PairingModeReply offer;
    offer.code = responder_->code();
    offer.endpoint = responder_->endpoint().str();
    offer.fingerprint = pairing::short_fingerprint(pairing_.identity->fingerprint());
    offer.window_seconds = static_cast<std::uint32_t>(pairing_.window.count());
    return offer;
  }

  Status unpair() {
    CLM_RETURN_IF_ERROR(core_.unpair_father());
    std::printf("CLUSTERLM_NODE_UNPAIRED\n");
    std::fflush(stdout);
    return Status::ok();
  }

  void stop_pairing() {
    std::lock_guard lk(pair_mu_);
    stop_pairing_locked();
  }

  int run() override {
    core_.run();
    stop_pairing();
    if (input_.joinable()) input_.detach();  // blocked on stdin; the process exits right after
    return 0;
  }

  void on_stop(platform::StopReason) override {
    core_.request_stop();
  }
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
      if (line == "pair") {
        if (auto st = start_pairing(); !st.is_ok()) std::fprintf(stderr, "pairing: %s\n", st.status().to_string().c_str());
      }
      if (line == "unpair") {
        if (auto st = unpair(); !st.is_ok()) std::fprintf(stderr, "unpair: %s\n", st.to_string().c_str());
      }
      if (line == "suspend") core_.handle_power_event({platform::PowerEventKind::kSuspend, true, false});
      if (line == "resume") core_.handle_power_event({platform::PowerEventKind::kResume, true, false});
      core_.helper_activity().submit({idle, locked, 0});
    }
    core_.request_stop();
  }

  // Runs on the pairing thread: waits for the mode to end and persists/applies a successful pairing.
  void await_pairing(pairing::PairingResponder* responder, std::chrono::seconds window) {
    auto peer = responder->wait(window + std::chrono::seconds(15));
    if (!peer.is_ok()) {
      std::printf("CLUSTERLM_NODE_PAIRING_ENDED state=%s\n", std::string(pairing::to_string(responder->state())).c_str());
      std::fflush(stdout);
      return;
    }
    config::PairedDevice d;
    d.fingerprint = peer->fingerprint;
    d.name = peer->info.name;
    d.role = "father";
    d.paired_at_unix = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // Persist first, then trust: a crash in between leaves the Father in the document but untrusted until the
    // next start (fail closed), never trusted without being recorded.
    Status st = pairing_.settings->update([&](config::NodeSettings& s) {
      s.paired_father = d;
      return Status::ok();
    });
    if (st.is_ok()) st = core_.set_trusted_fathers({d.fingerprint}, pairing::short_fingerprint(d.fingerprint));
    if (!st.is_ok()) {
      std::fprintf(stderr, "pairing: could not apply the new Father: %s\n", st.to_string().c_str());
      return;
    }
    std::printf("CLUSTERLM_NODE_PAIRED father=%s name=%s\n", pairing::short_fingerprint(d.fingerprint).c_str(), d.name.c_str());
    std::fflush(stdout);
  }

  void stop_pairing_locked() {
    if (responder_) responder_->cancel();
    if (pair_thread_.joinable()) pair_thread_.join();
    responder_.reset();
  }

  std::unique_ptr<platform::PowerMonitor> power_;
  node::ServiceCore core_;
  bool simulate_;
  std::uint32_t idle_required_;
  PairingSetup pairing_;
  bool start_paused_;
  std::thread input_;
  std::mutex pair_mu_;
  std::unique_ptr<pairing::PairingResponder> responder_;
  std::thread pair_thread_;
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
      for (const char* rule : {platform::kNodeRuleName, platform::kFatherRuleName, kPairingRuleName}) {
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
    auto specs = platform::firewall_rule_specs(fo);
    // Pairing listener: lives in the service executable (not the worker) and only exists while pairing mode runs.
    platform::FirewallRuleSpec pair_rule;
    pair_rule.name = kPairingRuleName;
    pair_rule.description = "Allows a ClusterLM Father on the local network to pair with this Node while pairing mode is active.";
    pair_rule.program_path = std::filesystem::path(std::wstring(exe, n)).string();
    pair_rule.local_port = static_cast<std::uint16_t>(args.integer("pair-port", port + 1u));
    pair_rule.remote_addresses = fo.remote_addresses.empty() ? std::vector<std::string>{"LocalSubnet"} : fo.remote_addresses;
    specs.push_back(std::move(pair_rule));
    if (auto st = platform::apply_firewall_specs(*fw, specs); !st.is_ok()) return fail(st);
    return 0;
  }
#endif

  auto paths = platform::default_paths();
  if (!paths.is_ok()) return fail(paths.status());

  // ---- persistent Node settings (docs/pairing.md): paired Father, participation policy, resource caps ----
  const std::filesystem::path settings_file = args.get("settings", config::node_settings_path(paths.value()).string());
  auto settings_store = config::NodeSettingsStore::open(settings_file);
  if (!settings_store.is_ok()) return fail(settings_store.status());
  const auto& load_report = settings_store.value()->report();
  if (load_report.outcome == config::LoadReport::Outcome::kRecoveredCorrupt)
    std::fprintf(stderr, "warning: %s (kept as %s)\n", load_report.note.c_str(), load_report.backup.string().c_str());
  if (args.has("unpair")) {
    auto st = settings_store.value()->update([](config::NodeSettings& s) {
      s.paired_father.reset();
      return Status::ok();
    });
    if (!st.is_ok()) return fail(st);
    std::printf("CLUSTERLM_NODE_UNPAIRED\n");
    return 0;
  }
  const config::NodeSettings settings = settings_store.value()->get();
  std::shared_ptr<config::NodeSettingsStore> settings_shared = std::move(settings_store).value();

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
    cfg.supervisor.worker_args = {"--name",    args.get("name", settings.name),
                                  "--listen",  "0.0.0.0:" + std::to_string(port),
                                  "--staging", staging.string(),
                                  "--identity", identity.string(),
                                  "--ram-gib", args.get("ram-gib", std::to_string(settings.caps.ram_gib)),
                                  "--vram-gib", args.get("vram-gib", std::to_string(settings.caps.vram_gib))};
    if (settings.temp_storage_limit_gib > 0) {
      cfg.supervisor.worker_args.push_back("--disk-gib");
      cfg.supervisor.worker_args.push_back(std::to_string(settings.temp_storage_limit_gib));
    }
    if (const auto threads = args.integer("threads", settings.caps.threads); threads > 0) {  // thread cap (0 = automatic)
      cfg.supervisor.worker_args.push_back("--threads");
      cfg.supervisor.worker_args.push_back(std::to_string(threads));
    }
  }
  // The identity pairing presents must be the worker's: take it from the worker arguments when they name one.
  std::filesystem::path pairing_identity_dir = identity;
  std::string pairing_name = args.get("name", settings.name);
  for (std::size_t i = 0; i + 1 < cfg.supervisor.worker_args.size(); ++i) {
    if (cfg.supervisor.worker_args[i] == "--identity") pairing_identity_dir = cfg.supervisor.worker_args[i + 1];
    if (cfg.supervisor.worker_args[i] == "--name") pairing_name = cfg.supervisor.worker_args[i + 1];
  }
  // The paired Father (if any) is the only peer the worker will talk to.
  if (settings.paired_father) {
    cfg.supervisor.trusted_peers = {settings.paired_father->fingerprint};
    cfg.paired_father = pairing::short_fingerprint(settings.paired_father->fingerprint);
  }
  cfg.staging_root = staging;
  cfg.settings = settings_shared;
  const auto idle_seconds = static_cast<std::uint32_t>(args.integer("idle-seconds", settings.idle_seconds));
  cfg.supervisor.policy.idle_seconds_required = idle_seconds;
  cfg.supervisor.policy.require_ac_power = !args.has("allow-battery") && settings.ac_only;
  cfg.supervisor.cooperative_deadline = std::chrono::milliseconds(args.integer("deadline-ms", 2000));
  cfg.helper_endpoint.socket_dir = args.get("ipc-dir", paths->ipc_dir.string());
  // Windows pipe names are machine-global (--ipc-dir does not apply there): side-by-side dev/test instances need
  // their own name. The Node UI and helper connect to the default name.
  if (args.has("helper-pipe")) cfg.helper_endpoint.name = args.get("helper-pipe");
  cfg.helper_auth.require_interactive_session = true;
  if (args.has("paired-father")) cfg.paired_father = args.get("paired-father");  // dev override of the status label

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

  // Pairing needs the device identity the worker presents. It is created here, before the worker starts, so the
  // two never race to generate different keys.
  PairingSetup pairing_setup;
  {
    auto ident = transport::DeviceIdentity::load_or_generate(pairing_identity_dir, pairing_name);
    if (!ident.is_ok()) return fail(ident.status());
    pairing_setup.identity = std::make_shared<const transport::DeviceIdentity>(std::move(ident).value());
  }
  pairing_setup.settings = settings_shared.get();
  pairing_setup.name = pairing_name;
  pairing_setup.listen = {"0.0.0.0", static_cast<std::uint16_t>(args.integer("pair-port", port + 1u))};
  pairing_setup.window = std::chrono::seconds(static_cast<std::int64_t>(args.integer("pair-window-seconds", 300)));
  pairing_setup.start_now = args.has("pair");
  const bool start_paused = settings.paused || !settings.allow_when_idle;

  NodeServiceApp app(std::move(cfg), std::move(power), std::move(job), simulate, idle_seconds,
                     std::move(pairing_setup), start_paused);
  platform::ServiceHostOptions options;
  Result<int> rc = console ? platform::run_in_console(app, options) : platform::run_as_service(app, options);
  if (!rc.is_ok()) return fail(rc.status());
  return rc.value();
}
