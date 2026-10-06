// clusterlm-node: the ClusterLM Node worker process.
//
// In the product this process is launched by the ClusterLM Node service inside a Job Object, with local
// activity reported by the session helper. In development it runs standalone: a process on 127.0.0.1 stands in
// for a remote Node execution domain ("localhost today, LAN tomorrow"). Local activity can be simulated on stdin.
//
// stdout protocol (for harnesses): one line "CLUSTERLM_NODE_LISTENING endpoint=<host:port> device_id=<id>".
// stdin commands: "activity", "idle", "status", "quit". EOF on stdin = quit (the parent harness went away).
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/node/node_worker.hpp"

using namespace clusterlm;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: clusterlm-node --staging DIR [--name NAME] [--listen HOST:PORT] [--ram-gib N] [--vram-gib N]\n"
               "                      [--disk-gib N] (--insecure-loopback | --identity DIR --trust FINGERPRINT...)\n"
               "                      [--impair PRESET] [--fault RULE]... [--start-busy] [--log debug|info|warn]\n"
               "                      [--backend reference|strata] [--cuda-device N] [--vram-reserve-mib N]\n"
               "                      [--strata-cpu-threads N]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  if (!args.has("staging")) return usage();
  node::NodeConfig cfg;
  cfg.name = args.get("name", "node");
  log::set_component(cfg.name);
  if (args.get("log") == "debug") log::set_level(log::Level::kDebug);
  if (args.get("log") == "warn") log::set_level(log::Level::kWarn);
  auto ep = transport::Endpoint::parse(args.get("listen", "127.0.0.1:0"));
  if (!ep.is_ok()) {
    std::fprintf(stderr, "%s\n", ep.status().to_string().c_str());
    return 2;
  }
  cfg.listen = ep.value();
  cfg.staging_root = args.get("staging");
  cfg.ram_allowance = static_cast<std::uint64_t>(args.number("ram-gib", 4) * static_cast<double>(cli::kGiB));
  cfg.vram_allowance = static_cast<std::uint64_t>(args.number("vram-gib", 0) * static_cast<double>(cli::kGiB));
  cfg.disk_allowance = static_cast<std::uint64_t>(args.number("disk-gib", 0) * static_cast<double>(cli::kGiB));
  cfg.start_busy = args.has("start-busy");
  // Backend of the middle-stage domains. An unknown or unbuilt backend refuses to start (NodeWorker::start).
  cfg.backend = args.get("backend", "reference");
  cfg.strata.cuda_device = static_cast<int>(args.integer("cuda-device", 0));
  cfg.strata.vram_reserve_mib = static_cast<std::uint32_t>(args.integer("vram-reserve-mib", 1024));
  cfg.strata.cpu_threads = static_cast<std::uint32_t>(args.integer("strata-cpu-threads", 0));

  if (args.has("insecure-loopback")) {
    cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  } else if (args.has("identity")) {
    auto id = transport::DeviceIdentity::load_or_generate(args.get("identity"), cfg.name);
    if (!id.is_ok()) {
      std::fprintf(stderr, "identity: %s\n", id.status().to_string().c_str());
      return 1;
    }
    cfg.security.mode = transport::SecurityConfig::Mode::kMutualTls;
    cfg.security.identity = std::make_shared<const transport::DeviceIdentity>(std::move(id).value());
    for (const auto& fp : args.all("trust")) cfg.security.trust(fp);
  } else {
    return usage();
  }
  if (args.has("impair")) {
    auto preset = transport::network_preset(args.get("impair"));
    if (!preset.is_ok()) {
      std::fprintf(stderr, "%s\n", preset.status().to_string().c_str());
      return 2;
    }
    cfg.impairment = preset.value();
  }
  if (args.has("fault")) {
    cfg.faults = std::make_shared<transport::FaultInjector>();
    for (const auto& rule : args.all("fault")) {
      auto r = transport::parse_fault_rule(rule);
      if (!r.is_ok()) {
        std::fprintf(stderr, "%s\n", r.status().to_string().c_str());
        return 2;
      }
      cfg.faults->add_rule(r.value());
    }
  }
  if (args.has("crash-at") || args.has("hang-at")) {
    // Fault-injection hooks for ClusterLM Bench: terminate abruptly (crash) or block forever (hung driver /
    // stuck worker) at a lifecycle phase. Never used in production.
    const std::string crash = args.get("crash-at"), hang = args.get("hang-at");
    cfg.phase_hook = [crash, hang](std::string_view phase) {
      if (phase == crash) std::_Exit(42);
      if (phase == hang)
        for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
    };
  }

  auto worker = node::NodeWorker::start(std::move(cfg));
  if (!worker.is_ok()) {
    std::fprintf(stderr, "start: %s\n", worker.status().to_string().c_str());
    return 1;
  }
  auto& w = *worker.value();
  std::printf("CLUSTERLM_NODE_LISTENING endpoint=%s device_id=%s\n", w.endpoint().str().c_str(), w.device_id().c_str());
  std::fflush(stdout);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "activity") {
      w.on_local_activity();
    } else if (line == "idle") {
      w.on_local_idle();
    } else if (line == "status") {
      auto s = w.status();
      std::printf("CLUSTERLM_NODE_STATUS state=%s lease=%llu windows=%llu forwarded=%llu stale=%llu census_bytes=%llu "
                  "storage_cleaned=%d\n",
                  std::string(node::to_string(s.state)).c_str(), static_cast<unsigned long long>(s.lease_generation),
                  static_cast<unsigned long long>(s.windows_executed),
                  static_cast<unsigned long long>(s.windows_forwarded),
                  static_cast<unsigned long long>(s.stale_rejections),
                  static_cast<unsigned long long>(s.staging_census_bytes), s.last_storage_cleaned ? 1 : 0);
      std::fflush(stdout);
    } else if (line == "quit") {
      break;
    }
  }
  w.stop();
  return 0;
}
