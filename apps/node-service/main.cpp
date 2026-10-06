// clusterlm-node-service: the ClusterLM Node service process.
//
// On Windows this runs as a service; local activity comes from the per-session helper and the worker runs
// inside a Job Object. In development (any OS) `--simulate-activity` reads "activity"/"idle" lines from stdin
// instead, so the policy and deadline logic can be exercised without a desktop session.
//
//   clusterlm-node-service --worker PATH --idle-seconds 300 --simulate-activity -- <clusterlm-node args...>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <thread>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/node/supervisor.hpp"
#include "clusterlm/platform/mock_adapters.hpp"

using namespace clusterlm;

int main(int argc, char** argv) {
  // Split "service options -- worker arguments".
  int split = argc;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--") split = i;
  cli::Args args(split, argv);
  log::set_component("node-service");
  node::SupervisorConfig cfg;
  cfg.worker_binary = args.get("worker", (platform::executable_dir() / "clusterlm-node").string());
  for (int i = split + 1; i < argc; ++i) cfg.worker_args.emplace_back(argv[i]);
  cfg.policy.idle_seconds_required = static_cast<std::uint32_t>(args.integer("idle-seconds", 300));
  cfg.policy.require_ac_power = !args.has("allow-battery");
  cfg.cooperative_deadline = std::chrono::milliseconds(args.integer("deadline-ms", 2000));

  std::unique_ptr<platform::ActivityMonitor> activity;
  std::unique_ptr<platform::PowerMonitor> power;
  std::unique_ptr<platform::ProcessJob> job;
  platform::MockActivityMonitor* simulated = nullptr;  // non-owning view when --simulate-activity
  std::mutex sim_mu;
  if (args.has("simulate-activity")) {
    auto sim = std::make_unique<platform::MockActivityMonitor>();
    simulated = sim.get();
    activity = std::move(sim);
    power = std::make_unique<platform::MockPowerMonitor>();
  } else {
#ifdef _WIN32
    activity = platform::make_windows_activity_monitor();
    power = platform::make_windows_power_monitor();
    job = platform::make_windows_process_job("ClusterLM-Node-Worker");
#else
    std::fprintf(stderr, "real activity monitoring is Windows-only; use --simulate-activity\n");
    return 2;
#endif
  }

  node::NodeSupervisor sup(cfg, *activity, *power, std::move(job));
  if (auto st = sup.start(); !st.is_ok()) {
    std::fprintf(stderr, "start: %s\n", st.to_string().c_str());
    return 1;
  }
  std::printf("CLUSTERLM_NODE_SERVICE worker_endpoint=%s device_id=%s\n", sup.worker_endpoint().c_str(),
              sup.worker_device_id().c_str());
  std::fflush(stdout);

  std::atomic<bool> quit{false};
  std::thread input;
  if (simulated) {
    input = std::thread([&] {
      std::string line;
      while (std::getline(std::cin, line)) {
        std::lock_guard lock(sim_mu);
        if (line == "activity") simulated->current.idle_seconds = 0;
        if (line == "idle") simulated->current.idle_seconds = cfg.policy.idle_seconds_required;
        if (line == "quit") break;
      }
      quit.store(true);
    });
  }
  while (!quit.load()) {
    std::vector<node::SupervisorEvent> events;
    {
      std::lock_guard lock(sim_mu);
      auto r = sup.tick();
      if (!r.is_ok()) {
        std::fprintf(stderr, "tick: %s\n", r.status().to_string().c_str());
      } else {
        events = std::move(r).value();
      }
    }
    for (const auto& e : events) {
      std::printf("CLUSTERLM_NODE_SERVICE_EVENT kind=%s latency_ms=%.1f residual_bytes=%llu\n",
                  std::string(node::to_string(e.kind)).c_str(), e.latency_ms,
                  static_cast<unsigned long long>(e.residual_bytes));
      std::fflush(stdout);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (input.joinable()) input.join();
  sup.stop();
  return 0;
}
