// clusterlm-bench: ClusterLM Bench — hardware, transport and runtime measurement and qualification suite.
#include <cstdio>
#include <string>
#include <vector>

#include "commands.hpp"
#include "clusterlm/common/log.hpp"
#include "expert_kernels.hpp"

using namespace clusterlm;

namespace {
int usage() {
  std::fprintf(stderr,
               "usage: clusterlm-bench <command> [options]\n"
               "measurements (results are Measured on the machine that ran them; host_role says whether it is a target):\n"
               "  calibrate       cpu + memory + gpu + (optional) network; writes profile-<machine>.json / network-<machine>.json\n"
               "  cpu             routed-expert kernel throughput, thread/q scaling, --sustained thermal run\n"
               "  memory          RAM, commit limit, bounded allocation probe, multi-threaded read bandwidth\n"
               "  gpu | pcie      CUDA device, VRAM, allocation overhead, H2D/D2H, pinned limit, cuBLAS shapes (exit 3 without CUDA)\n"
               "  transport       RTT/jitter/throughput; --peer (real link), --serve, --simulate-nodes, --node-to-node\n"
               "  profile         quick host probe\n"
               "software qualification (Synthetic: localhost processes, fixture model):\n"
               "  cluster         multi-process localhost cluster through the real protocol (--tier, --context, --compare-routing)\n"
               "  faults          kill/stall/revoke Nodes at protocol phases; verify invalidation and cleanup (--only <scenario|phase>,...)\n"
               "  nvml            NVML clocks/power/temperature/VRAM/throttle sampler to run beside a sustained workload\n"
               "  storage-census  --before|--after|--diff of CUDA/driver caches, temp and ClusterLM data (names and sizes)\n"
               "  placement       run the placement search over hardware profiles\n"
               "  qualification   list the hardware qualification registry\n"
               "  domain|baseline|numerics  hardware-backed experiments (require the Strata/llama backends; exit 3 here)\n"
               "common options: --out FILE (JSON result), --log debug|info|warn, --machine-id ID, --on-target\n");
  return 2;
}
}  // namespace

namespace {
int dispatch(const std::string& cmd, const cli::Args& args) {
  if (cmd == "profile") return bench::cmd_profile(args);
  if (cmd == "cpu") return bench::cmd_cpu(args);
  if (cmd == "memory") return bench::cmd_memory(args);
  if (cmd == "gpu") return bench::cmd_gpu(args, false);
  if (cmd == "pcie") return bench::cmd_gpu(args, true);
  if (cmd == "calibrate") return bench::cmd_calibrate(args);
  if (cmd == "transport") return bench::cmd_transport(args);
  if (cmd == "cluster") return bench::cmd_cluster(args);
  if (cmd == "faults") return bench::cmd_faults(args);
  if (cmd == "nvml") return bench::cmd_nvml(args);
  if (cmd == "storage-census") return bench::cmd_storage_census(args);
  if (cmd == "placement") return bench::cmd_placement(args);
  if (cmd == "placement-inputs") return bench::cmd_placement_inputs(args);
  if (cmd == "qualification") return bench::cmd_qualification(args);
  if (cmd == "baseline" && !args.positional().empty() && args.positional().front() == "llama-rpc")
    return bench::cmd_baseline_llama(args);
  return bench::cmd_hardware_only(cmd, args);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string cmd = argv[1];
  const auto* spec = bench::find_command_spec(cmd);
  if (!spec) return usage();
  std::vector<std::string> tokens;
  for (int i = 2; i < argc; ++i) tokens.emplace_back(argv[i]);
  if (const std::string bad = bench::check_command_tokens(*spec, tokens); !bad.empty()) {
    std::fprintf(stderr, "clusterlm-bench: %s\n", bad.c_str());
    return 2;
  }
  cli::Args args(argc, argv, 2);
  log::set_component("bench");
  log::set_level(args.get("log") == "debug" ? log::Level::kDebug
                 : args.get("log") == "info" ? log::Level::kInfo
                                             : log::Level::kWarn);
  bench::register_builtin_expert_providers();
  try {
    return dispatch(cmd, args);
  } catch (const std::exception& e) {
    // Malformed numeric options (std::stoul and friends) are usage errors, not crashes.
    std::fprintf(stderr, "clusterlm-bench %s: %s\n", cmd.c_str(), e.what());
    return 2;
  }
}
