// clusterlm-bench: ClusterLM Bench — hardware, transport and runtime qualification suite.
#include <cstdio>
#include <string>

#include "commands.hpp"
#include "clusterlm/common/log.hpp"

using namespace clusterlm;

namespace {
int usage() {
  std::fprintf(stderr,
               "usage: clusterlm-bench <command> [options]\n"
               "  profile         probe this host and emit a HardwareProfile (Measured for this host only)\n"
               "  transport       framed transport throughput/RTT (loopback with --impair, or --serve/--peer)\n"
               "  cluster         run a multi-process localhost cluster through the real protocol\n"
               "  faults          kill/stall/revoke Nodes at protocol phases; verify invalidation and cleanup\n"
               "  placement       run the placement search over hardware profiles\n"
               "  qualification   list the hardware qualification registry\n"
               "  domain|baseline|numerics  hardware-backed experiments (require the Strata/llama backends)\n"
               "common options: --out FILE (JSON result), --log debug|info|warn\n");
  return 2;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string cmd = argv[1];
  cli::Args args(argc, argv, 2);
  log::set_component("bench");
  log::set_level(args.get("log") == "debug" ? log::Level::kDebug
                 : args.get("log") == "info" ? log::Level::kInfo
                                             : log::Level::kWarn);
  if (cmd == "profile") return bench::cmd_profile(args);
  if (cmd == "transport") return bench::cmd_transport(args);
  if (cmd == "cluster") return bench::cmd_cluster(args);
  if (cmd == "faults") return bench::cmd_faults(args);
  if (cmd == "placement") return bench::cmd_placement(args);
  if (cmd == "qualification") return bench::cmd_qualification(args);
  if (cmd == "domain" || cmd == "baseline" || cmd == "numerics") return bench::cmd_hardware_only(cmd, args);
  return usage();
}
