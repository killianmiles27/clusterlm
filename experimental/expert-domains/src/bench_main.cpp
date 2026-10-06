// clusterlm-expert-domain-bench: EXPERIMENTAL grouped expert-domain topology (P0-C) vs the layer-domain pipeline.
//
// Everything it emits is Synthetic (fixture model, localhost, simulated links) except the analytic block, which is
// a labelled calculation. It never emits "Qualified". See docs/experimental/expert-domains.md.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "cli.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/expert_domains/analytic.hpp"
#include "clusterlm/expert_domains/simulation.hpp"
#include "clusterlm/transport/impairment.hpp"
#include "host_probe.hpp"

using namespace clusterlm;
namespace fs = std::filesystem;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: clusterlm-expert-domain-bench [options]\n"
               "  --out FILE            write the BenchmarkResult JSON (default: stdout)\n"
               "  --experiment ID       experiment id recorded in the result (default dev-expert-domains)\n"
               "  --q 1,2,3,4           verification widths to run\n"
               "  --presets a,b,c       network presets (default unlimited,gige-simulated,gige-degraded)\n"
               "  --remote-domains N    remote expert domains (default 2); Father is owner 0\n"
               "  --windows N           measured decode windows per (preset, q) (default 12)\n"
               "  --prompt-len N        prefill tokens before measuring (default 8)\n"
               "  --strided             round-robin expert ownership instead of contiguous ranges\n"
               "  --no-overlap          run Father's experts after the barrier instead of during it\n"
               "  --no-reference        skip the per-window comparison with the unsplit reference domain\n"
               "  --skip-layer-domain   do not spawn clusterlm-node processes for the layer-domain comparison\n"
               "  --layers N --hidden N --experts N --active N --expert-ff N --shared-ff N --seed N\n"
               "                        scale the generated fixture model (defaults: the standard fixture)\n"
               "  --tolerance X         reference tolerance, max |diff| / max |logit| (default 1e-5)\n"
               "  --work DIR            scratch directory for the model and node staging\n"
               "  --analytic-only       print the analytic model for the real geometry and exit\n"
               "  --quiet               no human-readable tables\n");
  return 2;
}

std::vector<std::uint32_t> parse_uints(const std::string& s) {
  std::vector<std::uint32_t> v;
  for (const auto& p : cli::split(s, ',')) v.push_back(static_cast<std::uint32_t>(std::stoul(p)));
  return v;
}

int analytic_only() {
  for (const char* preset : {"unlimited", "gige-simulated", "gige-degraded"}) {
    for (std::uint32_t q = 1; q <= 4; ++q) {
      expert_domains::AnalyticInputs in;
      in.q = q;
      if (std::string(preset) == "unlimited") {
        in.bandwidth_bytes_per_s = 1e12;
        in.one_way_latency_ms = 0;
      } else if (auto n = transport::network_preset(preset); n.is_ok()) {
        in.bandwidth_bytes_per_s = n->bandwidth_bytes_per_s;
        in.one_way_latency_ms = n->latency_ms + n->jitter_ms / 2.0;
      }
      std::printf("[%s]\n%s\n", preset, expert_domains::analytic_report(in, expert_domains::analytic_model(in)).c_str());
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv, 1);
  if (args.has("help") || args.has("h")) return usage();
  log::set_component("expert-domain-bench");
  log::set_level(log::Level::kWarn);
  if (args.has("analytic-only")) return analytic_only();

  Stopwatch total;
  expert_domains::SimulationOptions opt;
  try {
    if (args.has("q")) opt.q_values = parse_uints(args.get("q"));
    if (args.has("presets")) opt.presets = cli::split(args.get("presets"), ',');
    opt.remote_domains = static_cast<std::uint32_t>(args.integer("remote-domains", opt.remote_domains));
    opt.windows = static_cast<std::uint32_t>(args.integer("windows", opt.windows));
    opt.prompt_len = static_cast<std::uint32_t>(args.integer("prompt-len", opt.prompt_len));
    opt.fixture.n_layers = static_cast<std::uint32_t>(args.integer("layers", opt.fixture.n_layers));
    opt.fixture.hidden = static_cast<std::uint32_t>(args.integer("hidden", opt.fixture.hidden));
    opt.fixture.n_experts = static_cast<std::uint32_t>(args.integer("experts", opt.fixture.n_experts));
    opt.fixture.n_active = static_cast<std::uint32_t>(args.integer("active", opt.fixture.n_active));
    opt.fixture.expert_ff = static_cast<std::uint32_t>(args.integer("expert-ff", opt.fixture.expert_ff));
    opt.fixture.shared_expert_ff = static_cast<std::uint32_t>(args.integer("shared-ff", opt.fixture.shared_expert_ff));
    opt.fixture.seed = args.integer("seed", opt.fixture.seed);
    opt.reference_tolerance = args.number("tolerance", opt.reference_tolerance);
  } catch (const std::exception&) {
    std::fprintf(stderr, "invalid numeric option\n");
    return usage();
  }
  opt.strided = args.has("strided");
  opt.overlap_local = !args.has("no-overlap");
  opt.check_reference = !args.has("no-reference");
  opt.layer_domain = !args.has("skip-layer-domain");
  opt.work_dir = args.has("work") ? fs::path(args.get("work"))
                                  : fs::temp_directory_path() / ("clusterlm-expert-domains-" + std::to_string(monotonic_ns()));
  opt.report = args.has("quiet") ? nullptr : stderr;

  bench::BenchmarkResult result(args.get("experiment", "dev-expert-domains"), bench::probe_host());
  const Status st = expert_domains::run_simulation(opt, result);
  if (!st.is_ok()) {
    std::fprintf(stderr, "error: %s\n", st.to_string().c_str());
    return 1;
  }
  const bool ok = result.all_checks_passed();
  const auto doc = result.finish(total.elapsed_ms() / 1000.0);
  const std::string text = doc.dump(2) + "\n";
  if (args.has("out")) {
    std::ofstream f(args.get("out"), std::ios::binary);
    f << text;
    std::fprintf(stderr, "wrote %s (provenance %s)\n", args.get("out").c_str(), doc["provenance"].get<std::string>().c_str());
  } else {
    std::cout << text;
  }
  if (!args.has("work")) {
    std::error_code ec;
    fs::remove_all(opt.work_dir, ec);
  }
  return ok ? 0 : 1;
}
