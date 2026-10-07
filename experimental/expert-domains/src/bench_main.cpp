// clusterlm-expert-domain-bench: EXPERIMENTAL grouped expert-domain topology (P0-C) vs the layer-domain pipeline.
//
// Everything it emits is Synthetic (fixture model, localhost, simulated links) except the analytic block, which is
// a labelled calculation. It never emits "Qualified". See docs/experimental/expert-domains.md.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>
#include <fstream>
#include <iostream>

#include "cli.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/expert_domains/analytic.hpp"
#include "clusterlm/expert_domains/peer.hpp"
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
               "LAN peer mode (HQ-P0C-02; domains as separate processes/machines over the mutual-TLS transport):\n"
               "  --listen HOST:PORT    run ONE expert domain and serve Fathers (needs --domain-index and --remote-domains)\n"
               "  --domain-index I      this domain's owner index, 1..remote-domains (Father is owner 0)\n"
               "  --max-sessions N      with --listen: return after N Father sessions (default: serve until stdin EOF/SIGINT)\n"
               "  --peer NAME=HOST:PORT[@FINGERPRINT]  (Father; repeatable) connect to a running domain instead of a thread\n"
               "  --identity DIR        device identity directory (created if absent); --trust FP (repeatable) pinned peers\n"
               "  --insecure-loopback   plain TCP, loopback only (tests); default is mutual TLS\n"
               "  --max-window N        widest window the domains must accept (default 8; >= the largest --q)\n"
               "  --expert-kernel K     fixture (default) | iq3_s | iq2_xs: Strata CPU kernels on SYNTHETIC blobs (needs a\n"
               "                        -DCLUSTERLM_ENABLE_STRATA_CPU build and --hidden/--expert-ff multiples of 256)\n"
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

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

Result<transport::SecurityConfig> peer_security(const cli::Args& args) {
  transport::SecurityConfig sec;
  if (args.has("insecure-loopback")) {
    sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    return sec;
  }
  if (!args.has("identity")) return make_error(ErrorCode::kInvalidArgument, "mutual TLS needs --identity DIR (or --insecure-loopback)");
  CLM_ASSIGN_OR_RETURN(auto id, transport::DeviceIdentity::load_or_generate(args.get("identity"), "expert-domain-bench"));
  sec.identity = std::make_shared<const transport::DeviceIdentity>(std::move(id));
  sec.mode = transport::SecurityConfig::Mode::kMutualTls;
  for (const auto& fp : args.all("trust")) sec.trusted_peers.push_back(fp);
  return sec;
}

// --listen: one expert domain in this process.
int run_listen(const cli::Args& args, const objects::FixtureSpec& fixture, const fs::path& work) {
  expert_domains::PeerServeOptions po;
  po.fixture = fixture;
  po.work_dir = work;
  if (args.has("model")) po.model_dir = args.get("model");
  auto ep = transport::Endpoint::parse(args.get("listen"));
  auto sec = peer_security(args);
  auto kernel = expert_domains::parse_expert_kernel(args.get("expert-kernel", "fixture"), fixture.seed);
  if (!ep.is_ok() || !sec.is_ok() || !kernel.is_ok()) {
    const Status& st = !ep.is_ok() ? ep.status() : !sec.is_ok() ? sec.status() : kernel.status();
    std::fprintf(stderr, "error: %s\n", st.to_string().c_str());
    return 2;
  }
  po.listen = ep.value();
  po.security = sec.value();
  po.kernel = kernel.value();
  po.remote_domains = static_cast<std::uint32_t>(args.integer("remote-domains", 2));
  po.domain_index = static_cast<std::uint32_t>(args.integer("domain-index", 0));
  po.strided = args.has("strided");
  po.max_window = static_cast<std::uint32_t>(args.integer("max-window", 8));
  po.max_sessions = static_cast<std::uint32_t>(args.integer("max-sessions", 0));
  po.stop = &g_stop;
  po.on_listening = [](const transport::Endpoint& e, const std::string& id) {
    std::printf("EXPERT_DOMAIN_LISTENING endpoint=%s device_id=%s\n", e.str().c_str(), id.empty() ? "insecure-loopback" : id.c_str());
    std::fflush(stdout);
  };
  std::signal(SIGINT, on_signal);
  std::thread stdin_watch;
  if (args.has("exit-on-stdin-eof")) {
    stdin_watch = std::thread([] {
      char buf[64];
      while (std::fgets(buf, sizeof buf, stdin) != nullptr) {
      }
      g_stop = true;
    });
    stdin_watch.detach();
  }
  const Status st = expert_domains::serve_peer(po);
  if (!st.is_ok()) {
    std::fprintf(stderr, "error: %s\n", st.to_string().c_str());
    return 1;
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

  if (args.has("listen")) {
    const int rc = run_listen(args, opt.fixture, opt.work_dir);
    if (!args.has("work")) {
      std::error_code ec;
      fs::remove_all(opt.work_dir, ec);
    }
    return rc;
  }
  if (args.has("expert-kernel")) {
    auto k = expert_domains::parse_expert_kernel(args.get("expert-kernel"), opt.fixture.seed);
    if (!k.is_ok()) {
      std::fprintf(stderr, "error: %s\n", k.status().message().c_str());
      return 2;
    }
    opt.kernel = k.value();
    if (opt.kernel.quantized()) opt.check_reference = false;
  }
  if (args.has("peer")) {
    for (const auto& text : args.all("peer")) {
      auto p = expert_domains::parse_peer(text);
      if (!p.is_ok()) {
        std::fprintf(stderr, "error: %s\n", p.status().message().c_str());
        return 2;
      }
      opt.peers.push_back(std::move(p).value());
    }
    if (args.has("remote-domains") && opt.remote_domains != opt.peers.size()) {
      std::fprintf(stderr, "error: --remote-domains %u but %zu --peer given\n", opt.remote_domains, opt.peers.size());
      return 2;
    }
    opt.remote_domains = static_cast<std::uint32_t>(opt.peers.size());
    if (!args.has("presets")) opt.presets = {"unlimited"};
    auto sec = peer_security(args);
    if (!sec.is_ok()) {
      std::fprintf(stderr, "error: %s\n", sec.status().message().c_str());
      return 2;
    }
    opt.peer_security = sec.value();
  }

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
