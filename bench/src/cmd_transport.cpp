// clusterlm-bench transport: framed-transport RTT/jitter/throughput, concurrent peers and Node->Node links.
//
//   loopback self-test (development):   clusterlm-bench transport [--simulate-nodes 2] [--impair gige-simulated] [--tls]
//   real links (HQ-NET-01), on each peer: clusterlm-bench transport --serve 0.0.0.0:7400 --identity DIR --trust FP
//                    on the measuring machine: clusterlm-bench transport --peer nodeA=HOST:7400 --peer nodeB=HOST:7400
//                                              --identity DIR --trust FP --machine-id father --network-out network.json
//   Node->Node (development): clusterlm-bench transport --node-to-node
//
// Loopback, impaired and Node->Node-on-localhost runs are Synthetic; only --peer runs across a real link produce
// Measured results (and a Measured NetworkProfile).
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include "bench_common.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/transport/impairment.hpp"
#include "clusterlm/transport/security.hpp"
#include "commands.hpp"
#include "transport_bench.hpp"

namespace clusterlm::bench {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

Result<transport::SecurityConfig> security_from(const cli::Args& args, bool want_tls) {
  transport::SecurityConfig sec;
  if (!want_tls) {
    sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    return sec;
  }
  if (args.has("identity")) {
    CLM_ASSIGN_OR_RETURN(auto id, transport::DeviceIdentity::load_or_generate(args.get("identity"), "bench"));
    sec.identity = std::make_shared<const transport::DeviceIdentity>(std::move(id));
  } else {
    CLM_ASSIGN_OR_RETURN(auto id, transport::DeviceIdentity::generate("bench"));
    sec.identity = std::make_shared<const transport::DeviceIdentity>(std::move(id));
  }
  sec.mode = transport::SecurityConfig::Mode::kMutualTls;
  for (const auto& fp : args.all("trust")) sec.trusted_peers.push_back(fp);
  return sec;
}

struct PeerSpec {
  std::string name;
  transport::Endpoint endpoint;
  std::optional<std::string> expected_id;
};

Result<std::vector<PeerSpec>> parse_peers(const cli::Args& args) {
  std::vector<PeerSpec> out;
  const auto peers = args.all("peer");
  const auto ids = args.all("peer-id");
  for (std::size_t i = 0; i < peers.size(); ++i) {
    PeerSpec p;
    std::string text = peers[i];
    p.name = "peer" + std::to_string(i);
    if (auto eq = text.find('='); eq != std::string::npos) {
      p.name = text.substr(0, eq);
      text = text.substr(eq + 1);
    }
    CLM_ASSIGN_OR_RETURN(p.endpoint, transport::Endpoint::parse(text));
    if (i < ids.size() && !ids[i].empty()) p.expected_id = ids[i];
    out.push_back(std::move(p));
  }
  return out;
}

LinkBenchOptions link_options(const cli::Args& args) {
  LinkBenchOptions o;
  if (args.has("sizes")) {
    o.sizes.clear();
    for (auto s : parse_u32_list(args.get("sizes"))) o.sizes.push_back(s);
  }
  o.iterations = static_cast<unsigned>(args.integer("iterations", o.iterations));
  o.warmup = static_cast<unsigned>(args.integer("warmup", o.warmup));
  o.burst_bytes = args.integer("burst-mib", 32) << 20;
  o.duration_s = args.number("duration", 0);
  return o;
}

// Node A measures to Node B: two bench processes on this machine, orchestrated here. Localhost, hence Synthetic.
Status run_node_to_node(const cli::Args& args, BenchmarkResult& r) {
  const fs::path exe = platform::executable_dir() /
#ifdef _WIN32
                       "clusterlm-bench.exe";
#else
                       "clusterlm-bench";
#endif
  const fs::path work = fs::temp_directory_path() / ("clusterlm-bench-n2n-" + std::to_string(monotonic_ns()));
  struct Cleanup {
    fs::path p;
    ~Cleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{work};
  CLM_ASSIGN_OR_RETURN(auto id_a, transport::DeviceIdentity::load_or_generate(work / "nodeA", "nodeA"));
  CLM_ASSIGN_OR_RETURN(auto id_b, transport::DeviceIdentity::load_or_generate(work / "nodeB", "nodeB"));

  CLM_ASSIGN_OR_RETURN(auto server, platform::ChildProcess::spawn(exe, {"transport", "--serve", "127.0.0.1:0", "--identity",
                                                                        (work / "nodeB").string(), "--trust", id_a.fingerprint(),
                                                                        "--exit-on-stdin-eof"}));
  CLM_ASSIGN_OR_RETURN(auto line, server->read_until("CLUSTERLM_BENCH_SERVING", 15s));
  const auto at = line.find("endpoint=");
  if (at == std::string::npos) return make_error(ErrorCode::kProtocolError, "unexpected serve banner");
  const std::string endpoint = line.substr(at + 9, line.find(' ', at) - (at + 9));

  const fs::path out_file = work / "a-to-b.json";
  std::vector<std::string> a_args = {"transport", "--peer", endpoint, "--peer-id", id_b.fingerprint(), "--identity",
                                     (work / "nodeA").string(), "--trust", id_b.fingerprint(), "--out", out_file.string(),
                                     "--machine-id", "nodeA"};
  for (const char* flag : {"sizes", "iterations", "burst-mib", "warmup", "duration"})
    if (args.has(flag)) {
      a_args.push_back(std::string("--") + flag);
      a_args.push_back(args.get(flag));
    }
  CLM_ASSIGN_OR_RETURN(auto client, platform::ChildProcess::spawn(exe, a_args));
  auto code = client->wait(300s);
  server->close_stdin();
  server->kill();
  if (!code.is_ok()) {
    client->kill();
    return code.status();
  }
  if (code.value() != 0) return make_error(ErrorCode::kInternal, "node A measurement exited with code " + std::to_string(code.value()));
  std::ifstream f(out_file);
  const auto doc = nlohmann::json::parse(f, nullptr, false);
  if (doc.is_discarded() || !doc.contains("metrics")) return make_error(ErrorCode::kDataLoss, "node A produced no result");
  for (auto it = doc["metrics"].begin(); it != doc["metrics"].end(); ++it) r.metric("node_to_node." + it.key(), it.value());
  r.config("node_to_node_endpoint", endpoint);
  return Status::ok();
}

}  // namespace

int cmd_transport(const cli::Args& args) {
  auto specs = parse_peers(args);
  if (!specs.is_ok()) {
    std::fprintf(stderr, "%s\n", specs.status().to_string().c_str());
    return 2;
  }
  const bool remote = !specs->empty();
  const bool tls = (args.has("tls") || remote || args.has("serve") || args.has("node-to-node")) && !args.has("no-tls");
  auto sec = security_from(args, tls);
  if (!sec.is_ok()) {
    std::fprintf(stderr, "%s\n", sec.status().to_string().c_str());
    return 1;
  }

  if (args.has("serve")) {
    auto ep = transport::Endpoint::parse(args.get("serve"));
    if (!ep.is_ok()) return 2;
    auto listener = transport::listen(ep.value(), sec.value());
    if (!listener.is_ok()) {
      std::fprintf(stderr, "%s\n", listener.status().to_string().c_str());
      return 1;
    }
    std::printf("CLUSTERLM_BENCH_SERVING endpoint=%s device_id=%s\n", listener.value()->local_endpoint().str().c_str(),
                sec->identity ? sec->identity->fingerprint().c_str() : "insecure");
    std::fflush(stdout);
    if (args.has("exit-on-stdin-eof")) {
      // Orchestrated mode: the parent holds our stdin; when it closes (or the parent dies) we exit.
      std::thread([] {
        char buf[64];
        while (std::fgets(buf, sizeof buf, stdin)) {}
        std::_Exit(0);
      }).detach();
    }
    while (true) {
      auto c = listener.value()->accept(1000ms);
      if (c.is_ok()) std::thread(serve_connection, std::move(c).value()).detach();
    }
  }

  Stopwatch total;
  const RunContext ctx = make_run_context(args);
  BenchmarkResult result(experiment_id(args, ctx, "HQ-NET-01", "dev-transport"), probe_host());
  apply_run_context(result, ctx);
  const LinkBenchOptions lopt = link_options(args);
  result.config("tls", tls);
  result.config("iterations", lopt.iterations);
  result.config("warmup", lopt.warmup);
  result.config("sizes", lopt.sizes);
  result.config("burst_bytes", lopt.burst_bytes);

  if (args.has("node-to-node")) {
    result.mark_simulated("localhost_cluster", true);
    auto st = run_node_to_node(args, result);
    result.check("node_to_node_measured", st.is_ok(), st.to_string());
    result.pending("HQ-NET-01");
    return emit(args, result, total.elapsed_ms() / 1000.0);
  }

  // Targets: real peers, or an in-process loopback server with N connections.
  std::unique_ptr<transport::Listener> listener;
  std::thread server;
  std::atomic<bool> stop{false};
  std::vector<PeerSpec> peers = specs.value();
  bool any_loopback_peer = false;
  for (const auto& p : peers) any_loopback_peer |= p.endpoint.is_loopback();
  if (remote) {
    result.set_measured();
    // A peer on this machine is not a link between machines: the result is Synthetic and the profile quantities too.
    if (any_loopback_peer) result.mark_simulated("loopback", true);
  } else {
    auto server_sec = sec.value();
    if (tls) server_sec.trusted_peers.push_back(server_sec.identity->fingerprint());
    auto l = transport::listen({"127.0.0.1", 0}, server_sec);
    if (!l.is_ok()) {
      std::fprintf(stderr, "%s\n", l.status().to_string().c_str());
      return 1;
    }
    listener = std::move(l).value();
    server = std::thread([&] {
      while (!stop.load()) {
        auto c = listener->accept(100ms);
        if (c.is_ok()) std::thread(serve_connection, std::move(c).value()).detach();
      }
    });
    if (tls) sec->trusted_peers.push_back(sec->identity->fingerprint());
    const auto n = static_cast<std::size_t>(args.integer("simulate-nodes", 1));
    for (std::size_t i = 0; i < n; ++i) peers.push_back({"sim-node" + std::to_string(i), listener->local_endpoint(), std::nullopt});
    result.mark_simulated("loopback", true);
  }
  if (tls && remote)
    for (const auto& p : peers)
      if (p.expected_id) sec->trusted_peers.push_back(*p.expected_id);

  std::optional<transport::NetworkConditions> preset;
  if (args.has("impair")) {
    auto pr = transport::network_preset(args.get("impair"));
    if (!pr.is_ok()) return 2;
    preset = pr.value();
    result.mark_simulated("network_preset", args.get("impair"));
  }

  std::vector<std::unique_ptr<transport::Connection>> conns;
  for (const auto& p : peers) {
    auto conn = transport::connect(p.endpoint, sec.value(), p.expected_id, 10s);
    if (!conn.is_ok()) {
      std::fprintf(stderr, "connect %s: %s\n", p.endpoint.str().c_str(), conn.status().to_string().c_str());
      stop.store(true);
      if (server.joinable()) server.join();
      return 1;
    }
    std::unique_ptr<transport::Connection> c = std::move(conn).value();
    if (preset) c = transport::impair(std::move(c), *preset);
    conns.push_back(std::move(c));
  }
  result.config("peers", [&] {
    std::vector<std::string> v;
    v.reserve(peers.size());
    for (const auto& p : peers) v.push_back(p.name + "=" + p.endpoint.str());
    return v;
  }());

  bool ok = true;
  std::vector<PeerLink> links;
  for (std::size_t i = 0; i < conns.size(); ++i) {
    auto m = measure_link(*conns[i], lopt);
    if (!m.is_ok()) {
      result.check("link_" + peers[i].name, false, m.status().to_string());
      ok = false;
      continue;
    }
    emit_link_metrics(result, conns.size() == 1 ? "" : "link." + peers[i].name + ".", m.value());
    PeerLink pl;
    pl.peer_id = peers[i].name;
    pl.measure = std::move(m).value();
    pl.real_link = remote && !preset && !peers[i].endpoint.is_loopback();
    links.push_back(std::move(pl));
  }
  std::optional<ConcurrentMeasure> concurrent;
  if (ok && conns.size() >= 2) {
    std::vector<transport::Connection*> raw;
    raw.reserve(conns.size());
    for (auto& c : conns) raw.push_back(c.get());
    const std::uint32_t big = *std::max_element(lopt.sizes.begin(), lopt.sizes.end());
    auto cm = measure_concurrent(raw, big, lopt.burst_bytes);
    if (cm.is_ok()) {
      emit_concurrent_metrics(result, "", cm.value());
      concurrent = cm.value();
    } else {
      result.check("concurrent_peers", false, cm.status().to_string());
      ok = false;
    }
  }
  result.check("all_links_completed", ok);
  result.config("meets_min_repetitions", lopt.iterations >= 5);  // qualification runs need >= 5 RTT samples per size
  if (!(remote && !preset && !any_loopback_peer)) result.pending("HQ-NET-01");
  if (!remote || conns.size() < 2) result.pending("HQ-NET-02");

  if (args.has("network-out") && !links.empty()) {
    const auto profile = build_network_profile(ctx.machine_id, ctx.run_id, links, concurrent ? &*concurrent : nullptr);
    auto st = placement::save_network_profile(profile, args.get("network-out"));
    result.check("network_profile_written", st.is_ok(), st.to_string());
    result.metric("network_profile.weakest_provenance", std::string(placement::to_string(placement::weakest_provenance(profile))));
  }

  for (auto& c : conns) c->close();
  stop.store(true);
  if (server.joinable()) server.join();
  return emit(args, result, total.elapsed_ms() / 1000.0);
}

Status measure_remote_peers(const cli::Args& args, BenchmarkResult& result, std::vector<PeerLink>& links,
                            std::optional<ConcurrentMeasure>& concurrent) {
  CLM_ASSIGN_OR_RETURN(auto peers, parse_peers(args));
  if (peers.empty()) return make_error(ErrorCode::kInvalidArgument, "no --peer given");
  const bool tls = !args.has("no-tls");
  CLM_ASSIGN_OR_RETURN(auto sec, security_from(args, tls));
  if (tls)
    for (const auto& p : peers)
      if (p.expected_id) sec.trusted_peers.push_back(*p.expected_id);
  const LinkBenchOptions lopt = link_options(args);
  std::vector<std::unique_ptr<transport::Connection>> conns;
  for (const auto& p : peers) {
    CLM_ASSIGN_OR_RETURN(auto c, transport::connect(p.endpoint, sec, p.expected_id, 10s));
    conns.push_back(std::move(c));
  }
  for (std::size_t i = 0; i < conns.size(); ++i) {
    CLM_ASSIGN_OR_RETURN(auto m, measure_link(*conns[i], lopt));
    emit_link_metrics(result, "network.link." + peers[i].name + ".", m);
    PeerLink pl;
    pl.peer_id = peers[i].name;
    pl.measure = std::move(m);
    pl.real_link = !peers[i].endpoint.is_loopback();
    if (!pl.real_link) result.mark_simulated("loopback", true);
    links.push_back(std::move(pl));
  }
  if (conns.size() >= 2) {
    std::vector<transport::Connection*> raw;
    raw.reserve(conns.size());
    for (auto& c : conns) raw.push_back(c.get());
    const std::uint32_t big = *std::max_element(lopt.sizes.begin(), lopt.sizes.end());
    CLM_ASSIGN_OR_RETURN(auto cm, measure_concurrent(raw, big, lopt.burst_bytes));
    emit_concurrent_metrics(result, "network.", cm);
    concurrent = cm;
  }
  for (auto& c : conns) c->close();
  return Status::ok();
}

}  // namespace clusterlm::bench
