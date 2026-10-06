// clusterlm-bench transport: framed-transport throughput and latency.
//
//   loopback self-test (development):   clusterlm-bench transport [--impair gige-simulated] [--tls]
//   real link (HQ-NET-01), on the peer: clusterlm-bench transport --serve 0.0.0.0:7400 --identity DIR --trust FP
//                       on the other:  clusterlm-bench transport --peer HOST:7400 --identity DIR --trust FP
//
// Loopback and impaired runs are Synthetic; only --peer runs across a real link produce Measured results.
#include <atomic>
#include <cstdio>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/transport/impairment.hpp"
#include "clusterlm/transport/security.hpp"
#include "clusterlm/transport/transport.hpp"
#include "commands.hpp"

namespace clusterlm::bench {

using namespace std::chrono_literals;

namespace {

constexpr std::uint8_t kEcho = 0;     // reply with the same payload (latency)
constexpr std::uint8_t kSink = 1;     // consume (throughput)
constexpr std::uint8_t kSinkAck = 2;  // consume, then acknowledge (end of a throughput burst)

void serve_connection(std::unique_ptr<transport::Connection> c) {
  while (true) {
    auto f = c->receive(1000ms);
    if (!f.is_ok()) {
      if (f.status().code() == ErrorCode::kDeadlineExceeded) continue;
      return;
    }
    if (f->flags == kEcho) {
      if (!c->send(*f).is_ok()) return;
    } else if (f->flags == kSinkAck) {
      transport::Frame ack;
      ack.type = f->type;
      ack.flags = kSinkAck;
      ack.correlation = f->correlation;
      if (!c->send(ack).is_ok()) return;
    }
  }
}

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

struct SizeResult {
  Distribution rtt_ms;
  double bytes_per_s = 0;
};

Result<SizeResult> measure(transport::Connection& c, std::uint32_t size, int iterations, std::uint64_t burst_bytes) {
  SizeResult r;
  transport::Frame f;
  f.type = 1;
  f.payload.assign(size, 0x5A);
  for (int i = 0; i < iterations; ++i) {
    f.flags = kEcho;
    f.correlation = static_cast<std::uint64_t>(i);
    Stopwatch sw;
    CLM_RETURN_IF_ERROR(c.send(f));
    CLM_ASSIGN_OR_RETURN(auto back, c.receive(10s));
    if (back.payload.size() != size) return make_error(ErrorCode::kDataLoss, "echo size mismatch");
    r.rtt_ms.add(sw.elapsed_ms());
  }
  // One-way throughput: a burst of sink frames ending with an acknowledged frame.
  const std::uint64_t frames = std::max<std::uint64_t>(1, burst_bytes / std::max<std::uint32_t>(size, 1));
  Stopwatch sw;
  for (std::uint64_t i = 0; i < frames; ++i) {
    f.flags = i + 1 == frames ? kSinkAck : kSink;
    CLM_RETURN_IF_ERROR(c.send(f));
  }
  CLM_RETURN_IF_ERROR(c.receive(60s).status());
  r.bytes_per_s = static_cast<double>(frames * size) / (sw.elapsed_ms() / 1000.0);
  return r;
}

}  // namespace

int cmd_transport(const cli::Args& args) {
  const bool remote = args.has("peer");
  const bool tls = args.has("tls") || remote || args.has("serve");
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
    while (true) {
      auto c = listener.value()->accept(1000ms);
      if (c.is_ok()) std::thread(serve_connection, std::move(c).value()).detach();
    }
  }

  Stopwatch total;
  BenchmarkResult result(args.get("experiment", remote ? "HQ-NET-01" : "dev-transport"), probe_host());
  std::unique_ptr<transport::Listener> listener;
  std::thread server;
  std::atomic<bool> stop{false};
  transport::Endpoint target;
  if (remote) {
    auto ep = transport::Endpoint::parse(args.get("peer"));
    if (!ep.is_ok()) return 2;
    target = ep.value();
    result.set_measured();
  } else {
    // In-process loopback server sharing this process's identity (it trusts itself).
    auto server_sec = sec.value();
    if (tls) server_sec.trusted_peers.push_back(server_sec.identity->fingerprint());
    auto l = transport::listen({"127.0.0.1", 0}, server_sec);
    if (!l.is_ok()) {
      std::fprintf(stderr, "%s\n", l.status().to_string().c_str());
      return 1;
    }
    listener = std::move(l).value();
    target = listener->local_endpoint();
    server = std::thread([&] {
      while (!stop.load()) {
        auto c = listener->accept(100ms);
        if (c.is_ok()) std::thread(serve_connection, std::move(c).value()).detach();
      }
    });
    if (tls) sec->trusted_peers.push_back(sec->identity->fingerprint());
    result.mark_simulated("loopback", true);
  }

  std::optional<std::string> expected;
  if (tls && args.has("peer-id")) expected = args.get("peer-id");
  auto conn = transport::connect(target, sec.value(), expected, 10s);
  if (!conn.is_ok()) {
    std::fprintf(stderr, "connect: %s\n", conn.status().to_string().c_str());
    return 1;
  }
  std::unique_ptr<transport::Connection> c = std::move(conn).value();
  if (args.has("impair")) {
    auto preset = transport::network_preset(args.get("impair"));
    if (!preset.is_ok()) return 2;
    c = transport::impair(std::move(c), preset.value());
    result.mark_simulated("network_preset", args.get("impair"));
  }
  result.config("tls", tls);
  result.config("target", target.str());

  // Defaults: tiny control message, one Flash-Next boundary position (51,216 B), a q=4 window, 1 MiB.
  std::vector<std::uint32_t> sizes = {64, 51216, 204864, 1u << 20};
  if (args.has("sizes")) {
    sizes.clear();
    for (const auto& s : cli::split(args.get("sizes"), ',')) sizes.push_back(static_cast<std::uint32_t>(std::stoul(s)));
  }
  const int iterations = static_cast<int>(args.integer("iterations", 50));
  const std::uint64_t burst = args.integer("burst-mib", 32) << 20;
  bool ok = true;
  for (auto size : sizes) {
    auto r = measure(*c, size, iterations, burst);
    if (!r.is_ok()) {
      result.check("size_" + std::to_string(size), false, r.status().to_string());
      ok = false;
      continue;
    }
    result.metric("rtt_ms." + std::to_string(size), r->rtt_ms, "ms");
    result.metric("payload_bytes_per_s." + std::to_string(size), r->bytes_per_s);
  }
  result.check("all_sizes_completed", ok);
  if (!remote) result.pending("HQ-NET-01");
  c->close();
  stop.store(true);
  if (server.joinable()) server.join();
  return emit(args, result, total.elapsed_ms() / 1000.0);
}

}  // namespace clusterlm::bench
