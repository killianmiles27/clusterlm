#include "transport_bench.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

#include "clusterlm/common/clock.hpp"

namespace clusterlm::bench {

using namespace std::chrono_literals;

std::vector<std::uint32_t> default_message_sizes() { return {64, 51216, 204864, 1u << 20}; }

namespace {

void put_u32(Bytes& b, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
void put_u64(Bytes& b, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
std::uint64_t get_le(const Bytes& b, std::size_t off, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(b[off + static_cast<std::size_t>(i)]) << (8 * i);
  return v;
}

constexpr std::uint16_t kType = 1;

// One-way throughput this side -> peer: a burst of sink frames ending with an acknowledged frame.
Result<double> measure_tx(transport::Connection& c, std::uint32_t size, std::uint64_t burst_bytes) {
  transport::Frame f;
  f.type = kType;
  f.payload.assign(size, 0x5A);
  const std::uint64_t frames = std::max<std::uint64_t>(1, burst_bytes / std::max<std::uint32_t>(size, 1));
  Stopwatch sw;
  for (std::uint64_t i = 0; i < frames; ++i) {
    f.flags = i + 1 == frames ? kFlagSinkAck : kFlagSink;
    f.correlation = i;
    CLM_RETURN_IF_ERROR(c.send(f));
  }
  CLM_RETURN_IF_ERROR(c.receive(60s).status());
  return static_cast<double>(frames * size) / (sw.elapsed_ms() / 1000.0);
}

// One-way throughput peer -> this side: ask the server to source a burst.
Result<double> measure_rx(transport::Connection& c, std::uint32_t size, std::uint64_t burst_bytes) {
  const std::uint64_t frames = std::max<std::uint64_t>(1, burst_bytes / std::max<std::uint32_t>(size, 1));
  transport::Frame req;
  req.type = kType;
  req.flags = kFlagSource;
  put_u32(req.payload, size);
  put_u64(req.payload, frames);
  Stopwatch sw;
  CLM_RETURN_IF_ERROR(c.send(req));
  std::uint64_t got = 0;
  while (true) {
    CLM_ASSIGN_OR_RETURN(auto f, c.receive(60s));
    got += f.payload.size();
    if (f.flags == kFlagSourceEnd) break;
  }
  return static_cast<double>(got) / (sw.elapsed_ms() / 1000.0);
}

}  // namespace

void serve_connection(std::unique_ptr<transport::Connection> c) {
  while (true) {
    auto f = c->receive(1000ms);
    if (!f.is_ok()) {
      if (f.status().code() == ErrorCode::kDeadlineExceeded) continue;
      return;
    }
    if (f->flags == kFlagEcho) {
      if (!c->send(*f).is_ok()) return;
    } else if (f->flags == kFlagSinkAck) {
      transport::Frame ack;
      ack.type = f->type;
      ack.flags = kFlagSinkAck;
      ack.correlation = f->correlation;
      if (!c->send(ack).is_ok()) return;
    } else if (f->flags == kFlagSource && f->payload.size() == 12) {
      const auto size = static_cast<std::uint32_t>(get_le(f->payload, 0, 4));
      const std::uint64_t count = get_le(f->payload, 4, 8);
      transport::Frame out;
      out.type = f->type;
      out.payload.assign(size, 0xA5);
      for (std::uint64_t i = 0; i < count; ++i) {
        out.flags = i + 1 == count ? kFlagSourceEnd : kFlagSource;
        out.correlation = i;
        if (!c->send(out).is_ok()) return;
      }
    }
  }
}

const SizeMeasure* LinkMeasure::smallest() const {
  const SizeMeasure* best = nullptr;
  for (const auto& s : sizes)
    if (!best || s.size < best->size) best = &s;
  return best;
}

const SizeMeasure* LinkMeasure::largest() const {
  const SizeMeasure* best = nullptr;
  for (const auto& s : sizes)
    if (!best || s.size > best->size) best = &s;
  return best;
}

Result<LinkMeasure> measure_link(transport::Connection& c, const LinkBenchOptions& o) {
  LinkMeasure out;
  out.peer = c.peer().address;
  const double per_size_s = o.duration_s > 0 && !o.sizes.empty() ? o.duration_s / static_cast<double>(o.sizes.size()) : 0.0;
  for (std::uint32_t size : o.sizes) {
    SizeMeasure m;
    m.size = size;
    transport::Frame f;
    f.type = kType;
    f.flags = kFlagEcho;
    f.payload.assign(size, 0x5A);
    auto exchange = [&](std::uint64_t i) -> Result<double> {
      f.correlation = i;
      Stopwatch sw;
      CLM_RETURN_IF_ERROR(c.send(f));
      CLM_ASSIGN_OR_RETURN(auto back, c.receive(10s));
      if (back.payload.size() != size) return make_error(ErrorCode::kDataLoss, "echo size mismatch");
      return sw.elapsed_ms();
    };
    std::uint64_t seq = 0;
    for (unsigned i = 0; i < o.warmup; ++i) CLM_RETURN_IF_ERROR(exchange(seq++).status());
    Stopwatch window;
    for (unsigned i = 0; i < o.iterations || window.elapsed_ms() / 1000.0 < per_size_s; ++i) {
      CLM_ASSIGN_OR_RETURN(double ms, exchange(seq++));
      m.rtt_ms.add(ms);
    }
    CLM_ASSIGN_OR_RETURN(m.tx_bytes_per_s, measure_tx(c, size, o.burst_bytes));
    CLM_ASSIGN_OR_RETURN(m.rx_bytes_per_s, measure_rx(c, size, o.burst_bytes));
    out.sizes.push_back(std::move(m));
  }
  return out;
}

Result<ConcurrentMeasure> measure_concurrent(const std::vector<transport::Connection*>& conns, std::uint32_t size,
                                             std::uint64_t burst_bytes_per_conn) {
  ConcurrentMeasure out;
  out.connections = static_cast<unsigned>(conns.size());
  out.size = size;
  if (conns.empty()) return make_error(ErrorCode::kInvalidArgument, "no connections");
  const std::uint64_t frames = std::max<std::uint64_t>(1, burst_bytes_per_conn / std::max<std::uint32_t>(size, 1));
  const double bytes_per_conn = static_cast<double>(frames * size);

  auto run_phase = [&](bool tx) -> Result<double> {
    std::atomic<bool> go{false};
    std::vector<Status> status(conns.size());
    std::vector<std::thread> pool;
    for (std::size_t i = 0; i < conns.size(); ++i)
      pool.emplace_back([&, i] {
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        auto r = tx ? measure_tx(*conns[i], size, burst_bytes_per_conn) : measure_rx(*conns[i], size, burst_bytes_per_conn);
        status[i] = r.status();
      });
    Stopwatch sw;
    go.store(true, std::memory_order_release);
    for (auto& t : pool) t.join();
    const double secs = sw.elapsed_ms() / 1000.0;
    for (const auto& s : status) CLM_RETURN_IF_ERROR(s);
    return bytes_per_conn * static_cast<double>(conns.size()) / secs;
  };
  CLM_ASSIGN_OR_RETURN(out.egress_bytes_per_s, run_phase(true));
  CLM_ASSIGN_OR_RETURN(out.ingress_bytes_per_s, run_phase(false));
  return out;
}

void emit_link_metrics(BenchmarkResult& r, const std::string& prefix, const LinkMeasure& m) {
  for (const auto& s : m.sizes) {
    const std::string k = std::to_string(s.size);
    r.metric(prefix + "rtt_ms." + k, s.rtt_ms, "ms");
    r.metric(prefix + "jitter_ms." + k, s.jitter_ms());
    r.metric(prefix + "rtt_p99_minus_p50_ms." + k, s.tail_ms());
    r.metric(prefix + "tx_bytes_per_s." + k, s.tx_bytes_per_s);
    r.metric(prefix + "rx_bytes_per_s." + k, s.rx_bytes_per_s);
    // Kept for continuity with earlier result files.
    r.metric(prefix + "payload_bytes_per_s." + k, s.tx_bytes_per_s);
  }
}

void emit_concurrent_metrics(BenchmarkResult& r, const std::string& prefix, const ConcurrentMeasure& m) {
  r.metric(prefix + "concurrent.connections", m.connections);
  r.metric(prefix + "concurrent.message_bytes", m.size);
  r.metric(prefix + "concurrent.egress_bytes_per_s", m.egress_bytes_per_s);
  r.metric(prefix + "concurrent.ingress_bytes_per_s", m.ingress_bytes_per_s);
}

}  // namespace clusterlm::bench
