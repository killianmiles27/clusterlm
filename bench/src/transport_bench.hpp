#pragma once
// Link measurements over the framed transport: RTT distribution and jitter per message size, throughput per
// direction, and concurrent load across several connections (Father serving two Nodes). The serving side is
// an echo/sink/source endpoint (`clusterlm-bench transport --serve`).
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/transport/transport.hpp"
#include "result.hpp"

namespace clusterlm::bench {

// Frame flags understood by the bench server.
inline constexpr std::uint8_t kFlagEcho = 0;       // reply with the same payload (latency)
inline constexpr std::uint8_t kFlagSink = 1;       // consume (throughput)
inline constexpr std::uint8_t kFlagSinkAck = 2;    // consume, then acknowledge (end of a throughput burst)
inline constexpr std::uint8_t kFlagSource = 3;     // request: server sends `count` frames of `size` bytes
inline constexpr std::uint8_t kFlagSourceEnd = 4;  // last frame of a source burst

// Default message sizes: control message, one Flash-Next boundary position (51,216 B), a q=4 window, 1 MiB.
std::vector<std::uint32_t> default_message_sizes();

// Serves one connection until it closes (echo, sink and source requests).
void serve_connection(std::unique_ptr<transport::Connection> c);

struct LinkBenchOptions {
  std::vector<std::uint32_t> sizes = default_message_sizes();
  unsigned iterations = 50;          // measured RTT samples per size (>= 5 required for qualification runs)
  unsigned warmup = 5;               // discarded RTT exchanges per size
  std::uint64_t burst_bytes = 32ull << 20;
  double duration_s = 0;             // > 0: keep sampling RTT for at least this long, split across sizes
};

struct SizeMeasure {
  std::uint32_t size = 0;
  Distribution rtt_ms;
  double tx_bytes_per_s = 0;  // this side -> peer
  double rx_bytes_per_s = 0;  // peer -> this side
  double jitter_ms() const { return rtt_ms.stddev(); }
  double tail_ms() const { return rtt_ms.percentile(0.99) - rtt_ms.percentile(0.50); }
};

struct LinkMeasure {
  std::string peer;
  std::vector<SizeMeasure> sizes;
  const SizeMeasure* smallest() const;
  const SizeMeasure* largest() const;
};

Result<LinkMeasure> measure_link(transport::Connection& c, const LinkBenchOptions& options);

struct ConcurrentMeasure {
  unsigned connections = 0;
  std::uint32_t size = 0;
  double egress_bytes_per_s = 0;   // sum of this side's sends while all peers are loaded at once
  double ingress_bytes_per_s = 0;  // sum of receives, all peers at once
};

// Bursts to / from every connection simultaneously (one thread per connection).
Result<ConcurrentMeasure> measure_concurrent(const std::vector<transport::Connection*>& conns, std::uint32_t size,
                                             std::uint64_t burst_bytes_per_conn);

void emit_link_metrics(BenchmarkResult& r, const std::string& prefix, const LinkMeasure& m);
void emit_concurrent_metrics(BenchmarkResult& r, const std::string& prefix, const ConcurrentMeasure& m);

}  // namespace clusterlm::bench
