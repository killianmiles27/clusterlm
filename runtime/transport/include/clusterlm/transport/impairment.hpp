#pragma once
// Network emulation for development and benchmarks.
//
// SIMULATION ONLY: these parameters model a link; they are never a measurement of the physical network and
// must not be reported as one. Presets are synthetic development parameters.
//
// Delivery model on send: serialization (bytes / bandwidth, queued behind other traffic on the same
// SimulatedLink) + latency + U(0, jitter), with in-order delivery:
//   deliver_at = max(previous deliver_at, link_tx_done + latency + jitter_sample)
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/transport/fault_injector.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::transport {

struct NetworkConditions {
  std::string name;
  double bandwidth_bytes_per_s = 0;  // 0 = unlimited
  double latency_ms = 0;
  double jitter_ms = 0;              // extra delay uniformly drawn from [0, jitter_ms]
  std::uint64_t seed = 1;
};

Result<NetworkConditions> network_preset(std::string_view name);
std::vector<std::string> network_preset_names();

// One NIC's egress: shared by every connection that leaves through it, so concurrent senders split the
// bandwidth. Serialization is FIFO in reservation order.
class SimulatedLink {
 public:
  explicit SimulatedLink(double bandwidth_bytes_per_s) : bandwidth_(bandwidth_bytes_per_s) {}
  // Reserves transmission of `bytes` and returns when it finishes on the wire.
  std::chrono::steady_clock::time_point reserve(std::size_t bytes);
  double bandwidth_bytes_per_s() const { return bandwidth_; }

 private:
  double bandwidth_;
  std::mutex mu_;
  std::chrono::steady_clock::time_point busy_until_{};
};

// Decorates `inner`: impairments apply to send(); fault rules apply to both directions. send() enqueues into a
// bounded queue serviced by a sender thread (backpressure: blocks when full), so send errors surface
// asynchronously as kUnavailable on later calls. If `link` is null a private link at conditions.bandwidth is
// used; if given, the link's bandwidth governs serialization. close() drops frames still queued.
std::unique_ptr<Connection> impair(std::unique_ptr<Connection> inner, NetworkConditions conditions,
                                   std::shared_ptr<SimulatedLink> link = nullptr,
                                   std::shared_ptr<FaultInjector> faults = nullptr);

}  // namespace clusterlm::transport
