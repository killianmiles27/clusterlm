#pragma once
// LAN peer mode of the EXPERIMENTAL grouped expert-domain prototype (HQ-P0C-02): expert domains as separate processes
// or machines over the existing transport (mutual TLS with pinned fingerprints; insecure loopback only for tests).
//
//   domain machine i:  clusterlm-expert-domain-bench --listen 0.0.0.0:7500 --domain-index i --remote-domains R
//                          --identity DIR --trust <father-fingerprint> [--expert-kernel iq3_s]
//   Father:            clusterlm-expert-domain-bench --peer d1=HOST1:7500 --peer d2=HOST2:7500 --remote-domains R
//                          --identity DIR --trust <d1-fp> --trust <d2-fp> --presets unlimited ...
//
// Every process derives the SAME deterministic fixture model (same --seed and geometry flags) and the same expert
// ownership (contiguous ranges, or --strided), so a domain loads only the experts it owns from its own copy of the
// fixture, exactly like a plan-scoped lease would contain. There is no provisioning protocol here and no token data on
// the wire; the messages are wire.hpp's expert batches. Results stay Synthetic (fixture model).
//
// A domain process outlives its Fathers: it accepts one Father connection at a time, serves it until it closes, and
// returns to accepting (each connection restarts the window counter).
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/expert_domains/father_executor.hpp"
#include "clusterlm/expert_domains/quant_experts.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/transport/security.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::expert_domains {

struct PeerServeOptions {
  objects::FixtureSpec fixture;
  std::filesystem::path model_dir;  // existing canonical model; empty = write the fixture under work_dir
  std::filesystem::path work_dir;
  transport::Endpoint listen;
  transport::SecurityConfig security;
  std::uint32_t remote_domains = 2;
  std::uint32_t domain_index = 1;  // 1..remote_domains (owner index; Father is owner 0)
  bool strided = false;
  std::vector<std::uint32_t> shares;  // per owner, Father first; empty = equal
  ExpertKernelSpec kernel;
  Epoch epoch{1};
  std::uint32_t max_window = 8;  // must be >= the Father's window width
  const std::atomic<bool>* stop = nullptr;
  // Called once the listener is bound, before the first accept.
  std::function<void(const transport::Endpoint&, const std::string& device_id)> on_listening;
  // Return after this many Father sessions (0 = until `stop`).
  std::uint32_t max_sessions = 0;
};

// Loads this domain's experts and serves Fathers until `stop` is set or `max_sessions` sessions ended.
Status serve_peer(const PeerServeOptions& options);

struct PeerSpec {
  std::string name;
  transport::Endpoint endpoint;
  std::optional<std::string> expected_id;  // pinned device id (fingerprint); mutual TLS requires it to be trusted
};
// "NAME=HOST:PORT" or "NAME=HOST:PORT@FINGERPRINT".
Result<PeerSpec> parse_peer(const std::string& text);

// Connects to every peer (owner order = argument order). The connections carry the grouped wire messages.
Result<std::vector<RemoteLink>> connect_peers(const std::vector<PeerSpec>& peers, const transport::SecurityConfig& security,
                                              std::chrono::milliseconds timeout = std::chrono::seconds(10));

}  // namespace clusterlm::expert_domains
