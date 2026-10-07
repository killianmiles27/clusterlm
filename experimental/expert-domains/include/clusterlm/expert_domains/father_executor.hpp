#pragma once
// FatherExecutor: Father's side of the grouped expert-domain topology.
//
// Father runs the whole model: embedding, PLE, every layer's dense/mixer/router/shared-expert math, and the
// output head. Only the ROUTED experts are partitioned (ExpertAssignment; owner 0 = Father itself). For every
// layer, for all q positions of the window at once, Father
//   1. runs the mixer + router for each position (same math, same order as the reference backend),
//   2. splits the selected experts by owner and sends ONE ExpertBatch to each participating remote domain,
//   3. executes its own experts and the shared expert while the remote domains compute (overlap),
//   4. receives ONE ExpertResult per participating domain (blocking = the per-layer barrier),
//   5. combines the partial sums in a DETERMINISTIC order that depends only on the route, never on arrival
//      timing: owners ordered by their lowest selected expert id, then the shared expert.
//
// Reduction order vs the unsplit reference domain (docs/experimental/expert-domains.md, ADR 0120):
//   reference:  y = (((0 + w1 d1) + w2 d2) + ... + wK dK) + d_shared       (ascending expert id, one chain)
//   grouped:    y = (((0 + P_a) + P_b) + ...) + d_shared, P_o = (0 + w d) chain over owner o's experts
// If one owner executes every selected expert of a position the two are BITWISE identical (asserted by the
// tests). Otherwise float addition's non-associativity changes the last bits; the difference is bounded and
// tested against a documented tolerance. It cannot be removed without making the owners' work sequential.
//
// Failure model: a lost, stalled or erroring domain fails the window with an error naming the domain and marks
// the executor broken (kAborted afterwards). Nothing waits longer than `layer_timeout`.
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clusterlm/domain/execution_domain.hpp"
#include "clusterlm/expert_domains/assignment.hpp"
#include "clusterlm/expert_domains/quant_experts.hpp"
#include "clusterlm/expert_domains/wire.hpp"
#include "clusterlm/objects/provisioned.hpp"

namespace clusterlm::expert_domains {

struct FatherExecutorConfig {
  ExpertAssignment assignment;  // owner 0 = Father, owner i = remote connection i-1
  std::uint32_t max_context = 256;
  std::uint32_t max_window = 8;
  Epoch epoch;
  ExpertDecodeLimits limits;
  std::chrono::milliseconds layer_timeout{5000};  // per-layer wait for one domain's result
  // true: execute Father-owned experts + the shared expert between send and receive. false: after the results
  // arrive (ablation, shows what the overlap buys).
  bool overlap_local = true;
  // Father-owned routed experts: fixture FP32 (default) or the Strata CPU kernels on synthetic blobs.
  ExpertKernelSpec kernel;
};

struct RemoteLink {
  std::string name;
  std::unique_ptr<transport::Connection> connection;
};

// One layer of one window.
struct LayerExchange {
  std::uint32_t layer = 0;
  std::uint32_t positions = 0;
  std::uint32_t domains_participating = 0;  // remote domains that received a batch (== messages sent == received)
  std::uint64_t bytes_sent = 0;             // payload bytes Father -> domains
  std::uint64_t bytes_received = 0;         // payload bytes domains -> Father
  std::uint32_t selections = 0;             // (position, expert) pairs selected: q * K
  std::uint32_t distinct_experts = 0;       // union of selected experts over the q positions
  std::uint32_t remote_selections = 0;      // selections executed by remote domains
  std::uint64_t dense_ns = 0;               // mixer + router + finalize on Father
  std::uint64_t local_expert_ns = 0;        // Father-owned experts + shared expert
  std::uint64_t barrier_wait_ns = 0;        // Father blocked in receive AFTER its local work was done
  std::uint64_t exchange_ns = 0;            // first send -> last result received
  std::uint64_t moe_ns = 0;                 // routing split -> combined output (the whole MoE section)
  std::uint64_t remote_compute_ns_sum = 0;  // domain-reported compute, summed
  std::uint64_t remote_compute_ns_max = 0;
  std::uint64_t layer_ns = 0;
};

struct WindowMetrics {
  std::uint32_t positions = 0;
  std::vector<LayerExchange> layers;
  std::uint64_t total_ns = 0;
};

class FatherExecutor {
 public:
  // `resolver` must provide Father's objects: embedding, PLE, head, every dense + shared-expert object, and the
  // routed experts of owner 0. Remote experts are NOT resolved here (Father may not even have them resident).
  // remotes.size() + 1 must equal assignment.n_owners; every remote must own at least one expert.
  static Result<std::unique_ptr<FatherExecutor>> create(const objects::ModelManifest& manifest,
                                                        const objects::ObjectResolver& resolver,
                                                        FatherExecutorConfig config, std::vector<RemoteLink> remotes);
  ~FatherExecutor();
  FatherExecutor(const FatherExecutor&) = delete;
  FatherExecutor& operator=(const FatherExecutor&) = delete;

  // Runs one verification/prefill window of tokens.size() positions starting at committed_position().
  // Only one window may be outstanding; it is temporary until commit().
  Result<domain::Logits> run_window(std::span<const std::int32_t> tokens);
  // Keeps the first `accepted` (1..q) positions of the outstanding window, discards the rest.
  Status commit(std::uint32_t accepted);
  // Starts a fresh sequence (clears recurrent state, KV and token history). Does not repair a broken executor.
  void reset_session();

  std::uint64_t committed_position() const;
  bool broken() const;
  const WindowMetrics& last_window_metrics() const;
  // Closes every remote connection.
  void close();

  // Sequence epoch used on the wire; domains reject batches with another epoch.
  void set_epoch(Epoch epoch);

  struct Impl;

 private:
  explicit FatherExecutor(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace clusterlm::expert_domains
