#pragma once
// Placement search: admission -> cost model -> candidate ranking.
//
// Pipeline shape searched: Father prefix -> [Node A -> [Node B ->]] Father tail, cut at `granularity` layers,
// plus the Father-only plan. Every decision is a function of measured profiles and model inputs; no device
// name is ever consulted. Every output states its provenance, which is the weakest of every input used.
// The search does NOT claim optimality on real hardware: it ranks candidates under the stated cost model.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/digest.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/placement/model_inputs.hpp"
#include "clusterlm/placement/profile.hpp"
#include "clusterlm/placement/provenance.hpp"

namespace clusterlm::placement {

struct LayerRange {
  std::uint32_t begin = 0, end = 0;  // [begin, end)
  std::uint32_t size() const { return end - begin; }
  friend bool operator==(const LayerRange&, const LayerRange&) = default;
};

// How long a node is expected to stay leased, which decides how much of its preparation cost one request
// should bear.
struct LeaseExpectation {
  double expected_lease_s = 3600;
  double expected_tokens = 100000;
};

// Objective (lower is better):
//   score = preparation * (effective_prepare_s + overrun_penalty * lease_overrun_s)
//         + throughput  * (prefill_s + output_tokens / decode_tok_s)
struct ObjectiveWeights {
  double throughput = 1.0;
  double preparation = 1.0;
  double lease_overrun_penalty = 10.0;  // seconds of score per second a node's prep exceeds its lease
};

struct PlacementRequest {
  HardwareProfile father;
  std::vector<HardwareProfile> nodes;  // 0..N
  NetworkProfile network;
  ModelCostInputs model;

  std::uint32_t context_tokens = 4096;  // also the prompt length to prefill
  std::uint32_t q = 1;                  // speculative verify width
  Quantity acceptance = Quantity::synthetic(1.0);  // A: mean emitted tokens per round
  std::uint32_t output_tokens = 256;    // tokens one request generates (for T_request and lease share)
  std::uint32_t prefill_chunk = 512;
  std::uint32_t granularity = 4;        // cut points are multiples of this from the prefix end
  bool allow_node_orders = true;        // try all node permutations (false: request order only)
  std::vector<std::uint32_t> prefix_layers_options{4};
  // Fraction of min(cpu, dense+gpu) time hidden by overlap. 0 = conservative serial execution.
  double overlap_factor = 0.0;

  LeaseExpectation default_lease;
  std::map<std::string, LeaseExpectation> node_lease;  // by node id
  // Layers a node already holds in its lease store (objects reusable without transfer), by node id.
  std::map<std::string, std::vector<LayerRange>> already_provisioned;
  ObjectiveWeights weights;
};

enum class StageRole : std::uint8_t { kPrefix, kMiddle, kTail, kFull /* Father-only plan */ };
std::string_view to_string(StageRole r) noexcept;

struct PlanStage {
  std::string domain_id;
  StageRole role = StageRole::kPrefix;
  LayerRange layers;
  friend bool operator==(const PlanStage&, const PlanStage&) = default;
};

// Per-domain memory ledger (bytes). VRAM = dense + gpu_experts + state + scratch_vram (a domain without a
// usable GPU keeps dense+state in RAM instead). RAM = cpu_experts + staging + os_reserve + father_only
// (+ dense + state when no GPU).
struct MemoryLedger {
  std::string domain_id;
  std::uint64_t dense = 0, gpu_experts = 0, cpu_experts = 0, state = 0, scratch_vram = 0, staging_ram = 0;
  std::uint64_t os_reserve_ram = 0, father_only = 0;
  bool dense_on_gpu = true;
  std::uint64_t vram_budget = 0, ram_safe_allowance = 0;  // limits the ledger was admitted against

  std::uint64_t vram_used() const { return (dense_on_gpu ? dense + state : 0) + gpu_experts + scratch_vram; }
  std::uint64_t ram_used() const {
    return cpu_experts + staging_ram + os_reserve_ram + father_only + (dense_on_gpu ? 0 : dense + state);
  }
};

struct ExpertResidency {
  std::string domain_id;
  std::uint64_t gpu_expert_count = 0, cpu_expert_count = 0;
  // gpu_experts_by_layer[absolute layer] = ids of GPU-resident experts (empty for layers not hosted).
  std::vector<std::vector<std::uint16_t>> gpu_experts_by_layer;
};

struct StageMetrics {
  std::string domain_id;
  StageRole role = StageRole::kPrefix;
  LayerRange layers;
  double dense_ms = 0;                 // sum dense_ms(kind) * (1 + dense_q_scaling*(q-1))
  double cpu_ms = 0;                   // CPU-resident expert union bytes / (cpu_bw*sustained) * (1+q_scaling*(q-1))
  double gpu_ms = 0;                   // GPU-resident expert union bytes / gpu_bw
  double stage_ms = 0;                 // dense + cpu + gpu - overlap_factor*min(cpu, dense+gpu)
  double cpu_miss_bytes_expected = 0;  // expected CPU-served expert bytes per round (q-position union)
  double gpu_hit_bytes_expected = 0;
  double prefill_chunk_ms = 0;
};

struct TransportMetrics {
  std::string from, to;
  double bytes_per_round = 0;  // q * boundary bytes
  double ms = 0;               // bytes/bw + rtt/2
  double prefill_chunk_ms = 0;
};

struct PredictedMetrics {
  std::vector<StageMetrics> stages;
  std::vector<TransportMetrics> transports;
  double draft_ms = 0;
  double stage_ms_total = 0, transport_ms_total = 0, commit_ms = 0;  // commit = max RTT Father<->nodes
  double round_ms = 0;        // draft + stages + transports + commit
  double decode_tok_s = 0;    // 1000 * A / round_ms
  double transport_bytes_per_round = 0;
  double prefill_s = 0;       // pipelined chunks: first-chunk latency + (n-1) * slowest stage/link
  double provisioning_bytes = 0;  // new bytes Father must send to nodes
  double provisioning_s = 0;      // max(sum bytes / father_egress, max_node bytes / link bw): NOT parallel-additive
  double upload_s = 0;            // slowest domain's host->GPU upload of dense + GPU experts
  double prepare_s = 0;           // provisioning_s + upload_s
  double effective_prepare_s = 0; // prepare amortised by each node's expected lease (see ObjectiveWeights)
  double lease_overrun_s = 0;     // sum over nodes of max(0, node prep - expected lease)
  double t_request_s = 0;         // prepare + prefill + output_tokens / decode_tok_s
  double objective = 0;           // lower is better
  std::vector<std::string> flags; // e.g. "lease_overrun:<node>"
};

struct PlacementPlan {
  std::string key;  // human-readable, e.g. "F[0,4)>Node-A[4,20)>F[20,48)"
  std::vector<PlanStage> stages;
  std::vector<MemoryLedger> ledgers;      // one per domain, pipeline order
  std::vector<ExpertResidency> residency; // one per domain, pipeline order
  PredictedMetrics metrics;
  // Weakest provenance of every input quantity of the participating domains, links, model and acceptance.
  Provenance provenance = Provenance::kSynthetic;
  std::vector<std::string> non_qualified_inputs;  // which inputs hold the plan below Qualified
  Digest256 plan_hash;  // SHA-256 over the canonical assignment encoding (not over floating-point metrics)
  std::vector<std::string> node_ids() const;
};

struct RejectedPlan {
  std::string key;
  std::vector<PlanStage> stages;
  std::vector<std::string> reasons;
};

struct PlanEvaluation {
  std::optional<PlacementPlan> plan;  // set iff admitted
  std::vector<std::string> reasons;   // rejection reasons when !plan
  std::string key;
};

struct PlacementResult {
  std::vector<PlacementPlan> candidates;  // feasible, sorted by objective (ties by plan_hash)
  std::vector<RejectedPlan> rejected;
  std::vector<std::size_t> pareto;        // indices into candidates, prepare_s ascending (decode_tok_s ascending too)
  std::optional<std::size_t> best_throughput, best_preparation, recommended;
  Provenance provenance = Provenance::kSynthetic;  // weakest over the request's inputs

  const PlacementPlan* find(const std::string& key) const;
  const RejectedPlan* find_rejected(const std::string& key) const;
};

// Admits and costs one explicit stage assignment (tests, re-evaluation of a stored plan).
PlanEvaluation evaluate_plan(const PlacementRequest& request, const std::vector<PlanStage>& stages);

// Enumerates Father-only, single-node and two-node plans. Fails only for an invalid request.
Result<PlacementResult> search_placements(const PlacementRequest& request);

// Fails for any plan with a non-Qualified input. Synthetic or merely Measured plans must never be presented
// as qualified decisions.
Status require_qualified(const PlacementPlan& plan);

// Canonical key for a stage list.
std::string plan_key(const std::vector<PlanStage>& stages);

// JSON report; provenance is stated at the top level and on every plan.
std::string plan_to_json(const PlacementPlan& plan, bool include_expert_lists = false, int indent = 2);
std::string report_to_json(const PlacementResult& result, bool include_expert_lists = false, int indent = 2);

}  // namespace clusterlm::placement
