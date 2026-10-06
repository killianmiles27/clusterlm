#pragma once
// Analytic cost model for the grouped expert-domain topology versus the layer-domain pipeline, for a given
// geometry (default: the real Flash-Next geometry, 48 layers, H = 2560, two remote domains).
//
// THIS IS A CALCULATION, NOT A MEASUREMENT. Every hardware-dependent input is a Synthetic development
// estimate (the values in fixtures/profiles/*.json). The model exists to show how the cost scales with q, layers,
// bandwidth and latency, and which term dominates; HQ-P0C-01 replaces the inputs with measurements.
//
// Modelling assumptions (all stated, none hidden):
//  * Routing is uniform and independent per position: each of the K active experts of a position is a uniform
//    draw from E experts. Real routing is skewed and correlated across positions, which SHRINKS the union at
//    q > 1 (better for this topology) but concentrates load on hot experts (worse for balance).
//  * Experts are split equally over (1 + R) owners (Father + R remote domains).
//  * Expert compute is memory-bandwidth bound: each DISTINCT expert's weights are streamed once per layer per
//    window (valid while q is small), at `cpu_expert_bytes_per_s`.
//  * Both directions of a 1GbE link are independent (full duplex); Father's egress NIC serializes its R
//    requests in FIFO order; the R responses are serialized on Father's ingress NIC.
//  * Frame overhead per message = 24-byte transport header + fixed fields.
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace clusterlm::expert_domains {

struct AnalyticInputs {
  // Geometry (default = Flash-Next as described in the project brief).
  std::uint32_t layers = 48;
  std::uint32_t hidden = 2560;
  std::uint32_t residual_streams = 4;
  std::uint32_t experts = 512;
  std::uint32_t active = 10;
  std::uint32_t expert_ff = 640;
  std::uint32_t remote_domains = 2;
  std::uint32_t q = 1;
  // Weight format. IQ3_S is ~3.44 bits per weight nominally; an ASSUMPTION used only for compute-time scaling.
  double bits_per_weight = 3.4375;
  // Network (Synthetic: gige-simulated preset = 110 MB/s, 0.15 ms one-way).
  double bandwidth_bytes_per_s = 110e6;
  double one_way_latency_ms = 0.15;
  // CPU expert streaming rates (Synthetic: fixtures/profiles cpu.expert_bytes_per_s).
  double father_expert_bytes_per_s = 18e9;  // Father-4060Ti-7600 profile
  double remote_expert_bytes_per_s = 11e9;  // slower Node (3060-5600 profile): the barrier waits for the slowest
  // Layer-domain comparison: hops of one pipeline pass (Father -> A -> B -> Father with direct peer forwarding).
  std::uint32_t layer_domain_hops = 3;
};

struct AnalyticOutputs {
  // Per layer, per window of q positions.
  double expected_distinct_experts_per_domain = 0;
  double domain_participation_probability = 0;
  double request_bytes_per_domain = 0;
  double response_bytes_per_domain = 0;
  double bytes_per_layer = 0;           // all domains, both directions
  double messages_per_layer = 0;        // 2 per participating domain
  double remote_compute_ms = 0;
  double father_local_compute_ms = 0;
  double barrier_ms = 0;                // first send -> last result received (network + remote compute)
  double network_only_barrier_ms = 0;   // the same with zero compute: the RTT + serialization floor
  double exposed_wait_ms = 0;           // time Father waits beyond its own overlapped work
  // Per target pass (all layers).
  double barriers_per_pass = 0;
  double bytes_per_pass = 0;
  double messages_per_pass = 0;
  double exposed_wait_ms_per_pass = 0;
  double network_floor_ms_per_pass = 0;  // layers * network_only_barrier_ms
  // Layer-domain pipeline for the same q.
  double layer_domain_bytes_per_pass = 0;
  double layer_domain_messages_per_pass = 0;
  double layer_domain_network_ms_per_pass = 0;
};

AnalyticOutputs analytic_model(const AnalyticInputs& in);
nlohmann::json analytic_to_json(const AnalyticInputs& in, const AnalyticOutputs& out);
// Human-readable block, labelled as a calculation.
std::string analytic_report(const AnalyticInputs& in, const AnalyticOutputs& out);

}  // namespace clusterlm::expert_domains
