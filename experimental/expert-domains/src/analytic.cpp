#include "clusterlm/expert_domains/analytic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace clusterlm::expert_domains {

namespace {
constexpr double kFrameHeader = 24;
constexpr double kBatchFixed = 8 + 8 + 4 + 4 + 4;          // epoch, window, layer, q, H
constexpr double kResultTrailer = 8 + 4;                    // compute_ns, experts
constexpr double kLayerDomainFixedBytes = 128;              // RunWindow/StageResult fixed fields (upper estimate)
}  // namespace

AnalyticOutputs analytic_model(const AnalyticInputs& in) {
  AnalyticOutputs o;
  const double R = in.remote_domains, q = in.q, H = in.hidden, E = in.experts, K = in.active;
  const double s = 1.0 / (R + 1.0);
  const double p_miss = 1.0 - K / E;
  o.expected_distinct_experts_per_domain = E * s * (1.0 - std::pow(p_miss, q));
  o.domain_participation_probability = 1.0 - std::pow(1.0 - s, q * K);
  const double selections_per_domain = q * K * s;
  o.request_bytes_per_domain = kFrameHeader + kBatchFixed + q * H * 4 + q * 2 + selections_per_domain * 8;
  o.response_bytes_per_domain = kFrameHeader + kBatchFixed + q * H * 4 + kResultTrailer;
  o.messages_per_layer = 2 * R * o.domain_participation_probability;
  o.bytes_per_layer = R * o.domain_participation_probability * (o.request_bytes_per_domain + o.response_bytes_per_domain);

  const double expert_bytes = 3.0 * in.expert_ff * H * in.bits_per_weight / 8.0;
  o.remote_compute_ms = o.expected_distinct_experts_per_domain * expert_bytes / in.remote_expert_bytes_per_s * 1e3;
  // Father also runs the shared expert (one expert-equivalent) while it waits.
  o.father_local_compute_ms = (o.expected_distinct_experts_per_domain + 1.0) * expert_bytes / in.father_expert_bytes_per_s * 1e3;

  const double B = in.bandwidth_bytes_per_s, lat = in.one_way_latency_ms * 1e-3;
  auto barrier = [&](double compute_s) {
    double ingress_free = 0, last = 0;
    for (std::uint32_t i = 0; i < in.remote_domains; ++i) {
      const double request_done = (i + 1) * o.request_bytes_per_domain / B;  // Father egress, FIFO
      const double ready = request_done + lat + compute_s;
      const double end = std::max(ingress_free, ready + lat) + o.response_bytes_per_domain / B;
      ingress_free = end;
      last = end;
    }
    return last * 1e3;
  };
  o.barrier_ms = barrier(o.remote_compute_ms * 1e-3);
  o.network_only_barrier_ms = barrier(0.0);
  o.exposed_wait_ms = std::max(0.0, o.barrier_ms - o.father_local_compute_ms);

  o.barriers_per_pass = in.layers;
  o.bytes_per_pass = in.layers * o.bytes_per_layer;
  o.messages_per_pass = in.layers * o.messages_per_layer;
  o.exposed_wait_ms_per_pass = in.layers * o.exposed_wait_ms;
  o.network_floor_ms_per_pass = in.layers * o.network_only_barrier_ms;

  const double hop_bytes = kFrameHeader + kLayerDomainFixedBytes + q * (4.0 * in.hidden * (in.residual_streams + 1) + 4.0 * in.residual_streams);
  o.layer_domain_messages_per_pass = in.layer_domain_hops;
  o.layer_domain_bytes_per_pass = in.layer_domain_hops * hop_bytes;
  o.layer_domain_network_ms_per_pass = in.layer_domain_hops * (lat + hop_bytes / B) * 1e3;
  return o;
}

nlohmann::json analytic_to_json(const AnalyticInputs& in, const AnalyticOutputs& o) {
  nlohmann::json j;
  j["kind"] = "calculation";
  j["note"] = "Analytic model, not a measurement. Hardware inputs are Synthetic (fixtures/profiles/*.json).";
  j["inputs"] = {{"layers", in.layers},
                 {"hidden", in.hidden},
                 {"experts", in.experts},
                 {"active", in.active},
                 {"expert_ff", in.expert_ff},
                 {"remote_domains", in.remote_domains},
                 {"q", in.q},
                 {"bits_per_weight", in.bits_per_weight},
                 {"bandwidth_bytes_per_s", in.bandwidth_bytes_per_s},
                 {"one_way_latency_ms", in.one_way_latency_ms},
                 {"father_expert_bytes_per_s", in.father_expert_bytes_per_s},
                 {"remote_expert_bytes_per_s", in.remote_expert_bytes_per_s},
                 {"layer_domain_hops", in.layer_domain_hops}};
  j["per_layer"] = {{"expected_distinct_experts_per_domain", o.expected_distinct_experts_per_domain},
                    {"domain_participation_probability", o.domain_participation_probability},
                    {"request_bytes_per_domain", o.request_bytes_per_domain},
                    {"response_bytes_per_domain", o.response_bytes_per_domain},
                    {"bytes", o.bytes_per_layer},
                    {"messages", o.messages_per_layer},
                    {"remote_compute_ms", o.remote_compute_ms},
                    {"father_local_compute_ms", o.father_local_compute_ms},
                    {"barrier_ms", o.barrier_ms},
                    {"network_only_barrier_ms", o.network_only_barrier_ms},
                    {"exposed_wait_ms", o.exposed_wait_ms}};
  j["per_pass"] = {{"barriers", o.barriers_per_pass},
                   {"bytes", o.bytes_per_pass},
                   {"messages", o.messages_per_pass},
                   {"exposed_wait_ms", o.exposed_wait_ms_per_pass},
                   {"network_floor_ms", o.network_floor_ms_per_pass}};
  j["layer_domain_per_pass"] = {{"bytes", o.layer_domain_bytes_per_pass},
                                {"messages", o.layer_domain_messages_per_pass},
                                {"network_ms", o.layer_domain_network_ms_per_pass}};
  return j;
}

std::string analytic_report(const AnalyticInputs& in, const AnalyticOutputs& o) {
  char buf[2048];
  std::snprintf(buf, sizeof buf,
                "ANALYTIC MODEL (calculation, not a measurement; hardware inputs are Synthetic)\n"
                "  geometry: %u layers, H=%u, %u experts (%u active), expert ff %u, %u remote domains, q=%u\n"
                "  link: %.0f MB/s, %.2f ms one-way\n"
                "  per layer : %.1f messages, %.0f B, %.1f distinct experts/domain, remote compute %.3f ms,\n"
                "              Father local %.3f ms, barrier %.3f ms (network floor %.3f ms), exposed wait %.3f ms\n"
                "  per pass  : %.0f barriers, %.0f messages, %.0f B, exposed wait %.2f ms, network floor %.2f ms\n"
                "  layer-domain (same q): %.0f messages, %.0f B, network %.2f ms per pass\n",
                in.layers, in.hidden, in.experts, in.active, in.expert_ff, in.remote_domains, in.q,
                in.bandwidth_bytes_per_s / 1e6, in.one_way_latency_ms, o.messages_per_layer, o.bytes_per_layer,
                o.expected_distinct_experts_per_domain, o.remote_compute_ms, o.father_local_compute_ms, o.barrier_ms,
                o.network_only_barrier_ms, o.exposed_wait_ms, o.barriers_per_pass, o.messages_per_pass,
                o.bytes_per_pass, o.exposed_wait_ms_per_pass, o.network_floor_ms_per_pass,
                o.layer_domain_messages_per_pass, o.layer_domain_bytes_per_pass, o.layer_domain_network_ms_per_pass);
  return buf;
}

}  // namespace clusterlm::expert_domains
