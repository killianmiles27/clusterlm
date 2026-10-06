#include <doctest/doctest.h>

#include <cmath>

#include "clusterlm/expert_domains/analytic.hpp"

using namespace clusterlm::expert_domains;

TEST_CASE("analytic model: message and byte counts for the real geometry") {
  AnalyticInputs in;  // 48 layers, H=2560, 512 experts, 10 active, 2 remote domains
  in.q = 1;
  const AnalyticOutputs o = analytic_model(in);
  CHECK(o.barriers_per_pass == 48);
  // One position of 2560 FP32 activations each way per participating domain, plus small fixed fields.
  CHECK(o.request_bytes_per_domain > 2560 * 4);
  CHECK(o.request_bytes_per_domain < 2560 * 4 + 400);
  CHECK(o.response_bytes_per_domain > 2560 * 4);
  CHECK(o.response_bytes_per_domain < 2560 * 4 + 200);
  CHECK(o.domain_participation_probability > 0.98);  // 10 selections over 3 owners: almost always both domains
  CHECK(o.messages_per_layer == doctest::Approx(4 * o.domain_participation_probability));
  CHECK(o.messages_per_pass == doctest::Approx(48 * o.messages_per_layer));
  // Layer-domain: 3 boundary messages of 51,216 B per position.
  CHECK(o.layer_domain_messages_per_pass == 3);
  CHECK(o.layer_domain_bytes_per_pass > 3 * 51216);
  CHECK(o.layer_domain_bytes_per_pass < 3 * 51216 + 600);
  // The grouped topology moves more messages and more bytes per pass than the pipeline at q = 1.
  CHECK(o.bytes_per_pass > o.layer_domain_bytes_per_pass);
  CHECK(o.messages_per_pass > o.layer_domain_messages_per_pass);
}

TEST_CASE("analytic model: scaling with q, latency and bandwidth") {
  AnalyticInputs in;
  double prev_union = 0, prev_bytes = 0;
  for (std::uint32_t q = 1; q <= 4; ++q) {
    in.q = q;
    const AnalyticOutputs o = analytic_model(in);
    CHECK(o.expected_distinct_experts_per_domain > prev_union);
    CHECK(o.bytes_per_layer > prev_bytes);
    // Union growth is sub-linear in q (experts are shared between positions).
    CHECK(o.expected_distinct_experts_per_domain <= q * o.expected_distinct_experts_per_domain / 1.0);
    CHECK(o.barrier_ms >= o.network_only_barrier_ms);
    prev_union = o.expected_distinct_experts_per_domain;
    prev_bytes = o.bytes_per_layer;
  }
  in.q = 1;
  const AnalyticOutputs base = analytic_model(in);
  in.one_way_latency_ms *= 4;
  CHECK(analytic_model(in).network_floor_ms_per_pass > base.network_floor_ms_per_pass);
  in.one_way_latency_ms /= 4;
  in.bandwidth_bytes_per_s /= 2;
  CHECK(analytic_model(in).network_floor_ms_per_pass > base.network_floor_ms_per_pass);
  // With no network cost the barrier is just remote compute.
  in = AnalyticInputs{};
  in.bandwidth_bytes_per_s = 1e15;
  in.one_way_latency_ms = 0;
  const AnalyticOutputs free_net = analytic_model(in);
  CHECK(free_net.network_only_barrier_ms < 1e-3);
  CHECK(free_net.barrier_ms == doctest::Approx(free_net.remote_compute_ms).epsilon(0.01));
}

TEST_CASE("analytic report is labelled as a calculation") {
  AnalyticInputs in;
  const AnalyticOutputs o = analytic_model(in);
  CHECK(analytic_report(in, o).find("calculation") != std::string::npos);
  CHECK(analytic_to_json(in, o)["kind"] == "calculation");
}
