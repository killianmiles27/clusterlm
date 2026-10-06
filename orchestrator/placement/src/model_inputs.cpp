#include "clusterlm/placement/model_inputs.hpp"

#include <algorithm>
#include <cmath>

namespace clusterlm::placement {

std::string_view to_string(LayerKind k) noexcept { return k == LayerKind::kRecurrent ? "recurrent" : "attention"; }

namespace {

// Zipf-like routing row: p_r ∝ 1/(r+1)^s, scaled so the row sums to n_active, with every p <= 1 (a position
// selects an expert at most once). Capping is a water-fill: clamp the largest, rescale the rest, repeat.
// Expert id = (rank + rotation) % n_experts so layers do not share the same hot ids (deterministic).
std::vector<double> zipf_row(std::uint32_t n_experts, std::uint32_t n_active, double s, std::uint32_t rotation) {
  std::vector<double> by_rank(n_experts);
  for (std::uint32_t r = 0; r < n_experts; ++r) by_rank[r] = std::pow(static_cast<double>(r + 1), -s);
  std::vector<bool> capped(n_experts, false);
  double remaining = static_cast<double>(n_active);
  for (int iter = 0; iter < 64; ++iter) {
    double free_sum = 0;
    for (std::uint32_t r = 0; r < n_experts; ++r)
      if (!capped[r]) free_sum += by_rank[r];
    const double scale = remaining / free_sum;
    bool changed = false;
    for (std::uint32_t r = 0; r < n_experts; ++r) {
      if (!capped[r] && by_rank[r] * scale > 1.0) {
        capped[r] = true;
        remaining -= 1.0;
        changed = true;
      }
    }
    if (!changed) {
      std::vector<double> row(n_experts);
      for (std::uint32_t r = 0; r < n_experts; ++r)
        row[(r + rotation) % n_experts] = capped[r] ? 1.0 : by_rank[r] * scale;
      return row;
    }
  }
  // Unreachable for sane inputs (n_active < n_experts); fall back to uniform.
  return std::vector<double>(n_experts, static_cast<double>(n_active) / static_cast<double>(n_experts));
}

}  // namespace

Status validate(const ModelCostInputs& m) {
  auto err = [](std::string msg) { return make_error(ErrorCode::kInvalidArgument, "model inputs: " + std::move(msg)); };
  if (m.layers.empty()) return err("no layers");
  if (m.n_experts == 0 || m.n_active == 0 || m.n_active > m.n_experts) return err("bad expert counts");
  if (m.routing_freq.size() != m.layers.size()) return err("routing_freq rows != layers");
  for (std::size_t l = 0; l < m.layers.size(); ++l) {
    const auto& row = m.routing_freq[l];
    if (row.size() != m.n_experts) return err("routing_freq row " + std::to_string(l) + " has wrong width");
    double sum = 0;
    for (double p : row) {
      if (!(p >= 0 && p <= 1.0 + 1e-9)) return err("routing probability out of [0,1] in layer " + std::to_string(l));
      sum += p;
    }
    if (std::abs(sum - static_cast<double>(m.n_active)) > 1e-6 * static_cast<double>(m.n_active))
      return err("routing_freq row " + std::to_string(l) + " does not sum to n_active");
    if (m.layers[l].expert_bytes == 0) return err("zero expert_bytes in layer " + std::to_string(l));
    if (m.layers[l].quant.empty()) return err("empty quant in layer " + std::to_string(l));
  }
  if (m.boundary_bytes_per_position == 0) return err("zero boundary bytes");
  if (m.ple_layer >= static_cast<std::int32_t>(m.layers.size())) return err("ple_layer out of range");
  return Status::ok();
}

ModelCostInputs flash_next_planning_estimate() {
  ModelCostInputs m;
  m.name = "flash-next-planning-estimate";
  m.n_experts = 512;
  m.n_active = 10;
  // Published geometry: 48 layers, 3 recurrent : 1 attention, ~2.03 MB per expert, 51,216 B per position
  // (= (hc*H + H + hc) * 4 with hc=4, H=2560), ple_layer 2.
  // SYNTHETIC guesses: dense bytes, state sizes, routing skew, father-only sizes, draft time.
  for (std::uint32_t l = 0; l < 48; ++l) {
    LayerCost c;
    c.kind = (l % 4 == 3) ? LayerKind::kAttention : LayerKind::kRecurrent;
    c.dense_bytes = c.kind == LayerKind::kAttention ? (130ull << 20) : (90ull << 20);
    c.quant = "q4_k";
    c.expert_bytes = 2'030'000;
    c.fixed_state_bytes = c.kind == LayerKind::kRecurrent ? (8ull << 20) : 0;
    m.layers.push_back(std::move(c));
    m.routing_freq.push_back(zipf_row(m.n_experts, m.n_active, 0.8, (l * 37u) % m.n_experts));
  }
  m.state_bytes_per_context_token = {0, 4096};
  m.boundary_bytes_per_position = 51'216;
  m.batch_scratch_bytes_per_token = 512ull << 10;  // SYNTHETIC: per-token activation workspace on a GPU domain
  m.ple_layer = 2;
  m.father_only = {600ull << 20, 600ull << 20, 1500ull << 20, 512ull << 20};
  m.draft_ms = Quantity::synthetic(6.0, "synthetic: development estimate (draft + embed + head + sampling)");
  m.provenance = Provenance::kSynthetic;
  m.source = "synthetic: published geometry, guessed dense/state/routing";
  return m;
}

ModelCostInputs fixture_estimate(const FixtureEstimateSpec& s) {
  ModelCostInputs m;
  m.name = "fixture-estimate";
  m.n_experts = s.n_experts;
  m.n_active = s.n_active;
  for (std::uint32_t l = 0; l < s.n_layers; ++l) {
    LayerCost c;
    c.kind = (l % 4 == 3) ? LayerKind::kAttention : LayerKind::kRecurrent;
    c.dense_bytes = s.dense_bytes;
    c.quant = s.quant;
    c.expert_bytes = s.expert_bytes;
    c.fixed_state_bytes = c.kind == LayerKind::kRecurrent ? (1ull << 20) : 0;
    m.layers.push_back(std::move(c));
    m.routing_freq.push_back(zipf_row(s.n_experts, s.n_active, s.zipf_exponent, (l * 5u) % s.n_experts));
  }
  m.state_bytes_per_context_token = {0, 1024};
  m.boundary_bytes_per_position = s.boundary_bytes;
  m.ple_layer = -1;
  m.father_only = {1ull << 20, 1ull << 20, 0, 0};
  m.draft_ms = Quantity::synthetic(1.0);
  m.source = "synthetic: unit-test fixture";
  return m;
}

}  // namespace clusterlm::placement
