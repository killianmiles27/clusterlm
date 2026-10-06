#pragma once
// The model-side inputs of the placement cost model. Deliberately independent of runtime/objects: an
// integrator builds a ModelCostInputs from a real ModelManifest; tests and planning use the estimates below.
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/placement/provenance.hpp"

namespace clusterlm::placement {

enum class LayerKind : std::uint8_t { kRecurrent = 0, kAttention = 1 };
std::string_view to_string(LayerKind k) noexcept;  // "recurrent" / "attention" (profile dense_layer_ms keys)

struct LayerCost {
  LayerKind kind = LayerKind::kRecurrent;
  std::uint64_t dense_bytes = 0;        // attention/recurrent projections, router, shared experts
  std::string quant;                    // quant type of this layer's routed experts
  std::uint64_t expert_bytes = 0;       // bytes of ONE routed expert of this layer
  std::uint64_t fixed_state_bytes = 0;  // context-independent recurrent state
};

struct FatherOnlyBytes {
  std::uint64_t embedding = 0, head = 0, mtp = 0, lookup_working_set = 0;
  std::uint64_t total() const { return embedding + head + mtp + lookup_working_set; }
};

struct ModelCostInputs {
  std::string name;
  std::vector<LayerCost> layers;
  std::uint32_t n_experts = 0, n_active = 0;
  // routing_freq[layer][expert] = probability the expert is selected for ONE position; each row sums to
  // n_active. Aggregate frequencies only, never routing sequences (privacy).
  std::vector<std::vector<double>> routing_freq;
  // Sequence-state bytes per context token, indexed by LayerKind.
  std::array<std::uint64_t, 2> state_bytes_per_context_token{};
  std::uint64_t boundary_bytes_per_position = 0;
  // Highest layer index that consumes the per-layer-embedding lookup and therefore must stay
  // Father-resident (-1 = none). Placement keeps layers [0, ple_layer] in the Father prefix.
  std::int32_t ple_layer = -1;
  FatherOnlyBytes father_only;
  Quantity draft_ms;  // everything Father does per round outside layer stages: embed, MTP draft, head, sampling
  // Provenance of the structural numbers above (bytes, frequencies).
  Provenance provenance = Provenance::kSynthetic;
  std::string source;

  std::uint32_t n_layers() const { return static_cast<std::uint32_t>(layers.size()); }
};

// Structural validation (rows sum to n_active, sizes consistent).
Status validate(const ModelCostInputs& m);

// Published geometry of the planning target: 48 layers (3 recurrent : 1 attention), 512 experts, 10 active,
// ~2.03 MB per expert, 51,216 B boundary per position, ple_layer 2. Dense sizes, state sizes, routing
// frequencies (Zipf-like) and draft time are SYNTHETIC guesses.
ModelCostInputs flash_next_planning_estimate();

struct FixtureEstimateSpec {
  std::uint32_t n_layers = 8;
  std::uint32_t n_experts = 16;
  std::uint32_t n_active = 2;
  std::uint64_t dense_bytes = 8ull << 20;
  std::uint64_t expert_bytes = 1ull << 20;
  std::uint64_t boundary_bytes = 4096;
  double zipf_exponent = 0.8;
  std::string quant = "q4_k";
};
// Small synthetic model for unit tests (layer pattern R,R,R,A repeated).
ModelCostInputs fixture_estimate(const FixtureEstimateSpec& spec = {});

}  // namespace clusterlm::placement
