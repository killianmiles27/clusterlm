#pragma once
// Deterministic fixture model: a small, redistributable, generated (never real) MoE transformer used by
// tests, the reference backend and CI. Layer kinds mimic Flash-Next's 3:1 recurrent:full-attention mix.
//
// On-disk form mirrors a GGUF-style canonical store: per-layer *stacked* expert tensors
//   blk.L.exps_gate [E][ff][H], blk.L.exps_up [E][ff][H], blk.L.exps_down [E][H][ff]
// so one routed-expert object is THREE source ranges (a slice of each stacked tensor), never one
// contiguous range. Tensor starts are 64-byte aligned. Routed experts of odd layers use "q8_0-fixture".
#include <cstdint>
#include <filesystem>
#include <string>

#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::objects {

inline constexpr std::string_view kFixtureTransformerShard = "fixture-transformer.bin";
inline constexpr std::string_view kFixtureLookupShard = "fixture-lookup.bin";

struct FixtureSpec {
  std::string family = "clusterlm-fixture";
  std::uint32_t n_layers = 16;
  std::uint32_t hidden = 64;
  std::uint32_t residual_streams = 4;
  std::uint32_t n_experts = 32;
  std::uint32_t n_active = 4;
  std::uint32_t expert_ff = 32;
  std::uint32_t shared_expert_ff = 32;
  std::uint32_t n_heads = 4;
  std::uint32_t n_kv_heads = 2;
  std::uint32_t head_dim = 16;
  std::uint32_t vocab = 256;
  std::uint32_t ple_layer = 2;
  std::uint32_t ple_ngram = 3;
  std::uint32_t ple_rows = 1024;
  std::uint32_t mtp_layers = 1;
  std::uint64_t seed = 0x434C4D31;  // "CLM1"

  // Small configuration for fast tests (8 layers, hidden 32).
  static FixtureSpec tiny();

  // Layer pattern [R,R,R,A] repeating.
  ModelGeometry geometry() const;
  Status validate() const;
};

// Float offsets of each tensor inside a layer's dense object (see manifest of the fixture for the order).
struct DenseLayout {
  std::size_t norm = 0;                      // [H]
  std::size_t w_in = 0, decay = 0, w_out = 0;  // recurrent: [H][H], [H], [H][H]
  std::size_t wq = 0, wk = 0, wv = 0, wo = 0;  // attention: [nh*hd][H], [nkv*hd][H], [nkv*hd][H], [H][nh*hd]
  std::size_t router = 0;                    // [E][H]
  std::size_t inj = 0;                       // [hc][H]
  std::size_t total = 0;                     // floats
};
DenseLayout dense_layout(const ModelGeometry& g, LayerKind kind);

// Writes fixture-transformer.bin, fixture-lookup.bin and manifest.json into `dir` (created if needed) and
// returns the manifest. Same spec + seed => byte-identical files and identical digests.
Result<ModelManifest> write_fixture_model(const FixtureSpec& spec, const std::filesystem::path& dir);

}  // namespace clusterlm::objects
