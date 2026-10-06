#pragma once
// Model geometry: everything a domain needs to size its state, scratch and boundary buffers.
//
// Geometry is carried as data (from the model manifest) so that a different model family can supply a
// different layer mix and boundary schema later. Nothing in ClusterLM hard-codes Flash-Next dimensions
// outside fixtures/tests.
#include <cstdint>
#include <string>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"

namespace clusterlm::objects {

enum class LayerKind : std::uint8_t {
  kRecurrent = 0,      // GDN / linear attention: fixed-size recurrent + convolution state
  kFullAttention = 1,  // QSA / full attention: context-proportional KV (+ indexer) state
};

struct ModelGeometry {
  std::string family;                  // e.g. "qwen3.8-flash-next", "clusterlm-fixture"
  std::uint32_t n_layers = 0;
  std::uint32_t hidden_size = 0;       // width of one residual stream
  std::uint32_t residual_streams = 0;  // "hc": gated residual streams
  std::uint32_t n_experts = 0;         // routed experts per layer
  std::uint32_t n_active_experts = 0;  // routed experts selected per token per layer
  std::uint32_t expert_ff = 0;         // routed expert intermediate width
  std::uint32_t shared_expert_ff = 0;  // 0 = no shared expert
  std::uint32_t n_heads = 0;
  std::uint32_t n_kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t vocab_size = 0;        // Father-only material
  std::uint32_t ple_layer = 0;         // layer index whose block consumes the token-dependent PLE lookup
  std::uint32_t ple_ngram = 0;         // maximum n-gram order of the PLE lookup
  std::uint32_t ple_rows = 0;
  std::uint32_t mtp_layers = 0;
  std::vector<LayerKind> layer_kinds;  // size n_layers

  Status validate() const;
  void encode(ByteWriter& w) const;
  static Result<ModelGeometry> decode(ByteReader& r);
};

// Contiguous half-open layer range [begin, end).
struct LayerRange {
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
  std::uint32_t size() const { return end - begin; }
  bool empty() const { return end <= begin; }
  bool contains(std::uint32_t layer) const { return layer >= begin && layer < end; }
  friend bool operator==(const LayerRange&, const LayerRange&) = default;
};

}  // namespace clusterlm::objects
