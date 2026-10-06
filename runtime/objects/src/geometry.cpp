#include "clusterlm/objects/geometry.hpp"

namespace clusterlm::objects {

namespace {
constexpr std::uint32_t kMaxLayers = 1u << 16;
constexpr std::size_t kMaxFamilyLen = 256;
}  // namespace

Status ModelGeometry::validate() const {
  auto bad = [](const char* what) { return make_error(ErrorCode::kInvalidArgument, std::string("geometry: ") + what); };
  if (n_layers == 0 || n_layers > kMaxLayers) return bad("n_layers out of range");
  if (hidden_size == 0) return bad("hidden_size must be > 0");
  if (residual_streams == 0) return bad("residual_streams must be > 0");
  if (n_experts == 0 || n_active_experts == 0 || n_active_experts > n_experts)
    return bad("need 0 < n_active_experts <= n_experts");
  if (expert_ff == 0) return bad("expert_ff must be > 0");
  if (n_heads == 0 || n_kv_heads == 0 || head_dim == 0) return bad("attention dimensions must be > 0");
  if (n_heads % n_kv_heads != 0) return bad("n_heads must be a multiple of n_kv_heads");
  if (vocab_size == 0) return bad("vocab_size must be > 0");
  if (ple_layer >= n_layers) return bad("ple_layer out of range");
  if (ple_ngram == 0 || ple_ngram > 16) return bad("ple_ngram must be in [1,16]");
  if (ple_rows == 0) return bad("ple_rows must be > 0");
  if (layer_kinds.size() != n_layers) return bad("layer_kinds.size() != n_layers");
  for (LayerKind k : layer_kinds)
    if (k != LayerKind::kRecurrent && k != LayerKind::kFullAttention) return bad("unknown layer kind");
  if (family.size() > kMaxFamilyLen) return bad("family too long");
  return Status::ok();
}

void ModelGeometry::encode(ByteWriter& w) const {
  w.str(family);
  for (std::uint32_t v : {n_layers, hidden_size, residual_streams, n_experts, n_active_experts, expert_ff,
                          shared_expert_ff, n_heads, n_kv_heads, head_dim, vocab_size, ple_layer, ple_ngram,
                          ple_rows, mtp_layers})
    w.u32(v);
  w.u32(static_cast<std::uint32_t>(layer_kinds.size()));
  for (LayerKind k : layer_kinds) w.u8(static_cast<std::uint8_t>(k));
}

Result<ModelGeometry> ModelGeometry::decode(ByteReader& r) {
  ModelGeometry g;
  r.str(g.family, kMaxFamilyLen);
  for (std::uint32_t* v : {&g.n_layers, &g.hidden_size, &g.residual_streams, &g.n_experts, &g.n_active_experts,
                           &g.expert_ff, &g.shared_expert_ff, &g.n_heads, &g.n_kv_heads, &g.head_dim,
                           &g.vocab_size, &g.ple_layer, &g.ple_ngram, &g.ple_rows, &g.mtp_layers})
    r.u32(*v);
  std::uint32_t n_kinds = 0;
  r.u32(n_kinds);
  if (r.ok() && n_kinds > kMaxLayers) return make_error(ErrorCode::kProtocolError, "geometry: too many layer kinds");
  if (r.ok() && r.remaining() < n_kinds) return make_error(ErrorCode::kProtocolError, "geometry: truncated");
  for (std::uint32_t i = 0; i < n_kinds && r.ok(); ++i) {
    std::uint8_t k = 0;
    r.u8(k);
    if (k > 1) return make_error(ErrorCode::kProtocolError, "geometry: bad layer kind");
    g.layer_kinds.push_back(static_cast<LayerKind>(k));
  }
  if (!r.ok()) return make_error(ErrorCode::kProtocolError, "geometry: truncated or malformed");
  CLM_RETURN_IF_ERROR(g.validate());
  return g;
}

}  // namespace clusterlm::objects
