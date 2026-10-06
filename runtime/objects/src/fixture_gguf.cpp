#include "clusterlm/objects/fixture_gguf.hpp"

#include <fstream>

#include "clusterlm/objects/gguf_manifest.hpp"
#include "clusterlm/objects/gguf_writer.hpp"
#include "clusterlm/objects/tensor_codec.hpp"
#include "fixture_rng.hpp"

namespace clusterlm::objects {

namespace {

using detail::TensorRng;
using detail::fan_in_scale;
using Dims = std::vector<std::uint64_t>;

Bytes f32_bytes(const std::vector<float>& v) {
  Bytes b;
  encode_f32(v, b);
  return b;
}

Status add_f32(GgufWriter& w, const std::string& name, Dims dims, const std::vector<float>& v) {
  return w.add_tensor(name, std::move(dims), GgmlType::kF32, f32_bytes(v));
}

// Stacked experts [E][rows][cols] as GGUF dims [cols, rows, E].
Status add_stacked(GgufWriter& w, const std::string& name, std::uint64_t cols, std::uint64_t rows, std::uint64_t E,
                   const std::vector<float>& v, bool q8) {
  if (!q8) return add_f32(w, name, {cols, rows, E}, v);
  Bytes b;
  CLM_RETURN_IF_ERROR(encode_q8_0_ggml(v, b));
  return w.add_tensor(name, {cols, rows, E}, GgmlType::kQ8_0, std::move(b));
}

void add_common_metadata(GgufWriter& w, const FixtureSpec& spec, const std::string& artifact) {
  const std::string a = spec.family;
  w.add_string("general.architecture", a);
  w.add_string("general.name", artifact);
  w.add_string("general.license", "CC0-1.0 (synthetic generated data, not derived from any real model)");
  w.add_u32(a + ".block_count", spec.n_layers);
  w.add_u32(a + ".embedding_length", spec.hidden);
  w.add_u32(a + ".expert_count", spec.n_experts);
  w.add_u32(a + ".expert_used_count", spec.n_active);
  w.add_u32(a + ".expert_feed_forward_length", spec.expert_ff);
  w.add_u32(a + ".attention.head_count", spec.n_heads);
  w.add_u32(a + ".attention.head_count_kv", spec.n_kv_heads);
  w.add_u32(a + ".attention.key_length", spec.head_dim);
  w.add_u32(a + ".ple.ngram_size", spec.ple_ngram);
  w.add_u32(a + ".ple.layer", spec.ple_layer);
}

void add_split(GgufWriter& w, std::uint16_t no, std::uint16_t count, std::int32_t tensors) {
  w.add_u16("split.no", no);
  w.add_u16("split.count", count);
  w.add_i32("split.tensors.count", tensors);
}

Status build(const FixtureSpec& spec, const FixtureGgufOptions& opts, const std::string& artifact, GgufWriter& main,
             GgufWriter& lookup) {
  const ModelGeometry g = spec.geometry();
  const std::uint64_t H = g.hidden_size, E = g.n_experts, ff = g.expert_ff, sff = g.shared_expert_ff, V = g.vocab_size;
  const std::uint64_t qd = std::uint64_t{g.n_heads} * g.head_dim, kvd = std::uint64_t{g.n_kv_heads} * g.head_dim;

  add_common_metadata(main, spec, artifact);
  add_common_metadata(lookup, spec, artifact);

  {  // Father-only material: token embedding, PLE table (own shard), head, MTP drafter
    TensorRng r(spec.seed, "token_embd");
    CLM_RETURN_IF_ERROR(add_f32(main, "token_embd.weight", {H, V}, r.uniform_scaled(V * H, 1.0f)));
    TensorRng p(spec.seed, "ple_lookup");
    CLM_RETURN_IF_ERROR(add_f32(lookup, "per_layer_token_embd.weight", {H, g.ple_rows}, p.uniform_scaled(std::size_t{g.ple_rows} * H, 1.0f)));
    TensorRng hr(spec.seed, kHeadObjectName);
    CLM_RETURN_IF_ERROR(add_f32(main, "output_norm.weight", {H}, hr.affine(H, 1.0f, 0.1f)));
    CLM_RETURN_IF_ERROR(add_f32(main, "output.weight", {H, V}, hr.uniform_scaled(V * H, fan_in_scale(H))));
    TensorRng mr(spec.seed, kMtpObjectName);
    CLM_RETURN_IF_ERROR(add_f32(main, "mtp.norm.weight", {H}, mr.affine(H, 1.0f, 0.1f)));
    CLM_RETURN_IF_ERROR(add_f32(main, "mtp.head.weight", {H, V}, mr.uniform_scaled(V * H, fan_in_scale(H))));
  }

  for (std::uint32_t L = 0; L < g.n_layers; ++L) {
    const std::string lp = "blk." + std::to_string(L) + ".";
    auto rng = [&](const char* t) { return TensorRng(spec.seed, lp + t); };
    // Dense tensors are written in DenseLayout order: that is the object's physical (and so concatenation) order.
    CLM_RETURN_IF_ERROR(add_f32(main, lp + "hc_attn_norm.weight", {H}, rng("norm").affine(H, 1.0f, 0.1f)));
    if (g.layer_kinds[L] == LayerKind::kRecurrent) {
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "attn_qkv.weight", {H, H}, rng("w_in").uniform_scaled(H * H, fan_in_scale(H))));
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "ssm_a", {H}, rng("decay").affine(H, 0.725f, 0.2f)));
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "ssm_out.weight", {H, H}, rng("w_out").uniform_scaled(H * H, fan_in_scale(H))));
    } else {
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "attn_q.weight", {H, qd}, rng("wq").uniform_scaled(qd * H, fan_in_scale(H))));
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "attn_k.weight", {H, kvd}, rng("wk").uniform_scaled(kvd * H, fan_in_scale(H))));
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "attn_v.weight", {H, kvd}, rng("wv").uniform_scaled(kvd * H, fan_in_scale(H))));
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "attn_output.weight", {qd, H}, rng("wo").uniform_scaled(H * qd, fan_in_scale(qd))));
    }
    CLM_RETURN_IF_ERROR(add_f32(main, lp + "ffn_gate_inp.weight", {H, E}, rng("router").uniform_scaled(E * H, 4.0f * fan_in_scale(H))));
    CLM_RETURN_IF_ERROR(add_f32(main, lp + "hc_attn_inject.weight", {H, g.residual_streams},
                                rng("inj").uniform_scaled(std::size_t{g.residual_streams} * H, fan_in_scale(H))));
    if (sff > 0) {
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "ffn_gate_shexp.weight", {H, sff}, rng("shexp_gate").uniform_scaled(sff * H, fan_in_scale(H))));
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "ffn_up_shexp.weight", {H, sff}, rng("shexp_up").uniform_scaled(sff * H, fan_in_scale(H))));
      CLM_RETURN_IF_ERROR(add_f32(main, lp + "ffn_down_shexp.weight", {sff, H}, rng("shexp_down").uniform_scaled(H * sff, fan_in_scale(sff))));
    }
    const bool q8 = opts.q8_experts && (L % 2) == 1;
    CLM_RETURN_IF_ERROR(add_stacked(main, lp + "ffn_gate_exps.weight", H, ff, E, rng("exps_gate").uniform_scaled(ff * H * E, fan_in_scale(H)), q8));
    CLM_RETURN_IF_ERROR(add_stacked(main, lp + "ffn_up_exps.weight", H, ff, E, rng("exps_up").uniform_scaled(ff * H * E, fan_in_scale(H)), q8));
    CLM_RETURN_IF_ERROR(add_stacked(main, lp + "ffn_down_exps.weight", ff, H, E, rng("exps_down").uniform_scaled(H * ff * E, fan_in_scale(ff)), q8));
  }
  add_split(main, 0, 2, static_cast<std::int32_t>(main.tensor_count() + lookup.tensor_count()));
  add_split(lookup, 1, 2, static_cast<std::int32_t>(main.tensor_count() + lookup.tensor_count()));
  return Status::ok();
}

}  // namespace

Result<ModelManifest> write_fixture_gguf(const FixtureSpec& spec, const std::filesystem::path& dir, const FixtureGgufOptions& opts) {
  CLM_RETURN_IF_ERROR(spec.validate());
  if (opts.q8_experts && spec.n_layers > 1 && (std::uint64_t{spec.expert_ff} * spec.hidden) % 32 != 0)
    return make_error(ErrorCode::kInvalidArgument, "fixture gguf: expert_ff*hidden must be a multiple of 32");
  const std::string artifact = "clusterlm-fixture-gguf/" + std::to_string(spec.seed) + "/L" + std::to_string(spec.n_layers) +
                               (opts.q8_experts ? "" : "/f32");
  GgufWriter main, lookup;
  CLM_RETURN_IF_ERROR(build(spec, opts, artifact, main, lookup));

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return make_error(ErrorCode::kUnavailable, "cannot create " + dir.string() + ": " + ec.message());
  const auto p0 = dir / std::string(kFixtureGgufShard0), p1 = dir / std::string(kFixtureGgufShard1);
  CLM_RETURN_IF_ERROR(main.write(p0));
  CLM_RETURN_IF_ERROR(lookup.write(p1));

  ManifestBuildOptions bo;
  bo.hash_objects = true;
  bo.hash_shards = true;
  bo.artifact_id = artifact;
  CLM_ASSIGN_OR_RETURN(BuiltManifest built, build_manifest({p0, p1}, bo));
  const std::string json = built.manifest.to_json();
  std::ofstream f(dir / "manifest.json", std::ios::binary | std::ios::trunc);
  f.write(json.data(), static_cast<std::streamsize>(json.size()));
  f.flush();
  if (!f) return make_error(ErrorCode::kUnavailable, "cannot write manifest.json in " + dir.string());
  return std::move(built.manifest);
}

}  // namespace clusterlm::objects
