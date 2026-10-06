#pragma once
// (doctest-free core, shared with tests/fuzz/gen_gguf_seeds.cpp)
// Miniature "Flash-Next-shaped" GGUF models for tests: Flash-Next tensor names and metadata keys (arch
// "qwen4exp"), 3:1 linear:full attention layer pattern, stacked iq3_s experts, shared expert, PLE block on layer 1,
// gated-residual tensors, MTP tensors, Father-only tables. Tensor payloads are random bytes of the exact size.
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "clusterlm/objects/gguf.hpp"
#include "clusterlm/objects/gguf_writer.hpp"

namespace clusterlm::testutil {

using namespace clusterlm::objects;

struct MiniSpec {
  std::uint32_t layers = 4;       // pattern R R R A (every 4th layer is full attention)
  std::uint32_t hidden = 256;
  std::uint32_t experts = 4;
  std::uint32_t active = 2;
  std::uint32_t ff = 256;         // multiple of 256 so iq3_s rows are whole blocks
  std::uint32_t shared_ff = 64;   // 0 = no shared expert
  std::uint32_t hc = 4;
  std::uint32_t heads = 4, kv_heads = 2, head_dim = 64;
  std::uint32_t vocab = 512;
  std::uint32_t ple_rows = 1000;
  std::uint32_t ple_layer = 1;
  GgmlType gate_type = GgmlType::kIQ3_S, up_type = GgmlType::kIQ3_S, down_type = GgmlType::kIQ3_S;
  // Optional per-layer override of the down type (mixed-type experts).
  std::function<GgmlType(std::uint32_t layer, const char* which)> type_for;
  bool fused_gate_up = false;     // emit ffn_gate_up_exps instead of gate + up
  bool with_mtp = true;
  bool split = false;             // PLE lookup in a second shard
  std::uint64_t seed = 7;
  std::uint32_t alignment = 32;
};

struct MiniFiles {
  std::vector<std::filesystem::path> paths;
};

inline GgmlType mini_type(const MiniSpec& s, std::uint32_t L, const char* which, GgmlType dflt) {
  return s.type_for ? s.type_for(L, which) : dflt;
}

// Hook to add or alter tensors/metadata before writing (negative tests).
struct MiniHooks {
  std::function<void(GgufWriter&)> extra_main;
};

inline Status build_mini(const MiniSpec& s, GgufWriter& main, GgufWriter& lookup, const MiniHooks& hooks = {}) {
  const std::uint64_t H = s.hidden, E = s.experts, ff = s.ff, qd = std::uint64_t{s.heads} * s.head_dim, kvd = std::uint64_t{s.kv_heads} * s.head_dim;
  main.set_alignment(s.alignment);
  lookup.set_alignment(s.alignment);
  for (GgufWriter* w : {&main, &lookup}) {
    w->add_string("general.architecture", "qwen4exp");
    w->add_string("general.name", "mini-flash-next");
    w->add_u32("qwen4exp.block_count", s.layers);
    w->add_u32("qwen4exp.embedding_length", s.hidden);
    w->add_u32("qwen4exp.expert_count", s.experts);
    w->add_u32("qwen4exp.expert_used_count", s.active);
    w->add_u32("qwen4exp.attention.head_count", s.heads);
    w->add_u32("qwen4exp.attention.head_count_kv", s.kv_heads);
    w->add_u32("qwen4exp.ple.ngram_size", 3);
  }
  auto rnd = [&](GgufWriter& w, const std::string& name, std::vector<std::uint64_t> dims, GgmlType t) {
    return w.add_random_tensor(name, std::move(dims), t, s.seed);
  };
  CLM_RETURN_IF_ERROR(rnd(main, "token_embd.weight", {H, s.vocab}, GgmlType::kQ8_0));
  GgufWriter& ple_w = s.split ? lookup : main;
  CLM_RETURN_IF_ERROR(rnd(ple_w, "per_layer_token_embd.weight", {160, s.ple_rows}, GgmlType::kI8));
  CLM_RETURN_IF_ERROR(rnd(main, "output_norm.weight", {H}, GgmlType::kF32));
  CLM_RETURN_IF_ERROR(rnd(main, "output.weight", {H, s.vocab}, GgmlType::kQ6_K));
  CLM_RETURN_IF_ERROR(rnd(main, "output_hc_up.weight", {std::uint64_t{s.hc} * H, 32}, GgmlType::kF32));
  CLM_RETURN_IF_ERROR(rnd(main, "output_hc_down.weight", {32, std::uint64_t{s.hc} * H}, GgmlType::kF32));
  CLM_RETURN_IF_ERROR(rnd(main, "output_hc_norm.weight", {std::uint64_t{s.hc} * H}, GgmlType::kF32));
  if (s.with_mtp) {
    CLM_RETURN_IF_ERROR(rnd(main, "mtp.fc.weight", {2 * H, H}, GgmlType::kBF16));
    CLM_RETURN_IF_ERROR(rnd(main, "mtp.layers.0.mlp.experts.gate_up_proj", {H, 2 * ff, E}, GgmlType::kIQ2_XS));
    CLM_RETURN_IF_ERROR(rnd(main, "mtp.layers.0.mlp.experts.down_proj", {ff, H, E}, GgmlType::kIQ2_XS));
  }
  for (std::uint32_t L = 0; L < s.layers; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    const bool qsa = L % 4 == 3;
    CLM_RETURN_IF_ERROR(rnd(main, p + "hc_attn_down.weight", {std::uint64_t{s.hc} * H, 32}, GgmlType::kF32));
    CLM_RETURN_IF_ERROR(rnd(main, p + "hc_attn_up.weight", {32, std::uint64_t{s.hc} * H}, GgmlType::kF32));
    CLM_RETURN_IF_ERROR(rnd(main, p + "hc_attn_inject.weight", {std::uint64_t{s.hc} * H, s.hc}, GgmlType::kF32));
    CLM_RETURN_IF_ERROR(rnd(main, p + "hc_ffn_inject.weight", {std::uint64_t{s.hc} * H, s.hc}, GgmlType::kF32));
    CLM_RETURN_IF_ERROR(rnd(main, p + "hc_attn_norm.weight", {std::uint64_t{s.hc} * H}, GgmlType::kF32));
    CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_gate_inp.weight", {H, E}, GgmlType::kF32));
    if (qsa) {
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_q.weight", {H, 2 * qd}, GgmlType::kQ4_K));
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_k.weight", {H, kvd}, GgmlType::kQ4_K));
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_v.weight", {H, kvd}, GgmlType::kQ4_K));
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_output.weight", {qd, H}, GgmlType::kQ4_K));
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_q_norm.weight", {s.head_dim}, GgmlType::kF32));
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_k_norm.weight", {s.head_dim}, GgmlType::kF32));
      CLM_RETURN_IF_ERROR(rnd(main, p + "indexer.q_proj.weight", {H, 4 * 128}, GgmlType::kQ4_K));
      CLM_RETURN_IF_ERROR(rnd(main, p + "indexer.k_proj.weight", {H, 128}, GgmlType::kQ4_K));
    } else {
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_qkv.weight", {H, 512}, GgmlType::kQ8_0));
      CLM_RETURN_IF_ERROR(rnd(main, p + "attn_gate.weight", {H, 256}, GgmlType::kQ8_K));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ssm_out.weight", {256, H}, GgmlType::kQ8_K));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ssm_alpha.weight", {H, 8}, GgmlType::kBF16));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ssm_a", {8}, GgmlType::kF32));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ssm_dt.bias", {8}, GgmlType::kF32));
    }
    if (L == s.ple_layer) {
      CLM_RETURN_IF_ERROR(rnd(main, p + "ple_key.weight", {H, 160}, GgmlType::kIQ4_XS));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ple_value.weight", {H, H}, GgmlType::kIQ4_XS));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ple_conv1d.weight", {4, 160}, GgmlType::kF32));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ple_norm_query.weight", {160}, GgmlType::kF32));
    }
    if (s.shared_ff > 0) {
      CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_gate_shexp.weight", {H, s.shared_ff}, GgmlType::kQ8_0));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_up_shexp.weight", {H, s.shared_ff}, GgmlType::kQ8_0));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_down_shexp.weight", {s.shared_ff, H}, GgmlType::kQ8_0));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_gate_inp_shexp.weight", {H}, GgmlType::kBF16));
    }
    const GgmlType gt = mini_type(s, L, "gate", s.gate_type), ut = mini_type(s, L, "up", s.up_type), dt = mini_type(s, L, "down", s.down_type);
    if (s.fused_gate_up) {
      CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_gate_up_exps.weight", {H, 2 * ff, E}, gt));
    } else {
      CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_gate_exps.weight", {H, ff, E}, gt));
      CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_up_exps.weight", {H, ff, E}, ut));
    }
    CLM_RETURN_IF_ERROR(rnd(main, p + "ffn_down_exps.weight", {ff, H, E}, dt));
  }
  if (hooks.extra_main) hooks.extra_main(main);
  if (s.split) {
    const auto total = static_cast<std::int32_t>(main.tensor_count() + lookup.tensor_count());
    main.add_u16("split.no", 0);
    main.add_u16("split.count", 2);
    main.add_i32("split.tensors.count", total);
    lookup.add_u16("split.no", 1);
    lookup.add_u16("split.count", 2);
    lookup.add_i32("split.tensors.count", total);
  }
  return Status::ok();
}

}  // namespace clusterlm::testutil
