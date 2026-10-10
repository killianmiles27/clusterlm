#include "clusterlm/backends/llama_tiny_model.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "clusterlm/objects/gguf_writer.hpp"

namespace clusterlm::backends {
namespace {

using objects::GgmlType;
using objects::GgufWriter;

std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Uniform in [-1, 1).
float uniform(std::uint64_t& s) {
  return static_cast<float>(static_cast<double>(splitmix(s) >> 11) / 9007199254740992.0 * 2.0 - 1.0);
}

// IEEE binary16, round to nearest even; values here are small and finite.
std::uint16_t to_half(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
  const std::uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t m = mant | 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
    std::uint32_t half = m >> shift;
    const std::uint32_t rem = m & ((1u << shift) - 1u), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u))) ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7BFFu);
  std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
  const std::uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
  return static_cast<std::uint16_t>(sign | half);
}

Status add_matrix(GgufWriter& w, const std::string& name, std::vector<std::uint64_t> dims, float scale, bool f16,
                  std::uint64_t seed) {
  std::uint64_t n = 1;
  for (auto d : dims) n *= d;
  std::uint64_t s = seed;
  Bytes bytes;
  if (f16) {
    bytes.resize(n * 2);
    for (std::uint64_t i = 0; i < n; ++i) {
      const std::uint16_t h = to_half(uniform(s) * scale);
      std::memcpy(bytes.data() + i * 2, &h, 2);
    }
    return w.add_tensor(name, std::move(dims), GgmlType::kF16, std::move(bytes));
  }
  bytes.resize(n * 4);
  for (std::uint64_t i = 0; i < n; ++i) {
    const float v = uniform(s) * scale;
    std::memcpy(bytes.data() + i * 4, &v, 4);
  }
  return w.add_tensor(name, std::move(dims), GgmlType::kF32, std::move(bytes));
}

// Constant-plus-noise vector (used for the SSM decay "A" and skip "D", which must keep a recurrent state stable).
Status add_vec(GgufWriter& w, const std::string& name, std::vector<std::uint64_t> dims, float base, float spread,
               std::uint64_t seed) {
  std::uint64_t n = 1;
  for (auto d : dims) n *= d;
  std::uint64_t s = seed;
  Bytes bytes(n * 4);
  for (std::uint64_t i = 0; i < n; ++i) {
    const float v = base + spread * uniform(s);
    std::memcpy(bytes.data() + i * 4, &v, 4);
  }
  return w.add_tensor(name, std::move(dims), GgmlType::kF32, std::move(bytes));
}

Status add_norm(GgufWriter& w, const std::string& name, std::uint64_t n, std::uint64_t seed) {
  std::uint64_t s = seed;
  Bytes bytes(n * 4);
  for (std::uint64_t i = 0; i < n; ++i) {
    const float v = 1.0f + 0.1f * uniform(s);
    std::memcpy(bytes.data() + i * 4, &v, 4);
  }
  return w.add_tensor(name, {n}, GgmlType::kF32, std::move(bytes));
}

const char* arch_name(TinyArch a) {
  switch (a) {
    case TinyArch::kLlama: return "llama";
    case TinyArch::kLlamaMoe: return "llama";  // Mixtral-style: the llama architecture with expert tensors
    case TinyArch::kQwen2: return "qwen2";
    case TinyArch::kQwen3: return "qwen3";
    case TinyArch::kGemma: return "gemma";
    case TinyArch::kPhi3: return "phi3";
    case TinyArch::kMamba: return "mamba";
  }
  return "llama";
}

}  // namespace

const char* tiny_arch_name(TinyArch a) { return arch_name(a); }

namespace {

// Mamba (selective state space, no attention): a recurrent architecture. Expansion factor 2, conv kernel 4, state 8,
// time-step rank 4 -- the only shape family the pinned loader accepts ("only an expansion factor of 2").
Status write_tiny_mamba(const std::filesystem::path& path, const TinyLlamaSpec& sp) {
  const std::uint64_t H = sp.hidden, inner = 2 * H, d_conv = 4, d_state = 8, dt_rank = 4;
  GgufWriter w;
  w.add_string("general.architecture", "mamba");
  w.add_string("general.name", "clusterlm-tiny-mamba");
  w.add_u32("mamba.context_length", sp.context);
  w.add_u32("mamba.embedding_length", sp.hidden);
  w.add_u32("mamba.block_count", sp.layers);
  w.add_u32("mamba.feed_forward_length", 0);
  w.add_u32("mamba.attention.head_count", 0);
  w.add_f32("mamba.attention.layer_norm_rms_epsilon", 1e-5f);
  w.add_u32("mamba.ssm.conv_kernel", static_cast<std::uint32_t>(d_conv));
  w.add_u32("mamba.ssm.inner_size", static_cast<std::uint32_t>(inner));
  w.add_u32("mamba.ssm.state_size", static_cast<std::uint32_t>(d_state));
  w.add_u32("mamba.ssm.time_step_rank", static_cast<std::uint32_t>(dt_rank));
  w.add_u32("mamba.vocab_size", sp.vocab);
  w.add_string("tokenizer.ggml.model", "no_vocab");
  std::uint64_t seed = sp.seed * 1000003ull;
  auto next = [&] { return ++seed * 0x9E3779B97F4A7C15ull; };
  const float inv_h = 1.0f / std::sqrt(static_cast<float>(sp.hidden));
  const float inv_i = 1.0f / std::sqrt(static_cast<float>(inner));
  const bool f = sp.f16_matrices;
  CLM_RETURN_IF_ERROR(add_matrix(w, "token_embd.weight", {H, sp.vocab}, 1.0f, false, next()));
  CLM_RETURN_IF_ERROR(add_norm(w, "output_norm.weight", H, next()));
  CLM_RETURN_IF_ERROR(add_matrix(w, "output.weight", {H, sp.vocab}, 3.0f * inv_h, false, next()));
  for (std::uint32_t L = 0; L < sp.layers; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    CLM_RETURN_IF_ERROR(add_norm(w, p + "attn_norm.weight", H, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "ssm_in.weight", {H, 2 * inner}, 2.0f * inv_h, f, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "ssm_conv1d.weight", {d_conv, inner}, 0.5f, false, next()));
    CLM_RETURN_IF_ERROR(add_vec(w, p + "ssm_conv1d.bias", {inner}, 0.0f, 0.05f, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "ssm_x.weight", {inner, dt_rank + 2 * d_state}, 2.0f * inv_i, f, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "ssm_dt.weight", {dt_rank, inner}, 0.5f, false, next()));
    CLM_RETURN_IF_ERROR(add_vec(w, p + "ssm_dt.bias", {inner}, -1.0f, 0.2f, next()));
    CLM_RETURN_IF_ERROR(add_vec(w, p + "ssm_a", {d_state, inner}, -1.0f, 0.5f, next()));  // negative decay: stable
    CLM_RETURN_IF_ERROR(add_vec(w, p + "ssm_d", {inner}, 1.0f, 0.1f, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "ssm_out.weight", {inner, H}, 2.0f * inv_i, f, next()));
  }
  return w.write(path);
}

}  // namespace

Status write_tiny_llama_gguf(const std::filesystem::path& path, const TinyLlamaSpec& sp) {
  if (sp.layers == 0 || sp.hidden == 0 || sp.heads == 0 || sp.kv_heads == 0 || sp.heads % sp.kv_heads != 0 ||
      sp.hidden % sp.heads != 0 || sp.vocab < 2 || sp.ff == 0)
    return make_error(ErrorCode::kInvalidArgument, "tiny llama spec: inconsistent dimensions");
  const std::uint64_t H = sp.hidden, hd = sp.hidden / sp.heads, kvd = hd * sp.kv_heads;
  if (hd % 2 != 0) return make_error(ErrorCode::kInvalidArgument, "tiny llama spec: head dim must be even for RoPE");
  const TinyArch arch = sp.arch;
  const bool moe = arch == TinyArch::kLlamaMoe;
  if (moe && (sp.experts < 2 || sp.experts_used == 0 || sp.experts_used > sp.experts))
    return make_error(ErrorCode::kInvalidArgument, "tiny llama spec: MoE needs experts >= 2 and 1 <= used <= experts");
  const std::string a = arch_name(arch);
  if (arch == TinyArch::kMamba) return write_tiny_mamba(path, sp);
  const bool tied = sp.tied_embeddings || arch == TinyArch::kGemma;  // gemma never stores an output matrix
  GgufWriter w;
  w.add_string("general.architecture", a);
  w.add_string("general.name", std::string("clusterlm-tiny-") + a);
  w.add_u32(a + ".context_length", sp.context);
  w.add_u32(a + ".embedding_length", sp.hidden);
  w.add_u32(a + ".block_count", sp.layers);
  w.add_u32(a + ".feed_forward_length", sp.ff);
  w.add_u32(a + ".attention.head_count", sp.heads);
  w.add_u32(a + ".attention.head_count_kv", sp.kv_heads);
  w.add_f32(a + ".attention.layer_norm_rms_epsilon", 1e-5f);
  w.add_u32(a + ".rope.dimension_count", static_cast<std::uint32_t>(hd));
  w.add_f32(a + ".rope.freq_base", 10000.0f);
  if (arch == TinyArch::kGemma) {
    w.add_u32(a + ".attention.key_length", static_cast<std::uint32_t>(hd));
    w.add_u32(a + ".attention.value_length", static_cast<std::uint32_t>(hd));
  }
  if (moe) {
    w.add_u32(a + ".expert_count", sp.experts);
    w.add_u32(a + ".expert_used_count", sp.experts_used);
  }
  w.add_u32(a + ".vocab_size", sp.vocab);
  w.add_string("tokenizer.ggml.model", "no_vocab");

  std::uint64_t seed = sp.seed * 1000003ull;
  auto next = [&] { return ++seed * 0x9E3779B97F4A7C15ull; };
  const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(sp.hidden));
  CLM_RETURN_IF_ERROR(add_matrix(w, "token_embd.weight", {H, sp.vocab}, 1.0f, false, next()));
  CLM_RETURN_IF_ERROR(add_norm(w, "output_norm.weight", H, next()));
  if (!tied) CLM_RETURN_IF_ERROR(add_matrix(w, "output.weight", {H, sp.vocab}, 3.0f * inv_sqrt_h, false, next()));
  const float wq = 2.0f * inv_sqrt_h;
  for (std::uint32_t L = 0; L < sp.layers; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    const bool h = sp.f16_matrices;
    CLM_RETURN_IF_ERROR(add_norm(w, p + "attn_norm.weight", H, next()));
    if (arch == TinyArch::kPhi3) {
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_qkv.weight", {H, H + 2 * kvd}, wq, h, next()));
    } else {
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_q.weight", {H, H}, wq, h, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_k.weight", {H, kvd}, wq, h, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_v.weight", {H, kvd}, wq, h, next()));
    }
    if (arch == TinyArch::kQwen2) {  // Qwen2 carries biases on the Q/K/V projections
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_q.bias", {H}, 0.1f, false, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_k.bias", {kvd}, 0.1f, false, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_v.bias", {kvd}, 0.1f, false, next()));
    }
    if (arch == TinyArch::kQwen3) {  // Qwen3 normalises Q and K per head
      CLM_RETURN_IF_ERROR(add_norm(w, p + "attn_q_norm.weight", hd, next()));
      CLM_RETURN_IF_ERROR(add_norm(w, p + "attn_k_norm.weight", hd, next()));
    }
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_output.weight", {H, H}, wq, h, next()));
    CLM_RETURN_IF_ERROR(add_norm(w, p + "ffn_norm.weight", H, next()));
    const float wd = 2.0f / std::sqrt(static_cast<float>(sp.ff));
    if (moe) {
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_gate_inp.weight", {H, sp.experts}, wq, false, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_gate_exps.weight", {H, sp.ff, sp.experts}, wq, h, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_up_exps.weight", {H, sp.ff, sp.experts}, wq, h, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_down_exps.weight", {sp.ff, H, sp.experts}, wd, h, next()));
    } else if (arch == TinyArch::kPhi3) {  // fused gate+up
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_up.weight", {H, 2ull * sp.ff}, wq, h, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_down.weight", {sp.ff, H}, wd, h, next()));
    } else {
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_gate.weight", {H, sp.ff}, wq, h, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_up.weight", {H, sp.ff}, wq, h, next()));
      CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_down.weight", {sp.ff, H}, wd, h, next()));
    }
  }
  return w.write(path);
}

}  // namespace clusterlm::backends
