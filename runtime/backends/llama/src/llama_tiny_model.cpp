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

Status add_norm(GgufWriter& w, const std::string& name, std::uint64_t n, std::uint64_t seed) {
  std::uint64_t s = seed;
  Bytes bytes(n * 4);
  for (std::uint64_t i = 0; i < n; ++i) {
    const float v = 1.0f + 0.1f * uniform(s);
    std::memcpy(bytes.data() + i * 4, &v, 4);
  }
  return w.add_tensor(name, {n}, GgmlType::kF32, std::move(bytes));
}

}  // namespace

Status write_tiny_llama_gguf(const std::filesystem::path& path, const TinyLlamaSpec& sp) {
  if (sp.layers == 0 || sp.hidden == 0 || sp.heads == 0 || sp.kv_heads == 0 || sp.heads % sp.kv_heads != 0 ||
      sp.hidden % sp.heads != 0 || sp.vocab < 2 || sp.ff == 0)
    return make_error(ErrorCode::kInvalidArgument, "tiny llama spec: inconsistent dimensions");
  const std::uint64_t H = sp.hidden, hd = sp.hidden / sp.heads, kvd = hd * sp.kv_heads;
  if (hd % 2 != 0) return make_error(ErrorCode::kInvalidArgument, "tiny llama spec: head dim must be even for RoPE");
  GgufWriter w;
  w.add_string("general.architecture", "llama");
  w.add_string("general.name", "clusterlm-tiny-llama");
  w.add_u32("llama.context_length", sp.context);
  w.add_u32("llama.embedding_length", sp.hidden);
  w.add_u32("llama.block_count", sp.layers);
  w.add_u32("llama.feed_forward_length", sp.ff);
  w.add_u32("llama.attention.head_count", sp.heads);
  w.add_u32("llama.attention.head_count_kv", sp.kv_heads);
  w.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
  w.add_u32("llama.rope.dimension_count", static_cast<std::uint32_t>(hd));
  w.add_f32("llama.rope.freq_base", 10000.0f);
  w.add_u32("llama.vocab_size", sp.vocab);
  w.add_string("tokenizer.ggml.model", "no_vocab");

  std::uint64_t seed = sp.seed * 1000003ull;
  auto next = [&] { return ++seed * 0x9E3779B97F4A7C15ull; };
  const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(sp.hidden));
  CLM_RETURN_IF_ERROR(add_matrix(w, "token_embd.weight", {H, sp.vocab}, 1.0f, false, next()));
  CLM_RETURN_IF_ERROR(add_norm(w, "output_norm.weight", H, next()));
  CLM_RETURN_IF_ERROR(add_matrix(w, "output.weight", {H, sp.vocab}, 3.0f * inv_sqrt_h, false, next()));
  for (std::uint32_t L = 0; L < sp.layers; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    const bool h = sp.f16_matrices;
    CLM_RETURN_IF_ERROR(add_norm(w, p + "attn_norm.weight", H, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_q.weight", {H, H}, 2.0f * inv_sqrt_h, h, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_k.weight", {H, kvd}, 2.0f * inv_sqrt_h, h, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_v.weight", {H, kvd}, 2.0f * inv_sqrt_h, h, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "attn_output.weight", {H, H}, 2.0f * inv_sqrt_h, h, next()));
    CLM_RETURN_IF_ERROR(add_norm(w, p + "ffn_norm.weight", H, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_gate.weight", {H, sp.ff}, 2.0f * inv_sqrt_h, h, next()));
    CLM_RETURN_IF_ERROR(add_matrix(w, p + "ffn_up.weight", {H, sp.ff}, 2.0f * inv_sqrt_h, h, next()));
    CLM_RETURN_IF_ERROR(
        add_matrix(w, p + "ffn_down.weight", {sp.ff, H}, 2.0f / std::sqrt(static_cast<float>(sp.ff)), h, next()));
  }
  return w.write(path);
}

}  // namespace clusterlm::backends
