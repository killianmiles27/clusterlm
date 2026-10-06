// CpuExpertKernel over the pinned Strata CPU kernels (strata_kernels_cpu + ggml-cpu at the Strata ggml pin).
// Needs no GPU: built with CLUSTERLM_ENABLE_STRATA or CLUSTERLM_ENABLE_STRATA_CPU.
#include "clusterlm/backends/strata/cpu_expert_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "clusterlm/backends/strata/object_map.hpp"

#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

namespace clusterlm::backends::strata {
namespace {

namespace sk = ::strata::kernels::cpu;

// A quantized activation back to floats. ggml has no to_float for Q8_K (the i-quants' vec_dot type), so that one is
// spelled out: block_q8_K = { float d; int8_t qs[256]; int16_t bsums[16]; }, value = d * qs.
bool act_to_float(int type, const std::uint8_t* src, float* dst, std::int64_t n) {
  if (type == GGML_TYPE_Q8_K) {
    constexpr std::size_t kBlock = 4 + 256 + 16 * 2;
    for (std::int64_t b = 0; b < n / 256; ++b) {
      const std::uint8_t* blk = src + static_cast<std::size_t>(b) * kBlock;
      float d;
      std::memcpy(&d, blk, 4);
      for (int i = 0; i < 256; ++i) dst[b * 256 + i] = d * static_cast<float>(static_cast<std::int8_t>(blk[4 + i]));
    }
    return n % 256 == 0;
  }
  const auto* t = ggml_get_type_traits(static_cast<ggml_type>(type));
  if (t == nullptr || t->to_float == nullptr) return false;
  t->to_float(src, dst, n);
  return true;
}

class StrataCpuExpertKernel final : public CpuExpertKernel {
 public:
  StrataCpuExpertKernel(const GgmlType& gu, const GgmlType& d, sk::NativeFmt f) : gu_(gu), d_(d), f_(f) {}

  std::string_view gate_up_type() const override { return gu_.name; }
  std::string_view down_type() const override { return d_.name; }
  std::uint32_t hidden() const override { return static_cast<std::uint32_t>(f_.n_embd); }
  std::uint32_t ff() const override { return static_cast<std::uint32_t>(f_.n_ff); }
  std::uint64_t blob_bytes() const override { return f_.bytes; }

  std::string_view path(std::uint32_t tokens) const override {
    // The dispatch rule of strata native_gu_rows (src/kernels/cpu/native_expert.cpp) without its opt-in switches.
    const int mt_min = sk::native_gu_mt_min(f_.gu_type);
    const bool cpu512 = sk::cpu_avx512_ok();
    if (static_cast<int>(tokens) >= mt_min && (sk::iq512_supported(f_.gu_type) || (!cpu512 && sk::iq256_supported(f_.gu_type)))) {
      if (cpu512 && std::getenv("STRATA_NO_IQ512") == nullptr && sk::iq512_supported(f_.gu_type)) return "strata-iq512";
      if (sk::cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr && sk::iq256_supported(f_.gu_type))
        return "strata-iq256";
    }
    return "ggml-cpu";
  }

  Status check(ByteSpan blob, std::span<const float> x, std::uint32_t tokens, std::span<float> y) const {
    if (tokens == 0 || tokens > kCpuExpertMaxTokens)
      return make_error(ErrorCode::kInvalidArgument, "cpu expert: 1..8 tokens per call (Strata cpu::MAXT)");
    const std::size_t H = static_cast<std::size_t>(f_.n_embd);
    if (blob.size() != f_.bytes) return make_error(ErrorCode::kInvalidArgument, "cpu expert: blob is not the format's size");
    if (x.size() != H * tokens || y.size() != H * tokens)
      return make_error(ErrorCode::kInvalidArgument, "cpu expert: x/y must hold tokens * hidden floats");
    return Status::ok();
  }

  Status run(ByteSpan blob, std::span<const float> x, std::uint32_t tokens, std::span<float> y) const override {
    CLM_RETURN_IF_ERROR(check(blob, x, tokens, y));
    const std::size_t H = static_cast<std::size_t>(f_.n_embd), FF = static_cast<std::size_t>(f_.n_ff);
    // Strata's pool does exactly this per routed expert (expert_source.cpp, the multi-token native jobs).
    std::vector<std::uint8_t> act(tokens * sk::kNativeActBytes), hq(tokens * sk::kNativeHBytes);
    std::vector<float> ffv(tokens * FF);
    const void* a[kCpuExpertMaxTokens];
    float* fr[kCpuExpertMaxTokens];
    const void* h[kCpuExpertMaxTokens];
    float* out[kCpuExpertMaxTokens];
    for (std::uint32_t t = 0; t < tokens; ++t) {
      sk::native_quant_act(f_, x.data() + t * H, act.data() + t * sk::kNativeActBytes);
      a[t] = act.data() + t * sk::kNativeActBytes;
      fr[t] = ffv.data() + t * FF;
    }
    sk::native_gu_rows(f_, blob.data(), a, static_cast<int>(tokens), fr, 0, static_cast<int>(FF));
    for (std::uint32_t t = 0; t < tokens; ++t) {
      sk::native_quant_h(f_, fr[t], hq.data() + t * sk::kNativeHBytes);
      h[t] = hq.data() + t * sk::kNativeHBytes;
      out[t] = y.data() + t * H;
    }
    sk::native_down_rows(f_, blob.data(), h, static_cast<int>(tokens), out, 0, static_cast<int>(H));
    return Status::ok();
  }

  Status gate_up(ByteSpan blob, std::span<const float> x, std::uint32_t tokens, std::span<float> ffo) const override {
    CLM_RETURN_IF_ERROR(check_ff(blob, x, tokens, ffo));
    const std::size_t H = static_cast<std::size_t>(f_.n_embd), FF = static_cast<std::size_t>(f_.n_ff);
    std::vector<std::uint8_t> act(tokens * sk::kNativeActBytes);
    const void* a[kCpuExpertMaxTokens];
    float* fr[kCpuExpertMaxTokens];
    for (std::uint32_t t = 0; t < tokens; ++t) {
      sk::native_quant_act(f_, x.data() + t * H, act.data() + t * sk::kNativeActBytes);
      a[t] = act.data() + t * sk::kNativeActBytes;
      fr[t] = ffo.data() + t * FF;
    }
    sk::native_gu_rows(f_, blob.data(), a, static_cast<int>(tokens), fr, 0, static_cast<int>(FF));
    return Status::ok();
  }

  Status gate_up_reference(Reference kind, ByteSpan blob, std::span<const float> x, std::uint32_t tokens,
                           std::span<float> ffo) const override {
    CLM_RETURN_IF_ERROR(check_ff(blob, x, tokens, ffo));
    std::vector<float> y(x.size());
    return reference_impl(kind, blob, x, tokens, y, ffo);
  }

  Status check_ff(ByteSpan blob, std::span<const float> x, std::uint32_t tokens, std::span<float> ffo) const {
    if (tokens == 0 || tokens > kCpuExpertMaxTokens)
      return make_error(ErrorCode::kInvalidArgument, "cpu expert: 1..8 tokens per call (Strata cpu::MAXT)");
    if (blob.size() != f_.bytes) return make_error(ErrorCode::kInvalidArgument, "cpu expert: blob is not the format's size");
    if (x.size() != static_cast<std::size_t>(f_.n_embd) * tokens || ffo.size() != static_cast<std::size_t>(f_.n_ff) * tokens)
      return make_error(ErrorCode::kInvalidArgument, "cpu expert: x must hold tokens * hidden, ff tokens * ff floats");
    return Status::ok();
  }

  Status reference(Reference kind, ByteSpan blob, std::span<const float> x, std::uint32_t tokens,
                   std::span<float> y) const override {
    CLM_RETURN_IF_ERROR(check(blob, x, tokens, y));
    return reference_impl(kind, blob, x, tokens, y, {});
  }

  // y: the expert outputs; ff_out (optional): the gate/up results per token.
  Status reference_impl(Reference kind, ByteSpan blob, std::span<const float> x, std::uint32_t tokens, std::span<float> y,
                        std::span<float> ff_out) const {
    const std::size_t H = static_cast<std::size_t>(f_.n_embd), FF = static_cast<std::size_t>(f_.n_ff);
    const auto* tgu = ggml_get_type_traits_cpu(static_cast<ggml_type>(f_.gu_type));
    const auto* td = ggml_get_type_traits_cpu(static_cast<ggml_type>(f_.d_type));
    const auto* gw = ggml_get_type_traits(static_cast<ggml_type>(f_.gu_type));
    const auto* dw = ggml_get_type_traits(static_cast<ggml_type>(f_.d_type));
    if (tgu == nullptr || td == nullptr || tgu->vec_dot == nullptr || td->vec_dot == nullptr || gw == nullptr ||
        dw == nullptr || gw->to_float == nullptr || dw->to_float == nullptr)
      return make_error(ErrorCode::kUnimplemented, "cpu expert: ggml has no reference for these types");
    std::vector<std::uint8_t> act(sk::kNativeActBytes), hq(sk::kNativeHBytes);
    std::vector<float> ffv(FF), xa(H), ha(FF), wrow(std::max(H, FF));
    const std::uint8_t* g0 = blob.data();
    const std::uint8_t* u0 = blob.data() + f_.up_off;
    const std::uint8_t* d0 = blob.data() + f_.down_off;
    for (std::uint32_t t = 0; t < tokens; ++t) {
      sk::native_quant_act(f_, x.data() + t * H, act.data());
      if (kind == Reference::kScalarDequant && !act_to_float(f_.gu_act, act.data(), xa.data(), static_cast<std::int64_t>(H)))
        return make_error(ErrorCode::kUnimplemented, "cpu expert: no dequantizer for the gate/up activation type");
      for (std::size_t r = 0; r < FF; ++r) {
        float gs = 0.f, us = 0.f;
        if (kind == Reference::kExactGgml) {
          tgu->vec_dot(static_cast<int>(H), &gs, 0, g0 + r * f_.gu_row, 0, act.data(), 0, 1);
          tgu->vec_dot(static_cast<int>(H), &us, 0, u0 + r * f_.gu_row, 0, act.data(), 0, 1);
        } else {
          double gd = 0, ud = 0;
          gw->to_float(g0 + r * f_.gu_row, wrow.data(), static_cast<std::int64_t>(H));
          for (std::size_t c = 0; c < H; ++c) gd += static_cast<double>(wrow[c]) * xa[c];
          gw->to_float(u0 + r * f_.gu_row, wrow.data(), static_cast<std::int64_t>(H));
          for (std::size_t c = 0; c < H; ++c) ud += static_cast<double>(wrow[c]) * xa[c];
          gs = static_cast<float>(gd);
          us = static_cast<float>(ud);
        }
        ffv[r] = (gs / (1.f + std::exp(-gs))) * us;
      }
      if (!ff_out.empty()) std::copy(ffv.begin(), ffv.end(), ff_out.begin() + static_cast<std::ptrdiff_t>(t * FF));
      sk::native_quant_h(f_, ffv.data(), hq.data());
      if (kind == Reference::kScalarDequant && !act_to_float(f_.d_act, hq.data(), ha.data(), static_cast<std::int64_t>(FF)))
        return make_error(ErrorCode::kUnimplemented, "cpu expert: no dequantizer for the down activation type");
      for (std::size_t r = 0; r < H; ++r) {
        float s = 0.f;
        if (kind == Reference::kExactGgml) {
          td->vec_dot(static_cast<int>(FF), &s, 0, d0 + r * f_.d_row, 0, hq.data(), 0, 1);
        } else {
          double sd = 0;
          dw->to_float(d0 + r * f_.d_row, wrow.data(), static_cast<std::int64_t>(FF));
          for (std::size_t c = 0; c < FF; ++c) sd += static_cast<double>(wrow[c]) * ha[c];
          s = static_cast<float>(sd);
        }
        y[t * H + r] = s;
      }
    }
    return Status::ok();
  }

 private:
  const GgmlType& gu_;
  const GgmlType& d_;
  sk::NativeFmt f_;
};

}  // namespace

bool cpu_expert_kernels_available() { return sk::native_experts_available(); }

std::string cpu_expert_isa() {
  if (sk::cpu_avx512_ok()) return "avx512";
  if (sk::cpu_avx2_ok()) return "avx2";
  return "baseline";
}

Result<std::unique_ptr<CpuExpertKernel>> make_cpu_expert_kernel(std::string_view gate_up_type, std::string_view down_type,
                                                                std::uint32_t hidden, std::uint32_t ff) {
  const GgmlType* gu = ggml_type_by_name(gate_up_type);
  const GgmlType* d = ggml_type_by_name(down_type);
  if (gu == nullptr || d == nullptr) return make_error(ErrorCode::kInvalidArgument, "cpu expert: unknown ggml type");
  sk::NativeFmt f;
  std::string err;
  if (!sk::native_fmt(gu->id, d->id, hidden, ff, f, err)) return make_error(ErrorCode::kInvalidArgument, "cpu expert: " + err);
  // Cross-check the ClusterLM type table (object_map) against ggml's own row sizes.
  if (f.gu_row != ggml_bytes(*gu, hidden) || f.d_row != ggml_bytes(*d, ff))
    return make_error(ErrorCode::kInternal, "cpu expert: ClusterLM's ggml type table disagrees with ggml");
  return std::unique_ptr<CpuExpertKernel>(std::make_unique<StrataCpuExpertKernel>(*gu, *d, f));
}

}  // namespace clusterlm::backends::strata
