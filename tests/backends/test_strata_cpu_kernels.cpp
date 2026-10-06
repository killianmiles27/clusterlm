// Strata's CPU expert kernels through ClusterLM's CpuExpertKernel, on synthetic expert blobs at Flash-Next's real
// expert dimensions (H 2560, ff 640): IQ3_S and IQ2_XS gate/up with IQ4_NL or Q2_0 down.
//
// A random i-quant block is a valid block once its fp16 scale is sane: every IQ3_S / IQ2_XS grid and sign index is
// in range by construction, so random bytes exercise the whole codebook. Checks:
//   * against ggml-cpu's own per-token vec_dot on the same quantized activations (llama.cpp's CPU arithmetic):
//     bit-exact on ggml's path, and to float rounding where Strata's multi-token AVX kernels take over;
//   * against an implementation-independent scalar reference (ggml's reference dequantization, double accumulation);
//   * token independence: a token's output does not depend on the other tokens of the call (up to the documented
//     multi-token rounding, issue #152 of the pin).
// Runs without a GPU.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "clusterlm/backends/strata/cpu_expert_kernel.hpp"
#include "clusterlm/backends/strata/object_map.hpp"

using namespace clusterlm;
namespace bs = clusterlm::backends::strata;

namespace {

constexpr std::uint32_t kH = 2560, kFF = 640;

std::uint16_t fp16_bits(float f) {
  // small positive normal values only (the scales below)
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t e = ((x >> 23) & 0xff) - 127 + 15, m = (x >> 13) & 0x3ff;
  return static_cast<std::uint16_t>((e << 10) | m);
}

// `rows` rows of `elems` values of type `t`: random bytes with each block's leading fp16 scale set to a sane value.
Bytes random_rows(const bs::GgmlType& t, std::uint64_t elems, std::uint64_t rows, std::mt19937& rng, float scale) {
  const std::uint64_t blocks = elems / t.block_elems * rows;
  Bytes b(blocks * t.block_bytes);
  for (auto& x : b) x = static_cast<std::uint8_t>(rng());
  std::uniform_real_distribution<float> d(0.5f * scale, 1.5f * scale);
  for (std::uint64_t i = 0; i < blocks; ++i) {
    const std::uint16_t h = fp16_bits(d(rng));
    std::memcpy(b.data() + i * t.block_bytes, &h, 2);
  }
  return b;
}

Bytes random_expert(std::string_view gu, std::string_view down, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const bs::GgmlType& g = *bs::ggml_type_by_name(gu);
  const bs::GgmlType& d = *bs::ggml_type_by_name(down);
  Bytes blob = random_rows(g, kH, kFF, rng, 0.004f);   // gate
  Bytes up = random_rows(g, kH, kFF, rng, 0.004f);
  Bytes dn = random_rows(d, kFF, kH, rng, 0.02f);
  blob.insert(blob.end(), up.begin(), up.end());
  blob.insert(blob.end(), dn.begin(), dn.end());
  return blob;
}

std::vector<float> random_x(std::uint32_t tokens, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> n(0.0f, 1.0f);
  std::vector<float> x(std::size_t{tokens} * kH);
  for (float& v : x) v = n(rng);
  return x;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

bool all_finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

struct Config {
  const char* gu;
  const char* down;
};

}  // namespace

TEST_CASE("Strata CPU expert kernels are built and report this CPU's tier") {
  REQUIRE(bs::cpu_expert_kernels_available());
  MESSAGE("Strata CPU expert ISA on this host: " << bs::cpu_expert_isa());
  CHECK(bs::make_cpu_expert_kernel("iq3_s", "iq3_s", kH, kFF).status().code() == ErrorCode::kInvalidArgument);  // 640 / 256
  CHECK(bs::make_cpu_expert_kernel("nope", "iq4_nl", kH, kFF).status().code() == ErrorCode::kInvalidArgument);
}

TEST_CASE("IQ3_S / IQ2_XS experts: Strata's kernels against ggml-cpu and a scalar reference") {
  const Config configs[] = {{"iq3_s", "iq4_nl"}, {"iq2_xs", "iq4_nl"}, {"iq3_s", "q2_0"}, {"iq2_xs", "q2_0"}};
  std::uint32_t seed = 11;
  for (const Config& c : configs) {
    const std::string name = std::string(c.gu) + "+" + c.down;
    CAPTURE(name);
    auto k = bs::make_cpu_expert_kernel(c.gu, c.down, kH, kFF);
    REQUIRE_MESSAGE(k.is_ok(), k.status().to_string());
    const auto f = bs::expert_format({std::string(c.gu) + "+" + c.down, 256, 0, true},
                                     [] {
                                       objects::ModelGeometry g;
                                       g.hidden_size = kH;
                                       g.expert_ff = kFF;
                                       return g;
                                     }());
    REQUIRE(f.is_ok());
    CHECK((*k)->blob_bytes() == f->total());  // ClusterLM's expert object size == Strata's native blob
    const Bytes blob = random_expert(c.gu, c.down, seed++);
    REQUIRE(blob.size() == (*k)->blob_bytes());

    for (std::uint32_t tokens : {1u, 2u, 3u, 8u}) {
      CAPTURE(tokens);
      const auto x = random_x(tokens, seed++);
      std::vector<float> y(x.size()), exact(x.size()), scalar(x.size());
      REQUIRE((*k)->run(blob, x, tokens, y).is_ok());
      const Status st_exact = (*k)->reference(bs::CpuExpertKernel::Reference::kExactGgml, blob, x, tokens, exact);
      REQUIRE_MESSAGE(st_exact.is_ok(), st_exact.to_string());
      const Status st_scalar = (*k)->reference(bs::CpuExpertKernel::Reference::kScalarDequant, blob, x, tokens, scalar);
      REQUIRE_MESSAGE(st_scalar.is_ok(), st_scalar.to_string());
      REQUIRE(all_finite(y));
      double ymax = 0;
      for (float v : y) ymax = std::max(ymax, std::fabs(static_cast<double>(v)));
      CHECK(ymax > 0.0);
      const std::string_view path = (*k)->path(tokens);
      MESSAGE(name << " tokens " << tokens << " path " << path << " rel-L2 vs ggml "
                   << rel_l2(y, exact) << " vs scalar " << rel_l2(y, scalar));
      // gate/up rows - where the i-quant kernels work - before the hidden activation is re-quantized
      std::vector<float> ff(std::size_t{tokens} * kFF), ff_exact(ff.size()), ff_scalar(ff.size());
      REQUIRE((*k)->gate_up(blob, x, tokens, ff).is_ok());
      REQUIRE((*k)->gate_up_reference(bs::CpuExpertKernel::Reference::kExactGgml, blob, x, tokens, ff_exact).is_ok());
      REQUIRE((*k)->gate_up_reference(bs::CpuExpertKernel::Reference::kScalarDequant, blob, x, tokens, ff_scalar).is_ok());
      MESSAGE(name << " gate/up rel-L2 vs ggml " << rel_l2(ff, ff_exact) << " vs scalar " << rel_l2(ff, ff_scalar));
      CHECK(rel_l2(ff, ff_scalar) < 1e-5);
      if (path == "ggml-cpu") {
        // the same arithmetic as llama.cpp's CPU backend, bit for bit
        CHECK(std::memcmp(ff.data(), ff_exact.data(), ff.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(y.data(), exact.data(), y.size() * sizeof(float)) == 0);
      } else {
        // Strata's multi-token kernels: the same integer block dots, another float summation order...
        CHECK(rel_l2(ff, ff_exact) < 1e-5);
        // ...which may flip a hidden value's Q8 rounding before the down rows (Strata pin issue 152)
        CHECK(rel_l2(y, exact) < 2e-3);
      }
      CHECK(rel_l2(y, scalar) < 2e-3);
    }
  }
}

TEST_CASE("a token's expert output does not depend on the other tokens of the call") {
  auto k = bs::make_cpu_expert_kernel("iq3_s", "iq4_nl", kH, kFF);
  REQUIRE(k.is_ok());
  const Bytes blob = random_expert("iq3_s", "iq4_nl", 77);
  const auto x = random_x(4, 78);
  std::vector<float> all(x.size());
  REQUIRE((*k)->run(blob, x, 4, all).is_ok());
  for (std::uint32_t t = 0; t < 4; ++t) {
    std::vector<float> one(kH);
    REQUIRE((*k)->run(blob, std::span<const float>(x).subspan(std::size_t{t} * kH, kH), 1, one).is_ok());
    const std::vector<float> row(all.begin() + static_cast<std::ptrdiff_t>(t) * kH,
                                 all.begin() + static_cast<std::ptrdiff_t>(t + 1) * kH);
    // bit-exact when the same path computes both; otherwise the multi-token kernel's rounding (Strata #152)
    if ((*k)->path(4) == (*k)->path(1)) CHECK(std::memcmp(one.data(), row.data(), kH * sizeof(float)) == 0);
    else CHECK(rel_l2(row, one) < 1e-5);
  }
}

TEST_CASE("argument checks") {
  auto k = bs::make_cpu_expert_kernel("iq2_xs", "iq4_nl", kH, kFF);
  REQUIRE(k.is_ok());
  const Bytes blob = random_expert("iq2_xs", "iq4_nl", 5);
  std::vector<float> x(kH), y(kH);
  CHECK((*k)->run(ByteSpan(blob).first(blob.size() - 1), x, 1, y).code() == ErrorCode::kInvalidArgument);
  CHECK((*k)->run(blob, x, 0, y).code() == ErrorCode::kInvalidArgument);
  CHECK((*k)->run(blob, x, 9, y).code() == ErrorCode::kInvalidArgument);
  CHECK((*k)->run(blob, x, 2, y).code() == ErrorCode::kInvalidArgument);  // x/y hold one token
}
