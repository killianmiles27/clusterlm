#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "clusterlm/domain/reference_kernels.hpp"
#include "expert_kernels.hpp"

using namespace clusterlm;
using namespace clusterlm::bench;

namespace {

std::vector<float> ramp(std::size_t n, float scale, std::uint32_t seed) {
  std::vector<float> v(n);
  std::uint32_t s = seed * 2654435761u + 1;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * scale;
  }
  return v;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("reference_matmul_rows is bitwise identical to per-position reference_matvec") {
  const std::size_t rows = 24, cols = 64, P = 3;
  const auto w = ramp(rows * cols, 0.2f, 1);
  const auto x = ramp(P * cols, 1.0f, 2);
  std::vector<float> batched(P * rows), single(P * rows);
  domain::reference_matmul_rows(w.data(), rows, cols, x.data(), P, batched.data());
  for (std::size_t p = 0; p < P; ++p) domain::reference_matvec(w.data(), rows, cols, x.data() + p * cols, single.data() + p * rows);
  CHECK(bitwise_equal(batched, single));
}

TEST_CASE("reference_expert_forward equals the composition of its matvecs") {
  const std::size_t H = 32, ff = 16;
  const auto gate = ramp(ff * H, 0.3f, 3), up = ramp(ff * H, 0.3f, 4), down = ramp(H * ff, 0.3f, 5);
  const auto h = ramp(H, 1.0f, 6);
  std::vector<float> out(H), a(ff), b(ff), z(ff), expect(H);
  domain::ExpertScratch scratch;
  domain::reference_expert_forward(gate.data(), up.data(), down.data(), ff, H, h.data(), out.data(), scratch);
  domain::reference_matvec(gate.data(), ff, H, h.data(), a.data());
  domain::reference_matvec(up.data(), ff, H, h.data(), b.data());
  for (std::size_t i = 0; i < ff; ++i) z[i] = a[i] / (1.0f + std::exp(-a[i])) * b[i];
  domain::reference_matvec(down.data(), H, ff, z.data(), expect.data());
  CHECK(bitwise_equal(out, expect));
}

TEST_CASE("registry: reference provider is usable, strata-cpu is a clear stub, factories can be replaced") {
  register_builtin_expert_providers();
  auto& reg = ExpertKernelRegistry::instance();
  const auto ids = reg.ids();
  CHECK(std::find(ids.begin(), ids.end(), "reference") != ids.end());
  CHECK(std::find(ids.begin(), ids.end(), "strata-cpu") != ids.end());

  auto strata = reg.create("strata-cpu");
  REQUIRE(strata.is_ok());
#ifndef CLUSTERLM_BENCH_HAS_STRATA_CPU
  // Built without the Strata CPU kernels: the registration stub.
  auto bank = strata.value()->make_bank("iq3_s", ExpertShape{256, 64}, 4, 1);
  REQUIRE_FALSE(bank.is_ok());
  CHECK(bank.status().code() == ErrorCode::kHardwareUnavailable);
  CHECK(bank.status().message().find("registration stub") != std::string::npos);
#else
  // Built with them: the real provider (see the strata-cpu test case below).
  CHECK(strata.value()->isa() != "unavailable");
#endif

  CHECK_FALSE(reg.create("no-such-provider").is_ok());

  // A later workstream replaces a stub by registering under the same id; the newest factory wins.
  reg.register_factory("test-double", []() -> Result<std::unique_ptr<ExpertKernelProvider>> {
    return make_error(ErrorCode::kUnimplemented, "first");
  });
  reg.register_factory("test-double", []() -> Result<std::unique_ptr<ExpertKernelProvider>> {
    return ExpertKernelRegistry::instance().create("reference");
  });
  auto replaced = reg.create("test-double");
  REQUIRE(replaced.is_ok());
  CHECK(replaced.value()->id() == "reference");
}

TEST_CASE("reference banks: positions are independent, dequantization is timed separately") {
  register_builtin_expert_providers();
  auto provider = ExpertKernelRegistry::instance().create("reference");
  REQUIRE(provider.is_ok());
  const ExpertShape shape{64, 32};
  const std::size_t H = shape.hidden, P = 3;
  const auto in = ramp(P * H, 1.0f, 9);

  for (const std::string rep : {"f32", "q8_0-fixture"}) {
    CAPTURE(rep);
    auto bank = provider.value()->make_bank(rep, shape, 5, 42);
    REQUIRE(bank.is_ok());
    CHECK(bank.value()->experts() == 5);
    CHECK(bank.value()->bytes_per_expert() > 0);
    auto ctx = bank.value()->make_context();
    std::vector<float> batched(P * H), single(P * H);
    ExpertTiming timing;
    REQUIRE(bank.value()->run(2, P, in.data(), batched.data(), *ctx, &timing).is_ok());
    for (std::size_t p = 0; p < P; ++p)
      REQUIRE(bank.value()->run(2, 1, in.data() + p * H, single.data() + p * H, *ctx, nullptr).is_ok());
    CHECK(bitwise_equal(batched, single));
    CHECK(timing.gemv_ns > 0);
    if (rep == "f32") CHECK(timing.dequant_ns == 0);
    else CHECK(timing.dequant_ns > 0);
    CHECK_FALSE(bank.value()->run(5, 1, in.data(), single.data(), *ctx, nullptr).is_ok());
  }
  // q8 stores 36 bytes per 32 elements: 3 matrices of ff*H.
  auto q8 = provider.value()->make_bank("q8_0-fixture", shape, 1, 1);
  REQUIRE(q8.is_ok());
  CHECK(q8.value()->bytes_per_expert() == 3ull * shape.ff * shape.hidden / 32 * 36);
  auto f32 = provider.value()->make_bank("f32", shape, 1, 1);
  REQUIRE(f32.is_ok());
  CHECK(f32.value()->bytes_per_expert() == 3ull * shape.ff * shape.hidden * 4);
  CHECK_FALSE(provider.value()->make_bank("iq3_s", shape, 1, 1).is_ok());
}

#ifdef CLUSTERLM_BENCH_HAS_STRATA_CPU
TEST_CASE("strata-cpu provider: IQ3_S / IQ2_XS banks at the Flash-Next shape run, are deterministic and report their path") {
  register_builtin_expert_providers();
  auto p = ExpertKernelRegistry::instance().create("strata-cpu");
  REQUIRE(p.is_ok());
  CHECK(p.value()->id() == "strata-cpu");
  CHECK(p.value()->representations() == std::vector<std::string>{"iq3_s", "iq2_xs"});
  const std::string isa = p.value()->isa();
  CHECK((isa == "avx512" || isa == "avx2" || isa == "baseline"));
  const ExpertShape shape;  // 2560 x 640
  CHECK_FALSE(p.value()->make_bank("q4_k", shape, 2, 1).is_ok());
  CHECK(p.value()->make_bank("iq3_s", ExpertShape{100, 32}, 2, 1).status().code() == ErrorCode::kInvalidArgument);

  std::uint64_t bytes_per_expert[2] = {};
  int i = 0;
  for (const char* rep : {"iq3_s", "iq2_xs"}) {
    CAPTURE(rep);
    auto a = p.value()->make_bank(rep, shape, 3, 42);
    auto b = p.value()->make_bank(rep, shape, 3, 42);
    auto c = p.value()->make_bank(rep, shape, 3, 43);
    REQUIRE(a.is_ok());
    REQUIRE(b.is_ok());
    REQUIRE(c.is_ok());
    CHECK(a.value()->experts() == 3);
    bytes_per_expert[i++] = a.value()->bytes_per_expert();
    CHECK(a.value()->bytes_per_expert() > 0);
    CHECK_FALSE(a.value()->kernel_path(1).empty());
    CHECK_FALSE(a.value()->kernel_path(4).empty());
    const std::size_t H = shape.hidden;
    for (std::size_t positions : {1u, 2u, 11u}) {  // 11 > one Strata window: runs as two
      const auto x = ramp(positions * H, 2.0f, 7);
      std::vector<float> ya(positions * H), yb(positions * H), yc(positions * H);
      auto ca = a.value()->make_context(), cb = b.value()->make_context(), cc = c.value()->make_context();
      ExpertTiming t;
      REQUIRE(a.value()->run(1, positions, x.data(), ya.data(), *ca, &t).is_ok());
      REQUIRE(b.value()->run(1, positions, x.data(), yb.data(), *cb, nullptr).is_ok());
      REQUIRE(c.value()->run(1, positions, x.data(), yc.data(), *cc, nullptr).is_ok());
      CHECK(t.gemv_ns > 0);
      CHECK(bitwise_equal(ya, yb));
      CHECK_FALSE(bitwise_equal(ya, yc));
      for (float v : ya) REQUIRE(std::isfinite(v));
    }
    CHECK(a.value()->run(3, 1, ramp(H, 1.0f, 1).data(), std::vector<float>(H).data(), *a.value()->make_context(), nullptr).code() ==
          ErrorCode::kOutOfRange);
  }
  CHECK(bytes_per_expert[0] > bytes_per_expert[1]);  // IQ3_S is larger than IQ2_XS
}
#endif
