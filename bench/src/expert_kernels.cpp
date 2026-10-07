#include "expert_kernels.hpp"

#include <algorithm>
#include <cmath>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/domain/reference_kernels.hpp"
#include "clusterlm/objects/tensor_codec.hpp"

namespace clusterlm::bench {

namespace {

std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Uniform in [-scale, scale).
void fill_random(std::vector<float>& v, std::uint64_t& state, float scale) {
  for (float& x : v) {
    const double u = static_cast<double>(splitmix(state) >> 11) / 9007199254740992.0;  // [0,1)
    x = static_cast<float>((u * 2.0 - 1.0) * static_cast<double>(scale));
  }
}

struct RefContext final : ExpertContext {
  std::vector<float> gate, up, down;  // dequantized weights (q8 only)
  std::vector<float> a, b, z;         // [positions][ff]
};

// Batched SwiGLU over P positions: each weight row is streamed once and applied to every position.
void batched_swiglu(const float* gate, const float* up, const float* down, ExpertShape s, std::size_t P, const float* in,
                    float* out, RefContext& c) {
  const std::size_t H = s.hidden, ff = s.ff;
  c.a.resize(P * ff);
  c.b.resize(P * ff);
  c.z.resize(P * ff);
  // matmul_rows writes y[p*rows + i]; with rows = ff that is the [position][ff] layout used below.
  domain::reference_matmul_rows(gate, ff, H, in, P, c.a.data());
  domain::reference_matmul_rows(up, ff, H, in, P, c.b.data());
  for (std::size_t i = 0; i < P * ff; ++i) {
    const float g = c.a[i];
    c.z[i] = g / (1.0f + std::exp(-g)) * c.b[i];
  }
  domain::reference_matmul_rows(down, H, ff, c.z.data(), P, out);
}

class RefF32Bank final : public ExpertBank {
 public:
  RefF32Bank(ExpertShape s, std::size_t n, std::uint64_t seed) : shape_(s), n_(n) {
    const std::size_t per = std::size_t{s.ff} * s.hidden;
    w_.resize(n * 3 * per);
    std::uint64_t st = seed;
    std::vector<float> tmp(3 * per);
    for (std::size_t e = 0; e < n; ++e) {
      fill_random(tmp, st, 0.05f);
      std::copy(tmp.begin(), tmp.end(), w_.begin() + static_cast<std::ptrdiff_t>(e * 3 * per));
    }
  }
  std::size_t experts() const override { return n_; }
  std::uint64_t bytes_per_expert() const override { return 3ull * shape_.ff * shape_.hidden * sizeof(float); }
  std::unique_ptr<ExpertContext> make_context() const override { return std::make_unique<RefContext>(); }
  Status run(std::size_t e, std::size_t positions, const float* in, float* out, ExpertContext& ctx,
             ExpertTiming* timing) const override {
    if (e >= n_) return make_error(ErrorCode::kOutOfRange, "expert index out of range");
    const std::size_t per = std::size_t{shape_.ff} * shape_.hidden;
    const float* base = w_.data() + e * 3 * per;
    const auto t0 = monotonic_ns();
    batched_swiglu(base, base + per, base + 2 * per, shape_, positions, in, out, static_cast<RefContext&>(ctx));
    if (timing) timing->gemv_ns += monotonic_ns() - t0;
    return Status::ok();
  }

 private:
  ExpertShape shape_;
  std::size_t n_;
  std::vector<float> w_;
};

class RefQ8Bank final : public ExpertBank {
 public:
  static Result<std::unique_ptr<ExpertBank>> make(ExpertShape s, std::size_t n, std::uint64_t seed) {
    if (s.hidden % objects::kQ8BlockElems != 0 || s.ff % objects::kQ8BlockElems != 0)
      return make_error(ErrorCode::kInvalidArgument, "q8_0-fixture needs hidden and ff to be multiples of 32");
    auto bank = std::unique_ptr<RefQ8Bank>(new RefQ8Bank(s, n));
    const std::size_t per = std::size_t{s.ff} * s.hidden;
    std::uint64_t st = seed;
    std::vector<float> tmp(per);
    for (std::size_t e = 0; e < n; ++e)
      for (int m = 0; m < 3; ++m) {
        fill_random(tmp, st, 0.05f);
        CLM_RETURN_IF_ERROR(objects::encode_q8_fixture(tmp, bank->q_));
      }
    bank->per_matrix_bytes_ = objects::tensor_bytes(objects::kQuantQ8Fixture, per);
    if (bank->q_.size() != n * 3 * bank->per_matrix_bytes_)
      return make_error(ErrorCode::kInternal, "q8 encoder produced an unexpected size");
    return std::unique_ptr<ExpertBank>(std::move(bank));
  }
  std::size_t experts() const override { return n_; }
  std::uint64_t bytes_per_expert() const override { return 3 * per_matrix_bytes_; }
  std::unique_ptr<ExpertContext> make_context() const override { return std::make_unique<RefContext>(); }
  Status run(std::size_t e, std::size_t positions, const float* in, float* out, ExpertContext& ctx,
             ExpertTiming* timing) const override {
    if (e >= n_) return make_error(ErrorCode::kOutOfRange, "expert index out of range");
    auto& c = static_cast<RefContext&>(ctx);
    const std::size_t per = std::size_t{shape_.ff} * shape_.hidden;
    c.gate.resize(per);
    c.up.resize(per);
    c.down.resize(per);
    const std::uint8_t* base = q_.data() + e * 3 * per_matrix_bytes_;
    const auto t0 = monotonic_ns();
    float* dst[3] = {c.gate.data(), c.up.data(), c.down.data()};
    for (std::size_t m = 0; m < 3; ++m)
      CLM_RETURN_IF_ERROR(objects::decode_tensor(objects::kQuantQ8Fixture,
                                                 ByteSpan(base + m * per_matrix_bytes_, per_matrix_bytes_),
                                                 std::span<float>(dst[m], per)));
    const auto t1 = monotonic_ns();
    batched_swiglu(c.gate.data(), c.up.data(), c.down.data(), shape_, positions, in, out, c);
    const auto t2 = monotonic_ns();
    if (timing) {
      timing->dequant_ns += t1 - t0;
      timing->gemv_ns += t2 - t1;
    }
    return Status::ok();
  }

 private:
  RefQ8Bank(ExpertShape s, std::size_t n) : shape_(s), n_(n) {}
  ExpertShape shape_;
  std::size_t n_;
  std::uint64_t per_matrix_bytes_ = 0;
  Bytes q_;
};

class ReferenceProvider final : public ExpertKernelProvider {
 public:
  std::string id() const override { return "reference"; }
  std::string description() const override {
    return "reference backend FP32 expert math (deterministic scalar reduction order, no FMA); q8 dequantized per call";
  }
  std::string isa() const override { return "scalar-fp32"; }
  std::vector<std::string> representations() const override {
    return {std::string(objects::kQuantF32), std::string(objects::kQuantQ8Fixture)};
  }
  Result<std::unique_ptr<ExpertBank>> make_bank(const std::string& rep, ExpertShape shape, std::size_t n,
                                                std::uint64_t seed) const override {
    if (n == 0 || shape.hidden == 0 || shape.ff == 0) return make_error(ErrorCode::kInvalidArgument, "empty expert bank");
    if (rep == objects::kQuantF32) return std::unique_ptr<ExpertBank>(std::make_unique<RefF32Bank>(shape, n, seed));
    if (rep == objects::kQuantQ8Fixture) return RefQ8Bank::make(shape, n, seed);
    return make_error(ErrorCode::kInvalidArgument, "reference provider has no representation '" + rep + "'");
  }
};

class StrataCpuStub final : public ExpertKernelProvider {
 public:
  std::string id() const override { return "strata-cpu"; }
  std::string description() const override { return "Strata CPU IQ kernels (not built into this binary)"; }
  std::string isa() const override { return "unavailable"; }
  std::vector<std::string> representations() const override { return {"iq3_s", "iq2_xs"}; }
  Result<std::unique_ptr<ExpertBank>> make_bank(const std::string&, ExpertShape, std::size_t,
                                                std::uint64_t) const override {
    return make_error(ErrorCode::kHardwareUnavailable,
                      "provider 'strata-cpu' is a registration stub: the Strata CPU IQ3_S/IQ2_XS kernels are built by "
                      "another workstream and register through register_external_expert_providers()");
  }
};

}  // namespace

ExpertKernelRegistry& ExpertKernelRegistry::instance() {
  static ExpertKernelRegistry r;
  return r;
}

void ExpertKernelRegistry::register_factory(const std::string& id, ExpertProviderFactory factory) {
  for (auto& [k, f] : factories_)
    if (k == id) {
      f = std::move(factory);
      return;
    }
  factories_.emplace_back(id, std::move(factory));
}

Result<std::unique_ptr<ExpertKernelProvider>> ExpertKernelRegistry::create(const std::string& id) const {
  for (const auto& [k, f] : factories_)
    if (k == id) return f();
  std::string known;
  for (const auto& [k, f] : factories_) known += (known.empty() ? "" : ", ") + k;
  return make_error(ErrorCode::kNotFound, "unknown expert kernel provider '" + id + "' (registered: " + known + ")");
}

std::vector<std::string> ExpertKernelRegistry::ids() const {
  std::vector<std::string> out;
  out.reserve(factories_.size());
  for (const auto& [k, f] : factories_) out.push_back(k);
  return out;
}

void register_builtin_expert_providers() {
  auto& r = ExpertKernelRegistry::instance();
  r.register_factory("reference", []() -> Result<std::unique_ptr<ExpertKernelProvider>> {
    return std::unique_ptr<ExpertKernelProvider>(std::make_unique<ReferenceProvider>());
  });
  r.register_factory("strata-cpu", []() -> Result<std::unique_ptr<ExpertKernelProvider>> {
    return std::unique_ptr<ExpertKernelProvider>(std::make_unique<StrataCpuStub>());
  });
  register_external_expert_providers();
}

}  // namespace clusterlm::bench
