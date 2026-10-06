// Registration point for expert kernel providers contributed by other workstreams.
//
// "strata-cpu": Strata's CPU IQ expert kernels (IQ3_S / IQ2_XS gate+up, IQ4_NL down; AVX-512 / AVX-2 multi-token
// kernels where they apply, ggml-cpu's vec_dot otherwise) through ClusterLM's CpuExpertKernel
// (runtime/backends/strata, docs/backends/strata-port.md). It replaces the registration stub only in a build that has
// the kernels (-DCLUSTERLM_ENABLE_STRATA_CPU=ON or -DCLUSTERLM_ENABLE_STRATA=ON, which define
// CLUSTERLM_BENCH_HAS_STRATA_CPU for this library); without them the stub stays and reports kHardwareUnavailable.
//
// Banks hold deterministic pseudo-random expert blobs in the real GGUF slice layout [gate | up | down] at the
// requested shape (Flash-Next: H 2560, ff 640). Random bytes with a sane fp16 block scale are valid i-quant blocks, so
// every grid and sign index is exercised; the values are never model data. The `cpu` command, `calibrate` and the
// HardwareProfile writer select the provider by id and key the resulting cpu.expert_bytes_per_s entries by the
// representation names the provider reports ("iq3_s", "iq2_xs").
#include "expert_kernels.hpp"

#ifdef CLUSTERLM_BENCH_HAS_STRATA_CPU
#include <algorithm>
#include <cstring>
#include <random>

#include "clusterlm/backends/strata/cpu_expert_kernel.hpp"
#include "clusterlm/backends/strata/object_map.hpp"
#include "clusterlm/common/clock.hpp"

namespace clusterlm::bench {
namespace {

namespace bs = clusterlm::backends::strata;

struct StrataRepresentation {
  const char* name;
  const char* gate_up;
  const char* down;
};
constexpr StrataRepresentation kRepresentations[] = {{"iq3_s", "iq3_s", "iq4_nl"}, {"iq2_xs", "iq2_xs", "iq4_nl"}};

// fp16 bits of a small positive normal float (the block scales below).
std::uint16_t fp16_bits(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t e = ((x >> 23) & 0xff) - 127 + 15, m = (x >> 13) & 0x3ff;
  return static_cast<std::uint16_t>((e << 10) | m);
}

// `rows` rows of `elems` values of type `t`: random bytes with every block's leading fp16 scale set to a sane value.
void append_random_rows(Bytes& out, const bs::GgmlType& t, std::uint64_t elems, std::uint64_t rows, std::mt19937_64& rng,
                        float scale) {
  const std::uint64_t blocks = elems / t.block_elems * rows;
  const std::size_t at = out.size();
  out.resize(at + blocks * t.block_bytes);
  std::uint8_t* p = out.data() + at;
  for (std::uint64_t i = 0; i < blocks * t.block_bytes; ++i) p[i] = static_cast<std::uint8_t>(rng());
  std::uniform_real_distribution<float> d(0.5f * scale, 1.5f * scale);
  for (std::uint64_t i = 0; i < blocks; ++i) {
    const std::uint16_t h = fp16_bits(d(rng));
    std::memcpy(p + i * t.block_bytes, &h, 2);
  }
}

class StrataCpuBank final : public ExpertBank {
 public:
  StrataCpuBank(std::unique_ptr<bs::CpuExpertKernel> kernel, std::size_t n, std::uint64_t seed)
      : kernel_(std::move(kernel)), n_(n), per_(kernel_->blob_bytes()) {
    const bs::GgmlType& g = *bs::ggml_type_by_name(kernel_->gate_up_type());
    const bs::GgmlType& d = *bs::ggml_type_by_name(kernel_->down_type());
    const std::uint64_t H = kernel_->hidden(), FF = kernel_->ff();
    blobs_.reserve(n * per_);
    std::mt19937_64 rng(seed);
    for (std::size_t e = 0; e < n; ++e) {
      append_random_rows(blobs_, g, H, FF, rng, 0.004f);   // gate
      append_random_rows(blobs_, g, H, FF, rng, 0.004f);   // up
      append_random_rows(blobs_, d, FF, H, rng, 0.02f);    // down
    }
  }

  std::size_t experts() const override { return n_; }
  std::uint64_t bytes_per_expert() const override { return per_; }
  std::unique_ptr<ExpertContext> make_context() const override { return std::make_unique<ExpertContext>(); }
  std::string kernel_path(std::size_t positions) const override {
    return std::string(kernel_->path(static_cast<std::uint32_t>(std::min<std::size_t>(positions, bs::kCpuExpertMaxTokens))));
  }

  Status run(std::size_t e, std::size_t positions, const float* in, float* out, ExpertContext&,
             ExpertTiming* timing) const override {
    if (e >= n_) return make_error(ErrorCode::kOutOfRange, "expert index out of range");
    const std::size_t H = kernel_->hidden();
    const ByteSpan blob(blobs_.data() + e * per_, per_);
    const auto t0 = monotonic_ns();
    // One Strata window is at most kCpuExpertMaxTokens positions; a longer batch runs as consecutive windows.
    for (std::size_t at = 0; at < positions; at += bs::kCpuExpertMaxTokens) {
      const std::size_t n = std::min<std::size_t>(bs::kCpuExpertMaxTokens, positions - at);
      CLM_RETURN_IF_ERROR(kernel_->run(blob, std::span<const float>(in + at * H, n * H), static_cast<std::uint32_t>(n),
                                       std::span<float>(out + at * H, n * H)));
    }
    if (timing) timing->gemv_ns += monotonic_ns() - t0;  // quantization of the activations is inside the kernel
    return Status::ok();
  }

 private:
  std::unique_ptr<bs::CpuExpertKernel> kernel_;
  std::size_t n_;
  std::uint64_t per_;
  Bytes blobs_;
};

class StrataCpuProvider final : public ExpertKernelProvider {
 public:
  std::string id() const override { return "strata-cpu"; }
  std::string description() const override {
    return "Strata CPU IQ expert kernels (IQ3_S/IQ2_XS gate+up, IQ4_NL down) on synthetic blobs; includes per-call "
           "activation quantization, no weight dequantization pass";
  }
  std::string isa() const override { return bs::cpu_expert_isa(); }
  std::vector<std::string> representations() const override {
    std::vector<std::string> r;
    for (const auto& rep : kRepresentations) r.emplace_back(rep.name);
    return r;
  }
  Result<std::unique_ptr<ExpertBank>> make_bank(const std::string& representation, ExpertShape shape, std::size_t n,
                                                std::uint64_t seed) const override {
    if (n == 0) return make_error(ErrorCode::kInvalidArgument, "empty expert bank");
    for (const auto& rep : kRepresentations) {
      if (representation != rep.name) continue;
      CLM_ASSIGN_OR_RETURN(auto kernel, bs::make_cpu_expert_kernel(rep.gate_up, rep.down, shape.hidden, shape.ff));
      return std::unique_ptr<ExpertBank>(std::make_unique<StrataCpuBank>(std::move(kernel), n, seed));
    }
    return make_error(ErrorCode::kInvalidArgument, "strata-cpu has no representation '" + representation + "'");
  }
};

}  // namespace

void register_external_expert_providers() {
  if (!bs::cpu_expert_kernels_available()) return;  // the registration stub stays
  ExpertKernelRegistry::instance().register_factory("strata-cpu", []() -> Result<std::unique_ptr<ExpertKernelProvider>> {
    return std::unique_ptr<ExpertKernelProvider>(std::make_unique<StrataCpuProvider>());
  });
}

}  // namespace clusterlm::bench

#else

namespace clusterlm::bench {

void register_external_expert_providers() {}

}  // namespace clusterlm::bench

#endif
