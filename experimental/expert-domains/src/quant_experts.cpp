#include "clusterlm/expert_domains/quant_experts.hpp"

#include "expert_kernels.hpp"

namespace clusterlm::expert_domains {

namespace {
std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
}  // namespace

struct QuantExperts::Impl {
  std::uint32_t hidden = 0, first_layer = 0;
  std::vector<std::unique_ptr<bench::ExpertBank>> banks;  // [layer - first_layer]
  std::unique_ptr<bench::ExpertContext> ctx;
  std::vector<float> out;
};

QuantExperts::QuantExperts() : impl_(std::make_unique<Impl>()) {}
QuantExperts::~QuantExperts() = default;

Result<ExpertKernelSpec> parse_expert_kernel(const std::string& text, std::uint64_t seed) {
  ExpertKernelSpec s;
  s.seed = seed;
  std::string t = text;
  if (t.rfind("strata-", 0) == 0) t = t.substr(7);
  if (t.empty() || t == "fixture") return s;
  if (t == "iq3_s" || t == "iq2_xs") {
    s.representation = t;
    return s;
  }
  return make_error(ErrorCode::kInvalidArgument, "unknown expert kernel '" + text + "' (fixture, iq3_s, iq2_xs)");
}

Result<std::unique_ptr<QuantExperts>> QuantExperts::create(const ExpertKernelSpec& spec, std::uint32_t hidden,
                                                           std::uint32_t ff, std::uint32_t first_layer,
                                                           std::uint32_t end_layer, std::uint32_t n_experts,
                                                           std::uint64_t owner_key) {
  if (!spec.quantized()) return make_error(ErrorCode::kInvalidArgument, "QuantExperts needs a quantized representation");
  if (n_experts == 0 || first_layer >= end_layer) return make_error(ErrorCode::kInvalidArgument, "empty expert set");
  bench::register_builtin_expert_providers();
  CLM_ASSIGN_OR_RETURN(auto provider, bench::ExpertKernelRegistry::instance().create("strata-cpu"));
  std::unique_ptr<QuantExperts> q(new QuantExperts());
  q->impl_->hidden = hidden;
  q->impl_->first_layer = first_layer;
  for (std::uint32_t L = first_layer; L < end_layer; ++L) {
    CLM_ASSIGN_OR_RETURN(auto bank, provider->make_bank(spec.representation, bench::ExpertShape{hidden, ff}, n_experts,
                                                        mix64(spec.seed ^ mix64(L) ^ mix64(owner_key + 0x51ED))));
    q->resident_bytes_ += bank->bytes_per_expert() * bank->experts();
    if (q->kernel_path_.empty()) q->kernel_path_ = bank->kernel_path(1);
    if (!q->impl_->ctx) q->impl_->ctx = bank->make_context();
    q->impl_->banks.push_back(std::move(bank));
  }
  q->impl_->out.assign(hidden, 0.0f);
  return q;
}

Status QuantExperts::accumulate(std::uint32_t layer, std::uint32_t local_expert, const float* h, float weight, float* y) {
  Impl& m = *impl_;
  if (layer < m.first_layer || layer - m.first_layer >= m.banks.size())
    return make_error(ErrorCode::kOutOfRange, "layer not held by this expert set");
  CLM_RETURN_IF_ERROR(m.banks[layer - m.first_layer]->run(local_expert, 1, h, m.out.data(), *m.ctx, nullptr));
  for (std::uint32_t c = 0; c < m.hidden; ++c) y[c] += weight * m.out[c];
  return Status::ok();
}

}  // namespace clusterlm::expert_domains
