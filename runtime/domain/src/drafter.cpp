#include "clusterlm/domain/drafter.hpp"

#include "clusterlm/objects/manifest.hpp"
#include "clusterlm/objects/tensor_codec.hpp"
#include "reference_math.hpp"

namespace clusterlm::domain {

namespace {
std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

Result<std::vector<float>> load(const objects::ObjectResolver& r, std::string_view name, std::size_t offset_floats,
                                std::size_t n) {
  CLM_ASSIGN_OR_RETURN(objects::ProvisionedObject p, r.resolve(name));
  if (p.bytes.size() < (offset_floats + n) * 4) return make_error(ErrorCode::kDataLoss, "object too small");
  std::vector<float> out(n);
  CLM_RETURN_IF_ERROR(objects::decode_tensor(p.entry->representation.quant_type, p.bytes.subspan(offset_floats * 4, n * 4), out));
  return out;
}
}  // namespace

Result<std::unique_ptr<MtpFixtureDrafter>> MtpFixtureDrafter::create(const objects::ModelManifest& manifest,
                                                                     const objects::ObjectResolver& resolver) {
  std::unique_ptr<MtpFixtureDrafter> d(new MtpFixtureDrafter());
  d->hidden_ = manifest.geometry.hidden_size;
  d->vocab_ = manifest.geometry.vocab_size;
  const std::size_t H = d->hidden_, V = d->vocab_;
  CLM_ASSIGN_OR_RETURN(d->embd_, load(resolver, objects::kEmbeddingObjectName, 0, V * H));
  CLM_ASSIGN_OR_RETURN(d->norm_, load(resolver, objects::kMtpObjectName, 0, H));
  CLM_ASSIGN_OR_RETURN(d->head_, load(resolver, objects::kMtpObjectName, H, V * H));
  return d;
}

std::int32_t MtpFixtureDrafter::step(std::int32_t prev) const {
  const std::size_t H = hidden_;
  if (prev < 0 || static_cast<std::uint32_t>(prev) >= vocab_) return 0;
  std::vector<float> x(H);
  refmath::rmsnorm(embd_.data() + static_cast<std::size_t>(prev) * H, norm_.data(), H, x.data());
  std::int32_t best = 0;
  float best_logit = refmath::dot(head_.data(), x.data(), H);
  for (std::uint32_t v = 1; v < vocab_; ++v) {
    const float l = refmath::dot(head_.data() + std::size_t{v} * H, x.data(), H);
    if (l > best_logit) {  // ties resolve to the lowest token index
      best_logit = l;
      best = static_cast<std::int32_t>(v);
    }
  }
  return best;
}

std::vector<float> MtpFixtureDrafter::step_logits(std::int32_t prev) const {
  const std::size_t H = hidden_;
  std::vector<float> logits(vocab_, 0.0f);
  if (prev < 0 || static_cast<std::uint32_t>(prev) >= vocab_) return logits;
  std::vector<float> x(H);
  refmath::rmsnorm(embd_.data() + static_cast<std::size_t>(prev) * H, norm_.data(), H, x.data());
  for (std::uint32_t v = 0; v < vocab_; ++v) logits[v] = refmath::dot(head_.data() + std::size_t{v} * H, x.data(), H);
  return logits;
}

DraftProposal MtpFixtureDrafter::propose(std::span<const std::int32_t> committed, std::int32_t next_token,
                                         std::uint32_t count, const SamplingParams& params, Rng& rng) {
  if (params.greedy()) return DraftProposal{draft(committed, next_token, count), {}};
  DraftProposal out;
  std::int32_t prev = next_token;
  for (std::uint32_t i = 0; i < count; ++i) {
    auto probs = distribution(step_logits(prev), params);
    prev = sample_from(probs, rng);
    out.tokens.push_back(prev);
    out.probs.push_back(std::move(probs));
  }
  return out;
}

std::vector<std::int32_t> MtpFixtureDrafter::draft(std::span<const std::int32_t>, std::int32_t next_token,
                                                   std::uint32_t count) {
  std::vector<std::int32_t> out;
  std::int32_t prev = next_token;
  for (std::uint32_t i = 0; i < count; ++i) {
    prev = step(prev);
    out.push_back(prev);
  }
  return out;
}

std::vector<std::int32_t> ScriptedDrafter::draft(std::span<const std::int32_t> committed_tokens, std::int32_t,
                                                 std::uint32_t count) {
  std::vector<std::int32_t> out;
  const std::uint64_t base = committed_tokens.size();  // next_token sits at this position
  for (std::uint32_t j = 1; j <= count; ++j) {
    const std::uint64_t pos = base + j;
    std::int32_t tok = pos < ref_.size() ? ref_[pos] : 0;
    const std::uint64_t h = mix64(cfg_.seed ^ mix64(pos));
    bool corrupt = cfg_.corrupt_draft_index && *cfg_.corrupt_draft_index == j;
    for (std::uint64_t p : cfg_.corrupt_positions) corrupt = corrupt || p == pos;
    if (cfg_.corruption_rate > 0.0 && static_cast<double>(h >> 11) / 9007199254740992.0 < cfg_.corruption_rate) corrupt = true;
    if (corrupt && cfg_.vocab >= 2) {
      // Always a different token than the truth, deterministically.
      tok = static_cast<std::int32_t>((static_cast<std::uint64_t>(tok) + 1 + (h >> 8) % (cfg_.vocab - 1)) % cfg_.vocab);
    }
    out.push_back(tok);
  }
  return out;
}

}  // namespace clusterlm::domain
