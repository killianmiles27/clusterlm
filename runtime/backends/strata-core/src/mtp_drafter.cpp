#include "clusterlm/backends/strata/mtp_drafter.hpp"

#include <cmath>

namespace clusterlm::backends::strata {

void StrataMtpDrafter::fit(std::vector<std::int32_t>& drafts, std::int32_t next_token, std::uint32_t count) {
  if (drafts.size() > count) drafts.resize(count);
  const std::int32_t pad = drafts.empty() ? next_token : drafts.back();
  while (drafts.size() < count) drafts.push_back(pad);
}

std::vector<std::int32_t> StrataMtpDrafter::draft(std::span<const std::int32_t> committed_tokens, std::int32_t next_token,
                                                  std::uint32_t count) {
  std::vector<std::int32_t> drafts;
  if (count == 0) return drafts;
  last_ = engine_.mtp_draft(committed_tokens, next_token, count, nullptr, drafts, nullptr);
  if (!last_.is_ok()) {
    ++fallbacks_;
    drafts.clear();
  }
  fit(drafts, next_token, count);
  return drafts;
}

domain::DraftProposal StrataMtpDrafter::propose(std::span<const std::int32_t> committed_tokens, std::int32_t next_token,
                                                std::uint32_t count, const domain::SamplingParams& params,
                                                domain::Rng& rng) {
  (void)rng;  // Strata draws its draft samples with its own Philox counter (coupled draft sampling)
  if (params.greedy() || count == 0) return domain::DraftProposal{draft(committed_tokens, next_token, count), {}};
  domain::DraftProposal p;
  std::vector<std::vector<float>> probs;
  last_ = engine_.mtp_draft(committed_tokens, next_token, count, &params, p.tokens, &probs);
  const std::uint32_t V = engine_.mtp_vocab();
  bool usable = last_.is_ok() && p.tokens.size() == count && probs.size() == count;
  for (std::size_t j = 0; usable && j < probs.size(); ++j) {
    usable = probs[j].size() == V && p.tokens[j] >= 0 && static_cast<std::uint32_t>(p.tokens[j]) < V &&
             probs[j][static_cast<std::size_t>(p.tokens[j])] > 0.0f;
    double s = 0.0;
    for (float x : probs[j]) {
      if (!(x >= 0.0f) || !std::isfinite(x)) usable = false;
      s += x;
    }
    usable = usable && std::fabs(s - 1.0) < 1e-3;
  }
  if (!usable) {
    // Exactness never depends on the drafter: a deterministic proposal (one-hot Q) is always valid.
    if (last_.is_ok()) last_ = make_error(ErrorCode::kInternal, "strata mtp: sampled drafts without usable distributions");
    ++fallbacks_;
    fit(p.tokens, next_token, count);
    return domain::DraftProposal{std::move(p.tokens), {}};
  }
  p.probs = std::move(probs);
  return p;
}

}  // namespace clusterlm::backends::strata
