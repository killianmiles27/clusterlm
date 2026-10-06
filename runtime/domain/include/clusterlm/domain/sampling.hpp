#pragma once
// Father-local sampling and speculative verification.
//
// Everything here operates on vocabulary distributions and token IDs, so it runs only on Father (prefix/tail
// owner). It lives in the domain library because the MTP drafter (also Father-only) shares it.
//
// Speculative sampling follows Leviathan et al. 2023 / Chen et al. 2023: for drafted token d_j with drafter
// probability Q_j(d_j) and target probability P_{j-1}(d_j), accept with probability min(1, P/Q); on the first
// rejection emit a token from norm(max(0, P_{j-1} - Q_j)) and stop; if every draft is accepted emit a bonus token
// from P_k. The emitted sequence is then distributed exactly as sampling from the target alone, whatever the
// drafter does. A deterministic drafter is the special case Q_j = one-hot(d_j).
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clusterlm::domain {

struct SamplingParams {
  float temperature = 0.0f;  // 0 = greedy (argmax, ties -> lowest token id)
  std::uint32_t top_k = 0;   // 0 = disabled
  float top_p = 1.0f;        // 1 = disabled
  std::uint64_t seed = 0;
  bool greedy() const { return temperature <= 0.0f; }
};

// Deterministic, platform-independent RNG (xoshiro256**) so sampling is reproducible across machines.
class Rng {
 public:
  explicit Rng(std::uint64_t seed);
  std::uint64_t next();
  double uniform();  // [0, 1)

 private:
  std::uint64_t s_[4];
};

std::int32_t argmax_token(std::span<const float> logits);
// Target distribution after temperature, top-k and top-p (probabilities sum to 1; filtered tokens are 0).
std::vector<float> distribution(std::span<const float> logits, const SamplingParams& params);
std::int32_t sample_from(std::span<const float> probs, Rng& rng);

struct SpeculativeOutcome {
  std::uint32_t accepted_drafts = 0;  // drafts accepted before the first rejection (0..k)
  std::int32_t next_token = -1;       // the resampled token on rejection, or the bonus token if all accepted
};

// target_probs: k+1 distributions (positions x0..x_k of the window); draft_tokens: k drafted tokens;
// draft_probs: k drafter distributions, or empty for a deterministic (one-hot) drafter.
SpeculativeOutcome verify_speculative(std::span<const std::vector<float>> target_probs,
                                      std::span<const std::int32_t> draft_tokens,
                                      std::span<const std::vector<float>> draft_probs, Rng& rng);

}  // namespace clusterlm::domain
