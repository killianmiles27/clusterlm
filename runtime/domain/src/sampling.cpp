#include "clusterlm/domain/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace clusterlm::domain {

namespace {
std::uint64_t splitmix(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Rng::Rng(std::uint64_t seed) {
  for (auto& v : s_) v = splitmix(seed);
}

std::uint64_t Rng::next() {
  const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
  const std::uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

double Rng::uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

std::int32_t argmax_token(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i)
    if (logits[i] > logits[best]) best = i;
  return static_cast<std::int32_t>(best);
}

std::vector<float> distribution(std::span<const float> logits, const SamplingParams& p) {
  const std::size_t n = logits.size();
  std::vector<float> out(n, 0.0f);
  if (n == 0) return out;
  if (p.greedy()) {
    out[static_cast<std::size_t>(argmax_token(logits))] = 1.0f;
    return out;
  }
  // Softmax at temperature in double for stability, then top-k / top-p truncation on the sorted order.
  const float mx = *std::max_element(logits.begin(), logits.end());
  std::vector<double> e(n);
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    e[i] = std::exp(static_cast<double>(logits[i] - mx) / static_cast<double>(p.temperature));
    sum += e[i];
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return e[a] > e[b]; });
  std::size_t keep = n;
  if (p.top_k > 0) keep = std::min<std::size_t>(keep, p.top_k);
  if (p.top_p < 1.0f) {
    double acc = 0;
    for (std::size_t i = 0; i < keep; ++i) {
      acc += e[order[i]] / sum;
      if (acc >= static_cast<double>(p.top_p)) {
        keep = i + 1;
        break;
      }
    }
  }
  double kept = 0;
  for (std::size_t i = 0; i < keep; ++i) kept += e[order[i]];
  for (std::size_t i = 0; i < keep; ++i) out[order[i]] = static_cast<float>(e[order[i]] / kept);
  return out;
}

std::int32_t sample_from(std::span<const float> probs, Rng& rng) {
  double total = 0;
  for (float v : probs) total += static_cast<double>(v);
  const double r = rng.uniform() * total;
  double acc = 0;
  std::int32_t last_nonzero = 0;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    if (probs[i] <= 0.0f) continue;
    acc += static_cast<double>(probs[i]);
    last_nonzero = static_cast<std::int32_t>(i);
    if (r < acc) return last_nonzero;
  }
  return last_nonzero;  // rounding at the top end
}

SpeculativeOutcome verify_speculative(std::span<const std::vector<float>> target, std::span<const std::int32_t> drafts,
                                      std::span<const std::vector<float>> draft_probs, Rng& rng) {
  SpeculativeOutcome out;
  const bool one_hot = draft_probs.empty();
  for (std::size_t j = 0; j < drafts.size(); ++j) {
    const auto& P = target[j];
    const auto d = static_cast<std::size_t>(drafts[j]);
    const double p = d < P.size() ? static_cast<double>(P[d]) : 0.0;
    const double q = one_hot ? 1.0 : (d < draft_probs[j].size() ? static_cast<double>(draft_probs[j][d]) : 0.0);
    // q == 0 cannot happen for a token the drafter actually sampled; treat it as a rejection.
    if (q > 0.0 && rng.uniform() < std::min(1.0, p / q)) {
      ++out.accepted_drafts;
      continue;
    }
    // Rejected: sample from the normalized residual max(0, P - Q).
    std::vector<float> residual(P.size(), 0.0f);
    double mass = 0;
    for (std::size_t i = 0; i < P.size(); ++i) {
      const double qi = one_hot ? (i == d ? 1.0 : 0.0) : static_cast<double>(draft_probs[j][i]);
      const double r = std::max(0.0, static_cast<double>(P[i]) - qi);
      residual[i] = static_cast<float>(r);
      mass += r;
    }
    out.next_token = mass > 0 ? sample_from(residual, rng) : sample_from(P, rng);
    return out;
  }
  out.next_token = sample_from(target[drafts.size()], rng);
  return out;
}

}  // namespace clusterlm::domain
