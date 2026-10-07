#include "rank_agreement.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace clusterlm::bench {

namespace {

// 1-based ranks, ascending; equal values share the mean of the ranks they span.
std::vector<double> average_ranks(const std::vector<double>& v) {
  std::vector<std::size_t> order(v.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
  std::vector<double> ranks(v.size());
  for (std::size_t i = 0; i < order.size();) {
    std::size_t j = i;
    while (j + 1 < order.size() && v[order[j + 1]] == v[order[i]]) ++j;
    const double rank = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
    for (std::size_t k = i; k <= j; ++k) ranks[order[k]] = rank;
    i = j + 1;
  }
  return ranks;
}

std::size_t best_index(const std::vector<double>& v, bool higher_is_better) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i)
    if (higher_is_better ? v[i] > v[best] : v[i] < v[best]) best = i;
  return best;
}

}  // namespace

RankAgreement compare_rankings(const std::vector<double>& predicted, const std::vector<double>& measured, bool higher_is_better) {
  RankAgreement out;
  out.n = std::min(predicted.size(), measured.size());
  if (out.n < 2 || predicted.size() != measured.size()) return out;
  const auto rp = average_ranks(predicted), rm = average_ranks(measured);
  const double mean = (static_cast<double>(out.n) + 1.0) / 2.0;
  double cov = 0, vp = 0, vm = 0;
  for (std::size_t i = 0; i < out.n; ++i) {
    cov += (rp[i] - mean) * (rm[i] - mean);
    vp += (rp[i] - mean) * (rp[i] - mean);
    vm += (rm[i] - mean) * (rm[i] - mean);
  }
  if (vp == 0 || vm == 0) return out;  // a constant series orders nothing
  out.spearman = cov / std::sqrt(vp * vm);
  std::size_t pairs = 0, concordant = 0;
  for (std::size_t i = 0; i < out.n; ++i)
    for (std::size_t j = i + 1; j < out.n; ++j) {
      const double dp = predicted[i] - predicted[j], dm = measured[i] - measured[j];
      if (dp == 0 || dm == 0) continue;
      ++pairs;
      concordant += (dp > 0) == (dm > 0);
    }
  out.concordant_fraction = pairs > 0 ? static_cast<double>(concordant) / static_cast<double>(pairs) : 0.0;
  out.best_agrees = best_index(predicted, higher_is_better) == best_index(measured, higher_is_better);
  out.valid = true;
  return out;
}

}  // namespace clusterlm::bench
