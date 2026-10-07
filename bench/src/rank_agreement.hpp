#pragma once
// How well one ordering of candidates agrees with another (placement-validate: the placement model's predicted order
// against the measured order of the same plans).
#include <cstddef>
#include <vector>

namespace clusterlm::bench {

struct RankAgreement {
  bool valid = false;          // false: fewer than two candidates, or one of the series is constant (no ordering)
  std::size_t n = 0;
  double spearman = 0;         // rank correlation in [-1, 1] (ties get their average rank)
  double concordant_fraction = 0;  // of the pairs ordered by both series, the share ordered the same way
  bool best_agrees = false;    // the predicted best candidate is also the measured best (first of equal values)
};

// `predicted` and `measured` hold one value per candidate, in the same order. `higher_is_better`: throughput (true) or a
// time (false); it only decides which candidate is "best", the correlation is direction-independent.
RankAgreement compare_rankings(const std::vector<double>& predicted, const std::vector<double>& measured, bool higher_is_better);

}  // namespace clusterlm::bench
