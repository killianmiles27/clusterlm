#pragma once
// Provenance of every number the placement scheduler consumes.
//
// The scheduler accepts measurements; it never generates them. A quantity is Synthetic (a development
// estimate that lives only in fixtures), Measured (produced by a benchmark run, not yet qualified against
// the product acceptance procedure) or Qualified. Ordering is meaningful: a derived result is only as
// trustworthy as its weakest input, so aggregation is std::min over this enum.
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace clusterlm::placement {

enum class Provenance : std::uint8_t { kSynthetic = 0, kMeasured = 1, kQualified = 2 };

std::string_view to_string(Provenance p) noexcept;
bool provenance_from_string(std::string_view s, Provenance& out) noexcept;

inline Provenance weakest(Provenance a, Provenance b) noexcept { return a < b ? a : b; }

struct Quantity {
  double value = 0;
  Provenance provenance = Provenance::kSynthetic;
  // Where the number came from, e.g. "synthetic: development estimate" or "bench:run-2026-10-01-a".
  std::string source;

  static Quantity synthetic(double v, std::string src = "synthetic: development estimate") {
    return {v, Provenance::kSynthetic, std::move(src)};
  }
  static Quantity measured(double v, std::string src) { return {v, Provenance::kMeasured, std::move(src)}; }
  friend bool operator==(const Quantity&, const Quantity&) = default;
};

}  // namespace clusterlm::placement
