#include "clusterlm/expert_domains/assignment.hpp"

#include <algorithm>
#include <numeric>

namespace clusterlm::expert_domains {

namespace {
Status invalid(std::string m) { return make_error(ErrorCode::kInvalidArgument, std::move(m)); }
}  // namespace

Result<ExpertAssignment> ExpertAssignment::from_owners(std::vector<std::uint32_t> owner_of, std::uint32_t n_owners) {
  ExpertAssignment a;
  a.n_experts = static_cast<std::uint32_t>(owner_of.size());
  a.n_owners = n_owners;
  a.owner_of = std::move(owner_of);
  a.owned.assign(n_owners, {});
  a.local_index.assign(a.n_experts, 0);
  for (std::uint32_t e = 0; e < a.n_experts; ++e) {
    if (a.owner_of[e] >= n_owners) return invalid("expert owner out of range");
    a.local_index[e] = static_cast<std::uint32_t>(a.owned[a.owner_of[e]].size());
    a.owned[a.owner_of[e]].push_back(e);
  }
  CLM_RETURN_IF_ERROR(a.validate());
  return a;
}

Result<ExpertAssignment> ExpertAssignment::ranges(std::uint32_t n_experts, const std::vector<std::uint32_t>& shares) {
  if (shares.empty() || n_experts < shares.size()) return invalid("need at least one expert per owner");
  std::uint64_t total = 0;
  for (std::uint32_t s : shares) {
    if (s == 0) return invalid("every owner needs a positive share");
    total += s;
  }
  // Largest remainder: floor(share * E / total), leftovers to the biggest remainders (ties to the lower owner).
  // A share so small that it rounds to nothing takes one expert from the largest owner instead.
  std::vector<std::uint32_t> count(shares.size(), 0);
  std::vector<std::pair<std::uint64_t, std::size_t>> rem;  // (remainder numerator, owner)
  std::uint32_t given = 0;
  for (std::size_t i = 0; i < shares.size(); ++i) {
    const std::uint64_t num = std::uint64_t{shares[i]} * n_experts;
    count[i] = static_cast<std::uint32_t>(num / total);
    given += count[i];
    rem.emplace_back(num % total, i);
  }
  std::stable_sort(rem.begin(), rem.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  for (std::size_t k = 0; given < n_experts; ++k, ++given) ++count[rem[k % rem.size()].second];
  for (std::size_t i = 0; i < count.size(); ++i)
    if (count[i] == 0) {
      auto big = std::max_element(count.begin(), count.end());
      --*big;
      count[i] = 1;
    }
  std::vector<std::uint32_t> owner_of;
  owner_of.reserve(n_experts);
  for (std::size_t i = 0; i < count.size(); ++i)
    for (std::uint32_t k = 0; k < count[i]; ++k) owner_of.push_back(static_cast<std::uint32_t>(i));
  return from_owners(std::move(owner_of), static_cast<std::uint32_t>(shares.size()));
}

Result<ExpertAssignment> ExpertAssignment::strided(std::uint32_t n_experts, std::uint32_t n_owners) {
  if (n_owners == 0 || n_experts < n_owners) return invalid("need at least one expert per owner");
  std::vector<std::uint32_t> owner_of(n_experts);
  for (std::uint32_t e = 0; e < n_experts; ++e) owner_of[e] = e % n_owners;
  return from_owners(std::move(owner_of), n_owners);
}

Status ExpertAssignment::validate() const {
  if (n_owners == 0 || n_experts == 0) return invalid("empty assignment");
  if (owner_of.size() != n_experts || local_index.size() != n_experts || owned.size() != n_owners)
    return invalid("inconsistent assignment tables");
  std::size_t total = 0;
  for (std::uint32_t o = 0; o < n_owners; ++o) {
    if (!std::is_sorted(owned[o].begin(), owned[o].end())) return invalid("owned lists must be ascending");
    total += owned[o].size();
  }
  if (total != n_experts) return invalid("assignment does not cover every expert exactly once");
  return Status::ok();
}

}  // namespace clusterlm::expert_domains
