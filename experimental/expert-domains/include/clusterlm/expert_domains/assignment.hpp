#pragma once
// ExpertAssignment: which owner holds which routed expert. The same assignment applies to every layer.
//
// Owner 0 is Father (it executes its share locally, overlapped with the remote waits); owners 1..R are the
// remote expert domains. "Local expert id" on the wire is the index of an expert inside its owner's ascending
// list, so a domain never needs the global numbering.
#include <cstdint>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::expert_domains {

inline constexpr std::uint32_t kFatherOwner = 0;

struct ExpertAssignment {
  std::uint32_t n_experts = 0;
  std::uint32_t n_owners = 0;
  std::vector<std::uint32_t> owner_of;            // [expert]
  std::vector<std::uint32_t> local_index;         // [expert] index inside owned[owner_of[expert]]
  std::vector<std::vector<std::uint32_t>> owned;  // [owner] ascending global expert ids

  // Owners may hold no experts (Father can own none); remote domains must hold at least one (checked by their
  // constructors). Contiguous id ranges, sized proportionally to `shares` (owner order; every share > 0, shares.size() >=
  // 1, n_experts >= shares.size()). Largest-remainder rounding, so the ranges tile [0, n_experts) exactly.
  static Result<ExpertAssignment> ranges(std::uint32_t n_experts, const std::vector<std::uint32_t>& shares);
  // Expert e belongs to owner e % n_owners (spreads adjacent, often co-selected, experts across owners).
  static Result<ExpertAssignment> strided(std::uint32_t n_experts, std::uint32_t n_owners);
  // Explicit owner per expert (e.g. a hot/cold split computed from routing statistics).
  static Result<ExpertAssignment> from_owners(std::vector<std::uint32_t> owner_of, std::uint32_t n_owners);

  Status validate() const;
};

}  // namespace clusterlm::expert_domains
