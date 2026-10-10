#pragma once
// Cross-object validation of a profile/alias set (profile-schema-v1.md §3 level 1, the part that needs more than one
// document). Per-document rules are in profile.hpp (validate(Profile), validate(RoutingAlias)) and run on parse.
#include <string>
#include <vector>

#include "clusterlm/profiles/profile.hpp"

namespace clusterlm::profiles {

inline constexpr std::size_t kMaxProfiles = 256;
inline constexpr std::size_t kMaxAliases = 64;

// One cross-object problem, blamed on a single document (the later duplicate, the dangling referrer, a member of a cycle).
struct SetProblem {
  enum class Kind : std::uint8_t { kProfile, kAlias } kind = Kind::kProfile;
  std::size_t index = 0;   // index into the vector the document came from
  std::string message;
  bool dangling = false;   // a reference to a document that does not exist (tolerated at load, refused on create/update)
};
// Every per-document and cross-object problem, in a stable order. Used by the settings loader to quarantine just the
// offending documents instead of rejecting the whole file.
std::vector<SetProblem> find_set_problems(const std::vector<Profile>& profiles, const std::vector<RoutingAlias>& aliases);

// Unique profile/alias ids; api_model_id unique across profiles (also those with api:false) and aliases; every fallback
// profile exists, is not the profile itself and the chain is acyclic; every alias candidate exists and its
// workers_available names worker slots of that profile.
Status validate_set(const std::vector<Profile>& profiles, const std::vector<RoutingAlias>& aliases);

// Follows fallback_profile_id from `id`: [id, next, next2, ...] (stops at a profile whose on_worker_loss.then is not
// fallback-profile). Requires a validated set (acyclic).
std::vector<std::string> fallback_chain(const std::vector<Profile>& profiles, const std::string& id);

}  // namespace clusterlm::profiles
