#pragma once
// Cross-object validation of a profile/alias set (profile-schema-v1.md §3 level 1, the part that needs more than one
// document). Per-document rules are in profile.hpp (validate(Profile), validate(RoutingAlias)) and run on parse.
#include <string>
#include <vector>

#include "clusterlm/profiles/profile.hpp"

namespace clusterlm::profiles {

inline constexpr std::size_t kMaxProfiles = 256;
inline constexpr std::size_t kMaxAliases = 64;

// Unique profile/alias ids; api_model_id unique across profiles (also those with api:false) and aliases; every fallback
// profile exists, is not the profile itself and the chain is acyclic; every alias candidate exists and its
// workers_available names worker slots of that profile.
Status validate_set(const std::vector<Profile>& profiles, const std::vector<RoutingAlias>& aliases);

// Follows fallback_profile_id from `id`: [id, next, next2, ...] (stops at a profile whose on_worker_loss.then is not
// fallback-profile). Requires a validated set (acyclic).
std::vector<std::string> fallback_chain(const std::vector<Profile>& profiles, const std::string& id);

}  // namespace clusterlm::profiles
