#pragma once
// Fast/Strong/Ultra -> execution profiles, and settings v1 -> v2 (docs/interfaces/profile-schema-v1.md §5).
//
// Pure functions: no I/O, no clocks. The three tiers become the EXAMPLE SET (profiles with deterministic ids and
// `example_of: "tier:<id>"`), so the migration is idempotent and a fresh install seeds exactly what a migrated one has.
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/backend_descriptor.hpp"
#include "clusterlm/profiles/profile.hpp"

namespace clusterlm::migration {

inline constexpr std::string_view kExampleProfilePrefix = "prof_example_";

// The embedded example set: Fast, Strong, Ultra (in that order), untouched defaults (on-demand, 60 s idle release).
std::vector<profiles::Profile> example_profiles();

// Profile id for a tier id ("fast" -> "prof_example_fast").
std::string example_profile_id(std::string_view tier_id);
// Tier id for an example profile id; empty when `profile_id` is not one of the migrated tiers.
std::string tier_id_of_example(std::string_view profile_id);

struct KeepReady {
  bool enabled = false;
  std::uint32_t release_after_idle_minutes = 30;
};

// One catalog tier -> one profile (deterministic). `catalog` supplies the fallback order, `backends` the speculation
// ceiling of the tier's backend. `keep_ready` is the v1 settings policy applied to the lifecycle (it applied to whichever
// tier was prepared; v2 applies it to all migrated profiles).
Result<profiles::Profile> profile_from_tier(const catalog::TierEntry& tier, const catalog::Catalog& catalog,
                                            const domain::BackendRegistry& backends, const KeepReady* keep_ready = nullptr);
Result<std::vector<profiles::Profile>> profiles_from_catalog(const catalog::Catalog& catalog, const domain::BackendRegistry& backends,
                                                             const KeepReady* keep_ready = nullptr);

// Applies the v1 keep_ready policy to a profile's lifecycle (enabled=false -> on-demand + 60 s, v1's fixed idle release).
void apply_keep_ready(profiles::Profile& p, const KeepReady& k);

// settings v1 document text -> settings v2 document text. Every v1 value is mapped, kept under `legacy_v1`, or derived;
// nothing is dropped. A wrong-typed v1 field is an error (the store then treats the document as corrupt and keeps it).
Result<std::string> migrate_father_settings_v1_to_v2(std::string_view v1_document);

}  // namespace clusterlm::migration
