#pragma once
// Profile export / import (profile-schema-v1.md §7). Export never carries machine identities, paths, keys or the local
// library id; import validates everything, never auto-binds a Worker, never enables API exposure or LAN visibility, and
// says what the user still has to do.
#include <functional>
#include <string>
#include <vector>

#include "clusterlm/library/library.hpp"
#include "clusterlm/profiles/profile.hpp"
#include "clusterlm/profiles/topology.hpp"

namespace clusterlm::profiles {

// worker_name: display name of the paired Worker with that fingerprint ("" if unknown).
Profile export_profile(const Profile& p, const std::function<std::string(const std::string& fingerprint)>& worker_name);
// `worker:<slug>`: lower-case, runs of non [a-z0-9] become '-', trimmed to 32 characters; "worker:worker" when empty.
std::string worker_binding_slug(std::string_view display_name);

enum class ImportMode : std::uint8_t {
  kReject,     // same id with different content is an error
  kReplace,    // same id with different content replaces it (revision continues from the existing one)
  kDuplicate,  // always add as a new profile with a fresh id
};

struct ImportResult {
  enum class Action : std::uint8_t { kAdded, kUnchanged, kReplaced, kDuplicated } action = Action::kAdded;
  Profile profile;                  // as it should be stored
  std::vector<std::string> todo;    // what the user must still do, user-readable
};

struct ImportContext {
  const std::vector<Profile>* existing = nullptr;
  const std::vector<RoutingAlias>* aliases = nullptr;
  const library::ModelLibrary* library = nullptr;  // to tell whether the model is already present
  const BindingMap* bindings = nullptr;            // to tell which binding slots are unassigned
  std::string new_api_model_id;                    // chosen by the user when the document's id collides
};

Result<ImportResult> import_profile(std::string_view json, ImportMode mode, const ImportContext& ctx);

// A fresh profile id derived from `name`, unique among `existing`.
std::string fresh_profile_id(std::string_view name, const std::vector<Profile>& existing);

}  // namespace clusterlm::profiles
