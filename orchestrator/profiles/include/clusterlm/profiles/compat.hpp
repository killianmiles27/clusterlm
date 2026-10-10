#pragma once
// Profile validation level 2 (profile-schema-v1.md §3): does this profile's model, on this backend, with this topology,
// have a chance of running — decided from the Host library and the backend descriptor, before any provisioning.
// Every finding has a stable machine code and a human sentence; a profile with a blocker can never be reported ready.
#include <functional>
#include <string>
#include <vector>

#include "clusterlm/domain/backend_descriptor.hpp"
#include "clusterlm/library/library.hpp"
#include "clusterlm/profiles/profile.hpp"

namespace clusterlm::profiles {

// Descriptor-side facts of a library record (check_model input).
domain::ModelFacts model_facts(const library::ModelRecord& r, bool experimental_opt_in = false);

struct ProfileCompat {
  library::Resolution model;                 // how the identity resolved (never guessed)
  std::optional<domain::CompatReport> report;  // set when the backend and the model resolved
  std::vector<domain::Finding> findings;     // profile-level findings (the report carries model/backend ones)
  bool has_blocker() const;
  // The label to show: the model/backend label, or Unsupported when nothing could be checked.
  domain::CompatLabel label() const;
};

struct CompatContext {
  const library::ModelLibrary* library = nullptr;
  const domain::BackendRegistry* backends = nullptr;
  // Runtime status of a backend descriptor on this machine (built? runtime present?).
  std::function<domain::BackendRuntimeStatus(const domain::BackendDescriptor&)> runtime;
  // Per-model experimental opt-in (library record), by model id.
  std::function<bool(const std::string& model_id)> experimental_opt_in;
};

ProfileCompat check_profile(const Profile& p, const CompatContext& ctx);

// Validates profile.backend.options against the descriptor's declared options.
Status validate_backend_options(const BackendRef& ref, const domain::BackendDescriptor& d);

}  // namespace clusterlm::profiles
