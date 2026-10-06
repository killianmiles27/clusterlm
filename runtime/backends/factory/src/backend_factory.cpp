#include "clusterlm/backends/backend_factory.hpp"

namespace clusterlm::backends {

std::vector<std::string> known_backend_names() { return {"reference", "strata"}; }

bool backend_built(std::string_view name) {
  if (name == "reference") return true;
#if defined(CLUSTERLM_FACTORY_HAS_STRATA)
  if (name == "strata") return true;
#endif
  return false;
}

Status check_backend_name(std::string_view name) {
  bool is_known = false;
  std::string list;
  for (const auto& k : known_backend_names()) {
    is_known = is_known || k == name;
    list += (list.empty() ? "" : ", ") + k;
  }
  if (!is_known)
    return make_error(ErrorCode::kInvalidArgument, "unknown backend '" + std::string(name) + "' (known: " + list + ")");
  if (!backend_built(name))
    return make_error(ErrorCode::kUnimplemented,
                      "backend '" + std::string(name) + "' is not available: built without CLUSTERLM_ENABLE_STRATA");
  return Status::ok();
}

Result<std::unique_ptr<domain::BackendAdapter>> make_backend(const BackendOptions& options) {
  CLM_RETURN_IF_ERROR(check_backend_name(options.name));
  if (options.name == "reference") return domain::make_reference_backend();
#if defined(CLUSTERLM_FACTORY_HAS_STRATA)
  if (options.name == "strata") return make_strata_backend(options.strata);
#endif
  return make_error(ErrorCode::kInternal, "backend '" + options.name + "' has no factory");
}

Result<std::unique_ptr<domain::Drafter>> make_backend_drafter(std::string_view name, domain::ExecutionDomain& tail) {
  CLM_RETURN_IF_ERROR(check_backend_name(name));
#if defined(CLUSTERLM_FACTORY_HAS_STRATA)
  if (name == "strata") return make_strata_mtp_drafter(tail);
#else
  (void)tail;
#endif
  return make_error(ErrorCode::kUnimplemented, "backend '" + std::string(name) + "' has no domain-bound MTP drafter");
}

}  // namespace clusterlm::backends
