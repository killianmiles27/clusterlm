#include "clusterlm/domain/backend_adapter.hpp"
#include "clusterlm/domain/reference_domain.hpp"

namespace clusterlm::domain {

namespace {

class ReferenceBackend final : public BackendAdapter {
 public:
  BackendInfo info() const override {
    BackendInfo i;
    i.name = "reference";
    i.build_hash = "reference-cpu-fp32-v1";
    i.supports_gpu = false;
    i.hardware_available = true;
    return i;
  }

  Result<std::unique_ptr<ExecutionDomain>> create_domain(const objects::ModelManifest& manifest,
                                                         const DomainSpec& spec) override {
    CLM_ASSIGN_OR_RETURN(std::unique_ptr<ReferenceDomain> d, ReferenceDomain::create(manifest, spec));
    return std::unique_ptr<ExecutionDomain>(std::move(d));
  }
};

}  // namespace

std::unique_ptr<BackendAdapter> make_reference_backend() { return std::make_unique<ReferenceBackend>(); }

}  // namespace clusterlm::domain
