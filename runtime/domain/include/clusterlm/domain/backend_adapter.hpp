#pragma once
// BackendAdapter: factory for execution domains on one physical machine.
//
// Backends: "reference" (deterministic CPU FP32 math over fixture models, always available), "strata"
// (pinned Strata kernels; requires CUDA, compiled only with CLUSTERLM_ENABLE_STRATA), "llama" (pinned
// llama.cpp comparison arm). Callers never see raw backend pointers or CUDA launch details.
#include <memory>
#include <string>

#include "clusterlm/domain/execution_domain.hpp"

namespace clusterlm::domain {

struct BackendInfo {
  std::string name;             // "reference", "strata", ...
  std::string build_hash;       // identifies the exact backend build; part of PreparePlan admission
  bool supports_gpu = false;
  bool hardware_available = false;  // false when the required GPU/runtime is absent in this environment
};

class BackendAdapter {
 public:
  virtual ~BackendAdapter() = default;
  virtual BackendInfo info() const = 0;
  virtual Result<std::unique_ptr<ExecutionDomain>> create_domain(const objects::ModelManifest& manifest,
                                                                  const DomainSpec& spec) = 0;
};

// The always-available deterministic CPU reference backend.
std::unique_ptr<BackendAdapter> make_reference_backend();

}  // namespace clusterlm::domain
