#pragma once
// ReferenceDomain: the deterministic CPU FP32 execution domain over fixture models.
//
// Not thread-safe: one caller drives a domain at a time (as a Node's single worker or a test does).
// Layers are independent of the domain that runs them, so any split of the layer range produces bitwise
// identical logits. kGpuResident objects are *accounted* as GPU but executed on the CPU.
#include <memory>
#include <string>
#include <unordered_map>

#include "clusterlm/domain/execution_domain.hpp"

namespace clusterlm::domain {

class ReferenceDomain : public ExecutionDomain {
 public:
  // Role rules: prefix starts at layer 0 and contains ple_layer; middle is non-empty and excludes it; tail
  // excludes it and ends at n_layers (an empty tail range is a head-only tail).
  static Result<std::unique_ptr<ReferenceDomain>> create(const objects::ModelManifest& manifest, const DomainSpec& spec);

  // Per-object accounting target (default: everything CPU). Affects describe_requirements and the
  // experts_cpu/experts_gpu counters only. Call before describe_requirements()/prepare().
  virtual void set_allocation_targets(std::unordered_map<std::string, objects::AllocationTarget> targets) = 0;
  // Timing of the most recent run_prefix/run_window/run_tail call (counts only; never which experts).
  virtual StageTiming last_timing() const = 0;
};

}  // namespace clusterlm::domain
