#include "clusterlm/backends/strata_backend.hpp"

#include <utility>

#if defined(CLUSTERLM_STRATA_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

#ifndef CLUSTERLM_STRATA_PIN
#define CLUSTERLM_STRATA_PIN "unknown"  // set by CMake from third_party/upstream.json
#endif

namespace clusterlm::backends {
namespace {

using domain::BackendAdapter;
using domain::BackendInfo;
using domain::BoundaryLayout;
using domain::CommitAck;
using domain::CommitRequest;
using domain::DomainMetrics;
using domain::DomainRequirements;
using domain::DomainSpec;
using domain::ExecutionDomain;
using domain::Logits;
using domain::StageActivations;
using domain::StageRole;
using domain::WindowRequest;

Status pending(std::string_view what) {
  return make_error(ErrorCode::kUnimplemented,
                    std::string("strata backend: ") + std::string(what) +
                        " is QUALIFICATION-PENDING (needs the CUDA Verifier; see docs/backends/strata-port.md)");
}

bool cuda_device_available(int device) {
#if defined(CLUSTERLM_STRATA_HAVE_CUDA)
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && device >= 0 && device < count;
#else
  (void)device;
  return false;
#endif
}

// One stage of the model on one CUDA device. Intended to own what Strata's `GpuStage` owns
// (generate.cpp:990): WeightTable, NativeDense, NativeHead, SessionState, ExpertCache, Verifier, streams.
class StrataDomain final : public ExecutionDomain {
 public:
  StrataDomain(objects::ModelManifest manifest, DomainSpec spec, StrataBackendOptions options)
      : manifest_(std::move(manifest)), spec_(std::move(spec)), options_(options) {}

  const DomainSpec& spec() const override { return spec_; }
  BoundaryLayout boundary() const override { return BoundaryLayout::for_geometry(manifest_.geometry); }

  Result<DomainRequirements> describe_requirements() const override {
    // QUALIFICATION-PENDING: gpu/cpu weight bytes from manifest.total_bytes(spec_.layers) split by
    // AllocationTarget; state_bytes from Strata's session_bytes(g, max_context, k, lb, le) (pure arithmetic,
    // session.hpp:98); window/scratch/staging from Verifier::init sizing. Reporting guesses would corrupt admission.
    return pending("describe_requirements");
  }

  Status prepare(const objects::ObjectResolver&) override {
    // QUALIFICATION-PENDING: Strata's loaders take a pack directory (weights.cpp:148 index.txt,
    // expert_source.hpp:442 experts.bin). Needs loaders over resolver byte spans, restricted to spec_.layers
    // (WeightTable `skip`).
    if (!cuda_device_available(options_.cuda_device)) {
      return make_error(ErrorCode::kHardwareUnavailable, "strata backend: no usable CUDA device");
    }
    return pending("prepare");
  }

  Status open_session(Epoch, SessionId) override { return pending("open_session"); }

  Result<StageActivations> run_prefix(const WindowRequest&, std::span<const std::int32_t>) override {
    if (spec_.role != StageRole::kPrefix) {
      return make_error(ErrorCode::kFailedPrecondition, "run_prefix on a non-prefix domain");
    }
    // QUALIFICATION-PENDING: Verifier::run with set_stage(lb=0, le, nullptr, handoff_out), then transpose the
    // field-major hand-off buffer into position-major StageActivations (strata-port.md, finding 1).
    return pending("run_prefix");
  }

  Result<StageActivations> run_window(const WindowRequest&, const StageActivations&) override {
    if (spec_.role == StageRole::kPrefix) {
      return make_error(ErrorCode::kFailedPrecondition, "run_window on a prefix domain");
    }
    // QUALIFICATION-PENDING: transpose input into the mapped hand_in buffer, Verifier::run with a null/dummy
    // token array (finding 5), transpose hand_out back.
    return pending("run_window");
  }

  Result<Logits> run_tail(const WindowRequest&, const StageActivations&) override {
    if (spec_.role != StageRole::kTail) {
      return make_error(ErrorCode::kFailedPrecondition, "run_tail on a non-tail domain");
    }
    // QUALIFICATION-PENDING: head logits via Verifier::copy_logits; logits stay Father-local.
    return pending("run_tail");
  }

  Result<CommitAck> commit_window(const CommitRequest&) override {
    // QUALIFICATION-PENDING: Verifier::commit(accepted) once per window id; the idempotent replay of an
    // already-committed window must return the cached CommitAck without calling commit again (finding 4).
    return pending("commit_window");
  }

  Status abort_session(Epoch, SessionId) override {
    // QUALIFICATION-PENDING: Strata has no abort-without-commit (finding 3); the whole session state is discarded.
    return pending("abort_session");
  }

  Status release() override { return Status::ok(); }  // nothing is allocated by the skeleton

  DomainMetrics read_metrics() const override { return {}; }

 private:
  objects::ModelManifest manifest_;
  DomainSpec spec_;
  StrataBackendOptions options_;
};

class StrataBackend final : public BackendAdapter {
 public:
  explicit StrataBackend(StrataBackendOptions options) : options_(options) {}

  BackendInfo info() const override {
    BackendInfo i;
    i.name = "strata";
    i.build_hash = std::string("strata@") + CLUSTERLM_STRATA_PIN;
    i.supports_gpu = true;
    i.hardware_available = cuda_device_available(options_.cuda_device);
    return i;
  }

  Result<std::unique_ptr<ExecutionDomain>> create_domain(const objects::ModelManifest& manifest,
                                                         const DomainSpec& spec) override {
    const auto& g = manifest.geometry;
    if (spec.layers.empty() || spec.layers.end > g.n_layers) {
      return make_error(ErrorCode::kInvalidArgument, "strata backend: layer range outside the model");
    }
    if (spec.max_window == 0 || spec.max_window > options_.max_window_cap) {
      return make_error(ErrorCode::kInvalidArgument, "strata backend: max_window exceeds the Verifier window limit");
    }
    // The layer holding the PLE lookup consumes token-derived n-gram rows (Verifier::ple_stage), so it must stay on
    // the token-owning prefix domain; a token-free stage there would need token IDs on a Node.
    if (spec.role != StageRole::kPrefix && spec.layers.contains(g.ple_layer)) {
      return make_error(ErrorCode::kInvalidArgument,
                        "strata backend: the PLE layer must be owned by the Father prefix domain");
    }
    if (!cuda_device_available(options_.cuda_device)) {
      return make_error(ErrorCode::kHardwareUnavailable, "strata backend: no usable CUDA device");
    }
    // QUALIFICATION-PENDING: a real domain construction needs the CUDA Verifier, which cannot yet be
    // validated against hardware. Refuse rather than hand out a domain that cannot run.
    return pending("create_domain");
  }

 private:
  StrataBackendOptions options_;
};

}  // namespace

std::unique_ptr<domain::BackendAdapter> make_strata_backend(const StrataBackendOptions& options) {
  return std::make_unique<StrataBackend>(options);
}

}  // namespace clusterlm::backends
