#include "clusterlm/backends/llama_backend.hpp"

#include <utility>

#ifndef CLUSTERLM_LLAMA_PIN
#define CLUSTERLM_LLAMA_PIN "unknown"  // set by CMake from third_party/upstream.json
#endif

namespace clusterlm::backends {
namespace {

class LlamaBackend final : public domain::BackendAdapter {
 public:
  explicit LlamaBackend(LlamaBackendOptions options) : options_(std::move(options)) {}

  domain::BackendInfo info() const override {
    domain::BackendInfo i;
    i.name = "llama";
    i.build_hash = std::string("llama.cpp@") + CLUSTERLM_LLAMA_PIN + (options_.rpc_cache ? "+rpc-cache" : "");
    i.supports_gpu = true;
    // QUALIFICATION-PENDING: availability = both binaries exist and report the pinned build; not probed yet.
    i.hardware_available = false;
    return i;
  }

  Result<std::unique_ptr<domain::ExecutionDomain>> create_domain(const objects::ModelManifest&,
                                                                 const domain::DomainSpec&) override {
    return make_error(ErrorCode::kUnimplemented,
                      "llama backend: a comparison arm only; it cannot host ClusterLM execution domains "
                      "(see docs/backends/llama-rpc.md)");
  }

 private:
  LlamaBackendOptions options_;
};

}  // namespace

std::unique_ptr<domain::BackendAdapter> make_llama_backend(const LlamaBackendOptions& options) {
  return std::make_unique<LlamaBackend>(options);
}

}  // namespace clusterlm::backends
