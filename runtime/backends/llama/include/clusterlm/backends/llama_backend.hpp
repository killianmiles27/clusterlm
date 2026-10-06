#pragma once
// llama.cpp comparison arm: reports the pinned build and RPC settings used for throughput baselines.
//
// SKELETON. The arm is intended to drive stock llama-server / ggml-rpc-server as external processes
// (docs/backends/llama-rpc.md); it does not execute ClusterLM execution domains, because llama.cpp's RPC splits
// ggml graphs and moves raw ggml tensors, not the ClusterLM boundary ABI, and does not honour the
// "token IDs never leave Father" rule. create_domain therefore always refuses.
#include <memory>
#include <string>

#include "clusterlm/domain/backend_adapter.hpp"

namespace clusterlm::backends {

struct LlamaBackendOptions {
  std::string server_binary;       // path to llama-server (external process, not linked)
  std::string rpc_server_binary;   // path to ggml-rpc-server
  bool rpc_cache = false;          // `-c` on the RPC server; must stay false for lease-store-compatible runs
};

std::unique_ptr<domain::BackendAdapter> make_llama_backend(const LlamaBackendOptions& options);

}  // namespace clusterlm::backends
