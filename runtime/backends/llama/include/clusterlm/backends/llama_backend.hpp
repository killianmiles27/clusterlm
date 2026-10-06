#pragma once
// LlamaLocalBackend: the Fast tier's Father-local execution backend, built on the pinned llama.cpp's public C API.
//
// Fast runs the whole model on Father (docs/backends/llama-local.md, ADR 0300). The ExecutionDomain contract is
// satisfied with two Father domains over one shared llama.cpp context, exactly the shape the Coordinator already
// drives for any Father-only plan ("0-L@father,L-L@father"):
//   kPrefix [0, L)   consumes token IDs, runs the complete llama.cpp forward pass for the window and keeps the
//                    per-position logits Father-local; it returns an opaque Father-local handle as its
//                    "activations" (one dummy 3-float record per position — nothing model-sized, never sent);
//   kTail   [L, L)   head-only: hands the window's logits to the Coordinator.
// Both domains run the same speculative-window protocol through their WindowLedger. Only the prefix domain
// touches llama.cpp's KV cache: a window decodes q positions as one llama_batch with logits for every position;
// commit(n) keeps the first n positions with llama_memory_seq_rm(seq, base+n, -1); abort_window removes from the
// window base. Middle stages make no sense for a whole-model Father backend: create_domain refuses them and the
// activation-boundary entry points return kUnimplemented.
//
// Weights are never provisioned or requantized: the backend maps the GGUF files named by the manifest's shards
// from LlamaBackendOptions::model_dir (Father's canonical files; mmap is fine on Father). The ObjectResolver
// passed to prepare() is not used for weights.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "clusterlm/backends/llama_manifest.hpp"
#include "clusterlm/domain/backend_adapter.hpp"

namespace clusterlm::backends {

struct LlamaBackendOptions {
  // Directory holding the GGUF files named by the manifest's shards (Father's canonical model directory).
  std::filesystem::path model_dir;
  // Layers offloaded to the GPU by llama.cpp: 0 = CPU only (the only mode a CPU-only build supports), negative =
  // all. GPU use also needs a CUDA build (CLUSTERLM_LLAMA_CUDA); otherwise llama.cpp keeps everything on the CPU.
  std::int32_t n_gpu_layers = 0;
  std::int32_t n_threads = 0;         // decode threads; 0 = half the logical CPUs (at least 1)
  std::int32_t n_threads_batch = 0;   // prefill threads; 0 = n_threads
  bool use_mmap = true;
  // Windows longer than this are prefill chunks: only the final position's logits are computed and returned
  // (Logits::positions == 1), which is all the Coordinator reads from a prefill chunk, and avoids evaluating the
  // output head for every prompt position. Speculative verification windows (q <= this) return every position.
  std::uint32_t full_logits_max_positions = 32;
};

// Counts of live llama.cpp objects created by this backend, for leak checks: after every domain is released
// both are zero.
struct LlamaLiveCounts {
  std::int64_t models = 0;
  std::int64_t contexts = 0;
};
LlamaLiveCounts llama_live_counts();

struct LlamaDomainStats {
  std::uint64_t decode_calls = 0;
  std::uint64_t positions_decoded = 0;    // tokens passed through llama_decode (including recomputation)
  std::uint64_t kv_trims = 0;             // partial llama_memory_seq_rm calls
  std::uint64_t recompute_fallbacks = 0;  // seq_rm refused (recurrent state without rollback): re-decoded the prefix
};

// Implemented by both Father domains of the llama backend (dynamic_cast from ExecutionDomain).
class LlamaLocalDomain : public domain::ExecutionDomain {
 public:
  // What the model files contain, as stored (valid once the domain exists; the backend never converts tensors).
  virtual const LlamaModelReport& model_report() const = 0;
  virtual LlamaDomainStats llama_stats() const = 0;
  // The pinned llama.cpp commit this backend was built against.
  virtual std::string llama_pin() const = 0;
};

std::unique_ptr<domain::BackendAdapter> make_llama_backend(const LlamaBackendOptions& options);

}  // namespace clusterlm::backends
