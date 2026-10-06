#pragma once
// ModelManifest for a llama.cpp-loadable GGUF (any architecture the pinned llama.cpp supports), built from the
// GGUF's own metadata and tensor directory with exact byte ranges. No requantization, no copy.
//
// The ClusterLM manifest/geometry model is Flash-Next shaped (MoE, gated residual streams, PLE). A llama.cpp
// model maps onto it as follows (docs/backends/llama-local.md, ADR 0300):
//   residual_streams = 1; n_experts/n_active_experts = expert_count/expert_used_count (1/1 for dense models);
//   expert_ff = expert (or dense) feed-forward width; ple_layer = 0, ple_ngram = 1, ple_rows = 1 (placeholders:
//   llama models have no PLE); layer kinds: recurrent when the layer owns ssm_* tensors, else full attention.
// Objects: token_embd (kEmbedding), output_head (kOutputHead: output*, final norm and every tensor that belongs
// to no layer), and one kLayerDense object per layer holding all of the layer's tensors as ranges. These exist so
// the Coordinator, catalog and diagnostics see a normal manifest; the llama backend itself maps the GGUF files.
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::backends {

struct LlamaTypeStat {
  std::uint64_t tensors = 0;
  std::uint64_t bytes = 0;
};

// What the model file really contains, reported as stored (never what a backend might convert it to).
struct LlamaModelReport {
  std::string architecture;                          // general.architecture
  std::string name;                                  // general.name (may be empty)
  std::uint32_t n_layers = 0;                        // repeating layers (block_count minus nextn/MTP layers)
  std::uint32_t hidden_size = 0;
  std::uint32_t vocab_size = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t tensor_bytes = 0;
  std::map<std::string, LlamaTypeStat> tensor_types;  // ggml type name -> count and bytes
};

struct LlamaManifestOptions {
  std::string artifact_id;     // default: general.name, else the first file name
  bool hash_objects = false;   // stream every range into source/object digests
  bool hash_shards = false;
};

// `paths` are the GGUF files of one model (a split set in order, or a single file). All must share a directory.
Result<objects::ModelManifest> build_llama_manifest(const std::vector<std::filesystem::path>& paths,
                                                    const LlamaManifestOptions& options = {},
                                                    LlamaModelReport* report = nullptr);

// Builds the manifest and writes <dir>/manifest.json next to the GGUF files so a Coordinator can open `dir` as
// its model directory. Returns the manifest.
Result<objects::ModelManifest> write_llama_model_dir(const std::vector<std::filesystem::path>& paths,
                                                     const LlamaManifestOptions& options = {});

// Expands "<stem>-00001-of-0000N.gguf" to every shard (single files are returned unchanged).
Result<std::vector<std::filesystem::path>> llama_model_files(const std::filesystem::path& any_shard);

}  // namespace clusterlm::backends
