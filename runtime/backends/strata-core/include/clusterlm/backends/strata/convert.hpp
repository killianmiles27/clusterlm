#pragma once
// Father-side conversion of a Strata model into a ClusterLM model directory (ADR 0200).
//
// Input: Strata's pack directory (index.txt + dense.bin / embd.bin / extra.bin, from tools/iq_pack.py) and the model
// GGUF (first shard; the others are found beside it). Output, written into the GGUF's directory (`out` must be it:
// the manifest names shards relative to the model directory):
//   * strata-dense.bin - every strata-dense object (layer dense, shared expert, embedding, head), converted once;
//   * manifest.json    - the ClusterLM manifest: those objects (conversion_version 1, source == object bytes) and
//                        every routed expert as its three GGUF slices in place (conversion_version 0, no copy).
// The existing CanonicalModelStore and provisioning pipeline then serve Strata's representation unchanged.
// Built with the Strata sources (CLUSTERLM_ENABLE_STRATA or CLUSTERLM_ENABLE_STRATA_CPU): it reads the GGUF with
// Strata's own reader.
#include <filesystem>
#include <functional>
#include <string_view>

#include "clusterlm/common/status.hpp"
#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::backends::strata {

struct ConvertOptions {
  std::filesystem::path pack_dir;
  std::filesystem::path gguf;  // first shard
  std::filesystem::path out;   // empty: the GGUF's directory
};

Result<objects::ModelManifest> convert_model(const ConvertOptions& options,
                                             const std::function<void(std::string_view)>& progress = {});

}  // namespace clusterlm::backends::strata
