#pragma once
// The deterministic fixture model, emitted as a split GGUF (2 shards, like the real artifact: shard 2 holds the
// PLE lookup table) with Flash-Next style tensor names, and its manifest built through the real GGUF path
// (build_manifest + hashing). The reference backend runs from this manifest unchanged: dense objects keep the
// DenseLayout float order because tensors are written in that order, and the experts are F32 or ggml Q8_0.
//
// Weights are the same generator streams as write_fixture_model(); only the expert quantization of odd layers
// differs (ggml Q8_0, 34-byte blocks, instead of the fixture's private 36-byte q8_0-fixture) so logits are not
// bit-identical to the raw-shard fixture, but split-vs-unsplit equivalence holds on this model as on the other.
#include <filesystem>

#include "clusterlm/objects/fixture_model.hpp"

namespace clusterlm::objects {

struct FixtureGgufOptions {
  bool q8_experts = true;  // odd layers use ggml Q8_0 experts (as the raw fixture uses q8_0-fixture); false = all F32
};

inline constexpr std::string_view kFixtureGgufShard0 = "fixture-00001-of-00002.gguf";
inline constexpr std::string_view kFixtureGgufShard1 = "fixture-00002-of-00002.gguf";

// Writes the two shards and manifest.json (with object digests) into `dir`; returns the manifest.
Result<ModelManifest> write_fixture_gguf(const FixtureSpec& spec, const std::filesystem::path& dir,
                                         const FixtureGgufOptions& opts = {});

}  // namespace clusterlm::objects
