#pragma once
// A tiny random-weight model in the plain "llama" architecture, written as a GGUF that the pinned llama.cpp
// loader accepts. Tests and the bench proof run use it where no real model is available; it says nothing about
// quality or speed. Weights are deterministic in `seed` (F32 or F16, small magnitudes so logits stay finite and
// greedy decoding is well separated). The vocabulary type is "no_vocab" (llama.vocab_size only): the model is
// driven by token IDs and needs no tokenizer.
#include <cstdint>
#include <filesystem>

#include "clusterlm/common/status.hpp"

namespace clusterlm::backends {

// Architectures the generator can write. Each mirrors the tensor layout and metadata keys the pinned llama.cpp loader
// requires for that architecture (see third_party/upstream/llama.cpp/src/models/<arch>.cpp); the weights are random.
enum class TinyArch { kLlama, kLlamaMoe, kQwen2, kQwen3, kGemma, kPhi3, kMamba };
const char* tiny_arch_name(TinyArch a);  // the GGUF general.architecture string

struct TinyLlamaSpec {
  TinyArch arch = TinyArch::kLlama;
  bool tied_embeddings = false;  // no output.weight: the head reuses token_embd.weight (always true for gemma)
  std::uint32_t experts = 4;       // kLlamaMoe only
  std::uint32_t experts_used = 2;  // kLlamaMoe only
  std::uint32_t layers = 2;
  std::uint32_t hidden = 64;
  std::uint32_t heads = 4;
  std::uint32_t kv_heads = 2;   // heads must be a multiple of kv_heads; head dim = hidden / heads
  std::uint32_t ff = 128;
  std::uint32_t vocab = 256;
  std::uint32_t context = 512;  // llama.context_length (training context)
  std::uint64_t seed = 42;
  // F16 weights for the matrices instead of F32 (exercises a second tensor type and halves the file).
  bool f16_matrices = false;
};

// Writes one GGUF file.
Status write_tiny_llama_gguf(const std::filesystem::path& path, const TinyLlamaSpec& spec = {});

}  // namespace clusterlm::backends
