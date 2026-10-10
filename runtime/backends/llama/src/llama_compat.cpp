#include "clusterlm/backends/llama_compat.hpp"

#include <algorithm>
#include <mutex>

#include "llama.h"

namespace clusterlm::backends {
namespace {

const std::vector<std::string_view> kPinned = {
#include "llama_arch_table.inc"
};

// Keep in step with TinyArch in llama_tiny_model.hpp; the architecture test enforces it both ways.
const std::vector<std::string_view> kExercised = {"gemma", "llama", "phi3", "qwen2", "qwen3"};

}  // namespace

const std::vector<std::string_view>& llama_pinned_architectures() { return kPinned; }
const std::vector<std::string_view>& llama_exercised_architectures() { return kExercised; }

LlamaArchSupport llama_architecture_support(std::string_view architecture) {
  if (std::binary_search(kExercised.begin(), kExercised.end(), architecture)) return LlamaArchSupport::kExercised;
  if (std::binary_search(kPinned.begin(), kPinned.end(), architecture)) return LlamaArchSupport::kKnown;
  return LlamaArchSupport::kUnknown;
}

Status probe_llama_hparams(const std::filesystem::path& gguf) {
  static std::once_flag once;
  std::call_once(once, [] { llama_backend_init(); });
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* model = llama_model_load_from_file(gguf.string().c_str(), mp);
  if (model == nullptr)
    return make_error(ErrorCode::kInvalidArgument,
                      "the pinned llama.cpp cannot interpret " + gguf.filename().string() +
                          " (unknown architecture or inconsistent metadata; see its load log)");
  llama_model_free(model);
  return Status::ok();
}

}  // namespace clusterlm::backends
