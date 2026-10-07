#pragma once
// TierTokenizerProvider: builds (and caches) each tier's real tokenizer on Father from the tier model's first GGUF
// shard - the shard named first in the model directory's manifest - using GgufBpeTokenizer. The same provider backs
// both seams:
//   * ServiceDeps::tokenizer_for_tier   -> prompt encoding, decoding and stop tokens
//   * ProductionOptions::tokenizers     -> readiness: a tier whose tokenizer cannot be built is Unavailable with the
//                                          reason (unsupported pre-tokenizer, non-ChatML template, missing metadata)
// There is no fallback to the byte tokenizer, and the Fast tier uses GgufBpeTokenizer too (llama.cpp's tokenizer is
// not exposed through runtime/backends/llama). Tokenizers are cached per (file, size, mtime). Thread-safe.
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/config/store.hpp"
#include "clusterlm/father/gguf_bpe_tokenizer.hpp"

namespace clusterlm::father {

class TierTokenizerProvider {
 public:
  explicit TierTokenizerProvider(std::shared_ptr<config::FatherSettingsStore> settings, GgufBpeOptions options = {})
      : settings_(std::move(settings)), options_(std::move(options)) {}

  Result<std::shared_ptr<Tokenizer>> get(const catalog::TierEntry& tier);

 private:
  struct Entry {
    std::string key;  // path|size|mtime
    Result<std::shared_ptr<Tokenizer>> result = make_error(ErrorCode::kInternal, "unset");
  };
  std::shared_ptr<config::FatherSettingsStore> settings_;
  GgufBpeOptions options_;
  std::mutex mu_;
  std::map<std::string, Entry> cache_;  // tier id -> entry
};

}  // namespace clusterlm::father
