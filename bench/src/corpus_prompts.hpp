#pragma once
// Prompts for acceptance and throughput runs (HQ-MTP-01, HQ-PERF-*): a corpus FILE or a DIRECTORY of files turned into
// token prompts of a given length.
//
//   * Without a tokenizer (the fixture model) the bytes of the text are the token ids, folded into the vocabulary.
//   * With `--tokenizer-gguf FILE` the corpus is tokenized by the real Father tokenizer (father::GgufBpeTokenizer, built
//     from the GGUF's metadata only; docs/tokenizer.md). The text and the token ids exist only in this process: they are
//     never logged, never put in a result file and never sent to a Node. Results carry counts only.
//
// A directory gives one prompt per file (shorter text is repeated to the requested length, as before). A single file
// longer than twice the prompt length is cut into consecutive windows (at most kMaxCorpusPrompts), so a context sweep
// has distinct prompts; a shorter one gives one repeated prompt.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/father/tokenizer.hpp"

namespace clusterlm::bench {

inline constexpr std::size_t kMaxCorpusPrompts = 8;           // windows taken from one file
inline constexpr std::size_t kMaxCorpusBytesPerFile = 4u << 20;  // text read per file

struct CorpusPrompts {
  std::vector<std::vector<std::int32_t>> prompts;
  // Tokenizer ids were reduced modulo the model's vocabulary (only allowed for the fixture model, whose vocabulary is
  // far smaller than a real one): the run exercises the machinery, not the text. Forces a Synthetic result.
  bool ids_folded = false;
  std::string tokenizer = "fixture-bytes";  // "fixture-bytes" | "gguf-bpe"
  std::uint64_t source_tokens = 0;          // tokens of text before windowing/repeating (a count, never content)
};

// `tokenizer` null: the byte tokenizer. Otherwise kFailedPrecondition when the tokenizer's vocabulary is larger than
// the model's and the model is not the fixture. kNotFound when the path does not exist or yields no text.
Result<CorpusPrompts> make_corpus_prompts(const std::filesystem::path& corpus, std::uint32_t length, std::uint32_t model_vocab,
                                          const father::Tokenizer* tokenizer, bool fixture_model);

// The real tokenizer from a GGUF file (header metadata only); no chat template is required for plain text.
Result<std::shared_ptr<father::Tokenizer>> load_corpus_tokenizer(const std::filesystem::path& gguf);

}  // namespace clusterlm::bench
