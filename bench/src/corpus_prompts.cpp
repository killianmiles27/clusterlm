#include "corpus_prompts.hpp"

#include <algorithm>
#include <fstream>

#include "clusterlm/father/gguf_bpe_tokenizer.hpp"

namespace clusterlm::bench {

namespace fs = std::filesystem;

namespace {

std::string read_text(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::string text(kMaxCorpusBytesPerFile, '\0');
  in.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(in.gcount()));
  return text;
}

// A prompt of exactly `length` tokens from `tokens`: a prefix, repeated cyclically when the text is shorter.
std::vector<std::int32_t> fit(const std::vector<std::int32_t>& tokens, std::size_t length) {
  std::vector<std::int32_t> p(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(std::min(tokens.size(), length)));
  const std::size_t n0 = p.size();
  while (n0 > 0 && p.size() < length) p.push_back(p[p.size() % n0]);
  return p;
}

}  // namespace

Result<std::shared_ptr<father::Tokenizer>> load_corpus_tokenizer(const fs::path& gguf) {
  father::GgufBpeOptions options;
  options.require_chat_template = false;
  CLM_ASSIGN_OR_RETURN(auto tok, father::GgufBpeTokenizer::from_gguf_file(gguf, options));
  return std::shared_ptr<father::Tokenizer>(std::move(tok));
}

Result<CorpusPrompts> make_corpus_prompts(const fs::path& corpus, std::uint32_t length, std::uint32_t model_vocab,
                                          const father::Tokenizer* tokenizer, bool fixture_model) {
  if (model_vocab == 0) return make_error(ErrorCode::kInvalidArgument, "the model has an empty vocabulary");
  CorpusPrompts out;
  if (tokenizer) {
    out.tokenizer = "gguf-bpe";
    if (tokenizer->vocab_size() > model_vocab) {
      if (!fixture_model)
        return make_error(ErrorCode::kFailedPrecondition,
                          "the tokenizer's vocabulary (" + std::to_string(tokenizer->vocab_size()) + " tokens) is larger than the model's (" +
                              std::to_string(model_vocab) + "): it is not this model's tokenizer");
      out.ids_folded = true;
    }
  }
  std::error_code ec;
  std::vector<fs::path> files;
  const bool is_dir = fs::is_directory(corpus, ec);
  if (is_dir) {
    for (const auto& e : fs::directory_iterator(corpus, ec))
      if (e.is_regular_file()) files.push_back(e.path());
    std::sort(files.begin(), files.end());
  } else if (fs::is_regular_file(corpus, ec)) {
    files.push_back(corpus);
  } else {
    return make_error(ErrorCode::kNotFound, "corpus '" + corpus.string() + "' does not exist");
  }

  const auto vocab = static_cast<std::int32_t>(model_vocab);
  auto tokenize = [&](const std::string& text) {
    std::vector<std::int32_t> ids;
    if (tokenizer) {
      ids = tokenizer->encode(text);
    } else {
      ids.reserve(text.size());
      for (unsigned char c : text) ids.push_back(static_cast<std::int32_t>(c));
    }
    for (auto& id : ids) id %= vocab;  // no-op for a real vocabulary; folds the fixture's byte ids and a larger tokenizer
    return ids;
  };

  for (const auto& f : files) {
    const auto ids = tokenize(read_text(f));
    if (ids.empty()) continue;
    out.source_tokens += ids.size();
    if (!is_dir && length > 0 && ids.size() >= 2 * std::size_t{length}) {
      for (std::size_t w = 0; w < kMaxCorpusPrompts && (w + 1) * length <= ids.size(); ++w)
        out.prompts.emplace_back(ids.begin() + static_cast<std::ptrdiff_t>(w * length),
                                 ids.begin() + static_cast<std::ptrdiff_t>((w + 1) * length));
    } else {
      out.prompts.push_back(fit(ids, length));
    }
  }
  if (out.prompts.empty()) return make_error(ErrorCode::kNotFound, "corpus '" + corpus.string() + "' has no readable text");
  return out;
}

}  // namespace clusterlm::bench
