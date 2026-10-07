#pragma once
// A tiny SYNTHETIC byte-level BPE vocabulary (GGUF metadata only) for tests and the tokenizer fuzz target: the 256
// GPT-2 byte tokens, merges learned greedily from a small English corpus, ChatML control tokens and Qwen3-style
// <think> user-defined tokens. Not a real model vocabulary.
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "clusterlm/father/gguf_bpe_tokenizer.hpp"
#include "clusterlm/objects/gguf_writer.hpp"

namespace clusterlm::father::testing {

inline const char* kSyntheticChatTemplate =
    "{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}";

inline std::string synthetic_byte_char(unsigned b) {
  auto printable = [](unsigned x) { return (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255); };
  static std::map<unsigned, unsigned> cps;
  if (cps.empty()) {
    unsigned extra = 0;
    for (unsigned x = 0; x < 256; ++x) cps[x] = printable(x) ? x : 256 + extra++;
  }
  const unsigned cp = cps[b];
  std::string s;
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return s;
}

struct SyntheticVocabSpec {
  std::string pre = "qwen2";
  std::string model = "gpt2";
  std::string chat_template = kSyntheticChatTemplate;
  bool with_byte_tokens = true;
};

inline Bytes synthetic_vocab_gguf(const SyntheticVocabSpec& spec = {}) {
  std::vector<std::string> tokens;
  std::vector<std::uint32_t> types;
  if (spec.with_byte_tokens)
    for (unsigned b = 0; b < 256; ++b) {
      tokens.push_back(synthetic_byte_char(b));
      types.push_back(1);
    }
  // Greedy BPE training over a tiny corpus (deterministic: ties go to the lexicographically smallest pair).
  const char* corpus[] = {"\xC4\xA0" "the", "\xC4\xA0" "quick", "\xC4\xA0" "brown", "\xC4\xA0" "fox", "hello", "\xC4\xA0" "world", "\xC4\xA0" "the", "\xC4\xA0" "hello"};
  std::vector<std::vector<std::string>> words;
  for (const char* w : corpus) {
    std::vector<std::string> chars;
    for (const char* p = w; *p;) {
      const std::size_t n = (static_cast<unsigned char>(*p) >= 0xC0) ? 2 : 1;
      chars.emplace_back(p, n);
      p += n;
    }
    words.push_back(std::move(chars));
  }
  std::vector<std::string> merges;
  for (int round = 0; round < 40; ++round) {
    std::map<std::pair<std::string, std::string>, int> freq;
    for (const auto& w : words)
      for (std::size_t i = 0; i + 1 < w.size(); ++i) ++freq[{w[i], w[i + 1]}];
    if (freq.empty()) break;
    auto best = freq.begin();
    for (auto it = freq.begin(); it != freq.end(); ++it)
      if (it->second > best->second) best = it;
    const auto pr = best->first;
    merges.push_back(pr.first + " " + pr.second);
    tokens.push_back(pr.first + pr.second);
    types.push_back(1);
    for (auto& w : words)
      for (std::size_t i = 0; i + 1 < w.size(); ++i)
        if (w[i] == pr.first && w[i + 1] == pr.second) {
          w[i] += w[i + 1];
          w.erase(w.begin() + static_cast<std::ptrdiff_t>(i) + 1);
        }
  }
  tokens.push_back("<|endoftext|>"); types.push_back(3);
  tokens.push_back("<|im_start|>");  types.push_back(3);
  tokens.push_back("<|im_end|>");    types.push_back(3);
  tokens.push_back("<think>");       types.push_back(4);
  tokens.push_back("</think>");      types.push_back(4);

  objects::GgufWriter w;
  w.add_string("general.architecture", "qwen2");
  w.add_string("tokenizer.ggml.model", spec.model);
  w.add_string("tokenizer.ggml.pre", spec.pre);
  w.add_string_array("tokenizer.ggml.tokens", tokens);
  w.add_u32_array("tokenizer.ggml.token_type", types);
  w.add_string_array("tokenizer.ggml.merges", merges);
  w.add_u32("tokenizer.ggml.eos_token_id", static_cast<std::uint32_t>(tokens.size() - 3));
  w.add_u32("tokenizer.ggml.padding_token_id", static_cast<std::uint32_t>(tokens.size() - 5));
  if (!spec.chat_template.empty()) w.add_string("tokenizer.chat_template", spec.chat_template);
  auto bytes = w.serialize();
  return bytes.is_ok() ? std::move(bytes).value() : Bytes{};
}

inline Result<std::shared_ptr<GgufBpeTokenizer>> synthetic_tokenizer(const SyntheticVocabSpec& spec = {}, const GgufBpeOptions& opt = {}) {
  const Bytes b = synthetic_vocab_gguf(spec);
  CLM_ASSIGN_OR_RETURN(auto file, objects::parse_gguf_bytes(b, GgufBpeTokenizer::metadata_limits()));
  return GgufBpeTokenizer::from_gguf(file, opt);
}

}  // namespace clusterlm::father::testing
