#pragma once
// GgufBpeTokenizer: the model's real byte-level BPE tokenizer (GPT-2 style, as used by the Qwen families), built
// on Father from a GGUF file's METADATA only (tensors are never read). Token IDs, text and the vocabulary exist
// only on Father; nothing here is reachable from a Node. See docs/tokenizer.md.
//
// Behaviour is a re-implementation of llama.cpp's BPE tokenizer for `tokenizer.ggml.model == "gpt2"` with the
// pre-tokenizer selected by `tokenizer.ggml.pre`:
//   qwen2   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//           ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   qwen35  the same with \p{M} added to the letter class of the 2nd and 4th alternatives
// Any other `pre` (or a tokenizer model other than gpt2) is refused with kUnimplemented - the tokenizer never
// guesses a pre-tokenizer. The pattern is matched by hand over a compact Unicode category table
// (scripts/gen_unicode_tables.py); there is no std::regex.
//
// Verified token-for-token against llama.cpp's own vocab tests (ggml-vocab-qwen2/qwen35 .inp/.out); see
// tests/father_service/test_vocab_llamacpp.cpp.
//
// Text handling: bytes that are not valid UTF-8 are tokenized as single "other" units (llama.cpp throws); every
// byte maps through the GPT-2 byte<->unicode table, so decode(encode(x)) == x for ALL byte strings as long as the
// vocabulary contains the 256 byte-level base tokens (checked at construction).
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/father/chat_template.hpp"
#include "clusterlm/father/tokenizer.hpp"
#include "clusterlm/objects/gguf.hpp"

namespace clusterlm::father {

struct GgufBpeOptions {
  // Tier tokenizers need a renderable chat template (a tier whose prompt format cannot be produced is Unavailable).
  // Tests that only tokenize text can turn this off; encode_chat then returns an empty vector.
  bool require_chat_template = true;
  ChatTemplateOptions chat;  // enable_thinking, see chat_template.hpp
};

class GgufBpeTokenizer final : public Tokenizer {
 public:
  // GgufLimits that retain the full tokenizer arrays; pass to objects::read_gguf_file / parse_gguf_bytes.
  static objects::GgufLimits metadata_limits();

  // Reads only the GGUF header of `path` (the first shard of a split model carries all tokenizer metadata).
  static Result<std::shared_ptr<GgufBpeTokenizer>> from_gguf_file(const std::filesystem::path& path, const GgufBpeOptions& options = {});
  // `file` must have been parsed with metadata_limits() (otherwise only a 64-entry prefix of each array exists
  // and construction fails with kInvalidArgument).
  static Result<std::shared_ptr<GgufBpeTokenizer>> from_gguf(const objects::GgufFile& file, const GgufBpeOptions& options = {});

  // ---- Tokenizer ----
  std::uint32_t vocab_size() const override { return static_cast<std::uint32_t>(tokens_.size()); }
  // Plain text: control/unknown tokens in `text` are NOT parsed (they tokenize as ordinary text); user-defined
  // tokens are matched like llama.cpp. No BOS/EOS is added.
  std::vector<std::int32_t> encode(std::string_view text) const override { return encode_text(text, false); }
  // Control tokens render as nothing; everything else as its exact bytes.
  std::string decode(std::span<const std::int32_t> tokens) const override;
  // The ChatML prompt ending with the assistant opener. Empty when the model has no supported chat template.
  std::vector<std::int32_t> encode_chat(std::span<const ChatMessage> messages) const override;

  // `parse_special` true also matches control/unknown tokens in `text` (llama.cpp's parse_special). Never use it
  // on user-supplied text.
  std::vector<std::int32_t> encode_text(std::string_view text, bool parse_special) const;
  std::vector<std::int32_t> encode_chat(std::span<const ChatMessage> messages, const ChatTemplateOptions& options) const;

  // ---- vocabulary facts ----
  std::string_view pre_tokenizer() const { return pre_; }
  std::string_view token_text(std::int32_t id) const;  // byte-encoded form as stored in the GGUF; "" if out of range
  std::optional<std::int32_t> find_token(std::string_view stored_text) const;
  std::optional<std::int32_t> bos_id() const { return bos_; }
  std::optional<std::int32_t> eos_id() const { return eos_; }
  std::optional<std::int32_t> eot_id() const { return eot_; }
  std::optional<std::int32_t> pad_id() const { return pad_; }
  // Token ids that end a generation: eos, eot, <|im_end|>, <|endoftext|> (those that exist), without duplicates.
  // The Father service passes them as GenerationRequest::stop_tokens.
  const std::vector<std::int32_t>& stop_tokens() const { return stop_tokens_; }
  std::vector<std::int32_t> stop_token_ids() const override { return stop_tokens_; }
  bool has_chat_template() const { return chat_.has_value(); }
  const ChatMlTemplate* chat_template() const { return chat_ ? &*chat_ : nullptr; }
  std::size_t merge_count() const { return ranks_.size(); }

  // Same options-free decode into an existing buffer (used by the fuzzer and the streaming path).
  void decode_into(std::span<const std::int32_t> tokens, std::string& out) const;

 private:
  enum class Attr : std::uint8_t { kNormal, kUnknown, kControl, kUserDefined, kUnused, kByte };
  enum class Pre : std::uint8_t { kQwen2, kQwen35 };

  GgufBpeTokenizer() = default;
  Status init(const objects::GgufFile& file, const GgufBpeOptions& options);

  struct Fragment {
    bool is_token = false;
    std::int32_t token = 0;
    std::size_t off = 0, len = 0;
  };
  std::vector<Fragment> partition_special(std::string_view text, bool parse_special) const;
  void tokenize_plain(std::string_view text, std::vector<std::int32_t>& out) const;
  void bpe_word(std::string_view word_bytes_encoded, std::vector<std::int32_t>& out) const;
  bool is_special(std::int32_t id) const;

  Pre pre_kind_ = Pre::kQwen2;
  std::string pre_;
  std::vector<std::string> tokens_;
  std::vector<Attr> attrs_;
  std::unordered_map<std::string, std::int32_t> token_to_id_;
  std::unordered_map<std::string, std::int32_t> ranks_;  // "left right" -> merge rank
  std::vector<std::int32_t> special_order_;               // control/user-defined/unknown ids, longest text first
  std::optional<std::int32_t> bos_, eos_, eot_, pad_;
  bool add_bos_ = false;
  std::vector<std::int32_t> stop_tokens_;
  std::optional<ChatMlTemplate> chat_;
  ChatTemplateOptions chat_options_;
};

}  // namespace clusterlm::father
