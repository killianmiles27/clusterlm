#pragma once
// Father-local tokenizer interface. The real tokenizer belongs to the backend and is never reachable from a
// Node; text, roles and token IDs exist only on Father. A deterministic byte-level fixture tokenizer lets the
// whole chat path run on the fixture model.
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::father {

enum class ChatRole : std::uint8_t { kSystem, kUser, kAssistant };
std::string_view to_string(ChatRole r) noexcept;

// Roles and message text exist ONLY in the Father-local layer. Nothing below the Coordinator ever sees them.
struct ChatMessage {
  ChatRole role = ChatRole::kUser;
  std::string content;
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  virtual std::uint32_t vocab_size() const = 0;
  virtual std::vector<std::int32_t> encode(std::string_view text) const = 0;
  // Lossy for byte sequences that are not valid text; never throws.
  virtual std::string decode(std::span<const std::int32_t> tokens) const = 0;
  // Applies the chat template and appends the assistant-turn opener the model continues from.
  virtual std::vector<std::int32_t> encode_chat(std::span<const ChatMessage> messages) const = 0;
  // Token ids that end an answer (eos, <|im_end|>, ...); the service passes them as GenerationRequest::stop_tokens.
  // Empty for tokenizers without end-of-turn tokens (the byte fixture).
  virtual std::vector<std::int32_t> stop_token_ids() const { return {}; }
};

// Byte-level: token id == byte value (ids 0..255). Chat template "<role>: <content>\n" with a trailing
// "assistant: " opener. Requires vocab >= 256 (ids >= 256 are never produced by encode and decode as "?").
class FixtureByteTokenizer final : public Tokenizer {
 public:
  static Result<std::shared_ptr<FixtureByteTokenizer>> create(std::uint32_t vocab = 256);
  std::uint32_t vocab_size() const override { return vocab_; }
  std::vector<std::int32_t> encode(std::string_view text) const override;
  std::string decode(std::span<const std::int32_t> tokens) const override;
  std::vector<std::int32_t> encode_chat(std::span<const ChatMessage> messages) const override;

 private:
  explicit FixtureByteTokenizer(std::uint32_t vocab) : vocab_(vocab) {}
  std::uint32_t vocab_;
};

}  // namespace clusterlm::father
