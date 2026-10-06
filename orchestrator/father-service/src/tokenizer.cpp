#include "clusterlm/father/tokenizer.hpp"

namespace clusterlm::father {

std::string_view to_string(ChatRole r) noexcept {
  switch (r) {
    case ChatRole::kSystem: return "system";
    case ChatRole::kUser: return "user";
    case ChatRole::kAssistant: return "assistant";
  }
  return "user";
}

Result<std::shared_ptr<FixtureByteTokenizer>> FixtureByteTokenizer::create(std::uint32_t vocab) {
  if (vocab < 256) return make_error(ErrorCode::kInvalidArgument, "byte tokenizer needs a vocabulary of at least 256");
  return std::shared_ptr<FixtureByteTokenizer>(new FixtureByteTokenizer(vocab));
}

std::vector<std::int32_t> FixtureByteTokenizer::encode(std::string_view text) const {
  std::vector<std::int32_t> out;
  out.reserve(text.size());
  for (char c : text) out.push_back(static_cast<std::int32_t>(static_cast<unsigned char>(c)));
  return out;
}

std::string FixtureByteTokenizer::decode(std::span<const std::int32_t> tokens) const {
  std::string out;
  out.reserve(tokens.size());
  for (auto t : tokens) out.push_back(t >= 0 && t < 256 ? static_cast<char>(static_cast<unsigned char>(t)) : '?');
  return out;
}

std::vector<std::int32_t> FixtureByteTokenizer::encode_chat(std::span<const ChatMessage> messages) const {
  std::string text;
  for (const auto& m : messages) {
    text += to_string(m.role);
    text += ": ";
    text += m.content;
    text += '\n';
  }
  text += "assistant: ";
  return encode(text);
}

}  // namespace clusterlm::father
