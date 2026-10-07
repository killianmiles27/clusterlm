#include "clusterlm/father/chat_template.hpp"

namespace clusterlm::father {

namespace {
constexpr std::string_view kDefaultSystem = "You are a helpful assistant";
// Inside a GGUF's Jinja source the newlines of string literals are the two characters backslash + n.
constexpr std::string_view kEmptyThinkInSource = "<think>\\n\\n</think>\\n\\n";
constexpr std::string_view kOpenThinkInSource = "'<think>\\n'";
}  // namespace

Result<ChatMlTemplate> ChatMlTemplate::from_template_source(std::string_view src) {
  if (src.empty())
    return make_error(ErrorCode::kUnimplemented,
                      "the model has no tokenizer.chat_template; ClusterLM has no Jinja engine and will not guess a prompt format");
  if (src.find(kImStart) == std::string_view::npos)
    return make_error(ErrorCode::kUnimplemented,
                      "unsupported chat template: only Qwen ChatML (<|im_start|>role\\ncontent<|im_end|>) is implemented natively; "
                      "ClusterLM has no Jinja engine");
  ChatMlTemplate t;
  t.default_system_ = src.find(kDefaultSystem) != std::string_view::npos;
  t.thinking_toggle_ =
      src.find("enable_thinking") != std::string_view::npos && src.find(kEmptyThinkInSource) != std::string_view::npos;
  t.open_think_ = t.thinking_toggle_ && src.find(kOpenThinkInSource) != std::string_view::npos;
  return t;
}

std::vector<ChatPiece> ChatMlTemplate::render(std::span<const ChatMessage> messages, const ChatTemplateOptions& options) const {
  std::vector<ChatPiece> out;
  auto marker = [&](std::string_view m) { out.push_back({std::string(m), true}); };
  auto text = [&](std::string s) { out.push_back({std::move(s), false}); };
  auto turn = [&](std::string_view role, std::string_view content) {
    marker(kImStart);
    text(std::string(role) + "\n" + std::string(content));
    marker(kImEnd);
    text("\n");
  };
  if (default_system_ && (messages.empty() || messages.front().role != ChatRole::kSystem)) turn("system", kDefaultSystem);
  for (const auto& m : messages) turn(to_string(m.role), m.content);
  marker(kImStart);
  text("assistant\n");
  if (thinking_toggle_) {
    if (!options.enable_thinking) {
      marker("<think>");
      text("\n\n");
      marker("</think>");
      text("\n\n");
    } else if (open_think_) {
      marker("<think>");
      text("\n");
    }
  }
  return out;
}

}  // namespace clusterlm::father
