#pragma once
// Native chat-template rendering for Father. There is NO Jinja engine in ClusterLM: the only template family
// implemented is Qwen ChatML,
//
//     <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
//
// selected when the GGUF's `tokenizer.chat_template` contains "<|im_start|>". Any other template is refused
// (kUnimplemented, with the reason) - a wrong prompt format silently degrades the model, so Father never guesses.
//
// What is read from the template text (substring checks only, never evaluated):
//   * "You are a helpful assistant"  -> Qwen2/2.5-style default system turn when the conversation has no system
//     message (emitted exactly as the template does).
//   * "enable_thinking" together with a "<think>\n\n</think>\n\n" literal -> Qwen3-style thinking toggle:
//     enable_thinking=false appends the empty think block after the assistant opener.
//   * a quoted '<think>\n' literal (Qwen3.5-style) -> with thinking enabled the generation prompt ends with
//     "<think>\n" so the model continues inside its reasoning block.
// Not modelled: stripping earlier assistant turns' reasoning, tool calls/tool roles, multimodal parts. Callers pass
// assistant history as plain answer text (FatherService stores decoded answers).
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/father/tokenizer.hpp"

namespace clusterlm::father {

struct ChatTemplateOptions {
  // Honoured only when the template has the toggle (see above); ignored otherwise.
  bool enable_thinking = true;
};

// A rendered prompt is a sequence of pieces. `marker` pieces are template-literal special-token texts
// (<|im_start|>, <|im_end|>, <think>, </think>) that the tokenizer resolves to their token ids; every other piece
// is plain text (message content included) that is tokenized WITHOUT parsing control tokens, so user text can
// never inject <|im_start|>/<|im_end|>. Adjacent plain pieces are tokenized as one text, exactly as the Jinja
// output string would be.
struct ChatPiece {
  std::string text;
  bool marker = false;
};

class ChatMlTemplate {
 public:
  static constexpr std::string_view kImStart = "<|im_start|>";
  static constexpr std::string_view kImEnd = "<|im_end|>";

  // kUnimplemented (with the reason) when `jinja_source` is empty or not ChatML-shaped.
  static Result<ChatMlTemplate> from_template_source(std::string_view jinja_source);

  std::vector<ChatPiece> render(std::span<const ChatMessage> messages, const ChatTemplateOptions& options = {}) const;

  bool has_default_system() const { return default_system_; }
  bool has_thinking_toggle() const { return thinking_toggle_; }

 private:
  bool default_system_ = false;
  bool thinking_toggle_ = false;  // empty think block available for enable_thinking=false
  bool open_think_ = false;       // "<think>\n" opener when thinking is enabled
};

}  // namespace clusterlm::father
