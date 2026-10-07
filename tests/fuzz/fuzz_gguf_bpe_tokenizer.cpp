// GgufBpeTokenizer on a small synthetic vocabulary: encode arbitrary bytes (valid UTF-8 or not, with or without
// control-token text), never crash, decode(encode(x)) == x exactly, and chat rendering stays well formed.
#include "../father_service/synthetic_vocab.hpp"
#include "fuzz_util.hpp"

using namespace clusterlm;
using namespace clusterlm::father;

namespace {
const GgufBpeTokenizer& tokenizer() {
  static const std::shared_ptr<GgufBpeTokenizer> t = [] {
    auto r = testing::synthetic_tokenizer();
    CLM_FUZZ_CHECK(r.is_ok());
    return r.value();
  }();
  return *t;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const auto& t = tokenizer();
  const std::string text(fuzz::text_of(data, size));
  const auto ids = t.encode(text);
  for (auto id : ids) CLM_FUZZ_CHECK(id >= 0 && static_cast<std::uint32_t>(id) < t.vocab_size());
  // Control tokens are never produced from plain text, so the round trip is exact for ALL byte strings.
  CLM_FUZZ_CHECK(t.decode(ids) == text);
  (void)t.encode_text(text, true);  // parse_special: must not crash; ids may include control tokens

  const std::vector<ChatMessage> msgs = {{ChatRole::kUser, text}};
  const auto prompt = t.encode_chat(msgs);
  CLM_FUZZ_CHECK(!prompt.empty());
  return 0;
}
