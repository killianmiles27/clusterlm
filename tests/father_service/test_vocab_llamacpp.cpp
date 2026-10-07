// Verification of GgufBpeTokenizer against llama.cpp's own vocab tests.
//
// llama.cpp ships, next to each vocab-only GGUF (models/ggml-vocab-<name>.gguf), the inputs
// (<gguf>.inp; cases separated by "\n__ggml_vocab_test__\n") and the token ids its tokenizer produces
// (<gguf>.out; one line per case, space-separated ids). tests/test-tokenizer-0.cpp tokenizes every input with
// add_special=false, parse_special=false and requires exactly those ids; this test does the same with Father's
// tokenizer and requires exact equality for every case of the qwen2 and qwen35 vocabularies.
//
// The files come from the pinned llama.cpp checkout (python3 scripts/fetch_upstream.py llama.cpp). They are never
// committed; this executable is only built when they exist (see tests/father_service/CMakeLists.txt).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "clusterlm/father/gguf_bpe_tokenizer.hpp"

using namespace clusterlm;
using namespace clusterlm::father;

namespace {

struct Case {
  std::string text;
  std::vector<std::int32_t> ids;
};

std::string slurp(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Same parsing as read_tests() in llama.cpp's test-tokenizer-0.cpp.
std::vector<Case> read_cases(const std::filesystem::path& gguf) {
  const std::string raw = slurp(gguf.string() + ".inp");
  std::ifstream out_file(gguf.string() + ".out");
  std::vector<std::string> out_lines;
  for (std::string line; std::getline(out_file, line);) out_lines.push_back(line);

  const std::string sep = "\n__ggml_vocab_test__\n";
  std::vector<std::string> inputs;
  std::size_t pos = 0;
  while (pos < raw.size()) {
    const std::size_t next = raw.find(sep, pos);
    if (next == std::string::npos) {
      inputs.push_back(raw.substr(pos));
      break;
    }
    inputs.push_back(raw.substr(pos, next - pos));
    pos = next + sep.size();
  }
  REQUIRE(inputs.size() == out_lines.size());

  std::vector<Case> cases;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    Case c;
    c.text = inputs[i];
    const std::string& line = out_lines[i];
    std::size_t p = 0;
    while (p < line.size()) {
      std::size_t sp = line.find(' ', p);
      if (sp == std::string::npos) sp = line.size();
      if (sp > p) c.ids.push_back(std::stoi(line.substr(p, sp - p)));
      p = sp + 1;
    }
    cases.push_back(std::move(c));
  }
  return cases;
}

void run_vocab(const char* name, const char* expect_pre) {
  const std::filesystem::path gguf = std::filesystem::path(CLUSTERLM_LLAMACPP_VOCAB_DIR) / (std::string("ggml-vocab-") + name + ".gguf");
  GgufBpeOptions opt;
  opt.require_chat_template = false;  // the vocab-only files carry an old ChatML-with-default-system template; irrelevant here
  auto tok = GgufBpeTokenizer::from_gguf_file(gguf, opt);
  REQUIRE_MESSAGE(tok.is_ok(), tok.status().to_string());
  CHECK(tok.value()->pre_tokenizer() == expect_pre);

  const auto cases = read_cases(gguf);
  REQUIRE(!cases.empty());
  std::size_t passed = 0;
  std::vector<std::size_t> failed;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto got = tok.value()->encode_text(cases[i].text, /*parse_special=*/false);
    if (got == cases[i].ids) {
      ++passed;
    } else if (failed.size() < 10) {
      failed.push_back(i);
      std::string want, have;
      for (auto id : cases[i].ids) want += std::to_string(id) + " ";
      for (auto id : got) have += std::to_string(id) + " ";
      std::fprintf(stderr, "vocab %s case %zu mismatch\n  text: [%s]\n  want: %s\n  got:  %s\n", name, i, cases[i].text.c_str(),
                   want.c_str(), have.c_str());
    }
  }
  std::printf("VOCAB %s: %zu/%zu cases match llama.cpp exactly\n", name, passed, cases.size());
  CHECK(passed == cases.size());

  // Round trip: with control tokens not parsed, decode(encode(x)) is the identity on every case.
  std::size_t round_trip = 0;
  for (const auto& c : cases) {
    const auto ids = tok.value()->encode(c.text);
    if (tok.value()->decode(ids) == c.text) ++round_trip;
  }
  std::printf("VOCAB %s: %zu/%zu cases round-trip decode(encode(x)) == x\n", name, round_trip, cases.size());
  CHECK(round_trip == cases.size());
}

}  // namespace

TEST_CASE("llama.cpp vocab test: qwen2") { run_vocab("qwen2", "qwen2"); }
TEST_CASE("llama.cpp vocab test: qwen35") { run_vocab("qwen35", "qwen35"); }
