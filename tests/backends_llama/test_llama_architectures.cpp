// Architecture breadth of the llama.cpp backend: the same contract checks as test_llama_backend.cpp, repeated on tiny
// random-weight GGUFs in each architecture the generator can write (Llama, Llama MoE, Qwen2, Qwen3, Gemma, Phi3, and a
// tied-embedding Llama). This proves the pinned llama.cpp loads the layout and that the backend reproduces its decode
// exactly. It is mechanics evidence on CPU fixtures: it says nothing about real checkpoints, quality or speed, and it
// never raises a compatibility label above "Supported, awaiting hardware qualification".
#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/backends/llama_compat.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "llama_test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::llamatest;

namespace {

struct ArchCase {
  const char* label;
  backends::TinyLlamaSpec spec;
  const char* architecture;
  std::uint32_t experts;
};

std::vector<ArchCase> cases() {
  using backends::TinyArch;
  std::vector<ArchCase> v;
  auto add = [&](const char* label, TinyArch a, auto tweak, std::uint32_t experts) {
    backends::TinyLlamaSpec s;
    s.arch = a;
    tweak(s);
    v.push_back({label, s, backends::tiny_arch_name(a), experts});
  };
  add("llama", TinyArch::kLlama, [](auto&) {}, 1);
  add("llama-tied", TinyArch::kLlama, [](auto& s) { s.tied_embeddings = true; }, 1);
  add("llama-moe", TinyArch::kLlamaMoe, [](auto&) {}, 4);
  add("qwen2", TinyArch::kQwen2, [](auto&) {}, 1);
  add("qwen3", TinyArch::kQwen3, [](auto&) {}, 1);
  add("gemma", TinyArch::kGemma, [](auto&) {}, 1);
  add("phi3", TinyArch::kPhi3, [](auto&) {}, 1);
  add("qwen2-f16", TinyArch::kQwen2, [](auto& s) { s.f16_matrices = true; }, 1);
  return v;
}

}  // namespace

TEST_CASE("architecture breadth: manifest identifies the architecture and covers every tensor byte") {
  for (const ArchCase& c : cases()) {
    CAPTURE(c.label);
    TinyModel m(std::string("arch-manifest-") + c.label, c.spec);
    CHECK(m.manifest.geometry.family == std::string("llama.cpp:") + c.architecture);
    CHECK(m.manifest.geometry.n_experts == c.experts);
    backends::LlamaModelReport rep;
    REQUIRE(backends::build_llama_manifest({m.gguf}, {}, &rep).is_ok());
    CHECK(rep.architecture == c.architecture);
    std::uint64_t total = 0;
    for (const auto& o : m.manifest.objects) total += o.byte_size;
    CHECK(total == rep.tensor_bytes);
  }
}

TEST_CASE("architecture breadth: Coordinator greedy generation equals llama.cpp's own decode") {
  for (const ArchCase& c : cases()) {
    CAPTURE(c.label);
    TinyModel m(std::string("arch-greedy-") + c.label, c.spec);
    RefLlama ref(m.gguf, 256);
    for (std::size_t prompt_len : {1u, 9u, 40u}) {
      CAPTURE(prompt_len);
      const auto prompt = prompt_of(prompt_len, m.vocab());
      const auto expect = ref.greedy(prompt, 16);
      auto father = make_father(m);
      coordinator::GenerationRequest r;
      r.prompt = prompt;
      r.max_new_tokens = 16;
      r.prefill_chunk = 7;
      auto g = father->generate(r);
      REQUIRE_MESSAGE(g.is_ok(), g.status().to_string());
      CHECK(g->tokens == expect);
      REQUIRE(father->release().is_ok());
    }
  }
}

TEST_CASE("architecture breadth: every speculative acceptance length gives the q=1 tokens") {
  for (const ArchCase& c : cases()) {
    CAPTURE(c.label);
    TinyModel m(std::string("arch-accept-") + c.label, c.spec);
    const auto prompt = prompt_of(10, m.vocab());
    const std::uint32_t max_new = 12;
    auto father = make_father(m);
    coordinator::GenerationRequest base;
    base.prompt = prompt;
    base.max_new_tokens = max_new + 6;
    auto ref = father->generate(base);
    REQUIRE(ref.is_ok());
    std::vector<std::int32_t> sequence = prompt;
    sequence.insert(sequence.end(), ref->tokens.begin(), ref->tokens.end());
    const std::vector<std::int32_t> expect(ref->tokens.begin(), ref->tokens.begin() + max_new);
    const std::uint32_t q = 3;
    for (std::uint32_t j = 0; j <= q - 1; ++j) {
      CAPTURE(j);
      domain::ScriptedDrafter::Config dc;
      dc.vocab = m.vocab();
      if (j > 0) dc.corrupt_draft_index = j;
      coordinator::GenerationRequest r;
      r.prompt = prompt;
      r.max_new_tokens = max_new;
      r.q = q;
      r.drafter = std::make_shared<domain::ScriptedDrafter>(sequence, dc);
      auto out = father->generate(r);
      REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());
      CHECK(out->tokens == expect);
    }
    REQUIRE(father->release().is_ok());
  }
}

TEST_CASE("architecture breadth: the exercised list is exactly the architectures this file tests") {
  std::vector<std::string_view> tested;
  for (const ArchCase& c : cases()) tested.push_back(c.architecture);
  std::sort(tested.begin(), tested.end());
  tested.erase(std::unique(tested.begin(), tested.end()), tested.end());
  CHECK(tested == backends::llama_exercised_architectures());
  for (auto a : tested) {
    CAPTURE(a);
    CHECK(backends::llama_architecture_support(a) == backends::LlamaArchSupport::kExercised);
    // every exercised architecture is one the pinned build names
    CHECK(std::binary_search(backends::llama_pinned_architectures().begin(),
                             backends::llama_pinned_architectures().end(), a));
  }
}

TEST_CASE("architecture breadth: unknown and merely-known architectures are told apart") {
  using backends::LlamaArchSupport;
  CHECK(backends::llama_architecture_support("llama") == LlamaArchSupport::kExercised);
  CHECK(backends::llama_architecture_support("mamba") == LlamaArchSupport::kKnown);
  CHECK(backends::llama_architecture_support("not-an-architecture") == LlamaArchSupport::kUnknown);
  CHECK(backends::llama_architecture_support("clip") == LlamaArchSupport::kUnknown);  // not a model architecture
  CHECK(backends::llama_architecture_support("") == LlamaArchSupport::kUnknown);
}

TEST_CASE("architecture breadth: the generated table matches the pinned checkout") {
  // The table is generated from third_party/upstream/llama.cpp/src/llama-arch.cpp; re-derive it here with a plain
  // scan so a stale table fails the llama CI job rather than silently mislabelling an architecture.
  const std::filesystem::path src = std::filesystem::path(CLUSTERLM_LLAMA_SRC_DIR) / "src" / "llama-arch.cpp";
  std::ifstream in(src);
  REQUIRE_MESSAGE(in.good(), "missing " << src.string());
  std::vector<std::string> names;
  std::string line;
  bool inside = false;
  while (std::getline(in, line)) {
    if (line.find("LLM_ARCH_NAMES") != std::string::npos && line.find('{') != std::string::npos) inside = true;
    if (!inside) continue;
    if (line.rfind("};", 0) == 0) break;
    const auto id = line.find("LLM_ARCH_");
    const auto q1 = line.find('"');
    const auto q2 = q1 == std::string::npos ? q1 : line.find('"', q1 + 1);
    if (id == std::string::npos || q2 == std::string::npos) continue;
    std::string n = line.substr(q1 + 1, q2 - q1 - 1);
    if (n != "clip" && n != "(unknown)") names.push_back(n);
  }
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  const auto& table = backends::llama_pinned_architectures();
  REQUIRE(names.size() == table.size());
  for (std::size_t i = 0; i < names.size(); ++i) CHECK(names[i] == table[i]);
}

TEST_CASE("architecture breadth: the hyper-parameter probe accepts fixtures and refuses other files") {
  for (const ArchCase& c : cases()) {
    CAPTURE(c.label);
    TinyModel m(std::string("arch-probe-") + c.label, c.spec);
    CHECK(backends::probe_llama_hparams(m.gguf).is_ok());
  }
  const auto dir = unique_dir("arch-probe-bad");
  const auto junk = dir / "junk.gguf";
  {
    std::ofstream o(junk, std::ios::binary);
    o << "this is not a gguf file";
  }
  CHECK_FALSE(backends::probe_llama_hparams(junk).is_ok());
  CHECK_FALSE(backends::probe_llama_hparams(dir / "missing.gguf").is_ok());
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
