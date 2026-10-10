// Which stored tensor types the pinned llama.cpp CPU build actually executes through the ClusterLM llama backend.
//
// A tiny llama-architecture model (hidden 256, so every block-quantized row length is valid) is written with its layer
// matrices quantized to one ggml type via ggml's own quantizer. For each type the backend must load it and reproduce
// llama.cpp's own greedy decode with finite logits. The list of types that pass is the llama-local descriptor's
// `tensor_formats` (runtime/backends/llama/descriptor/llama-local.descriptor.json); a test below keeps the two equal.
// Mechanics on random weights only: not an accuracy statement about any quantization of any real model.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "clusterlm/objects/gguf_writer.hpp"
#include "ggml.h"
#include "llama_test_support.hpp"

using namespace clusterlm;
using namespace clusterlm::llamatest;
using objects::GgmlType;

namespace {

constexpr std::uint64_t kH = 256, kFF = 256, kVocab = 128, kLayers = 2, kHeads = 4, kKvHeads = 2;

std::vector<float> rand_floats(std::uint64_t n, float scale, std::uint64_t seed) {
  std::vector<float> v(n);
  std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  for (auto& x : v) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    x = (static_cast<float>(s >> 40) / 8388608.0f - 1.0f) * scale;
  }
  return v;
}

Bytes f32_bytes(const std::vector<float>& f) {
  Bytes b(f.size() * 4);
  std::memcpy(b.data(), f.data(), b.size());
  return b;
}

// Quantize `ne0 x rows` floats to `t` with ggml's reference quantizer.
Result<Bytes> quantize(GgmlType t, const std::vector<float>& src, std::uint64_t ne0, std::uint64_t rows) {
  const auto gt = static_cast<ggml_type>(static_cast<std::uint32_t>(t));
  std::vector<float> imatrix;
  if (ggml_quantize_requires_imatrix(gt)) imatrix.assign(ne0, 1.0f);  // uniform importance: layout/mechanics only
  CLM_ASSIGN_OR_RETURN(std::uint64_t row_bytes, objects::ggml_row_bytes(t, ne0));
  Bytes out(row_bytes * rows);
  ggml_quantize_init(gt);
  const std::size_t n = ggml_quantize_chunk(gt, src.data(), out.data(), 0, static_cast<std::int64_t>(rows),
                                            static_cast<std::int64_t>(ne0), imatrix.empty() ? nullptr : imatrix.data());
  if (n != out.size()) return make_error(ErrorCode::kInternal, "quantizer returned an unexpected size");
  return out;
}

Status write_model(const std::filesystem::path& path, GgmlType t) {
  objects::GgufWriter w;
  w.add_string("general.architecture", "llama");
  w.add_string("general.name", "clusterlm-tensor-type-fixture");
  w.add_u32("llama.context_length", 512);
  w.add_u32("llama.embedding_length", kH);
  w.add_u32("llama.block_count", kLayers);
  w.add_u32("llama.feed_forward_length", kFF);
  w.add_u32("llama.attention.head_count", kHeads);
  w.add_u32("llama.attention.head_count_kv", kKvHeads);
  w.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
  w.add_u32("llama.rope.dimension_count", kH / kHeads);
  w.add_f32("llama.rope.freq_base", 10000.0f);
  w.add_u32("llama.vocab_size", kVocab);
  w.add_string("tokenizer.ggml.model", "no_vocab");
  const float inv = 1.0f / std::sqrt(static_cast<float>(kH));
  std::uint64_t seed = 7;
  auto f32 = [&](const std::string& n, std::vector<std::uint64_t> dims, float scale) {
    std::uint64_t cnt = 1;
    for (auto d : dims) cnt *= d;
    return w.add_tensor(n, std::move(dims), GgmlType::kF32, f32_bytes(rand_floats(cnt, scale, ++seed)));
  };
  auto norm = [&](const std::string& n, std::uint64_t len) {
    auto v = rand_floats(len, 0.1f, ++seed);
    for (auto& x : v) x += 1.0f;
    return w.add_tensor(n, {len}, GgmlType::kF32, f32_bytes(v));
  };
  auto mat = [&](const std::string& n, std::uint64_t ne0, std::uint64_t rows, float scale) -> Status {
    if (t == GgmlType::kF32) return f32(n, {ne0, rows}, scale);
    CLM_ASSIGN_OR_RETURN(Bytes b, quantize(t, rand_floats(ne0 * rows, scale, ++seed), ne0, rows));
    return w.add_tensor(n, {ne0, rows}, t, std::move(b));
  };
  CLM_RETURN_IF_ERROR(f32("token_embd.weight", {kH, kVocab}, 1.0f));
  CLM_RETURN_IF_ERROR(norm("output_norm.weight", kH));
  CLM_RETURN_IF_ERROR(f32("output.weight", {kH, kVocab}, 3.0f * inv));
  const std::uint64_t kvd = kH / kHeads * kKvHeads;
  for (std::uint32_t L = 0; L < kLayers; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    CLM_RETURN_IF_ERROR(norm(p + "attn_norm.weight", kH));
    CLM_RETURN_IF_ERROR(mat(p + "attn_q.weight", kH, kH, 2 * inv));
    CLM_RETURN_IF_ERROR(mat(p + "attn_k.weight", kH, kvd, 2 * inv));
    CLM_RETURN_IF_ERROR(mat(p + "attn_v.weight", kH, kvd, 2 * inv));
    CLM_RETURN_IF_ERROR(mat(p + "attn_output.weight", kH, kH, 2 * inv));
    CLM_RETURN_IF_ERROR(norm(p + "ffn_norm.weight", kH));
    CLM_RETURN_IF_ERROR(mat(p + "ffn_gate.weight", kH, kFF, 2 * inv));
    CLM_RETURN_IF_ERROR(mat(p + "ffn_up.weight", kH, kFF, 2 * inv));
    CLM_RETURN_IF_ERROR(mat(p + "ffn_down.weight", kFF, kH, 2 * inv));
  }
  return w.write(path);
}

// Every weight type the pinned table lists, minus the activation-only quantizer types (Q8_1, Q8_K) and the
// non-weight scalar types (I8..I64, F64) that llama.cpp does not accept as model weights.
std::vector<GgmlType> weight_types() {
  std::vector<GgmlType> v;
  for (const auto& ti : objects::all_ggml_types()) {
    switch (ti.type) {
      case GgmlType::kQ8_1: case GgmlType::kQ8_K: case GgmlType::kI8: case GgmlType::kI16: case GgmlType::kI32:
      case GgmlType::kI64: case GgmlType::kF64:
        continue;
      default: v.push_back(ti.type);
    }
  }
  return v;
}

}  // namespace

TEST_CASE("every candidate weight type: load, decode equal to llama.cpp, finite logits") {
  const char* only = std::getenv("CLM_TENSOR_TYPE");  // exploration aid: run exactly one type in this process
  std::set<std::string> passed;
  for (GgmlType t : weight_types()) {
    const std::string name(objects::ggml_type_info(t).name);
    if (only != nullptr && name != only) continue;
    CAPTURE(name);
    const auto dir = unique_dir("types-" + name);
    const auto gguf = dir / "m.gguf";
    auto st = write_model(gguf, t);
    REQUIRE_MESSAGE(st.is_ok(), st.to_string());
    auto man = backends::write_llama_model_dir({gguf});
    REQUIRE_MESSAGE(man.is_ok(), man.status().to_string());
    backends::LlamaModelReport rep;
    REQUIRE(backends::build_llama_manifest({gguf}, {}, &rep).is_ok());
    CHECK(rep.tensor_types.count(name) == 1);  // the stored type is reported as stored
    {
      RefLlama ref(gguf, 128);
      const auto prompt = prompt_of(12, static_cast<std::uint32_t>(kVocab));
      const auto expect = ref.greedy(prompt, 8);
      const auto logits = ref.logits_last(prompt);
      for (float x : logits) REQUIRE(std::isfinite(x));
      coordinator::CoordinatorConfig cfg;
      cfg.model_dir = dir;
      cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
      backends::LlamaBackendOptions o;
      o.model_dir = dir;
      o.n_threads = 2;
      cfg.backend = std::shared_ptr<domain::BackendAdapter>(backends::make_llama_backend(o));
      auto c = coordinator::Coordinator::create(std::move(cfg));
      REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
      const auto layers = man->geometry.n_layers;
      auto plan = coordinator::ClusterPlan::parse("0-" + std::to_string(layers) + "@father," + std::to_string(layers) +
                                                      "-" + std::to_string(layers) + "@father",
                                                  layers);
      REQUIRE(plan.is_ok());
      plan.value().max_context = 128;
      plan.value().max_window = 64;
      REQUIRE_MESSAGE(c.value()->prepare(plan.value()).is_ok(), "prepare failed for " << name);
      coordinator::GenerationRequest r;
      r.prompt = prompt;
      r.max_new_tokens = 8;
      auto g = c.value()->generate(r);
      REQUIRE_MESSAGE(g.is_ok(), g.status().to_string());
      CHECK(g->tokens == expect);
      REQUIRE(c.value()->release().is_ok());
    }
    passed.insert(name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  MESSAGE("executed types: " << passed.size());
}

#include <nlohmann/json.hpp>

#include "clusterlm/backends/llama_compat.hpp"

TEST_CASE("the shipped llama-local descriptor says exactly what the tests above exercised") {
  const auto doc = nlohmann::json::parse(backends::llama_local_descriptor_json());
  CHECK(doc.at("id") == "llama-local");
  CHECK(doc.at("factory_name") == "llama");
  // tensor_formats == the candidate weight types (every one passed above, or this binary's first test failed)
  std::set<std::string> listed;
  for (const auto& f : doc.at("model_support").at("tensor_formats")) {
    std::string n = f.get<std::string>();
    for (auto& c : n) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    listed.insert(n);
  }
  std::set<std::string> tested;
  for (GgmlType t : weight_types()) tested.insert(std::string(objects::ggml_type_info(t).name));
  CHECK(listed == tested);
  // families that carry an architecture == the architectures test_llama_architectures exercises
  std::set<std::string> archs, exercised;
  for (const auto& fam : doc.at("model_support").at("families"))
    if (fam.contains("architecture")) archs.insert(fam.at("architecture").get<std::string>());
  for (auto a : backends::llama_exercised_architectures()) exercised.insert(std::string(a));
  CHECK(archs == exercised);
}
