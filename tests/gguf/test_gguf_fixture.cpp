#include <doctest/doctest.h>

#include <cmath>
#include <set>

#include "clusterlm/objects/fixture_gguf.hpp"
#include "clusterlm/objects/gguf_manifest.hpp"
#include "clusterlm/objects/tensor_codec.hpp"
#include "pipeline_harness.hpp"

using namespace clusterlm;
using namespace clusterlm::domain;
using namespace clusterlm::objects;
using namespace clusterlm::testutil;

namespace {

const std::vector<std::int32_t> kPrompt = {17, 3, 59, 50, 4, 4, 4, 31, 12, 40, 7};

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct GgufFixture {
  GgufFixture(bool q8, const std::string& tag) : dir(tag) {
    FixtureGgufOptions o;
    o.q8_experts = q8;
    auto m = write_fixture_gguf(FixtureSpec::tiny(), dir.path(), o);
    REQUIRE_MESSAGE(m.is_ok(), m.status().to_string());
    manifest = std::move(m).value();
  }
  std::unique_ptr<CanonicalModelStore> open_store() const {
    auto s = CanonicalModelStore::open(dir.path());
    REQUIRE_MESSAGE(s.is_ok(), s.status().to_string());
    return std::move(s).value();
  }
  TempDir dir;
  ModelManifest manifest;
};

const GgufFixture& q8_fixture() {
  static const GgufFixture f(true, "gguf-fixture-q8");
  return f;
}
const GgufFixture& f32_fixture() {
  static const GgufFixture f(false, "gguf-fixture-f32");
  return f;
}

}  // namespace

TEST_CASE("fixture GGUF: the manifest built from the GGUF files matches the fixture's geometry and object set") {
  const ModelManifest& raw = tiny_fixture().manifest;
  for (const GgufFixture* f : {&q8_fixture(), &f32_fixture()}) {
    const ModelManifest& m = f->manifest;
    CHECK(m.validate().is_ok());
    const ModelGeometry &a = m.geometry, &b = raw.geometry;
    CHECK(a.family == b.family);
    CHECK(a.n_layers == b.n_layers);
    CHECK(a.hidden_size == b.hidden_size);
    CHECK(a.residual_streams == b.residual_streams);
    CHECK(a.n_experts == b.n_experts);
    CHECK(a.n_active_experts == b.n_active_experts);
    CHECK(a.expert_ff == b.expert_ff);
    CHECK(a.shared_expert_ff == b.shared_expert_ff);
    CHECK(a.n_heads == b.n_heads);
    CHECK(a.n_kv_heads == b.n_kv_heads);
    CHECK(a.head_dim == b.head_dim);
    CHECK(a.vocab_size == b.vocab_size);
    CHECK(a.ple_layer == b.ple_layer);
    CHECK(a.ple_ngram == b.ple_ngram);
    CHECK(a.ple_rows == b.ple_rows);
    CHECK(a.mtp_layers == b.mtp_layers);
    CHECK(a.layer_kinds == b.layer_kinds);
    CHECK(m.objects.size() == raw.objects.size());
    std::set<std::string> names_a, names_b;
    for (const auto& o : m.objects) names_a.insert(o.name);
    for (const auto& o : raw.objects) names_b.insert(o.name);
    CHECK(names_a == names_b);
    CHECK(m.shards.size() == 2);
    CHECK(m.shards[0].file_name == kFixtureGgufShard0);
    CHECK(m.shards[1].file_name == kFixtureGgufShard1);
    // The PLE table is in shard 2 like the real artifact.
    CHECK(m.find(kPleObjectName)->source_ranges[0].shard == 1);
    for (const auto& o : m.objects) {
      CHECK_FALSE(o.source_digest.is_zero());
      const ManifestObject* r = raw.find(o.name);
      REQUIRE(r != nullptr);
      CHECK(o.kind == r->kind);
      CHECK(o.source_ranges.size() == r->source_ranges.size());
      if (o.kind != ObjectKind::kRoutedExpert) CHECK(o.byte_size == r->byte_size);
    }
  }
  // F32 experts have the same byte size as the raw fixture's even layers; q8_0 experts are 34/36 of the
  // fixture's private 36-byte blocks.
  CHECK(f32_fixture().manifest.find(expert_object_name(1, 3))->representation.quant_type == "f32");
  CHECK(q8_fixture().manifest.find(expert_object_name(1, 3))->representation.quant_type == "q8_0");
  CHECK(q8_fixture().manifest.find(expert_object_name(1, 3))->representation.block_size == 32);
  CHECK(q8_fixture().manifest.find(expert_object_name(0, 3))->representation.quant_type == "f32");
  CHECK(q8_fixture().manifest.find(expert_object_name(1, 3))->byte_size * 36 == raw.find(expert_object_name(1, 3))->byte_size * 34);
  CHECK(f32_fixture().manifest.find(expert_object_name(1, 3))->byte_size == 3 * 32 * 32 * 4);
}

TEST_CASE("fixture GGUF: q8_0 experts dequantize close to the f32 originals (same generator streams)") {
  auto sq = q8_fixture().open_store();
  auto sf = f32_fixture().open_store();
  const std::string name = expert_object_name(3, 5);
  auto q = sq->resolve(name);
  auto f = sf->resolve(name);
  REQUIRE(q.is_ok());
  REQUIRE(f.is_ok());
  const std::size_t n = 32 * 32;  // ff * H per tensor
  std::vector<float> dq(n), df(n);
  for (std::size_t part = 0; part < 3; ++part) {
    const std::uint64_t qb = tensor_bytes("q8_0", n), fb = tensor_bytes("f32", n);
    REQUIRE(decode_tensor("q8_0", q->bytes.subspan(part * qb, qb), dq).is_ok());
    REQUIRE(decode_tensor("f32", f->bytes.subspan(part * fb, fb), df).is_ok());
    float amax = 0;
    for (float v : df) amax = std::max(amax, std::fabs(v));
    float worst = 0;
    for (std::size_t i = 0; i < n; ++i) worst = std::max(worst, std::fabs(dq[i] - df[i]));
    CHECK(worst <= amax / 127.0f * 0.51f + amax * 2e-3f);
    CHECK(worst > 0.0f);  // it really is quantized
  }
}

TEST_CASE("fixture GGUF: the reference backend runs from the GGUF-derived manifest; split == unsplit bitwise") {
  for (const GgufFixture* f : {&q8_fixture(), &f32_fixture()}) {
    auto store = f->open_store();
    const ModelGeometry& g = f->manifest.geometry;
    Pipeline unsplit(f->manifest, *store, {{0, 8, 8}});
    Pipeline split(f->manifest, *store, {{0, 3, 5, 6, 8}});
    const auto a = generate(unsplit, Epoch{1}, SessionId{1}, kPrompt, 12, nullptr, 1, g.vocab_size, 8);
    const auto b = generate(split, Epoch{1}, SessionId{1}, kPrompt, 12, nullptr, 1, g.vocab_size, 8);
    CHECK(a.generated.size() == 12);
    CHECK(a.generated == b.generated);
    REQUIRE(a.trace.size() == b.trace.size());
    for (std::size_t i = 0; i < a.trace.size(); ++i) {
      CHECK_MESSAGE(bit_equal(a.trace[i].logits, b.trace[i].logits), "window " << i);
      for (float v : a.trace[i].logits) REQUIRE(std::isfinite(v));
    }
  }
}

TEST_CASE("fixture GGUF: deterministic bytes for the same spec and seed") {
  TempDir other("gguf-fixture-again");
  auto again = write_fixture_gguf(FixtureSpec::tiny(), other.path(), FixtureGgufOptions{});
  REQUIRE(again.is_ok());
  CHECK(again->root_hash() == q8_fixture().manifest.root_hash());
  FixtureSpec s2 = FixtureSpec::tiny();
  s2.seed = 99;
  TempDir third("gguf-fixture-seed");
  auto diff = write_fixture_gguf(s2, third.path(), FixtureGgufOptions{});
  REQUIRE(diff.is_ok());
  CHECK(diff->root_hash() != q8_fixture().manifest.root_hash());
}
