#include <doctest/doctest.h>

#include <cmath>

#include "clusterlm/objects/tensor_codec.hpp"
#include "test_util.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;

namespace {
const ModelManifest& M() { return testutil::tiny_fixture().manifest; }
}  // namespace

TEST_CASE("geometry encode/decode round trip and validation") {
  const ModelGeometry g = FixtureSpec::tiny().geometry();
  ByteWriter w;
  g.encode(w);
  ByteReader r(w.bytes());
  auto back = ModelGeometry::decode(r);
  REQUIRE(back.is_ok());
  CHECK(back->n_layers == g.n_layers);
  CHECK(back->layer_kinds == g.layer_kinds);
  CHECK(r.at_end());

  for (std::size_t cut = 0; cut + 1 < w.size(); cut += 7) {
    ByteReader tr(ByteSpan(w.bytes()).subspan(0, cut));
    CHECK_FALSE(ModelGeometry::decode(tr).is_ok());
  }
  ModelGeometry bad = g;
  bad.n_active_experts = bad.n_experts + 1;
  CHECK(bad.validate().code() == ErrorCode::kInvalidArgument);
  bad = g;
  bad.layer_kinds.pop_back();
  CHECK_FALSE(bad.validate().is_ok());
  bad = g;
  bad.n_kv_heads = bad.n_heads + 1;  // n_heads not a multiple of n_kv_heads
  CHECK_FALSE(bad.validate().is_ok());
}

TEST_CASE("fixture geometry follows the 3:1 recurrent:attention pattern") {
  const ModelGeometry g = FixtureSpec{}.geometry();
  REQUIRE(g.layer_kinds.size() == 16);
  for (std::uint32_t i = 0; i < 16; ++i)
    CHECK((g.layer_kinds[i] == LayerKind::kFullAttention) == (i % 4 == 3));
  CHECK(g.hidden_size == 64);
  CHECK(g.ple_layer == 2);
}

TEST_CASE("manifest binary round trip") {
  ByteWriter w;
  M().encode(w);
  ByteReader r(w.bytes());
  auto back = ModelManifest::decode(r);
  REQUIRE(back.is_ok());
  CHECK(r.at_end());
  CHECK(back->root_hash() == M().root_hash());
  CHECK(back->to_json() == M().to_json());
  CHECK(back->objects.size() == M().objects.size());
}

TEST_CASE("manifest decode is bounded and rejects truncation") {
  ByteWriter w;
  M().encode(w);
  for (std::size_t cut = 0; cut < w.size(); cut += 101) {
    ByteReader r(ByteSpan(w.bytes()).subspan(0, cut));
    CHECK_FALSE(ModelManifest::decode(r).is_ok());
  }
  // Absurd object count claims must not allocate or crash.
  Bytes junk = w.bytes();
  junk.resize(40);
  ByteReader r(junk);
  CHECK_FALSE(ModelManifest::decode(r).is_ok());
}

TEST_CASE("manifest JSON round trip") {
  auto back = ModelManifest::from_json(M().to_json());
  REQUIRE(back.is_ok());
  CHECK(back->root_hash() == M().root_hash());
  CHECK(back->find("token_embd") != nullptr);
  CHECK_FALSE(ModelManifest::from_json("{").is_ok());
  CHECK_FALSE(ModelManifest::from_json("[]").is_ok());
  CHECK_FALSE(ModelManifest::from_json("{\"format_version\":1}").is_ok());
}

TEST_CASE("JSON with a tampered root_hash or object is rejected") {
  std::string json = M().to_json();
  const std::string needle = "\"byte_size\": ";
  const auto pos = json.find(needle, json.find("\"objects\""));
  REQUIRE(pos != std::string::npos);
  json[pos + needle.size()] = json[pos + needle.size()] == '9' ? '8' : '9';  // changes an object's byte_size
  CHECK_FALSE(ModelManifest::from_json(json).is_ok());
}

TEST_CASE("root hash is stable and sensitive to every field") {
  const Digest256 base = M().root_hash();
  CHECK(M().root_hash() == base);

  auto changed = [&](auto&& mutate) {
    ModelManifest m = M();
    mutate(m);
    return m.root_hash() != base;
  };
  CHECK(changed([](ModelManifest& m) { m.artifact_id += "x"; }));
  CHECK(changed([](ModelManifest& m) { m.license += "x"; }));
  CHECK(changed([](ModelManifest& m) { m.geometry.family += "x"; }));
  CHECK(changed([](ModelManifest& m) { m.geometry.hidden_size += 1; }));
  CHECK(changed([](ModelManifest& m) { m.geometry.ple_layer += 1; }));
  CHECK(changed([](ModelManifest& m) { m.geometry.layer_kinds[0] = LayerKind::kFullAttention; }));
  CHECK(changed([](ModelManifest& m) { m.shards[0].file_name += "x"; }));
  CHECK(changed([](ModelManifest& m) { m.shards[0].byte_size += 1; }));
  CHECK(changed([](ModelManifest& m) { m.shards[1].digest.bytes[31] ^= 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].name += "x"; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].kind = ObjectKind::kMtp; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].layer = 3; }));
  CHECK(changed([](ModelManifest& m) { m.objects.back().expert = 0; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].representation.quant_type += "x"; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].representation.block_size += 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].representation.conversion_version += 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].representation.little_endian = false; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].source_ranges[0].offset += 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].source_ranges[0].length += 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].source_ranges[0].shard += 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].byte_size += 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].alignment *= 2; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].peak_workspace += 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].source_digest.bytes[0] ^= 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].object_digest.bytes[0] ^= 1; }));
  CHECK(changed([](ModelManifest& m) { m.objects[5].dependencies.push_back("x"); }));
  CHECK(changed([](ModelManifest& m) { std::swap(m.objects[4], m.objects[5]); }));
}

TEST_CASE("a routed expert is described by three separate source ranges") {
  const ManifestObject* e = M().find(expert_object_name(3, 5));
  REQUIRE(e != nullptr);
  CHECK(e->kind == ObjectKind::kRoutedExpert);
  CHECK(*e->layer == 3);
  CHECK(*e->expert == 5);
  REQUIRE(e->source_ranges.size() == 3);
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    total += e->source_ranges[i].length;
    if (i > 0) {  // slices live in different stacked tensors: never adjacent
      const auto& prev = e->source_ranges[i - 1];
      CHECK(e->source_ranges[i].offset > prev.offset + prev.length);
    }
  }
  CHECK(total == e->byte_size);
  CHECK(e->source_digest == e->object_digest);

  // Dense and shared objects are also made of several tensors.
  CHECK(M().find(dense_object_name(0))->source_ranges.size() == 6);  // norm,w_in,decay,w_out,router,inj
  CHECK(M().find(dense_object_name(3))->source_ranges.size() == 7);  // norm,wq,wk,wv,wo,router,inj
  CHECK(M().find(shared_expert_object_name(0))->source_ranges.size() == 3);
}

TEST_CASE("odd layers use q8_0-fixture experts, even layers f32") {
  for (std::uint32_t L = 0; L < M().geometry.n_layers; ++L) {
    const ManifestObject* e = M().find(expert_object_name(L, 0));
    REQUIRE(e != nullptr);
    CHECK(e->representation.quant_type == (L % 2 ? "q8_0-fixture" : "f32"));
    CHECK(e->representation.block_size == (L % 2 ? 32u : 0u));
  }
}

TEST_CASE("object naming, layer_objects and total_bytes") {
  const LayerRange r{2, 4};
  auto objs = M().layer_objects(r);
  const std::size_t per_layer = 1 + 1 + M().geometry.n_experts;
  CHECK(objs.size() == 2 * per_layer);
  std::uint64_t sum = 0;
  for (const ManifestObject* o : objs) {
    CHECK_FALSE(o->father_only());
    CHECK(r.contains(*o->layer));
    sum += o->byte_size;
  }
  CHECK(M().total_bytes(r) == sum);
  CHECK(dense_object_name(7) == "blk.7.dense");
  CHECK(expert_object_name(7, 9) == "blk.7.exp.9");
  CHECK(shared_expert_object_name(7) == "blk.7.shared");
  CHECK(M().find("nope") == nullptr);
  CHECK(M().find(kPleObjectName)->kind == ObjectKind::kPleLookup);
  CHECK(M().find(kPleObjectName)->source_ranges[0].shard == 1);
  CHECK(M().find(kHeadObjectName)->father_only());
}

TEST_CASE("validation rejects bad ranges and fields") {
  REQUIRE(M().validate().is_ok());
  auto rejects = [](auto&& mutate) {
    ModelManifest m = M();
    mutate(m);
    return !m.validate().is_ok();
  };
  const std::size_t exp_idx = static_cast<std::size_t>(M().find(expert_object_name(1, 1)) - M().objects.data());

  CHECK(rejects([&](ModelManifest& m) {  // overlapping ranges inside one object
    auto& rs = m.objects[exp_idx].source_ranges;
    rs[1].offset = rs[0].offset + 8;
  }));
  CHECK(rejects([&](ModelManifest& m) {  // range extends past the shard
    auto& r = m.objects[exp_idx].source_ranges[2];
    r.offset = m.shards[0].byte_size - 4;
  }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].source_ranges[0].shard = 9; }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].source_ranges[0].length = 0; }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].source_ranges[0].offset = ~std::uint64_t{0} - 2; }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].byte_size += 1; }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].source_ranges.clear(); }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].expert = m.geometry.n_experts; }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].layer.reset(); }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].name = m.objects[exp_idx + 1].name; }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].alignment = 48; }));
  CHECK(rejects([&](ModelManifest& m) { m.objects[exp_idx].dependencies.push_back("missing"); }));
  CHECK(rejects([&](ModelManifest& m) { m.shards[0].file_name = "../escape.bin"; }));
  CHECK(rejects([&](ModelManifest& m) { m.shards[0].file_name = "/abs/path.bin"; }));
  CHECK(rejects([&](ModelManifest& m) { m.shards.clear(); }));
  CHECK(rejects([&](ModelManifest& m) { m.geometry.vocab_size = 0; }));
}

TEST_CASE("q8_0-fixture codec dequantizes exactly and is block-local") {
  std::vector<float> v(64);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(static_cast<float>(i)) * (i < 32 ? 1.0f : 100.0f);
  Bytes enc;
  REQUIRE(encode_q8_fixture(v, enc).is_ok());
  CHECK(enc.size() == tensor_bytes("q8_0-fixture", 64));
  CHECK(enc.size() == 2 * 36);
  std::vector<float> dec(64);
  REQUIRE(decode_tensor("q8_0-fixture", enc, dec).is_ok());
  // Re-encoding the decoded values is lossless: dequantization is exact scale*q.
  Bytes enc2;
  REQUIRE(encode_q8_fixture(dec, enc2).is_ok());
  CHECK(enc2 == enc);
  for (std::size_t i = 0; i < 64; ++i) CHECK(std::fabs(dec[i] - v[i]) <= (i < 32 ? 1.0f : 100.0f) / 127.0f * 0.51f);
  // Hand-built block: scale 0.5, q = i-16.
  Bytes blk;
  const float scale = 0.5f;
  encode_f32(std::span<const float>(&scale, 1), blk);
  for (int i = 0; i < 32; ++i) blk.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(i - 16)));
  std::vector<float> out(32);
  REQUIRE(decode_tensor("q8_0-fixture", blk, out).is_ok());
  for (int i = 0; i < 32; ++i) CHECK(out[static_cast<std::size_t>(i)] == 0.5f * static_cast<float>(i - 16));
  CHECK(decode_tensor("q8_0-fixture", ByteSpan(blk).subspan(0, 35), out).code() == ErrorCode::kDataLoss);
  CHECK_FALSE(decode_tensor("mystery", blk, out).is_ok());
}
