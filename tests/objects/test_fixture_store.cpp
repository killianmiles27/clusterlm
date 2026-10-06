#include <doctest/doctest.h>

#include <fstream>
#include <iterator>

#include "clusterlm/objects/tensor_codec.hpp"
#include "test_util.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;
using clusterlm::testutil::TempDir;

namespace {
Bytes slurp(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  return Bytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
void spit(const std::filesystem::path& p, const Bytes& b) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}
}  // namespace

TEST_CASE("fixture generation is deterministic in the seed") {
  FixtureSpec spec = FixtureSpec::tiny();
  TempDir a("det-a"), b("det-b"), c("det-c");
  auto ma = write_fixture_model(spec, a.path());
  auto mb = write_fixture_model(spec, b.path());
  REQUIRE(ma.is_ok());
  REQUIRE(mb.is_ok());
  CHECK(ma->root_hash() == mb->root_hash());
  CHECK(slurp(a.path() / "fixture-transformer.bin") == slurp(b.path() / "fixture-transformer.bin"));
  CHECK(slurp(a.path() / "fixture-lookup.bin") == slurp(b.path() / "fixture-lookup.bin"));
  CHECK(slurp(a.path() / "manifest.json") == slurp(b.path() / "manifest.json"));
  for (std::size_t i = 0; i < ma->objects.size(); ++i) {
    CHECK(ma->objects[i].source_digest == mb->objects[i].source_digest);
    CHECK(ma->objects[i].object_digest == mb->objects[i].object_digest);
  }
  spec.seed += 1;
  auto mc = write_fixture_model(spec, c.path());
  REQUIRE(mc.is_ok());
  CHECK(mc->root_hash() != ma->root_hash());
  CHECK(mc->objects[0].object_digest != ma->objects[0].object_digest);
}

TEST_CASE("fixture layout: aligned tensors, two shards, ranges inside shards") {
  const ModelManifest& m = testutil::tiny_fixture().manifest;
  REQUIRE(m.shards.size() == 2);
  CHECK(m.shards[0].file_name == kFixtureTransformerShard);
  CHECK(m.shards[1].file_name == kFixtureLookupShard);
  for (const ManifestObject& o : m.objects) {
    if (o.kind == ObjectKind::kRoutedExpert) continue;
    for (const SourceRange& r : o.source_ranges) CHECK(r.offset % 64 == 0);
  }
  // Experts of one layer are slices of three stacked tensors, so consecutive experts' gate ranges abut.
  const auto& e0 = m.find(expert_object_name(0, 0))->source_ranges;
  const auto& e1 = m.find(expert_object_name(0, 1))->source_ranges;
  CHECK(e1[0].offset == e0[0].offset + e0[0].length);
  CHECK(e1[2].offset == e0[2].offset + e0[2].length);
  CHECK(m.find(kPleObjectName)->byte_size == std::uint64_t{m.geometry.ple_rows} * m.geometry.hidden_size * 4);
}

TEST_CASE("fixture parameters are in range") {
  const auto& f = testutil::tiny_fixture();
  auto store = f.open_store();
  const ModelGeometry& g = f.manifest.geometry;
  auto dense = store->resolve(dense_object_name(0));
  REQUIRE(dense.is_ok());
  const DenseLayout lay = dense_layout(g, g.layer_kinds[0]);
  std::vector<float> v(lay.total);
  REQUIRE(decode_tensor("f32", dense->bytes, v).is_ok());
  for (std::size_t i = 0; i < g.hidden_size; ++i) {
    CHECK(v[lay.decay + i] > 0.5f);
    CHECK(v[lay.decay + i] < 0.95f);
  }
}

TEST_CASE("full-size default fixture matches the specified geometry") {
  const ModelManifest& m = testutil::full_fixture().manifest;
  const ModelGeometry& g = m.geometry;
  CHECK(g.n_layers == 16);
  CHECK(g.hidden_size == 64);
  CHECK(g.residual_streams == 4);
  CHECK(g.n_experts == 32);
  CHECK(g.n_active_experts == 4);
  CHECK(g.expert_ff == 32);
  CHECK(g.vocab_size == 256);
  CHECK(g.ple_rows == 1024);
  CHECK(m.objects.size() == 4 + 16 * (2 + 32));
  CHECK(m.validate().is_ok());
}

TEST_CASE("CanonicalModelStore resolves lazily, verifies, and counts what it loads") {
  const auto& f = testutil::tiny_fixture();
  auto store = f.open_store();
  CHECK(store->loaded_bytes() == 0);
  CHECK(store->loaded_object_count() == 0);

  auto r = store->resolve(dense_object_name(2));
  REQUIRE(r.is_ok());
  CHECK(r->entry->name == dense_object_name(2));
  CHECK(r->bytes.size() == r->entry->byte_size);
  CHECK(store->loaded_object_count() == 1);
  CHECK(store->loaded_bytes() == r->entry->byte_size);

  auto again = store->resolve(dense_object_name(2));  // cached: no growth, same storage
  REQUIRE(again.is_ok());
  CHECK(again->bytes.data() == r->bytes.data());
  CHECK(store->loaded_object_count() == 1);

  // read_object_bytes never caches.
  auto raw = store->read_object_bytes(expert_object_name(1, 3));
  REQUIRE(raw.is_ok());
  CHECK(raw->size() == f.manifest.find(expert_object_name(1, 3))->byte_size);
  CHECK(store->loaded_object_count() == 1);

  CHECK(store->resolve("blk.99.dense").status().code() == ErrorCode::kNotFound);
  store->evict_all();
  CHECK(store->loaded_bytes() == 0);

  // Every object of the model verifies against its digests.
  for (const ManifestObject& o : f.manifest.objects) {
    auto b = store->read_object_bytes(o.name);
    CHECK_MESSAGE(b.is_ok(), o.name);
  }
}

TEST_CASE("an expert object is the concatenation of its three source ranges") {
  const auto& f = testutil::tiny_fixture();
  auto store = f.open_store();
  const ManifestObject* e = f.manifest.find(expert_object_name(2, 6));
  auto obj = store->read_object_bytes(e->name);
  REQUIRE(obj.is_ok());
  const Bytes shard = slurp(f.dir.path() / "fixture-transformer.bin");
  Bytes expect;
  for (const SourceRange& r : e->source_ranges)
    expect.insert(expect.end(), shard.begin() + static_cast<std::ptrdiff_t>(r.offset),
                  shard.begin() + static_cast<std::ptrdiff_t>(r.offset + r.length));
  CHECK(*obj == expect);
}

TEST_CASE("a corrupted byte is detected as data loss, only for the affected object") {
  const auto& f = testutil::tiny_fixture();
  TempDir copy("corrupt");
  for (const char* n : {"fixture-transformer.bin", "fixture-lookup.bin", "manifest.json"})
    std::filesystem::copy_file(f.dir.path() / n, copy.path() / n);
  const ManifestObject* victim = f.manifest.find(expert_object_name(1, 2));
  Bytes shard = slurp(copy.path() / "fixture-transformer.bin");
  shard[static_cast<std::size_t>(victim->source_ranges[1].offset) + 5] ^= 0x40;
  spit(copy.path() / "fixture-transformer.bin", shard);

  auto store = CanonicalModelStore::open(copy.path());
  REQUIRE(store.is_ok());
  CHECK((*store)->resolve(victim->name).status().code() == ErrorCode::kDataLoss);
  CHECK((*store)->loaded_object_count() == 0);
  CHECK((*store)->read_object_bytes(victim->name).status().code() == ErrorCode::kDataLoss);
  CHECK((*store)->resolve(expert_object_name(1, 3)).is_ok());  // neighbours unaffected
}

TEST_CASE("store open rejects a missing manifest and a resized shard") {
  TempDir empty("empty");
  CHECK(CanonicalModelStore::open(empty.path()).status().code() == ErrorCode::kNotFound);

  const auto& f = testutil::tiny_fixture();
  TempDir copy("short");
  for (const char* n : {"fixture-transformer.bin", "fixture-lookup.bin", "manifest.json"})
    std::filesystem::copy_file(f.dir.path() / n, copy.path() / n);
  Bytes shard = slurp(copy.path() / "fixture-lookup.bin");
  shard.resize(shard.size() - 1);
  spit(copy.path() / "fixture-lookup.bin", shard);
  CHECK(CanonicalModelStore::open(copy.path()).status().code() == ErrorCode::kDataLoss);
}

TEST_CASE("InMemoryResolver verifies digests on add and only resolves what was added") {
  const auto& f = testutil::tiny_fixture();
  auto store = f.open_store();
  InMemoryResolver mem(f.manifest);
  CHECK(mem.resolve(dense_object_name(0)).status().code() == ErrorCode::kNotFound);

  auto bytes = store->read_object_bytes(dense_object_name(0));
  REQUIRE(bytes.is_ok());
  Bytes bad = *bytes;
  bad[3] ^= 1;
  CHECK(mem.add(dense_object_name(0), bad).code() == ErrorCode::kDataLoss);
  CHECK(mem.add("not-in-manifest", *bytes).code() == ErrorCode::kNotFound);
  CHECK(mem.size() == 0);

  REQUIRE(mem.add(dense_object_name(0), *bytes).is_ok());
  auto r = mem.resolve(dense_object_name(0));
  REQUIRE(r.is_ok());
  CHECK(r->entry == mem.manifest().find(dense_object_name(0)));
  CHECK(mem.bytes() == bytes->size());
  CHECK(mem.remove(dense_object_name(0)));
  CHECK(mem.resolve(dense_object_name(0)).status().code() == ErrorCode::kNotFound);
}

TEST_CASE("stream_object yields exactly the object bytes in bounded chunks and detects corruption") {
  TempDir dir("stream");
  auto m = write_fixture_model(FixtureSpec::tiny(), dir.path());
  REQUIRE(m.is_ok());
  auto store = CanonicalModelStore::open(dir.path());
  REQUIRE(store.is_ok());
  for (const char* name : {"blk.1.exp.3", "blk.0.dense"}) {
    const auto* obj = store.value()->manifest().find(name);
    if (obj == nullptr) continue;
    auto whole = store.value()->read_object_bytes(name);
    REQUIRE(whole.is_ok());
    Bytes streamed;
    std::size_t max_chunk = 0;
    auto st = store.value()->stream_object(name, 100, [&](std::uint64_t off, ByteSpan c) {
      CHECK(off == streamed.size());
      max_chunk = std::max(max_chunk, c.size());
      streamed.insert(streamed.end(), c.begin(), c.end());
      return Status::ok();
    });
    REQUIRE(st.is_ok());
    CHECK(max_chunk <= 100);
    CHECK(streamed == whole.value());
  }
  // Corrupt one byte of the transformer shard: streaming completes but reports kDataLoss.
  const auto* obj = store.value()->manifest().find("blk.1.exp.3");
  REQUIRE(obj != nullptr);
  {
    std::fstream f(dir.path() / "fixture-transformer.bin", std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(obj->source_ranges[0].offset));
    char c = 0x7f;
    f.write(&c, 1);
  }
  auto st = store.value()->stream_object("blk.1.exp.3", 64, [](std::uint64_t, ByteSpan) { return Status::ok(); });
  CHECK(st.code() == ErrorCode::kDataLoss);
}
