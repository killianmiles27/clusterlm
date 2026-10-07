// Father-side conversion (ADR 0200) end to end on a synthetic Strata model: a pack directory (index.txt + pack files)
// and a GGUF written with Strata's own test fixture writer. The converted directory must open in ClusterLM's
// CanonicalModelStore, every object must resolve with a verified digest, routed experts must be the GGUF's
// [gate | up | down] slices byte for byte, and dense objects the pack rows plus their GGUF-form copies.
#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

#include "clusterlm/backends/strata/convert.hpp"
#include "clusterlm/backends/strata/object_map.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "gguf_fixture.hpp"  // strata/tests/core: synthetic GGUF v3 files

using namespace clusterlm;
namespace bs = clusterlm::backends::strata;
namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kH = 256, kFF = 64, kE = 4, kV = 32, kLayers = 4;

struct Model {
  fs::path dir;
  fs::path pack;
  fs::path gguf;
  fixture::Written written;
  std::vector<fixture::Tensor> tensors;
  std::map<std::string, Bytes> pack_bytes;

  Model() {
    dir = fs::temp_directory_path() / ("clusterlm-strata-convert-" + std::to_string(std::random_device{}()));
    pack = dir / "pack";
    fs::create_directories(pack);
    gguf = dir / "model.gguf";
    std::uint8_t seed = 1;
    for (std::uint64_t L = 0; L < kLayers; ++L) {
      const std::string b = "blk." + std::to_string(L) + ".";
      tensors.push_back({b + "ffn_gate_exps.weight", {kH, kFF, kE}, 21, seed++});  // IQ3_S
      tensors.push_back({b + "ffn_up_exps.weight", {kH, kFF, kE}, 21, seed++});
      tensors.push_back({b + "ffn_down_exps.weight", {kFF, kH, kE}, 20, seed++});  // IQ4_NL
      tensors.push_back({b + "attn_qkv.weight", {kH, 64}, 8, seed++});             // Q8_0, served natively
      tensors.push_back({b + "ffn_gate_shexp.weight", {kH, kFF}, 8, seed++});
    }
    tensors.push_back({"token_embd.weight", {kH, kV}, 8, seed++});
    tensors.push_back({"output.weight", {kH, kV}, 8, seed++});
    written = fixture::write(gguf, {fixture::str("general.architecture", "qwen35moe")}, tensors);

    // the pack: what tools/iq_pack.py would hold for these tensors (synthetic bytes, real row format)
    std::string index = "# align 256 pool 1048576 tensors 0\n";
    std::map<int, Bytes> files;
    std::mt19937 rng(7);
    auto row = [&](const std::string& name, int file, int kind, long long ne0, long long ne1, int bits,
                   std::uint64_t codes, std::uint64_t scales) {
      std::uint64_t src = codes + scales / 2, dst = codes + scales;
      if (bits == 0) src = dst = static_cast<std::uint64_t>(ne0 * (ne1 > 0 ? ne1 : 1)) * 4;
      Bytes b(src);
      for (auto& x : b) x = static_cast<std::uint8_t>(rng());
      Bytes& f = files[file];
      char line[512];
      std::snprintf(line, sizeof line, "%s %d %d %llu %llu 0 %llu %lld %lld %d 0 0 0 0 %llu %llu 0 %d 0\n", name.c_str(),
                    file, kind, static_cast<unsigned long long>(f.size()), static_cast<unsigned long long>(src),
                    static_cast<unsigned long long>(dst), ne0, ne1, bits, static_cast<unsigned long long>(codes),
                    static_cast<unsigned long long>(scales), bits ? 1 : 0);
      index += line;
      f.insert(f.end(), b.begin(), b.end());
      pack_bytes[name] = std::move(b);
    };
    for (std::uint64_t L = 0; L < kLayers; ++L) {
      const std::string b = "blk." + std::to_string(L) + ".";
      row(b + "attn_qkv.weight", 0, 0, kH, 64, 2, kH * 64 / 4, kH * 64 / 64 * 4);
      row(b + "hc_attn_norm.weight", 0, 2, 4 * kH, 0, 0, 0, 0);
      row(b + "ffn_gate_shexp.weight", 0, 0, kH, kFF, 2, kH * kFF / 4, kH * kFF / 64 * 4);
      row(b + "ffn_gate_exps.weight", 2, 0, kH, kFF * kE, 2, kH * kFF * kE / 4, kH * kFF * kE / 64 * 4);
    }
    row("token_embd.weight", 1, 0, kH, kV, 2, kH * kV / 4, kH * kV / 64 * 4);
    row("output.weight", 3, 0, kH, kV, 2, kH * kV / 4, kH * kV / 64 * 4);
    const char* names[] = {"dense.bin", "embd.bin", "experts.bin", "extra.bin"};
    for (auto& [id, b] : files)
      std::ofstream(pack / names[id], std::ios::binary).write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    std::ofstream(pack / "index.txt") << index;
  }
  ~Model() { { std::error_code ec_rm; fs::remove_all(dir, ec_rm); } }

  // the GGUF payload bytes of tensor `name` in [off, off+n)
  Bytes payload(const std::string& name, std::uint64_t off, std::uint64_t n) const {
    for (std::size_t i = 0; i < tensors.size(); ++i)
      if (tensors[i].name == name) {
        Bytes b(n);
        for (std::uint64_t j = 0; j < n; ++j) b[j] = fixture::pattern(tensors[i].seed, off + j);
        return b;
      }
    return {};
  }
};

}  // namespace

TEST_CASE("convert: a Strata pack + GGUF become a ClusterLM model directory the canonical store serves") {
  Model mdl;
  std::vector<std::string> progress;
  auto m = bs::convert_model({mdl.pack, mdl.gguf, {}}, [&](std::string_view s) { progress.emplace_back(s); });
  REQUIRE_MESSAGE(m.is_ok(), m.status().to_string());
  CHECK(progress.size() == kLayers);
  CHECK(m->geometry.n_layers == kLayers);
  CHECK(m->geometry.hidden_size == kH);
  CHECK(m->geometry.expert_ff == kFF);
  CHECK(m->geometry.n_experts == kE);
  CHECK(m->geometry.vocab_size == kV);
  CHECK(m->geometry.shared_expert_ff == kFF);
  CHECK(m->geometry.layer_kinds[3] == objects::LayerKind::kFullAttention);
  CHECK(m->objects.size() == 1 + kLayers * (2 + kE) + 1);

  auto store = objects::CanonicalModelStore::open(mdl.dir);
  REQUIRE_MESSAGE(store.is_ok(), store.status().to_string());
  for (const auto& o : (*store)->manifest().objects) {
    auto p = (*store)->resolve(o.name);  // reads every range and verifies the object digest
    REQUIRE_MESSAGE(p.is_ok(), (o.name + ": " + p.status().to_string()));
  }

  // a routed expert is the three GGUF slices back to back, unchanged
  const std::uint64_t gu_row = 110, d_row = 2 * 18;  // IQ3_S row of 256, IQ4_NL row of 64
  for (std::uint32_t e : {0u, 3u}) {
    auto p = (*store)->resolve(objects::expert_object_name(2, e));
    REQUIRE(p.is_ok());
    CHECK(p->entry->representation.quant_type == "iq3_s+iq4_nl");
    CHECK(p->entry->representation.conversion_version == 0);
    Bytes want = mdl.payload("blk.2.ffn_gate_exps.weight", e * kFF * gu_row, kFF * gu_row);
    const Bytes up = mdl.payload("blk.2.ffn_up_exps.weight", e * kFF * gu_row, kFF * gu_row);
    const Bytes dn = mdl.payload("blk.2.ffn_down_exps.weight", e * kH * d_row, kH * d_row);
    want.insert(want.end(), up.begin(), up.end());
    want.insert(want.end(), dn.begin(), dn.end());
    CHECK(Bytes(p->bytes.begin(), p->bytes.end()) == want);
  }

  // a dense object holds the layer's pack rows (byte-exact) and the GGUF-form copy of what Strata serves natively
  auto d = (*store)->resolve("blk.1.dense");
  REQUIRE(d.is_ok());
  CHECK(d->entry->representation.quant_type == "strata-dense");
  CHECK(d->entry->representation.conversion_version == 1);
  auto parsed = bs::parse_dense_object(d->bytes);
  REQUIRE(parsed.is_ok());
  REQUIRE(parsed->rows.size() == 2);
  for (const auto& r : parsed->rows) {
    const ByteSpan c = parsed->canonical(r);
    CHECK(Bytes(c.begin(), c.end()) == mdl.pack_bytes[r.name]);
  }
  const auto& qkv = parsed->rows[0];
  REQUIRE(qkv.name == "blk.1.attn_qkv.weight");
  REQUIRE(qkv.has_native());
  CHECK(qkv.native_type == 8);
  const ByteSpan nb = parsed->native(qkv);
  CHECK(Bytes(nb.begin(), nb.end()) == mdl.payload("blk.1.attn_qkv.weight", 0, 64 * (kH / 32) * 34));
  auto sh = (*store)->resolve("blk.1.shared");
  REQUIRE(sh.is_ok());
  CHECK(bs::parse_dense_object(sh->bytes)->rows[0].has_native());  // shared-expert projections are native too
  auto hd = (*store)->resolve("output_head");
  REQUIRE(hd.is_ok());
  CHECK(bs::parse_dense_object(hd->bytes)->rows[0].native_ne1 == kV);
}

TEST_CASE("convert refuses an output directory the GGUF is not in, and a missing pack") {
  Model mdl;
  const fs::path elsewhere = mdl.dir / "elsewhere";
  fs::create_directories(elsewhere);
  CHECK(bs::convert_model({mdl.pack, mdl.gguf, elsewhere}).status().code() == ErrorCode::kInvalidArgument);
  CHECK(bs::convert_model({mdl.dir / "nopack", mdl.gguf, {}}).status().code() == ErrorCode::kNotFound);
}

TEST_CASE("convert: a pack with experts.bin under the model directory gives one contiguous range per expert") {
  Model mdl;
  const std::uint64_t gu_row = 110, d_row = 36, blob = 2 * kFF * gu_row + kH * d_row;
  // experts.bin as tools/iq_pack.py lays it out: per layer, kE blobs of [gate | up | down]
  Bytes bin;
  std::string txt = "# strata native experts v3: layer gate_up_type down_type offset blob (n_expert 4)\n";
  for (std::uint64_t L = 0; L < kLayers; ++L) {
    txt += std::to_string(L) + " 21 20 " + std::to_string(bin.size()) + " " + std::to_string(blob) + "\n";
    const std::string b = "blk." + std::to_string(L) + ".";
    for (std::uint64_t e = 0; e < kE; ++e) {
      for (const Bytes& part : {mdl.payload(b + "ffn_gate_exps.weight", e * kFF * gu_row, kFF * gu_row),
                                mdl.payload(b + "ffn_up_exps.weight", e * kFF * gu_row, kFF * gu_row),
                                mdl.payload(b + "ffn_down_exps.weight", e * kH * d_row, kH * d_row)})
        bin.insert(bin.end(), part.begin(), part.end());
    }
  }
  std::ofstream(mdl.pack / "experts.bin", std::ios::binary)
      .write(reinterpret_cast<const char*>(bin.data()), static_cast<std::streamsize>(bin.size()));
  std::ofstream(mdl.pack / "native_experts.txt") << txt;

  auto m = bs::convert_model({mdl.pack, mdl.gguf, {}});
  REQUIRE_MESSAGE(m.is_ok(), m.status().to_string());
  REQUIRE(m->shards.size() == 3);
  CHECK(m->shards[2].file_name == "pack/experts.bin");
  const auto* o = m->find(objects::expert_object_name(3, 2));
  REQUIRE(o != nullptr);
  REQUIRE(o->source_ranges.size() == 1);
  CHECK(o->source_ranges[0].shard == 2);
  CHECK(o->source_ranges[0].offset == 3 * kE * blob + 2 * blob);
  auto store = objects::CanonicalModelStore::open(mdl.dir);
  REQUIRE(store.is_ok());
  auto p = (*store)->resolve(o->name);
  REQUIRE(p.is_ok());
  const auto first = bin.begin() + static_cast<std::ptrdiff_t>(3 * kE * blob + 2 * blob);
  CHECK(Bytes(p->bytes.begin(), p->bytes.end()) == Bytes(first, first + static_cast<std::ptrdiff_t>(blob)));
}
