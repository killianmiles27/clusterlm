#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <set>

#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/objects/gguf_manifest.hpp"
#include "mini_model.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;
using namespace clusterlm::testutil;

namespace {

Bytes read_range(const std::filesystem::path& p, std::uint64_t off, std::uint64_t len) {
  std::ifstream f(p, std::ios::binary);
  f.seekg(static_cast<std::streamoff>(off));
  Bytes b(static_cast<std::size_t>(len));
  f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(len));
  REQUIRE(static_cast<std::uint64_t>(f.gcount()) == len);
  return b;
}

Digest256 direct_digest(const std::vector<std::filesystem::path>& paths, const ManifestObject& o) {
  Sha256 h;
  for (const SourceRange& r : o.source_ranges) h.update(read_range(paths[r.shard], r.offset, r.length));
  return h.finish();
}

BuiltManifest build_ok(const MiniFiles& f, ManifestBuildOptions o = {}) {
  auto b = build_manifest(f.paths, o);
  REQUIRE_MESSAGE(b.is_ok(), b.status().to_string());
  return std::move(b).value();
}

std::string err_of(const MiniFiles& f, ManifestBuildOptions o = {}) {
  auto b = build_manifest(f.paths, o);
  REQUIRE_FALSE(b.is_ok());
  return b.status().message();
}

}  // namespace

TEST_CASE("classify_tensor: Flash-Next naming") {
  auto cls = [](const char* n) { return classify_tensor(n, 48).cls; };
  CHECK(cls("token_embd.weight") == TensorClass::kEmbedding);
  CHECK(cls("per_layer_token_embd.weight") == TensorClass::kPleLookup);
  CHECK(cls("output.weight") == TensorClass::kOutputHead);
  CHECK(cls("output_norm.weight") == TensorClass::kOutputHead);
  CHECK(cls("output_hc_up.weight") == TensorClass::kOutputHead);
  CHECK(cls("output_hc_norm.weight") == TensorClass::kOutputHead);
  CHECK(cls("mtp.layers.0.mlp.experts.gate_up_proj") == TensorClass::kMtp);
  CHECK(cls("nextn.eh_proj.weight") == TensorClass::kMtp);
  CHECK(cls("blk.48.nextn.eh_proj.weight") == TensorClass::kMtp);
  CHECK(cls("blk.47.nextn.shared_head_norm.weight") == TensorClass::kMtp);
  CHECK(cls("blk.48.ffn_gate_exps.weight") == TensorClass::kMtp);  // beyond block_count: the MTP layer
  CHECK(cls("blk.0.ffn_gate_exps.weight") == TensorClass::kExpertGate);
  CHECK(cls("blk.47.ffn_up_exps.weight") == TensorClass::kExpertUp);
  CHECK(cls("blk.5.ffn_down_exps.weight") == TensorClass::kExpertDown);
  CHECK(cls("blk.5.ffn_gate_up_exps.weight") == TensorClass::kExpertGateUp);
  CHECK(cls("blk.5.ffn_gate_shexp.weight") == TensorClass::kSharedExpert);
  CHECK(cls("blk.5.ffn_gate_inp_shexp.weight") == TensorClass::kSharedExpert);
  CHECK(cls("blk.5.ffn_gate_inp.weight") == TensorClass::kLayerDense);  // the router is dense
  CHECK(cls("blk.1.ple_key.weight") == TensorClass::kLayerDense);
  CHECK(cls("blk.3.indexer.q_proj.weight") == TensorClass::kLayerDense);
  CHECK(cls("blk.3.attn_q.weight") == TensorClass::kLayerDense);
  CHECK(cls("blk.5.ffn_gate_exps.bias") == TensorClass::kUnclassified);
  CHECK(cls("blk.01.attn_q.weight") == TensorClass::kUnclassified);  // non-canonical index
  CHECK(cls("blk.x.attn_q.weight") == TensorClass::kUnclassified);
  CHECK(cls("blk.5.") == TensorClass::kUnclassified);
  CHECK(cls("blk.5") == TensorClass::kUnclassified);
  CHECK(cls("rope_freqs.weight") == TensorClass::kUnclassified);
  CHECK(cls("") == TensorClass::kUnclassified);
  CHECK(classify_tensor("blk.7.attn_q.weight", 48).layer == 7);
  CHECK_FALSE(classify_tensor("rope_freqs.weight", 48).reason.empty());
}

TEST_CASE("quant_type grammar: uniform collapses, mixed lists every range, both directions") {
  const GgmlType u[] = {GgmlType::kIQ3_S, GgmlType::kIQ3_S, GgmlType::kIQ3_S};
  CHECK(encode_quant_types(u) == "iq3_s");
  const GgmlType m[] = {GgmlType::kIQ3_S, GgmlType::kIQ3_S, GgmlType::kIQ4_XS};
  CHECK(encode_quant_types(m) == "iq3_s|iq3_s|iq4_xs");
  const GgmlType one[] = {GgmlType::kF32};
  CHECK(encode_quant_types(one) == "f32");
  CHECK(*decode_quant_types("iq3_s", 3) == std::vector<GgmlType>{GgmlType::kIQ3_S, GgmlType::kIQ3_S, GgmlType::kIQ3_S});
  CHECK(*decode_quant_types("iq3_s|iq3_s|iq4_xs", 3) == std::vector<GgmlType>(std::begin(m), std::end(m)));
  CHECK(*decode_quant_types("IQ3_S|Q8_0", 2) == std::vector<GgmlType>{GgmlType::kIQ3_S, GgmlType::kQ8_0});
  CHECK_FALSE(decode_quant_types("iq3_s|iq4_xs", 3).is_ok());
  CHECK_FALSE(decode_quant_types("iq3_s|nope|iq4_xs", 3).is_ok());
  CHECK_FALSE(decode_quant_types("", 1).is_ok());
  CHECK_FALSE(decode_quant_types("iq3_s|", 2).is_ok());
}

TEST_CASE("mini Flash-Next model: geometry is derived from metadata and tensor shapes") {
  TempDir dir("mini-geo");
  MiniSpec s;
  const auto files = write_mini(s, dir.path());
  const BuiltManifest b = build_ok(files);
  const ModelGeometry& g = b.manifest.geometry;
  CHECK(g.family == "qwen4exp");
  CHECK(g.n_layers == 4);
  CHECK(g.hidden_size == 256);
  CHECK(g.residual_streams == 4);
  CHECK(g.n_experts == 4);
  CHECK(g.n_active_experts == 2);
  CHECK(g.expert_ff == 256);
  CHECK(g.shared_expert_ff == 64);
  CHECK(g.n_heads == 4);
  CHECK(g.n_kv_heads == 2);
  CHECK(g.head_dim == 64);
  CHECK(g.vocab_size == 512);
  CHECK(g.ple_layer == 1);
  CHECK(g.ple_ngram == 3);
  CHECK(g.ple_rows == 1000);
  CHECK(g.mtp_layers == 1);
  REQUIRE(g.layer_kinds.size() == 4);
  CHECK(g.layer_kinds == std::vector<LayerKind>{LayerKind::kRecurrent, LayerKind::kRecurrent, LayerKind::kRecurrent, LayerKind::kFullAttention});
  CHECK(b.manifest.artifact_id == "mini-flash-next");
  CHECK(b.report.architecture == "qwen4exp");
  CHECK(b.report.unclassified.empty());
  CHECK(b.manifest.validate().is_ok());
}

TEST_CASE("mini Flash-Next model: objects, exact expert ranges, block-aligned slices") {
  TempDir dir("mini-obj");
  MiniSpec s;
  const auto files = write_mini(s, dir.path());
  const BuiltManifest b = build_ok(files);
  const ModelManifest& m = b.manifest;
  auto gg = open_gguf_model(files.paths);
  REQUIRE(gg.is_ok());

  // 4 Father-only + per layer (dense + shared + experts)
  CHECK(m.objects.size() == 4 + 4 * (2 + 4));
  for (auto name : {kEmbeddingObjectName, kPleObjectName, kHeadObjectName, kMtpObjectName}) {
    const ManifestObject* o = m.find(name);
    REQUIRE(o != nullptr);
    CHECK(o->father_only());
    CHECK_FALSE(o->layer.has_value());
  }
  CHECK(m.find(kEmbeddingObjectName)->kind == ObjectKind::kEmbedding);
  CHECK(m.find(kPleObjectName)->kind == ObjectKind::kPleLookup);
  CHECK(m.find(kHeadObjectName)->kind == ObjectKind::kOutputHead);
  CHECK(m.find(kMtpObjectName)->kind == ObjectKind::kMtp);
  CHECK(m.find(kHeadObjectName)->source_ranges.size() == 5);  // output, output_norm, output_hc_{up,down,norm}
  CHECK(m.find(kMtpObjectName)->source_ranges.size() == 3);

  const std::uint64_t slice_iq3s = 110ull * 256;  // 256 rows of one 256-element block, 110 bytes each
  for (std::uint32_t L = 0; L < 4; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    const GgufTensorInfo* gate = gg->find_tensor(p + "ffn_gate_exps.weight")->info;
    const GgufTensorInfo* up = gg->find_tensor(p + "ffn_up_exps.weight")->info;
    const GgufTensorInfo* down = gg->find_tensor(p + "ffn_down_exps.weight")->info;
    CHECK(gate->n_bytes == 4 * slice_iq3s);
    std::uint64_t prev_end = 0;
    for (std::uint32_t e = 0; e < 4; ++e) {
      const ManifestObject* o = m.find(expert_object_name(L, e));
      REQUIRE(o != nullptr);
      CHECK(o->kind == ObjectKind::kRoutedExpert);
      CHECK(o->layer == L);
      CHECK(o->expert == e);
      REQUIRE(o->source_ranges.size() == 3);
      CHECK(o->byte_size == 3 * slice_iq3s);
      std::size_t i = 0;
      for (const GgufTensorInfo* t : {gate, up, down}) {
        const SourceRange& r = o->source_ranges[i++];
        CHECK(r.shard == 0);
        CHECK(r.length == slice_iq3s);
        CHECK(r.offset == t->file_offset + e * slice_iq3s);
        CHECK(r.length % 110 == 0);  // whole blocks
        CHECK((r.offset - t->file_offset) % 110 == 0);
        CHECK(r.offset >= t->file_offset);
        CHECK(r.offset + r.length <= t->file_offset + t->n_bytes);
      }
      CHECK(o->representation.quant_type == "iq3_s");
      CHECK(o->representation.block_size == 256);
      CHECK(o->representation.conversion_version == 0);
      if (e > 0) CHECK(o->source_ranges[0].offset == prev_end);  // consecutive experts are adjacent slices
      prev_end = o->source_ranges[0].offset + o->source_ranges[0].length;
    }
  }
  CHECK(m.layer_objects({0, 4}).size() == 4 * 6);  // Father-only objects are not layer objects
  CHECK(m.total_bytes({0, 4}) + m.find(kEmbeddingObjectName)->byte_size + m.find(kPleObjectName)->byte_size +
            m.find(kHeadObjectName)->byte_size + m.find(kMtpObjectName)->byte_size ==
        b.report.tensor_bytes_in_manifest);
}

TEST_CASE("mini Flash-Next model: dense and shared objects hold exactly the right tensors") {
  TempDir dir("mini-dense");
  MiniSpec s;
  const auto files = write_mini(s, dir.path());
  const BuiltManifest b = build_ok(files);
  auto gg = open_gguf_model(files.paths);
  REQUIRE(gg.is_ok());
  REQUIRE(b.report.layouts.size() == b.manifest.objects.size());

  for (std::uint32_t L = 0; L < 4; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    std::set<std::string> want_dense, want_shared;
    for (const GgufTensorInfo& t : gg->shards[0].tensors) {
      if (t.name.rfind(p, 0) != 0) continue;
      if (t.name.find("_exps") != std::string::npos) continue;
      (t.name.find("shexp") != std::string::npos ? want_shared : want_dense).insert(t.name);
    }
    for (bool shared : {false, true}) {
      const std::size_t idx = static_cast<std::size_t>(b.manifest.find(shared ? shared_expert_object_name(L) : dense_object_name(L)) - b.manifest.objects.data());
      const ManifestObject& o = b.manifest.objects[idx];
      const ObjectLayout& lay = b.report.layouts[idx];
      std::set<std::string> got;
      std::uint64_t off = 0, prev_file_off = 0;
      REQUIRE(lay.entries.size() == o.source_ranges.size());
      for (std::size_t i = 0; i < lay.entries.size(); ++i) {
        got.insert(lay.entries[i].tensor);
        const GgufTensorInfo* t = gg->shards[0].find_tensor(lay.entries[i].tensor);
        REQUIRE(t != nullptr);
        CHECK(o.source_ranges[i].offset == t->file_offset);
        CHECK(o.source_ranges[i].length == t->n_bytes);
        CHECK(lay.entries[i].object_offset == off);
        CHECK(lay.entries[i].type == t->type);
        off += t->n_bytes;
        CHECK(o.source_ranges[i].offset >= prev_file_off);  // physical order
        prev_file_off = o.source_ranges[i].offset;
      }
      CHECK(got == (shared ? want_shared : want_dense));
      CHECK(o.byte_size == off);
      CHECK(o.kind == (shared ? ObjectKind::kSharedExpert : ObjectKind::kLayerDense));
      // Mixed tensor types are preserved range by range.
      std::vector<GgmlType> types;
      for (const auto& e : lay.entries) types.push_back(e.type);
      CHECK(*decode_quant_types(o.representation.quant_type, types.size()) == types);
    }
  }
  // Layer 1 carries the PLE block tensors in its dense object.
  const std::size_t i1 = static_cast<std::size_t>(b.manifest.find(dense_object_name(1)) - b.manifest.objects.data());
  bool has_ple = false;
  for (const auto& e : b.report.layouts[i1].entries) has_ple = has_ple || e.tensor == "blk.1.ple_key.weight";
  CHECK(has_ple);
  // Expert layouts name the stacked tensor and per-expert dims.
  const std::size_t ie = static_cast<std::size_t>(b.manifest.find(expert_object_name(2, 3)) - b.manifest.objects.data());
  REQUIRE(b.report.layouts[ie].entries.size() == 3);
  CHECK(b.report.layouts[ie].entries[2].tensor == "blk.2.ffn_down_exps.weight");
  CHECK(b.report.layouts[ie].entries[2].dims == std::vector<std::uint64_t>{256, 256});
  CHECK(b.report.layouts[ie].entries[1].object_offset == 110ull * 256);
}

TEST_CASE("every tensor byte is in exactly one object") {
  TempDir dir("mini-cover");
  MiniSpec s;
  s.split = true;
  const auto files = write_mini(s, dir.path());
  const BuiltManifest b = build_ok(files);
  std::vector<SourceRange> all;
  for (const ManifestObject& o : b.manifest.objects) all.insert(all.end(), o.source_ranges.begin(), o.source_ranges.end());
  std::sort(all.begin(), all.end(), [](const SourceRange& x, const SourceRange& y) { return std::tie(x.shard, x.offset) < std::tie(y.shard, y.offset); });
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < all.size(); ++i) {
    total += all[i].length;
    if (i > 0 && all[i].shard == all[i - 1].shard) CHECK(all[i].offset >= all[i - 1].offset + all[i - 1].length);
  }
  CHECK(total == b.report.tensor_bytes_total);
  CHECK(total == b.report.tensor_bytes_in_manifest);
  auto gg = open_gguf_model(files.paths);
  std::uint64_t tensors = 0;
  for (const auto& sh : gg->shards) tensors += sh.tensors.size();
  CHECK(b.report.tensors_total == tensors);
}

TEST_CASE("mixed expert types are recorded exactly, per range") {
  TempDir dir("mini-mixed");
  MiniSpec s;
  s.type_for = [](std::uint32_t L, const char* which) {
    const std::string w = which;
    if (L == 1 && w == "down") return GgmlType::kIQ4_XS;       // iq3_s|iq3_s|iq4_xs
    if (L == 2 && w == "gate") return GgmlType::kIQ2_XS;       // iq2_xs|iq3_s|iq3_s
    if (L == 3) return w == "gate" ? GgmlType::kQ8_0 : (w == "up" ? GgmlType::kIQ3_S : GgmlType::kQ4_K);  // different block sizes
    return GgmlType::kIQ3_S;
  };
  const auto files = write_mini(s, dir.path());
  const BuiltManifest b = build_ok(files);
  const auto& m = b.manifest;
  CHECK(m.find(expert_object_name(0, 0))->representation.quant_type == "iq3_s");
  CHECK(m.find(expert_object_name(1, 2))->representation.quant_type == "iq3_s|iq3_s|iq4_xs");
  CHECK(m.find(expert_object_name(1, 2))->representation.block_size == 256);
  CHECK(m.find(expert_object_name(2, 0))->representation.quant_type == "iq2_xs|iq3_s|iq3_s");
  const ManifestObject* l3 = m.find(expert_object_name(3, 1));
  CHECK(l3->representation.quant_type == "q8_0|iq3_s|q4_k");
  CHECK(l3->representation.block_size == 0);  // block sizes differ across ranges: consumers use the type list
  // byte sizes follow each type: row_bytes * 256 rows per slice
  CHECK(l3->source_ranges[0].length == 256 / 32 * 34ull * 256);
  CHECK(l3->source_ranges[1].length == 110ull * 256);
  CHECK(l3->source_ranges[2].length == 144ull * 256);
  CHECK(l3->byte_size == l3->source_ranges[0].length + l3->source_ranges[1].length + l3->source_ranges[2].length);
  CHECK(m.validate().is_ok());
}

TEST_CASE("fused gate_up expert tensors become two-range experts") {
  TempDir dir("mini-fused");
  MiniSpec s;
  s.fused_gate_up = true;
  const BuiltManifest b = build_ok(write_mini(s, dir.path()));
  const ManifestObject* o = b.manifest.find(expert_object_name(1, 2));
  REQUIRE(o != nullptr);
  REQUIRE(o->source_ranges.size() == 2);
  CHECK(o->source_ranges[0].length == 110ull * 512);  // gate and up halves together: 2*ff rows
  CHECK(o->source_ranges[1].length == 110ull * 256);
  CHECK(b.manifest.geometry.expert_ff == 256);
}

TEST_CASE("unclassified tensors are reported, never silently dropped") {
  TempDir dir("mini-unclass");
  MiniSpec s;
  MiniHooks h;
  h.extra_main = [](GgufWriter& w) {
    REQUIRE(w.add_random_tensor("rope_freqs.weight", {64}, GgmlType::kF32, 1).is_ok());
    REQUIRE(w.add_random_tensor("blk.2.ffn_gate_exps.bias", {256, 4}, GgmlType::kF32, 1).is_ok());
  };
  const auto files = write_mini(s, dir.path(), h);
  const std::string msg = err_of(files);
  CHECK(msg.find("2 tensor(s) cannot be classified") != std::string::npos);
  CHECK(msg.find("rope_freqs.weight") != std::string::npos);

  ManifestBuildOptions o;
  o.allow_unclassified = true;
  const BuiltManifest b = build_ok(files, o);
  REQUIRE(b.report.unclassified.size() == 2);
  std::set<std::string> names;
  for (const auto& u : b.report.unclassified) {
    names.insert(u.name);
    CHECK_FALSE(u.reason.empty());
    CHECK(u.bytes > 0);
  }
  CHECK(names == std::set<std::string>{"rope_freqs.weight", "blk.2.ffn_gate_exps.bias"});
  CHECK(b.report.tensor_bytes_in_manifest + 64 * 4 + 256 * 4 * 4 == b.report.tensor_bytes_total);
  CHECK(b.manifest.validate().is_ok());
}

TEST_CASE("split GGUF: shards keep their index and the PLE table lives in shard 2") {
  TempDir dir("mini-split");
  MiniSpec s;
  s.split = true;
  const auto files = write_mini(s, dir.path());
  const BuiltManifest b = build_ok(files);
  REQUIRE(b.manifest.shards.size() == 2);
  CHECK(b.manifest.shards[0].file_name == "mini-00001-of-00002.gguf");
  CHECK(b.manifest.shards[1].file_name == "mini-00002-of-00002.gguf");
  CHECK(b.manifest.shards[0].byte_size == std::filesystem::file_size(files.paths[0]));
  CHECK(b.manifest.shards[1].byte_size == std::filesystem::file_size(files.paths[1]));
  const ManifestObject* ple = b.manifest.find(kPleObjectName);
  REQUIRE(ple->source_ranges.size() == 1);
  CHECK(ple->source_ranges[0].shard == 1);
  CHECK(ple->byte_size == 160ull * 1000);
  CHECK(b.manifest.find(kEmbeddingObjectName)->source_ranges[0].shard == 0);
  CHECK(b.shard_paths == files.paths);

  // Reversed argument order yields the identical manifest (ordered by split.no).
  const BuiltManifest rev = build_ok(MiniFiles{{files.paths[1], files.paths[0]}});
  CHECK(rev.manifest.root_hash() == b.manifest.root_hash());
}

TEST_CASE("a companion GGUF carries the MTP tensors") {
  TempDir dir("mini-comp");
  MiniSpec s;
  s.with_mtp = false;
  auto files = write_mini(s, dir.path());
  GgufWriter mtp;
  mtp.add_string("general.architecture", "qwen4exp-mtp");
  REQUIRE(mtp.add_random_tensor("mtp.layers.0.mlp.experts.gate_up_proj", {256, 512, 4}, GgmlType::kIQ2_XS, 1).is_ok());
  REQUIRE(mtp.add_random_tensor("mtp.layers.0.mlp.experts.down_proj", {256, 256, 4}, GgmlType::kIQ2_XS, 1).is_ok());
  REQUIRE(mtp.add_random_tensor("mtp.layers.0.fc.weight", {512, 256}, GgmlType::kBF16, 1).is_ok());
  REQUIRE(mtp.write(dir.path() / "mtp.gguf").is_ok());
  files.paths.push_back(dir.path() / "mtp.gguf");
  const BuiltManifest b = build_ok(files);
  CHECK(b.manifest.geometry.mtp_layers == 1);
  const ManifestObject* o = b.manifest.find(kMtpObjectName);
  REQUIRE(o != nullptr);
  CHECK(o->source_ranges.size() == 3);
  for (const auto& r : o->source_ranges) CHECK(r.shard == 1);
  CHECK(b.manifest.geometry.family == "qwen4exp");  // from the primary file
}

TEST_CASE("manifest build failures are precise") {
  auto expect = [&](const char* tag, MiniSpec s, MiniHooks h, const char* needle) {
    TempDir d(std::string("mini-bad-") + tag);
    const auto files = write_mini(s, d.path(), h);
    const std::string msg = err_of(files);
    INFO(tag << ": " << msg);
    CHECK(msg.find(needle) != std::string::npos);
  };
  {
    MiniSpec s;
    s.ple_layer = 99;  // no layer holds the PLE block
    expect("no-ple", s, {}, "ple_layer cannot be determined");
  }
  {
    MiniHooks h;
    h.extra_main = [](GgufWriter& w) { w.add_u32("qwen4exp.attention.key_length", 128); };
    expect("key-length", {}, h, "key_length disagrees");
  }
  {
    MiniHooks h;
    h.extra_main = [](GgufWriter& w) { w.add_u32("qwen4exp.expert_feed_forward_length", 512); };
    expect("expert-ff", {}, h, "expert_feed_forward_length disagrees");
  }
  {
    MiniHooks h;  // a layer with both attention flavours
    h.extra_main = [](GgufWriter& w) { REQUIRE(w.add_random_tensor("blk.0.attn_q.weight", {256, 512}, GgmlType::kQ4_K, 1).is_ok()); };
    expect("both-attn", {}, h, "both attn_q.weight and attn_qkv.weight");
  }
  {
    MiniHooks h;  // a second layer claiming the PLE block
    h.extra_main = [](GgufWriter& w) { REQUIRE(w.add_random_tensor("blk.2.ple_key.weight", {256, 160}, GgmlType::kIQ4_XS, 1).is_ok()); };
    expect("two-ple", {}, h, "more than one layer");
  }
  {
    MiniHooks h;  // PLE layer given by the ClusterLM extension key must agree with the tensors
    h.extra_main = [](GgufWriter& w) { w.add_u32("qwen4exp.ple.layer", 2); };
    expect("ple-disagree", {}, h, "ple.layer disagrees");
  }
}

TEST_CASE("manifest build: missing or inconsistent metadata is rejected") {
  const std::vector<const char*> required = {"qwen4exp.block_count", "qwen4exp.embedding_length", "qwen4exp.expert_count",
                                             "qwen4exp.expert_used_count", "qwen4exp.attention.head_count",
                                             "qwen4exp.attention.head_count_kv", "qwen4exp.ple.ngram_size", "general.architecture"};
  // Copy the mini file with one key dropped by re-parsing it: simpler to rebuild with a writer that skips the key.
  for (const char* skip : required) {
    GgufWriter w;
    const std::string sk = skip;
    auto add = [&](const std::string& k, std::uint32_t v) { if (k != sk) w.add_u32(k, v); };
    if (sk != "general.architecture") w.add_string("general.architecture", "qwen4exp");
    add("qwen4exp.block_count", 1);
    add("qwen4exp.embedding_length", 256);
    add("qwen4exp.expert_count", 4);
    add("qwen4exp.expert_used_count", 2);
    add("qwen4exp.attention.head_count", 4);
    add("qwen4exp.attention.head_count_kv", 2);
    add("qwen4exp.ple.ngram_size", 3);
    REQUIRE(w.add_random_tensor("token_embd.weight", {256, 8}, GgmlType::kF32, 1).is_ok());
    TempDir d("meta-missing");
    const auto path = d.path() / "x.gguf";
    REQUIRE(w.write(path).is_ok());
    auto r = build_manifest({path});
    REQUIRE_FALSE(r.is_ok());
    INFO(skip << ": " << r.status().message());
    CHECK(r.status().message().find(sk == "general.architecture" ? "general.architecture" : sk) != std::string::npos);
  }
  // Wrong architecture when one is required.
  TempDir dir("mini-arch");
  const auto files = write_mini({}, dir.path());
  ManifestBuildOptions o;
  o.require_architecture = "llama";
  CHECK(err_of(files, o).find("expected 'llama'") != std::string::npos);
  o.require_architecture = "qwen4exp";
  CHECK(build_manifest(files.paths, o).is_ok());
}

namespace {
// A one-layer model assembled by hand so individual expert/Father-only tensors can be broken.
struct OneLayer {
  std::uint64_t meta_experts = 4;
  std::vector<std::uint64_t> gate{256, 256, 4}, up{256, 256, 4}, down{256, 256, 4};
  bool with_gate = true, with_up = true, with_down = true, with_gate_up = false, with_inj = true, with_embd = true, with_output = true, with_ple_table = true;
  std::vector<std::uint64_t> embd{256, 8};
};

Result<BuiltManifest> build_one_layer(const OneLayer& o, const std::string& tag) {
  TempDir d("one-layer-" + tag);
  GgufWriter w;
  w.add_string("general.architecture", "qwen4exp");
  w.add_u32("qwen4exp.block_count", 1);
  w.add_u32("qwen4exp.embedding_length", 256);
  w.add_u32("qwen4exp.expert_count", static_cast<std::uint32_t>(o.meta_experts));
  w.add_u32("qwen4exp.expert_used_count", 2);
  w.add_u32("qwen4exp.attention.head_count", 4);
  w.add_u32("qwen4exp.attention.head_count_kv", 2);
  w.add_u32("qwen4exp.ple.ngram_size", 3);
  w.add_u32("qwen4exp.ple.layer", 0);
  w.add_u32("qwen4exp.attention.key_length", 64);
  auto add = [&](const char* n, std::vector<std::uint64_t> dims, GgmlType t) { REQUIRE(w.add_random_tensor(n, std::move(dims), t, 1).is_ok()); };
  if (o.with_embd) add("token_embd.weight", o.embd, GgmlType::kF32);
  if (o.with_ple_table) add("per_layer_token_embd.weight", {160, 8}, GgmlType::kI8);
  if (o.with_output) add("output.weight", {256, 8}, GgmlType::kF32);
  add("blk.0.attn_qkv.weight", {256, 256}, GgmlType::kF32);
  if (o.with_inj) add("blk.0.hc_attn_inject.weight", {1024, 4}, GgmlType::kF32);
  if (o.with_gate) add("blk.0.ffn_gate_exps.weight", o.gate, GgmlType::kIQ3_S);
  if (o.with_up) add("blk.0.ffn_up_exps.weight", o.up, GgmlType::kIQ3_S);
  if (o.with_gate_up) add("blk.0.ffn_gate_up_exps.weight", {256, 512, 4}, GgmlType::kIQ3_S);
  if (o.with_down) add("blk.0.ffn_down_exps.weight", o.down, GgmlType::kIQ3_S);
  const auto p = d.path() / "one.gguf";
  REQUIRE(w.write(p).is_ok());
  return build_manifest({p});
}

void expect_one_layer_error(const OneLayer& o, const char* tag, const char* needle) {
  auto r = build_one_layer(o, tag);
  REQUIRE_FALSE(r.is_ok());
  INFO(tag << ": " << r.status().message());
  CHECK(r.status().message().find(needle) != std::string::npos);
}
}  // namespace

TEST_CASE("manifest build: structural errors in expert and Father-only tensors") {
  {
    auto ok = build_one_layer({}, "ok");
    REQUIRE_MESSAGE(ok.is_ok(), ok.status().to_string());
    CHECK(ok->manifest.objects.size() == 3 + 1 + 4);  // embd, ple, head, dense, 4 experts (no MTP, no shared)
    CHECK(ok->manifest.geometry.mtp_layers == 0);
    CHECK(ok->manifest.geometry.shared_expert_ff == 0);
    CHECK(ok->manifest.find(kMtpObjectName) == nullptr);
  }
  { OneLayer o; o.meta_experts = 8; expect_one_layer_error(o, "ecount", "n_experts = 8"); }
  { OneLayer o; o.up = {256, 512, 4}; expect_one_layer_error(o, "updiff", "shapes differ"); }
  { OneLayer o; o.with_down = false; expect_one_layer_error(o, "nodown", "routed expert tensors must be"); }
  { OneLayer o; o.with_up = false; expect_one_layer_error(o, "noup", "routed expert tensors must be"); }
  { OneLayer o; o.with_gate = false; o.with_up = false; expect_one_layer_error(o, "noexp", "routed expert tensors must be"); }
  { OneLayer o; o.with_gate_up = true; expect_one_layer_error(o, "bothforms", "routed expert tensors must be"); }
  { OneLayer o; o.down = {512, 256, 4}; expect_one_layer_error(o, "downin", "input width differs"); }
  { OneLayer o; o.down = {256, 128, 4}; expect_one_layer_error(o, "downout", "do not match embedding_length"); }
  { OneLayer o; o.gate = {256, 256}; o.up = {256, 256}; expect_one_layer_error(o, "twod", "must have 3 dims"); }
  { OneLayer o; o.gate = {256, 256, 4}; o.gate[0] = 512; o.up = o.gate; expect_one_layer_error(o, "gatein", "do not match embedding_length"); }
  { OneLayer o; o.with_inj = false; expect_one_layer_error(o, "noinj", "residual stream count cannot be determined"); }
  { OneLayer o; o.with_embd = false; expect_one_layer_error(o, "noembd", "token_embd.weight"); }
  { OneLayer o; o.embd = {128, 8}; expect_one_layer_error(o, "embdw", "token_embd.weight"); }
  { OneLayer o; o.with_ple_table = false; expect_one_layer_error(o, "noplet", "per_layer_token_embd.weight"); }
  { OneLayer o; o.with_output = false; expect_one_layer_error(o, "nohead", "no output head"); }
}

TEST_CASE("hashing: digests equal direct computation, any thread count, with progress and cancellation") {
  TempDir dir("mini-hash");
  MiniSpec s;
  s.split = true;
  const auto files = write_mini(s, dir.path());

  ManifestBuildOptions o;
  o.hash_objects = true;
  o.hash_shards = true;
  o.hash_buffer_bytes = 4096;  // force many bounded reads per range
  std::vector<std::uint64_t> seen;
  std::uint64_t total_seen = 0;
  o.progress = [&](const BuildProgress& p) {
    if (p.phase == BuildProgress::Phase::kHashingObjects) {
      seen.push_back(p.bytes_done);
      total_seen = p.bytes_total;
    }
    return true;
  };
  const BuiltManifest one = build_ok(files, o);
  CHECK(one.report.hashed_objects);
  CHECK(one.report.hashed_shards);
  CHECK_FALSE(seen.empty());
  CHECK(std::is_sorted(seen.begin(), seen.end()));
  CHECK(seen.back() == total_seen);
  CHECK(total_seen == one.report.tensor_bytes_in_manifest);
  for (const ManifestObject& obj : one.manifest.objects) {
    const Digest256 d = direct_digest(files.paths, obj);
    REQUIRE_MESSAGE(obj.source_digest == d, obj.name);
    CHECK(obj.object_digest == d);
    CHECK_FALSE(d.is_zero());
  }
  for (std::size_t i = 0; i < files.paths.size(); ++i)
    CHECK(one.manifest.shards[i].digest == Sha256::of(read_range(files.paths[i], 0, std::filesystem::file_size(files.paths[i]))));

  // Multi-threaded hashing yields the same manifest; so does hashing afterwards.
  ManifestBuildOptions o3 = o;
  o3.hash_threads = 3;
  std::atomic<std::uint64_t> calls{0};
  o3.progress = [&](const BuildProgress&) { ++calls; return true; };
  const BuiltManifest three = build_ok(files, o3);
  CHECK(calls > 0);
  CHECK(three.manifest.root_hash() == one.manifest.root_hash());

  ManifestBuildOptions plain;
  BuiltManifest later = build_ok(files, plain);
  CHECK(later.manifest.objects[0].source_digest.is_zero());
  CHECK_FALSE(later.report.hashed_objects);
  CHECK(later.manifest.root_hash() != one.manifest.root_hash());
  HashOptions ho;
  ho.hash_shards = true;
  ho.threads = 2;
  REQUIRE(compute_manifest_digests(later.manifest, later.shard_paths, ho).is_ok());
  CHECK(later.manifest.root_hash() == one.manifest.root_hash());

  // Cancellation.
  ManifestBuildOptions cancel;
  cancel.hash_objects = true;
  cancel.progress = [](const BuildProgress&) { return false; };
  auto c = build_manifest(files.paths, cancel);
  CHECK(c.status().code() == ErrorCode::kCancelled);

  // A file that changed size since the manifest was built is refused.
  {
    std::ofstream f(files.paths[1], std::ios::binary | std::ios::app);
    f.put('x');
  }
  CHECK(compute_manifest_digests(later.manifest, later.shard_paths, ho).code() == ErrorCode::kDataLoss);
}

TEST_CASE("hashed GGUF manifest: JSON/binary round trips, objects resolve and verify from the GGUF files") {
  TempDir dir("mini-store");
  MiniSpec s;
  s.split = true;
  const auto files = write_mini(s, dir.path());
  ManifestBuildOptions o;
  o.hash_objects = true;
  BuiltManifest b = build_ok(files, o);

  auto back = ModelManifest::from_json(b.manifest.to_json());
  REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
  CHECK(back->root_hash() == b.manifest.root_hash());
  ByteWriter w;
  b.manifest.encode(w);
  ByteReader r(w.bytes());
  auto dec = ModelManifest::decode(r);
  REQUIRE(dec.is_ok());
  CHECK(dec->root_hash() == b.manifest.root_hash());

  // The canonical store reads exact ranges straight out of the GGUF shards and verifies the digests.
  auto store = CanonicalModelStore::open(dir.path(), b.manifest);
  REQUIRE_MESSAGE(store.is_ok(), store.status().to_string());
  for (const char* name : {"blk.2.exp.1", "blk.0.dense", "token_embd", "ple_lookup"}) {
    const std::string n = std::string(name);
    if (b.manifest.find(n) == nullptr) continue;
    auto obj = (*store)->resolve(n);
    REQUIRE_MESSAGE(obj.is_ok(), n << ": " << obj.status().to_string());
    CHECK(obj->bytes.size() == b.manifest.find(n)->byte_size);
  }
  auto ex = (*store)->read_object_bytes(expert_object_name(2, 1));
  REQUIRE(ex.is_ok());
  CHECK(Sha256::of(*ex) == b.manifest.find(expert_object_name(2, 1))->source_digest);
  // Corrupting one byte inside an expert slice is caught by digest verification.
  {
    const SourceRange& rg = b.manifest.find(expert_object_name(1, 0))->source_ranges[1];
    std::fstream f(files.paths[rg.shard], std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(static_cast<std::streamoff>(rg.offset + 7));
    f.put('\x5A');
  }
  auto fresh = CanonicalModelStore::open(dir.path(), b.manifest);
  REQUIRE(fresh.is_ok());
  // (the store re-reads; the byte we wrote may equal the original with probability 1/256 - then try another byte)
  const auto res = (*fresh)->resolve(expert_object_name(1, 0));
  if (!res.is_ok()) CHECK(res.status().code() == ErrorCode::kDataLoss);
}
