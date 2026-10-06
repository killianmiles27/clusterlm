// Object mapping for the Strata backend: ggml type sizes, expert formats, the strata-dense container, tensor
// placement, and Father-side conversion from a synthetic Strata pack (index.txt + pack files) - no real weights.
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS  // std::sscanf reads Strata's index.txt rows exactly as Strata's loader does
#endif
#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

#include "clusterlm/backends/strata/object_map.hpp"
#include "strata_test_support.hpp"

using namespace clusterlm;
namespace bs = clusterlm::backends::strata;

namespace {

Bytes random_bytes(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Bytes b(n);
  for (auto& x : b) x = static_cast<std::uint8_t>(rng());
  return b;
}

std::filesystem::path temp_dir(const char* name) {
  auto p = std::filesystem::temp_directory_path() / (std::string("clusterlm-strata-") + name + "-" +
                                                     std::to_string(std::random_device{}()));
  std::filesystem::create_directories(p);
  return p;
}

// A synthetic Strata pack: rows in the format tools/pack_index.py writes, bytes in dense.bin / embd.bin / extra.bin.
struct SyntheticPack {
  std::filesystem::path dir;
  std::map<std::string, Bytes> bytes;  // per tensor, its pack bytes

  explicit SyntheticPack(const char* name) : dir(temp_dir(name)) {
    struct Spec {
      const char* name;
      int file, kind;
      std::int64_t ne0, ne1;
      int code_bits;
      std::uint64_t codes, scales;
      int fp16;
    };
    const Spec rows[] = {
        {"blk.0.attn_qkv.weight", 0, 0, 64, 4, 2, 64, 16, 1},         // S2 planes, fp16 scales widened by the loader
        {"blk.0.hc_attn_norm.weight", 0, 2, 32, 0, 0, 0, 0, 0},        // F32
        {"blk.0.ffn_gate_inp_shexp.weight", 0, 1, 16, 0, 0, 0, 0, 0},  // BF16 promoted to F32
        {"blk.0.ffn_gate_shexp.weight", 0, 0, 64, 2, 2, 32, 8, 1},
        {"blk.0.ffn_gate_exps.weight", 2, 0, 64, 8, 2, 128, 32, 1},    // routed experts: never in a dense object
        {"blk.1.ple_key.weight", 3, 4, 16, 4, 0, 0, 0, 0},              // raw BF16 (native pack)
        {"token_embd.weight", 1, 0, 64, 8, 4, 256, 32, 1},
        {"output.weight", 3, 0, 64, 8, 4, 256, 32, 1},
    };
    std::map<int, Bytes> files;
    std::string index = "# align 256 pool 65536 tensors 8\n";
    std::uint32_t seed = 1;
    for (const Spec& s : rows) {
      const std::uint64_t elems = static_cast<std::uint64_t>(s.ne0) * static_cast<std::uint64_t>(s.ne1 > 0 ? s.ne1 : 1);
      std::uint64_t src = 0, dst = 0;
      if (s.code_bits != 0) {
        src = s.codes + (s.fp16 ? s.scales / 2 : s.scales);
        dst = s.codes + s.scales;
      } else if (s.kind == 2) {
        src = dst = elems * 4;
      } else if (s.kind == 1) {
        src = elems * 4;
        dst = elems * 2;
      } else {
        src = dst = elems * 2;
      }
      Bytes b = random_bytes(src, seed++);
      Bytes& f = files[s.file];
      const std::uint64_t off = f.size();
      f.insert(f.end(), b.begin(), b.end());
      char line[512];
      std::snprintf(line, sizeof line, "%s %d %d %llu %llu %llu %llu %lld %lld %d 0 0 0 0 %llu %llu 0 %d 0\n", s.name,
                    s.file, s.kind, static_cast<unsigned long long>(off), static_cast<unsigned long long>(src), 0ull,
                    static_cast<unsigned long long>(dst), static_cast<long long>(s.ne0), static_cast<long long>(s.ne1),
                    s.code_bits, static_cast<unsigned long long>(s.codes), static_cast<unsigned long long>(s.scales),
                    s.fp16);
      index += line;
      bytes[s.name] = std::move(b);
    }
    const char* names[] = {"dense.bin", "embd.bin", "experts.bin", "extra.bin"};
    for (auto& [id, b] : files) std::ofstream(dir / names[id], std::ios::binary).write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    std::ofstream(dir / "index.txt") << index;
  }
  ~SyntheticPack() { std::filesystem::remove_all(dir); }
};

objects::ManifestObject dense_object(std::string name, objects::ObjectKind kind) {
  objects::ManifestObject o;
  o.name = std::move(name);
  o.kind = kind;
  o.representation = {std::string(bs::kDenseQuantType), 0, bs::kDenseConversionVersion, true};
  return o;
}

}  // namespace

TEST_CASE("ggml type table matches the block geometry Strata's packs use") {
  const auto* iq3s = bs::ggml_type_by_name("iq3_s");
  const auto* iq2xs = bs::ggml_type_by_name("iq2_xs");
  const auto* iq4nl = bs::ggml_type_by_name("iq4_nl");
  REQUIRE(iq3s != nullptr);
  REQUIRE(iq2xs != nullptr);
  REQUIRE(iq4nl != nullptr);
  CHECK(iq3s->id == 21);
  CHECK(bs::ggml_bytes(*iq3s, 2560) == 10 * 110);
  CHECK(bs::ggml_bytes(*iq2xs, 2560) == 10 * 74);
  CHECK(bs::ggml_bytes(*iq4nl, 640) == 20 * 18);
  CHECK(bs::ggml_bytes(*iq3s, 640) == 0);  // 640 is not a whole number of 256-blocks
  CHECK(bs::ggml_type_by_id(42)->name == "q2_0");
  CHECK(bs::ggml_type_by_name("nope") == nullptr);
}

TEST_CASE("routed experts are the GGUF slices unchanged: [gate | up | down]") {
  const auto g = strata_test::flash_next_geometry();
  auto f = bs::expert_format({"iq3_s+iq4_nl", 256, 0, true}, g);
  REQUIRE(f.is_ok());
  CHECK(f->gate_bytes == 640ull * 1100);
  CHECK(f->up_bytes == f->gate_bytes);
  CHECK(f->down_bytes == 2560ull * 360);
  CHECK(f->total() == 2 * 704000 + 921600);
  CHECK(bs::expert_quant_type(*f->gate_up, *f->down) == "iq3_s+iq4_nl");
  CHECK(bs::expert_format({"iq3_s", 256, 0, true}, g).status().code() == ErrorCode::kInvalidArgument);  // down at ff 640
  CHECK(bs::expert_format({"iq3_s+iq4_nl", 256, 1, true}, g).status().code() == ErrorCode::kInvalidArgument);  // no transform
  CHECK(bs::expert_format({"iq9+iq4_nl", 256, 0, true}, g).status().code() == ErrorCode::kInvalidArgument);
  auto x = bs::expert_format({"iq2_xs+iq4_nl", 256, 0, true}, g);
  REQUIRE(x.is_ok());
  CHECK(x->gate_bytes == 640ull * 740);
}

TEST_CASE("tensor placement: which object owns each Strata tensor") {
  using bs::TensorHome;
  CHECK(bs::place_tensor("blk.12.attn_qkv.weight").object_name == "blk.12.dense");
  CHECK(bs::place_tensor("blk.1.ple_key.weight").object_name == "blk.1.dense");
  CHECK(bs::place_tensor("blk.7.ffn_up_shexp.weight").object_name == "blk.7.shared");
  CHECK(bs::place_tensor("blk.7.ffn_gate_inp_shexp.weight").home == TensorHome::kSharedExpert);
  CHECK(bs::place_tensor("blk.7.ffn_down_exps.weight").home == TensorHome::kRoutedExperts);
  CHECK(bs::place_tensor("blk.7.ffn_down_exps.weight").object_name.empty());
  CHECK(bs::place_tensor("token_embd.weight").object_name == "token_embd");
  CHECK(bs::place_tensor("output_hc_down.weight").object_name == "output_head");
  CHECK(bs::place_tensor("output.weight").home == TensorHome::kHead);
  CHECK(bs::place_tensor("mtp.fc.weight").home == TensorHome::kOther);
  CHECK(bs::place_tensor("blk.x.foo").home == TensorHome::kOther);
  CHECK(bs::served_natively("blk.3.attn_qkv.weight"));
  CHECK(bs::served_natively("output.weight"));
  CHECK_FALSE(bs::served_natively("blk.3.hc_attn_norm.weight"));
  CHECK_FALSE(bs::served_natively("blk.1.ple_key.weight"));
}

TEST_CASE("strata-dense container: deterministic, bounds-checked round trip") {
  bs::DenseRow a;
  a.name = "blk.3.attn_qkv.weight";
  a.ne0 = 64;
  a.ne1 = 4;
  a.code_bits = 2;
  a.codes_bytes = 64;
  a.scales_bytes = 16;
  a.scales_fp16 = 1;
  a.dst_bytes = 80;
  a.native_type = 8;  // q8_0
  a.native_ne0 = 64;
  a.native_ne1 = 4;
  bs::DenseRow b;
  b.name = "blk.3.hc_attn_norm.weight";
  b.kind = 2;
  b.ne0 = 8;
  b.dst_bytes = 32;
  const Bytes ca = random_bytes(72, 1), cb = random_bytes(32, 2), na = random_bytes(4 * 68, 3);
  auto enc = bs::encode_dense_object(256, {a, b}, {ByteSpan(ca), ByteSpan(cb)}, {ByteSpan(na), ByteSpan()});
  REQUIRE_MESSAGE(enc.is_ok(), enc.status().to_string());
  auto enc2 = bs::encode_dense_object(256, {a, b}, {ByteSpan(ca), ByteSpan(cb)}, {ByteSpan(na), ByteSpan()});
  REQUIRE(enc2.is_ok());
  CHECK(*enc == *enc2);
  auto p = bs::parse_dense_object(*enc);
  REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
  REQUIRE(p->rows.size() == 2);
  CHECK(p->align == 256);
  CHECK(p->rows[0].name == a.name);
  CHECK(p->rows[0].has_native());
  const ByteSpan pa = p->canonical(p->rows[0]), pn = p->native(p->rows[0]), pb = p->canonical(p->rows[1]);
  CHECK(Bytes(pa.begin(), pa.end()) == ca);
  CHECK(Bytes(pn.begin(), pn.end()) == na);
  CHECK(Bytes(pb.begin(), pb.end()) == cb);
  CHECK(p->rows[0].payload_offset % 64 == 0);
  CHECK(p->rows[0].native_offset % 64 == 0);

  // corruption is refused, never decoded into plausible weights
  Bytes bad = *enc;
  bad[0] ^= 1;
  CHECK(bs::parse_dense_object(bad).status().code() == ErrorCode::kDataLoss);
  Bytes cut(enc->begin(), enc->end() - 1);
  CHECK(bs::parse_dense_object(cut).status().code() == ErrorCode::kDataLoss);
  bs::DenseRow wrong = a;
  wrong.dst_bytes = 81;  // planes no longer add up to the arena bytes
  CHECK(bs::encode_dense_object(256, {wrong}, {ByteSpan(ca)}, {ByteSpan(na)}).status().code() == ErrorCode::kDataLoss);
  bs::DenseRow badnative = a;
  CHECK(bs::encode_dense_object(256, {badnative}, {ByteSpan(ca)}, {ByteSpan(na).first(100)}).status().code() ==
        ErrorCode::kDataLoss);

  // a row renders as one index.txt line of Strata's 19 fields
  const std::string line = bs::index_line(p->rows[0], 0, 4096, 512);
  char name[256];
  int file = -1, kind = -1, cbits = 0, cbias = 0, ge = 0, cbk = 0, hoff = 0, f16 = 0, act = 0;
  unsigned long long so = 0, sb = 0, dofs = 0, db = 0, codes = 0, scales = 0, offs = 0;
  long long ne0 = 0, ne1 = 0;
  CHECK(std::sscanf(line.c_str(), "%255s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d", name,
                    &file, &kind, &so, &sb, &dofs, &db, &ne0, &ne1, &cbits, &cbias, &ge, &cbk, &hoff, &codes, &scales,
                    &offs, &f16, &act) == 19);
  CHECK(so == 4096);
  CHECK(sb == 72);
  CHECK(db == 80);
  CHECK(f16 == 1);
}

TEST_CASE("Father conversion: a layer's pack rows + GGUF-form copies become its strata-dense object") {
  SyntheticPack pack("convert");
  auto reader = bs::StrataPackReader::open(pack.dir);
  REQUIRE_MESSAGE(reader.is_ok(), reader.status().to_string());
  CHECK(reader->rows().size() == 8);
  int native_calls = 0;
  const bs::NativeTensorProvider native = [&](std::string_view name) -> Result<std::optional<bs::NativeTensor>> {
    ++native_calls;
    if (name != "blk.0.attn_qkv.weight" && name != "output.weight") return std::optional<bs::NativeTensor>{};
    bs::NativeTensor t;
    t.type = 8;
    t.ne0 = 64;
    t.ne1 = name == "output.weight" ? 8 : 4;
    t.bytes = random_bytes(68 * t.ne1, 9);
    return std::optional<bs::NativeTensor>(std::move(t));
  };

  auto dense = bs::convert_object(dense_object("blk.0.dense", objects::ObjectKind::kLayerDense), *reader, native);
  REQUIRE_MESSAGE(dense.is_ok(), dense.status().to_string());
  auto parsed = bs::parse_dense_object(*dense);
  REQUIRE(parsed.is_ok());
  REQUIRE(parsed->rows.size() == 2);  // the shared-expert and routed rows are not this object's
  CHECK(parsed->rows[0].name == "blk.0.attn_qkv.weight");
  CHECK(parsed->rows[0].has_native());
  CHECK(parsed->rows[1].name == "blk.0.hc_attn_norm.weight");
  CHECK_FALSE(parsed->rows[1].has_native());
  for (const auto& r : parsed->rows) {
    const ByteSpan c = parsed->canonical(r);
    CHECK(Bytes(c.begin(), c.end()) == pack.bytes[r.name]);  // byte-exact: never requantized
  }
  CHECK(native_calls == 1);  // asked only for the tensors the engine serves natively

  auto shared = bs::convert_object(dense_object("blk.0.shared", objects::ObjectKind::kSharedExpert), *reader, native);
  REQUIRE(shared.is_ok());
  CHECK(bs::parse_dense_object(*shared)->rows.size() == 2);
  auto l1 = bs::convert_object(dense_object("blk.1.dense", objects::ObjectKind::kLayerDense), *reader, native);
  REQUIRE(l1.is_ok());
  CHECK(bs::parse_dense_object(*l1)->rows[0].kind == 4);  // the raw-BF16 PLE key keeps its pack kind
  auto head = bs::convert_object(dense_object("output_head", objects::ObjectKind::kOutputHead), *reader, native);
  REQUIRE(head.is_ok());
  CHECK(bs::parse_dense_object(*head)->rows[0].native_ne1 == 8);

  objects::ManifestObject exp;
  exp.name = "blk.0.exp.3";
  exp.kind = objects::ObjectKind::kRoutedExpert;
  CHECK(bs::convert_object(exp, *reader, native).status().code() == ErrorCode::kInvalidArgument);
  objects::ManifestObject ple = dense_object("ple_lookup", objects::ObjectKind::kPleLookup);
  CHECK(bs::convert_object(ple, *reader, native).status().code() == ErrorCode::kInvalidArgument);
  CHECK(bs::convert_object(dense_object("blk.5.dense", objects::ObjectKind::kLayerDense), *reader, native).status().code() ==
        ErrorCode::kNotFound);
  CHECK(bs::StrataPackReader::open(pack.dir / "missing").status().code() == ErrorCode::kNotFound);
}

TEST_CASE("required objects: a partial domain binds only its own layers") {
  const auto g = strata_test::flash_next_geometry();
  domain::DomainSpec mid;
  mid.role = domain::StageRole::kMiddle;
  mid.layers = {12, 24};
  const auto names = bs::required_objects(g, mid);
  CHECK(names.size() == 12 * (2 + 512));
  CHECK(names.front() == "blk.12.dense");
  CHECK(names.back() == "blk.23.exp.511");
  domain::DomainSpec tail;
  tail.role = domain::StageRole::kTail;
  tail.layers = {36, 48};
  const auto t = bs::required_objects(g, tail, {true});
  CHECK(t.front() == "token_embd");  // the MTP drafter embeds its draft tokens (Father-local)
  CHECK(t.back() == "output_head");
  CHECK(bs::required_objects(g, tail).front() == "blk.36.dense");
}
