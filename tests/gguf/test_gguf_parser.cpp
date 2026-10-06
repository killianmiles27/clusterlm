#include <doctest/doctest.h>

#include <cstring>
#include <fstream>
#include <limits>

#include "clusterlm/objects/gguf.hpp"
#include "clusterlm/objects/gguf_writer.hpp"
#include "../objects/test_util.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;

namespace {

// Hand-assembled little-endian bytes for malformed-file tests.
struct Raw {
  Bytes b;
  Raw& u8(std::uint8_t v) { b.push_back(v); return *this; }
  Raw& u32(std::uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i))); return *this; }
  Raw& u64(std::uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i))); return *this; }
  Raw& str(const std::string& s) { u64(s.size()); b.insert(b.end(), s.begin(), s.end()); return *this; }
  Raw& zeros(std::size_t n) { b.resize(b.size() + n, 0); return *this; }
  Raw& pad_to(std::size_t align) { b.resize((b.size() + align - 1) / align * align, 0); return *this; }
  Raw& header(std::uint32_t version, std::uint64_t n_tensors, std::uint64_t n_kv) { return u32(0x46554747u).u32(version).u64(n_tensors).u64(n_kv); }
  Raw& tensor(const std::string& name, const std::vector<std::uint64_t>& dims, std::uint32_t type, std::uint64_t off) {
    str(name).u32(static_cast<std::uint32_t>(dims.size()));
    for (auto d : dims) u64(d);
    return u32(type).u64(off);
  }
};

Result<GgufFile> parse(const Bytes& b, const GgufLimits& l = {}) { return parse_gguf_bytes(b, l); }

void expect_error(const Bytes& b, ErrorCode code, const GgufLimits& l = {}) {
  auto r = parse(b, l);
  CHECK_FALSE(r.is_ok());
  if (!r.is_ok()) CHECK_MESSAGE(r.status().code() == code, r.status().to_string());
}

// One tensor, no metadata, data region of `data_bytes` zero bytes.
Bytes one_tensor(const std::string& name, const std::vector<std::uint64_t>& dims, std::uint32_t type, std::uint64_t off, std::size_t data_bytes) {
  Raw r;
  r.header(3, 1, 0).tensor(name, dims, type, off).pad_to(32).zeros(data_bytes);
  return r.b;
}

Bytes rich_file() {
  GgufWriter w;
  w.set_alignment(64);
  w.add_string("general.architecture", "test");
  w.add_u16("u16", 65535);
  w.add_u32("u32", 4000000000u);
  w.add_u64("u64", 0x1122334455667788ull);
  w.add_i32("i32", -5);
  w.add_f32("f32", 1.5f);
  w.add_bool("flag", true);
  w.add_u32_array("arr.u32", {1, 2, 3});
  w.add_f32_array("arr.f32", {0.5f, 0.25f});
  std::vector<std::string> toks;
  for (int i = 0; i < 100; ++i) toks.push_back("tok" + std::to_string(i));
  w.add_string_array("tokens", toks);
  REQUIRE(w.add_random_tensor("a.f32", {4, 3}, GgmlType::kF32, 1).is_ok());
  REQUIRE(w.add_random_tensor("b.q8", {64, 2}, GgmlType::kQ8_0, 1).is_ok());
  REQUIRE(w.add_random_tensor("c.iq3s", {256, 2, 3}, GgmlType::kIQ3_S, 1).is_ok());
  auto bytes = w.serialize();
  REQUIRE(bytes.is_ok());
  return *bytes;
}

}  // namespace

TEST_CASE("gguf: writer/parser round trip of metadata and tensors") {
  const Bytes file = rich_file();
  auto r = parse(file);
  REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
  CHECK(r->version == 3);
  CHECK(r->alignment == 64);
  CHECK(r->data_start % 64 == 0);
  CHECK(r->data_start >= r->header_bytes);
  CHECK(r->data_start - r->header_bytes < 64);
  CHECK(r->file_size == file.size());
  CHECK(r->find("general.architecture")->s == "test");
  CHECK(r->find("u16")->as_u64() == 65535);
  CHECK(r->find("u32")->as_u64() == 4000000000u);
  CHECK(r->find("u64")->as_u64() == 0x1122334455667788ull);
  CHECK_FALSE(r->find("i32")->as_u64().has_value());
  CHECK(r->find("i32")->i == -5);
  CHECK(r->find("i32")->as_double() == -5.0);
  CHECK(r->find("f32")->f == 1.5);
  CHECK(r->find("flag")->u == 1);
  CHECK_FALSE(r->find("flag")->is_integer());
  const GgufValue* arr = r->find("arr.u32");
  REQUIRE(arr != nullptr);
  CHECK(arr->is_array());
  CHECK(arr->count == 3);
  CHECK(arr->elem_type == GgufValueType::kU32);
  CHECK(arr->items.size() == 3);
  CHECK(arr->items[2].u == 3);
  CHECK(r->find("arr.f32")->items[1].f == 0.25);
  const GgufValue* toks = r->find("tokens");
  CHECK(toks->count == 100);
  CHECK(toks->items.size() == GgufValue::kRetainedArrayItems);
  CHECK(toks->items[63].s == "tok63");
  CHECK(r->find("missing") == nullptr);

  REQUIRE(r->tensors.size() == 3);
  const GgufTensorInfo* a = r->find_tensor("a.f32");
  const GgufTensorInfo* b = r->find_tensor("b.q8");
  const GgufTensorInfo* c = r->find_tensor("c.iq3s");
  REQUIRE((a && b && c));
  CHECK(a->n_bytes == 48);
  CHECK(a->n_elements == 12);
  CHECK(b->n_bytes == 2 * 2 * 34);
  CHECK(c->n_bytes == 3 * 2 * 110);
  CHECK(c->n_dims == 3);
  CHECK(c->dims[0] == 256);
  CHECK(c->row_bytes() == 110);
  for (const auto* t : {a, b, c}) {
    CHECK(t->offset % 64 == 0);
    CHECK(t->file_offset == r->data_start + t->offset);
    CHECK(t->file_offset + t->n_bytes <= file.size());
  }
  CHECK(a->offset == 0);
  CHECK(r->find_tensor("nope") == nullptr);

  // Payload bytes are exactly what the writer was given.
  const Bytes expect = pseudo_random_bytes(48, [] {
    std::uint64_t h = 1;
    for (char ch : std::string("a.f32")) h = (h ^ static_cast<std::uint8_t>(ch)) * 0x100000001B3ull;
    return h;
  }());
  CHECK(std::memcmp(file.data() + a->file_offset, expect.data(), 48) == 0);
}

TEST_CASE("gguf: version 2 is accepted; version 1, unknown and big-endian files are rejected") {
  GgufWriter w2(2);
  w2.add_string("k", "v");
  REQUIRE(w2.add_random_tensor("t", {32}, GgmlType::kF32, 1).is_ok());
  auto f2 = w2.serialize();
  REQUIRE(f2.is_ok());
  auto r = parse(*f2);
  REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
  CHECK(r->version == 2);

  for (std::uint32_t v : {0u, 1u, 4u, 99u, 0x03000000u, 0xFFFFFFFFu}) {
    Raw raw;
    raw.header(v, 0, 0).pad_to(32);
    expect_error(raw.b, ErrorCode::kVersionMismatch);
  }
  Bytes bad_magic = *f2;
  bad_magic[0] = 'X';
  expect_error(bad_magic, ErrorCode::kInvalidArgument);
  expect_error({}, ErrorCode::kOutOfRange);
  expect_error({'G', 'G', 'U'}, ErrorCode::kOutOfRange);
}

TEST_CASE("gguf: every truncation of a valid file is rejected, never read out of bounds") {
  const Bytes file = rich_file();
  REQUIRE(parse(file).is_ok());
  for (std::size_t cut = 0; cut < file.size(); ++cut) {
    auto r = parse_gguf_bytes(ByteSpan(file).subspan(0, cut));
    REQUIRE_MESSAGE(!r.is_ok(), "cut " << cut << " of " << file.size() << " parsed");
  }
}

TEST_CASE("gguf: single-byte corruption anywhere in the header never crashes or over-reads") {
  const Bytes file = rich_file();
  auto good = parse(file);
  REQUIRE(good.is_ok());
  std::size_t accepted = 0;
  for (std::size_t i = 0; i < good->header_bytes; ++i)
    for (std::uint8_t v : {std::uint8_t{0x00}, std::uint8_t{0xFF}, std::uint8_t{0x80}, std::uint8_t{0x41}}) {
      Bytes m = file;
      m[i] = v;
      auto r = parse(m);
      if (r.is_ok()) {
        ++accepted;
        for (const auto& t : r->tensors) CHECK(t.file_offset + t.n_bytes <= m.size());
      }
    }
  CHECK(accepted < good->header_bytes * 4);  // most corruptions are caught
}

TEST_CASE("gguf: count and length limits are enforced before allocation") {
  // 2^40 tensors / kv entries: rejected on the count alone, from a 24-byte file.
  Raw a;
  a.header(3, 1ull << 40, 0);
  expect_error(a.b, ErrorCode::kResourceExhausted);
  Raw b;
  b.header(3, 0, 1ull << 40);
  expect_error(b.b, ErrorCode::kResourceExhausted);
  // Within the limit but more than the file can hold.
  Raw c;
  c.header(3, 1000000, 0);
  expect_error(c.b, ErrorCode::kOutOfRange);
  Raw c2;
  c2.header(3, 0, 1000000);
  expect_error(c2.b, ErrorCode::kOutOfRange);

  GgufLimits small;
  small.max_tensors = 1;
  GgufWriter w;
  REQUIRE(w.add_random_tensor("a", {32}, GgmlType::kF32, 1).is_ok());
  REQUIRE(w.add_random_tensor("b", {32}, GgmlType::kF32, 1).is_ok());
  auto two = w.serialize();
  REQUIRE(two.is_ok());
  expect_error(*two, ErrorCode::kResourceExhausted, small);
  CHECK(parse(*two).is_ok());

  // Strings: 1 GB declared in a tiny file; and a string just over the limit.
  Raw s;
  s.header(3, 0, 1).u64(1ull << 30);
  expect_error(s.b, ErrorCode::kResourceExhausted);
  Raw s2;
  s2.header(3, 0, 1).u64(65537).zeros(70000);
  expect_error(s2.b, ErrorCode::kResourceExhausted);
  Raw s3;  // exactly at the limit, but past EOF
  s3.header(3, 0, 1).u64(65536).zeros(100);
  expect_error(s3.b, ErrorCode::kOutOfRange);
  Raw s4;  // a valid long key at the limit
  s4.header(3, 0, 1).str(std::string(65536, 'k')).u32(4).u32(1).pad_to(32);
  CHECK(parse(s4.b).is_ok());

  // Arrays: element count over the limit; count that cannot fit in the file; element-by-element string arrays.
  Raw ar;
  ar.header(3, 0, 1).str("a").u32(9).u32(4).u64(16u << 20).u64(0);
  CHECK(parse(ar.b).status().code() == ErrorCode::kOutOfRange);  // exactly at the limit: allowed, but no bytes
  Raw ar2;
  ar2.header(3, 0, 1).str("a").u32(9).u32(4).u64((16u << 20) + 1);
  expect_error(ar2.b, ErrorCode::kResourceExhausted);
  Raw ar3;
  ar3.header(3, 0, 1).str("a").u32(9).u32(10).u64(std::numeric_limits<std::uint64_t>::max());
  expect_error(ar3.b, ErrorCode::kResourceExhausted);
  GgufLimits few;
  few.max_array_elements = 2;
  Raw ar4;
  ar4.header(3, 0, 1).str("a").u32(9).u32(4).u64(3).u32(1).u32(2).u32(3).pad_to(32);
  expect_error(ar4.b, ErrorCode::kResourceExhausted, few);
  CHECK(parse(ar4.b).is_ok());
  Raw ar5;  // 16M string elements declared, 4 bytes of payload
  ar5.header(3, 0, 1).str("a").u32(9).u32(8).u64(16u << 20).u64(0);
  CHECK_FALSE(parse(ar5.b).is_ok());

  // Total header size.
  GgufLimits hdr;
  hdr.max_header_bytes = 40;
  GgufWriter big;
  big.add_string("some.long.key.name", "a value that makes the header exceed the tiny limit");
  auto bf = big.serialize();
  REQUIRE(bf.is_ok());
  expect_error(*bf, ErrorCode::kResourceExhausted, hdr);
  CHECK(parse(*bf).is_ok());
}

TEST_CASE("gguf: malformed metadata values are rejected") {
  Raw nested;
  nested.header(3, 0, 1).str("a").u32(9).u32(9).u64(1).u32(4).u64(0).pad_to(32);
  expect_error(nested.b, ErrorCode::kInvalidArgument);
  Raw bad_type;
  bad_type.header(3, 0, 1).str("a").u32(99).u32(0).pad_to(32);
  expect_error(bad_type.b, ErrorCode::kInvalidArgument);
  Raw bad_elem;
  bad_elem.header(3, 0, 1).str("a").u32(9).u32(77).u64(1).u32(0).pad_to(32);
  expect_error(bad_elem.b, ErrorCode::kInvalidArgument);
  Raw bad_bool;
  bad_bool.header(3, 0, 1).str("a").u32(7).u8(2).pad_to(32);
  expect_error(bad_bool.b, ErrorCode::kInvalidArgument);
  Raw dup;
  dup.header(3, 0, 2).str("a").u32(4).u32(1).str("a").u32(4).u32(2).pad_to(32);
  expect_error(dup.b, ErrorCode::kInvalidArgument);
  Raw empty_key;
  empty_key.header(3, 0, 1).str("").u32(4).u32(1).pad_to(32);
  expect_error(empty_key.b, ErrorCode::kInvalidArgument);
}

TEST_CASE("gguf: alignment key is validated and drives the data start") {
  for (std::uint32_t a : {0u, 3u, 48u, 100u}) {
    Raw r;
    r.header(3, 0, 1).str("general.alignment").u32(4).u32(a).pad_to(64);
    expect_error(r.b, ErrorCode::kInvalidArgument);
  }
  Raw str;
  str.header(3, 0, 1).str("general.alignment").u32(8).str("32").pad_to(64);
  expect_error(str.b, ErrorCode::kInvalidArgument);
  Raw huge;
  huge.header(3, 0, 1).str("general.alignment").u32(10).u64(1ull << 40).pad_to(64);
  expect_error(huge.b, ErrorCode::kResourceExhausted);
  for (std::uint32_t a : {1u, 8u, 16u, 64u, 4096u}) {
    Raw r;
    r.header(3, 1, 1).str("general.alignment").u32(4).u32(a).tensor("t", {8}, 0, 0);
    const std::size_t header = r.b.size();
    r.pad_to(a).zeros(32);
    auto p = parse(r.b);
    REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
    CHECK(p->data_start == (header + a - 1) / a * a);
  }
  // File ends inside the padding: data section would start past EOF.
  Raw eof;
  eof.header(3, 0, 1).str("k").u32(4).u32(1);
  expect_error(eof.b, ErrorCode::kOutOfRange);
}

TEST_CASE("gguf: tensor directory validation") {
  CHECK(parse(one_tensor("t", {8}, 0, 0, 32)).is_ok());
  CHECK(parse(one_tensor("t", {8}, 0, 0, 40)).is_ok());                       // trailing bytes are fine
  expect_error(one_tensor("t", {8}, 0, 0, 31), ErrorCode::kOutOfRange);       // one byte short
  expect_error(one_tensor("t", {8}, 0, 4, 64), ErrorCode::kInvalidArgument);  // offset not aligned
  expect_error(one_tensor("t", {8}, 0, 32, 32), ErrorCode::kOutOfRange);      // starts at the end
  expect_error(one_tensor("t", {8}, 0, 0xFFFFFFFFFFFFFFE0ull, 64), ErrorCode::kOutOfRange);
  expect_error(one_tensor("t", {8}, 0, 0xFFFFFFFFFFFFFFFFull, 64), ErrorCode::kInvalidArgument);
  expect_error(one_tensor("", {8}, 0, 0, 32), ErrorCode::kInvalidArgument);
  expect_error(one_tensor("t", {}, 0, 0, 32), ErrorCode::kInvalidArgument);                    // n_dims 0
  expect_error(one_tensor("t", {1, 1, 1, 1, 1}, 0, 0, 32), ErrorCode::kInvalidArgument);       // n_dims 5
  expect_error(one_tensor("t", {0}, 0, 0, 32), ErrorCode::kInvalidArgument);                   // zero dim
  expect_error(one_tensor("t", {4, 0}, 0, 0, 32), ErrorCode::kInvalidArgument);
  expect_error(one_tensor("t", {1ull << 63}, 0, 0, 32), ErrorCode::kInvalidArgument);          // > INT64_MAX
  expect_error(one_tensor("t", {1ull << 40, 1ull << 40}, 0, 0, 32), ErrorCode::kOutOfRange);   // element overflow
  expect_error(one_tensor("t", {1ull << 62, 3}, 0, 0, 32), ErrorCode::kOutOfRange);            // bytes overflow
  expect_error(one_tensor("t", {1ull << 40, 1ull << 20}, 0, 0, 32), ErrorCode::kOutOfRange);   // bigger than the file
  expect_error(one_tensor("t", {100}, 21, 0, 4096), ErrorCode::kInvalidArgument);              // iq3_s row not whole blocks
  expect_error(one_tensor("t", {33}, 8, 0, 4096), ErrorCode::kInvalidArgument);                // q8_0
  for (std::uint32_t bad : {4u, 5u, 31u, 36u, 43u, 999u, 0xFFFFFFFFu})
    expect_error(one_tensor("t", {32}, bad, 0, 4096), ErrorCode::kInvalidArgument);
  CHECK(parse(one_tensor("t", {256}, 21, 0, 110)).is_ok());  // exactly one iq3_s block
  expect_error(one_tensor("t", {256}, 21, 0, 109), ErrorCode::kOutOfRange);

  Raw dup;
  dup.header(3, 2, 0).tensor("t", {8}, 0, 0).tensor("t", {8}, 0, 32).pad_to(32).zeros(64);
  expect_error(dup.b, ErrorCode::kInvalidArgument);
  Raw overlap;
  overlap.header(3, 2, 0).tensor("a", {16}, 0, 0).tensor("b", {8}, 0, 32).pad_to(32).zeros(96);
  expect_error(overlap.b, ErrorCode::kInvalidArgument);
  Raw same_offset;
  same_offset.header(3, 2, 0).tensor("a", {8}, 0, 0).tensor("b", {8}, 0, 0).pad_to(32).zeros(64);
  expect_error(same_offset.b, ErrorCode::kInvalidArgument);
  Raw ok_order;  // directory order need not match offset order
  ok_order.header(3, 2, 0).tensor("b", {8}, 0, 32).tensor("a", {8}, 0, 0).pad_to(32).zeros(64);
  CHECK(parse(ok_order.b).is_ok());
}

namespace {
// Records the extent of every read so a test can prove tensor data is never consumed.
class CountingSource final : public GgufSource {
 public:
  explicit CountingSource(const Bytes& b) : b_(b) {}
  std::uint64_t size() const override { return b_.size(); }
  Status read(std::uint64_t offset, void* dst, std::size_t n) override {
    if (offset + n > b_.size()) return make_error(ErrorCode::kOutOfRange, "oob");
    std::memcpy(dst, b_.data() + offset, n);
    max_end = std::max(max_end, offset + n);
    total += n;
    return Status::ok();
  }
  std::uint64_t max_end = 0, total = 0;

 private:
  const Bytes& b_;
};
}  // namespace

TEST_CASE("gguf: only the header region of a large file is read") {
  GgufWriter w;
  w.add_string("general.architecture", "test");
  REQUIRE(w.add_random_tensor("big.0", {1024, 1024}, GgmlType::kF32, 1).is_ok());  // 4 MiB
  REQUIRE(w.add_random_tensor("big.1", {1024, 1024}, GgmlType::kF32, 2).is_ok());  // 4 MiB
  auto file = w.serialize();
  REQUIRE(file.is_ok());
  CHECK(file->size() > 8u << 20);
  CountingSource src(*file);
  auto r = parse_gguf(src);
  REQUIRE(r.is_ok());
  // The header is a few hundred bytes; the reader fetches one bounded chunk from the head and nothing more.
  CHECK(src.max_end <= (1u << 20));
  CHECK(src.total <= (1u << 20));
  CHECK(src.total < file->size() / 8);
}

TEST_CASE("gguf: file reader matches the in-memory parser and reports missing files") {
  testutil::TempDir dir("gguf-parse");
  const Bytes file = rich_file();
  const auto path = dir.path() / "rich.gguf";
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
  }
  auto a = read_gguf_file(path);
  auto b = parse(file);
  REQUIRE(a.is_ok());
  REQUIRE(b.is_ok());
  CHECK(a->data_start == b->data_start);
  CHECK(a->tensors.size() == b->tensors.size());
  CHECK(a->metadata.size() == b->metadata.size());
  CHECK(read_gguf_file(dir.path() / "absent.gguf").status().code() == ErrorCode::kNotFound);
  // A file that is truncated on disk is rejected the same way.
  const auto cut_path = dir.path() / "cut.gguf";
  {
    std::ofstream f(cut_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size() - 5));
  }
  CHECK(read_gguf_file(cut_path).status().code() == ErrorCode::kOutOfRange);
}

// ---- split GGUF ---------------------------------------------------------------------------------------------------

namespace {
struct SplitSet {
  testutil::TempDir dir{"gguf-split"};
  std::vector<std::filesystem::path> paths;
};

void shard_writer(GgufWriter& w, int no, int count, int total, bool arch, const std::string& tensor) {
  if (arch) w.add_string("general.architecture", "test");
  if (count > 0) {
    w.add_u16("split.no", static_cast<std::uint16_t>(no));
    w.add_u16("split.count", static_cast<std::uint16_t>(count));
    w.add_i32("split.tensors.count", total);
  }
  REQUIRE(w.add_random_tensor(tensor, {32}, GgmlType::kF32, 3).is_ok());
}

std::filesystem::path write(const std::filesystem::path& dir, const std::string& name, const GgufWriter& w) {
  const auto p = dir / name;
  REQUIRE(w.write(p).is_ok());
  return p;
}
}  // namespace

TEST_CASE("gguf split: shards are ordered by split.no and cross-validated") {
  SplitSet s;
  GgufWriter w0, w1, w2;
  shard_writer(w0, 0, 3, 3, true, "t0");
  shard_writer(w1, 1, 3, 3, false, "t1");
  shard_writer(w2, 2, 3, 3, false, "t2");
  const auto p0 = write(s.dir.path(), "m-00001-of-00003.gguf", w0), p1 = write(s.dir.path(), "m-00002-of-00003.gguf", w1),
             p2 = write(s.dir.path(), "m-00003-of-00003.gguf", w2);

  auto ok = open_gguf_model({p2, p0, p1});  // arguments out of order
  REQUIRE_MESSAGE(ok.is_ok(), ok.status().to_string());
  CHECK(ok->is_split);
  REQUIRE(ok->shards.size() == 3);
  CHECK(ok->paths == std::vector<std::filesystem::path>{p0, p1, p2});
  CHECK(ok->meta().find("general.architecture") != nullptr);
  CHECK(ok->tensor_count() == 3);
  CHECK(ok->find_tensor("t2")->shard == 2);
  CHECK_FALSE(ok->find_tensor("t9").has_value());

  CHECK(open_gguf_model({p0, p1}).status().code() == ErrorCode::kFailedPrecondition);  // a shard is missing
  CHECK(open_gguf_model({p0}).status().code() == ErrorCode::kFailedPrecondition);      // one shard opened as a whole model
  CHECK(open_gguf_model({p0, p0, p1}).status().code() == ErrorCode::kInvalidArgument); // split.no repeated

  // split.tensors.count that does not add up.
  GgufWriter b0, b1;
  shard_writer(b0, 0, 2, 5, true, "t0");
  shard_writer(b1, 1, 2, 5, false, "t1");
  CHECK(open_gguf_model({write(s.dir.path(), "bad-00001-of-00002.gguf", b0), write(s.dir.path(), "bad-00002-of-00002.gguf", b1)}).status().code() ==
        ErrorCode::kDataLoss);

  // Shards disagreeing on split.tensors.count.
  GgufWriter c0, c1;
  shard_writer(c0, 0, 2, 2, true, "t0");
  shard_writer(c1, 1, 2, 3, false, "t1");
  CHECK(open_gguf_model({write(s.dir.path(), "dis-00001-of-00002.gguf", c0), write(s.dir.path(), "dis-00002-of-00002.gguf", c1)}).status().code() ==
        ErrorCode::kInvalidArgument);

  // The same tensor in two shards.
  GgufWriter d0, d1;
  shard_writer(d0, 0, 2, 2, true, "same");
  shard_writer(d1, 1, 2, 2, false, "same");
  CHECK(open_gguf_model({write(s.dir.path(), "dup-00001-of-00002.gguf", d0), write(s.dir.path(), "dup-00002-of-00002.gguf", d1)}).status().code() ==
        ErrorCode::kAlreadyExists);

  // First shard (split.no 0) lacking the architecture.
  GgufWriter e0, e1;
  shard_writer(e0, 0, 2, 2, false, "t0");
  shard_writer(e1, 1, 2, 2, false, "t1");
  CHECK(open_gguf_model({write(s.dir.path(), "noarch-00001-of-00002.gguf", e0), write(s.dir.path(), "noarch-00002-of-00002.gguf", e1)}).status().code() ==
        ErrorCode::kInvalidArgument);

  // Some files with split keys and some without.
  GgufWriter f1;
  shard_writer(f1, 0, 0, 0, true, "solo");
  CHECK(open_gguf_model({p0, write(s.dir.path(), "plain.gguf", f1)}).status().code() == ErrorCode::kInvalidArgument);
}

TEST_CASE("gguf split: companion files (no split keys) merge by tensor name") {
  SplitSet s;
  GgufWriter a, b, c;
  shard_writer(a, 0, 0, 0, true, "main.t");
  shard_writer(b, 0, 0, 0, true, "mtp.t");
  shard_writer(c, 0, 0, 0, true, "main.t");
  const auto pa = write(s.dir.path(), "a.gguf", a), pb = write(s.dir.path(), "b.gguf", b), pc = write(s.dir.path(), "c.gguf", c);
  auto ok = open_gguf_model({pa, pb});
  REQUIRE(ok.is_ok());
  CHECK_FALSE(ok->is_split);
  CHECK(ok->find_tensor("mtp.t")->shard == 1);
  CHECK(open_gguf_model({pa, pc}).status().code() == ErrorCode::kAlreadyExists);
  CHECK(open_gguf_model({}).status().code() == ErrorCode::kInvalidArgument);
  CHECK(open_gguf_model({s.dir.path() / "absent.gguf"}).status().code() == ErrorCode::kNotFound);
}

TEST_CASE("gguf split: shard path expansion") {
  SplitSet s;
  std::vector<std::filesystem::path> want;
  for (int i = 1; i <= 3; ++i) {
    char name[64];
    std::snprintf(name, sizeof name, "model-Q-%05d-of-00003.gguf", i);
    want.push_back(s.dir.path() / name);
    std::ofstream(want.back()).put('x');
  }
  for (const auto& p : want) {
    auto r = expand_split_paths(p);
    REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
    CHECK(*r == want);
  }
  std::filesystem::remove(want[1]);
  CHECK(expand_split_paths(want[0]).status().code() == ErrorCode::kNotFound);
  const auto plain = s.dir.path() / "model.gguf";
  CHECK(*expand_split_paths(plain) == std::vector<std::filesystem::path>{plain});
  CHECK(*expand_split_paths(s.dir.path() / "x-0000a-of-00003.gguf") == std::vector<std::filesystem::path>{s.dir.path() / "x-0000a-of-00003.gguf"});
  CHECK_FALSE(expand_split_paths(s.dir.path() / "x-00001-of-00000.gguf").is_ok());
}
