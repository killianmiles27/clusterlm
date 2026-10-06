#include <doctest/doctest.h>

#include <cstring>

#include "../lease_store/test_support.hpp"
#include "clusterlm/platform/durable_file.hpp"
#include "clusterlm/platform/mapped_file.hpp"

using namespace clusterlm;
using namespace clusterlm::platform;
namespace fs = std::filesystem;

TEST_CASE("MappedFile create, write, remap read-only") {
  testing::TempDir t;
  const fs::path p = t.path() / "f.bin";
  const Bytes data = testing::pattern_bytes(10000);
  {
    auto mf = MappedFile::create(p, data.size());
    REQUIRE(mf.is_ok());
    CHECK(mf->is_mapped());
    REQUIRE(mf->writable_span().size() == data.size());
    std::memcpy(mf->writable_span().data(), data.data(), data.size());
    CHECK(mf->flush().is_ok());
    CHECK(mf->close().is_ok());
    CHECK_FALSE(mf->is_open());
    CHECK(mf->close().is_ok());  // idempotent
  }
  auto ro = MappedFile::open(p, MapMode::kReadOnly);
  REQUIRE(ro.is_ok());
  CHECK(ro->writable_span().empty());
  REQUIRE(ro->span().size() == data.size());
  CHECK(std::memcmp(ro->span().data(), data.data(), data.size()) == 0);
  CHECK(ro->unmap().is_ok());
  CHECK(ro->span().empty());
  CHECK(ro->close().is_ok());
}

TEST_CASE("MappedFile create is exclusive and handles zero length") {
  testing::TempDir t;
  const fs::path p = t.path() / "z.bin";
  auto a = MappedFile::create(p, 0);
  REQUIRE(a.is_ok());
  CHECK(a->span().empty());
  auto b = MappedFile::create(p, 16);
  REQUIRE_FALSE(b.is_ok());
  CHECK(b.status().code() == ErrorCode::kAlreadyExists);
  CHECK_FALSE(MappedFile::open(t.path() / "nope", MapMode::kReadOnly).is_ok());
}

TEST_CASE("MappedFile moves transfer ownership") {
  testing::TempDir t;
  auto a = MappedFile::create(t.path() / "m.bin", 64);
  REQUIRE(a.is_ok());
  MappedFile b = std::move(*a);
  CHECK(b.is_mapped());
  CHECK_FALSE(a->is_open());
}

TEST_CASE("DurableAppendFile appends and tolerates reopen; write_file_atomic replaces") {
  testing::TempDir t;
  const fs::path p = t.path() / "j.log";
  {
    auto f = DurableAppendFile::open(p);
    REQUIRE(f.is_ok());
    CHECK(f->append(std::string_view("one\n")).is_ok());
    CHECK(f->close().is_ok());
  }
  {
    auto f = DurableAppendFile::open(p);
    REQUIRE(f.is_ok());
    CHECK(f->size() == 4);
    CHECK(f->append(std::string_view("two\n")).is_ok());
  }
  auto r = read_file_bytes(p);
  REQUIRE(r.is_ok());
  CHECK(std::string(r->begin(), r->end()) == "one\ntwo\n");

  const std::string repl = "new\n";
  CHECK(write_file_atomic(p, ByteSpan(reinterpret_cast<const std::uint8_t*>(repl.data()), repl.size())).is_ok());
  auto r2 = read_file_bytes(p);
  REQUIRE(r2.is_ok());
  CHECK(std::string(r2->begin(), r2->end()) == repl);
  CHECK_FALSE(fs::exists(fs::path(p.string() + ".tmp")));
  CHECK(read_file_bytes(t.path() / "absent").status().code() == ErrorCode::kNotFound);
}
