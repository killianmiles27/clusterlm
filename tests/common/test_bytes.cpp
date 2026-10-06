#include <doctest/doctest.h>

#include "clusterlm/common/bytes.hpp"

using namespace clusterlm;

TEST_CASE("ByteWriter/ByteReader round-trip explicit little-endian fields") {
  ByteWriter w;
  w.u8(7);
  w.u16(0xBEEF);
  w.u32(0xDEADBEEF);
  w.u64(0x0123456789ABCDEFull);
  w.f32(-1.5f);
  w.str("stage");
  const float values[] = {1.0f, 2.5f, -0.0f};
  w.f32_array(values);
  Bytes b = std::move(w).take();
  // Little-endian on the wire regardless of host.
  CHECK(b[1] == 0xEF);
  CHECK(b[2] == 0xBE);

  ByteReader r(b);
  std::uint8_t a;
  std::uint16_t c;
  std::uint32_t d;
  std::uint64_t e;
  float f;
  std::string s;
  std::vector<float> arr;
  CHECK(r.u8(a));
  CHECK(r.u16(c));
  CHECK(r.u32(d));
  CHECK(r.u64(e));
  CHECK(r.f32(f));
  CHECK(r.str(s, 64));
  CHECK(r.f32_array(arr, 16));
  CHECK(r.finish("test").is_ok());
  CHECK(a == 7);
  CHECK(c == 0xBEEF);
  CHECK(d == 0xDEADBEEF);
  CHECK(e == 0x0123456789ABCDEFull);
  CHECK(f == -1.5f);
  CHECK(s == "stage");
  REQUIRE(arr.size() == 3);
  CHECK(arr[1] == 2.5f);
}

TEST_CASE("ByteReader rejects truncation, oversize and trailing bytes") {
  ByteWriter w;
  w.str("hello");
  Bytes b = std::move(w).take();
  {
    ByteReader r(ByteSpan(b).first(b.size() - 1));
    std::string s;
    CHECK_FALSE(r.str(s, 64));
    CHECK(r.finish("t").code() == ErrorCode::kProtocolError);
  }
  {
    ByteReader r(b);
    std::string s;
    CHECK_FALSE(r.str(s, 4));  // exceeds declared maximum
  }
  {
    b.push_back(0);
    ByteReader r(b);
    std::string s;
    CHECK(r.str(s, 64));
    CHECK_FALSE(r.finish("t").is_ok());
  }
}

TEST_CASE("hex round trip") {
  Bytes b = {0x00, 0xAB, 0xFF};
  CHECK(to_hex(b) == "00abff");
  auto back = from_hex("00abff");
  REQUIRE(back.is_ok());
  CHECK(*back == b);
  CHECK_FALSE(from_hex("0g").is_ok());
}
