#include <doctest/doctest.h>

#include "clusterlm/common/digest.hpp"

using namespace clusterlm;

TEST_CASE("SHA-256 known answers and incremental hashing") {
  CHECK(Sha256::of(std::string_view("")).hex() ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(Sha256::of(std::string_view("abc")).hex() ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  Sha256 h;
  h.update(std::string_view("a"));
  h.update(std::string_view("bc"));
  CHECK(h.finish() == Sha256::of(std::string_view("abc")));
  auto parsed = Digest256::from_hex(Sha256::of(std::string_view("abc")).hex());
  REQUIRE(parsed.is_ok());
  CHECK(*parsed == Sha256::of(std::string_view("abc")));
}
