#include <doctest/doctest.h>

#include "clusterlm/father/tokenizer.hpp"

using namespace clusterlm;

TEST_CASE("fixture byte tokenizer is deterministic and round-trips") {
  auto t = father::FixtureByteTokenizer::create(256);
  REQUIRE(t.is_ok());
  const std::string text = "hello \xC3\xA9 world";
  auto ids = t.value()->encode(text);
  CHECK(ids.size() == text.size());
  for (auto id : ids) {
    CHECK(id >= 0);
    CHECK(id < 256);
  }
  CHECK(t.value()->decode(ids) == text);
  CHECK(t.value()->encode(text) == ids);
  CHECK(t.value()->decode(std::vector<std::int32_t>{300, -1}) == "??");
  CHECK_FALSE(father::FixtureByteTokenizer::create(100).is_ok());
}

TEST_CASE("chat template is role-tagged and ends with the assistant opener") {
  auto t = father::FixtureByteTokenizer::create(256);
  REQUIRE(t.is_ok());
  std::vector<father::ChatMessage> msgs{{father::ChatRole::kSystem, "s"}, {father::ChatRole::kUser, "u"}};
  auto ids = t.value()->encode_chat(msgs);
  CHECK(t.value()->decode(ids) == "system: s\nuser: u\nassistant: ");
}
