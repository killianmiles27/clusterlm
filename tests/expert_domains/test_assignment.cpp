#include <doctest/doctest.h>

#include "clusterlm/expert_domains/assignment.hpp"

using namespace clusterlm;
using namespace clusterlm::expert_domains;

TEST_CASE("range assignment tiles the experts exactly and proportionally") {
  auto a = ExpertAssignment::ranges(512, {1, 1, 1});
  REQUIRE(a.is_ok());
  CHECK(a->validate().is_ok());
  std::size_t total = 0;
  for (const auto& o : a->owned) {
    total += o.size();
    CHECK(o.size() >= 170);
    CHECK(o.size() <= 171);
    for (std::size_t i = 1; i < o.size(); ++i) CHECK(o[i] == o[i - 1] + 1);  // contiguous
  }
  CHECK(total == 512);
  // Owner order == ascending expert ranges, so owner order == ascending expert id order.
  CHECK(a->owned[0].front() == 0);
  CHECK(a->owned[1].front() == a->owned[0].back() + 1);
  CHECK(a->owned[2].back() == 511);

  auto skew = ExpertAssignment::ranges(32, {1, 3});
  REQUIRE(skew.is_ok());
  CHECK(skew->owned[0].size() == 8);
  CHECK(skew->owned[1].size() == 24);
}

TEST_CASE("local indices map back to the owner's list") {
  auto a = ExpertAssignment::strided(10, 3);
  REQUIRE(a.is_ok());
  for (std::uint32_t e = 0; e < 10; ++e) {
    CHECK(a->owner_of[e] == e % 3);
    CHECK(a->owned[a->owner_of[e]][a->local_index[e]] == e);
  }
}

TEST_CASE("invalid assignments are rejected") {
  CHECK_FALSE(ExpertAssignment::ranges(2, {1, 1, 1}).is_ok());  // fewer experts than owners
  CHECK_FALSE(ExpertAssignment::ranges(8, {1, 0}).is_ok());
  CHECK_FALSE(ExpertAssignment::ranges(8, {}).is_ok());
  CHECK_FALSE(ExpertAssignment::strided(8, 0).is_ok());
  CHECK_FALSE(ExpertAssignment::from_owners({0, 1, 5}, 2).is_ok());  // owner out of range
  auto ok = ExpertAssignment::from_owners({1, 1, 1, 1}, 2);          // Father may own nothing
  REQUIRE(ok.is_ok());
  CHECK(ok->owned[0].empty());
}
