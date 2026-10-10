#include <doctest/doctest.h>

#include "clusterlm/scheduler/fair_queue.hpp"

using namespace clusterlm::scheduler;

TEST_CASE("round robin between clients, FIFO within a client") {
  FairQueue q;
  for (TicketId i = 1; i <= 8; ++i) q.push({i, "A", Priority::kApi});
  q.push({9, "B", Priority::kApi});
  q.push({10, "C", Priority::kApi});
  CHECK(q.order({}) == std::vector<TicketId>{1, 9, 10, 2, 3, 4, 5, 6, 7, 8});
  CHECK(q.depth() == 10);
  CHECK(q.client_depth("A") == 8);
}

TEST_CASE("served() advances the cursor, erase() of a middle ticket keeps order") {
  FairQueue q;
  q.push({1, "A", Priority::kApi});
  q.push({2, "A", Priority::kApi});
  q.push({3, "B", Priority::kApi});
  CHECK(*q.head({}) == 1);
  q.served("A", Priority::kApi);
  q.erase(1);
  CHECK(*q.head({}) == 3);       // B's turn
  q.served("B", Priority::kApi);
  q.erase(3);
  CHECK(*q.head({}) == 2);       // wraps back to A
  CHECK(q.erase(2));
  CHECK_FALSE(q.erase(2));
  CHECK_FALSE(q.head({}).has_value());
}

TEST_CASE("interactive class goes first; demoted clients go last but are not starved when alone") {
  FairQueue q;
  q.push({1, "A", Priority::kApi});
  q.push({2, "B", Priority::kApi});
  q.push({3, "UI", Priority::kInteractive});
  CHECK(q.order({}) == std::vector<TicketId>{3, 1, 2});
  auto demote_a = [](const std::string& c) { return c == "A"; };
  CHECK(q.order(demote_a) == std::vector<TicketId>{3, 2, 1});
  FairQueue only_a;
  only_a.push({1, "A", Priority::kApi});
  CHECK(*only_a.head(demote_a) == 1);
}

TEST_CASE("a re-pushed ticket (T17) returns to its original position") {
  FairQueue q;
  q.push({1, "A", Priority::kApi});
  q.push({2, "A", Priority::kApi});
  q.served("A", Priority::kApi);
  q.erase(1);
  q.push({1, "A", Priority::kApi});
  CHECK(*q.head({}) == 1);
}
