#include <thread>
#include <vector>

#include "clusterlm/transport/impairment.hpp"
#include "test_helpers.hpp"

using namespace clm_test;

namespace {

using Clock = std::chrono::steady_clock;

NetworkConditions cond(double bw, double latency_ms, double jitter_ms, std::uint64_t seed = 1) {
  NetworkConditions c;
  c.name = "test";
  c.bandwidth_bytes_per_s = bw;
  c.latency_ms = latency_ms;
  c.jitter_ms = jitter_ms;
  c.seed = seed;
  return c;
}

}  // namespace

TEST_CASE("network presets") {
  const auto names = network_preset_names();
  CHECK(names.size() == 5);
  for (const auto& n : names) CHECK(network_preset(n).is_ok());
  auto g = network_preset("gige-simulated");
  REQUIRE(g.is_ok());
  CHECK(g->bandwidth_bytes_per_s == 110e6);
  CHECK(g->latency_ms == 0.15);
  CHECK(g->jitter_ms == 0.05);
  CHECK(network_preset("unlimited")->bandwidth_bytes_per_s == 0);
  CHECK(network_preset("slow-link")->bandwidth_bytes_per_s == 20e6);
  CHECK(network_preset("gige-degraded")->latency_ms == 0.4);
  CHECK(network_preset("gige-high-latency")->latency_ms == 2.0);
  CHECK(network_preset("nope").status().code() == ErrorCode::kNotFound);
}

TEST_CASE("impairment: bandwidth serialization delay") {
  Pair p = insecure_pair();
  auto tx = impair(std::move(p.client), cond(20e6, 0, 0));
  const auto t0 = Clock::now();
  REQUIRE(tx->send(make_frame(1, 2'000'000)).is_ok());
  auto r = p.server->receive(5s);
  const auto dt = Clock::now() - t0;
  REQUIRE(r.is_ok());
  CHECK(dt >= 90ms);  // 2 MB / 20 MB/s = 100 ms
  CHECK(dt < 1s);
}

TEST_CASE("impairment: latency is at least the configured value") {
  Pair p = insecure_pair();
  auto tx = impair(std::move(p.client), cond(0, 40, 0));
  const auto t0 = Clock::now();
  REQUIRE(tx->send(make_frame(1, 10)).is_ok());
  REQUIRE(p.server->receive(5s).is_ok());
  const auto dt = Clock::now() - t0;
  CHECK(dt >= 39ms);
  CHECK(dt < 500ms);
}

TEST_CASE("impairment: ordering is preserved under jitter") {
  Pair p = insecure_pair();
  auto tx = impair(std::move(p.client), cond(0, 1, 8, 42));
  constexpr int kN = 60;
  for (int i = 0; i < kN; ++i) REQUIRE(tx->send(make_frame(1, 8, 0, static_cast<std::uint64_t>(i))).is_ok());
  for (int i = 0; i < kN; ++i) {
    auto r = p.server->receive(5s);
    REQUIRE(r.is_ok());
    CHECK(r->correlation == static_cast<std::uint64_t>(i));
  }
}

TEST_CASE("impairment: a shared SimulatedLink splits bandwidth between connections") {
  Pair p1 = insecure_pair();
  Pair p2 = insecure_pair();
  auto link = std::make_shared<SimulatedLink>(20e6);
  auto tx1 = impair(std::move(p1.client), cond(20e6, 0, 0), link);
  auto tx2 = impair(std::move(p2.client), cond(20e6, 0, 0), link);
  const auto t0 = Clock::now();
  REQUIRE(tx1->send(make_frame(1, 1'000'000)).is_ok());
  REQUIRE(tx2->send(make_frame(1, 1'000'000)).is_ok());
  REQUIRE(p1.server->receive(5s).is_ok());
  REQUIRE(p2.server->receive(5s).is_ok());
  // 2 MB total through one 20 MB/s NIC: ~100 ms, versus ~50 ms if each connection had its own link.
  CHECK(Clock::now() - t0 >= 90ms);
}

TEST_CASE("impairment: close() unblocks a sender stuck on a full queue") {
  Pair p = insecure_pair();
  auto tx = impair(std::move(p.client), cond(2000, 0, 0));  // ~50 ms per 100-byte frame: the queue fills up
  Connection* raw = tx.get();
  std::atomic<int> sent{0};
  Status last;
  std::thread sender([&] {
    for (int i = 0; i < 1000; ++i) {
      last = raw->send(make_frame(1, 100));
      if (!last.is_ok()) return;
      ++sent;
    }
  });
  std::this_thread::sleep_for(150ms);
  CHECK(sent.load() >= 60);  // the bound is ~64 frames: backpressure engaged
  CHECK(sent.load() < 1000);
  const auto t0 = Clock::now();
  raw->close();
  sender.join();
  CHECK(Clock::now() - t0 < 2s);
  CHECK(last.code() == ErrorCode::kUnavailable);
}

TEST_CASE("faults: parse_fault_rule") {
  auto r = parse_fault_rule("close-before:type=22:nth=3:dir=send");
  REQUIRE(r.is_ok());
  CHECK(r->action == FaultAction::kCloseBefore);
  CHECK(r->frame_type == std::optional<std::uint16_t>(22));
  CHECK(r->nth == 3);
  CHECK(r->dir == FaultDirection::kSend);
  auto d = parse_fault_rule("delay:ms=25:dir=recv:channel=2");
  REQUIRE(d.is_ok());
  CHECK(d->action == FaultAction::kDelay);
  CHECK(d->delay_ms == 25);
  CHECK(d->dir == FaultDirection::kReceive);
  CHECK(d->channel == std::optional<std::uint8_t>(2));
  CHECK(parse_fault_rule("stall").is_ok());
  CHECK(parse_fault_rule("corrupt:nth=2").is_ok());
  CHECK(parse_fault_rule("close-after").is_ok());
  CHECK_FALSE(parse_fault_rule("explode").is_ok());
  CHECK_FALSE(parse_fault_rule("delay").is_ok());  // delay needs ms=
  CHECK_FALSE(parse_fault_rule("stall:nth=0").is_ok());
  CHECK_FALSE(parse_fault_rule("stall:bogus=1").is_ok());
  CHECK_FALSE(parse_fault_rule("stall:type=abc").is_ok());
}

TEST_CASE("faults: close-before the nth matching frame") {
  Pair p = insecure_pair();
  auto faults = std::make_shared<FaultInjector>();
  faults->add_rule(parse_fault_rule("close-before:type=22:nth=3:dir=send").value());
  auto tx = impair(std::move(p.client), cond(0, 0, 0), nullptr, faults);
  REQUIRE(tx->send(make_frame(22, 4, 0, 1)).is_ok());
  REQUIRE(tx->send(make_frame(5, 4, 0, 99)).is_ok());  // different type: not counted
  REQUIRE(tx->send(make_frame(22, 4, 0, 2)).is_ok());
  CHECK(faults->fired_count() == 0);
  CHECK(tx->send(make_frame(22, 4, 0, 3)).code() == ErrorCode::kUnavailable);
  CHECK(faults->fired_count() == 1);
  CHECK(tx->send(make_frame(22, 4, 0, 4)).code() == ErrorCode::kUnavailable);
  // Earlier frames may or may not have drained before the close; the peer must then see the connection end.
  for (;;) {
    auto r = p.server->receive(2s);
    if (!r.is_ok()) {
      CHECK(r.status().code() == ErrorCode::kUnavailable);
      break;
    }
    CHECK(r->correlation != 3);
  }
}

TEST_CASE("faults: close-after delivers the frame, then closes") {
  Pair p = insecure_pair();
  auto faults = std::make_shared<FaultInjector>();
  faults->add_rule(parse_fault_rule("close-after:nth=2").value());
  auto tx = impair(std::move(p.client), cond(0, 0, 0), nullptr, faults);
  REQUIRE(tx->send(make_frame(1, 4, 0, 1)).is_ok());
  REQUIRE(tx->send(make_frame(1, 4, 0, 2)).is_ok());
  CHECK(p.server->receive(2s)->correlation == 1);
  CHECK(p.server->receive(2s)->correlation == 2);
  CHECK(p.server->receive(2s).status().code() == ErrorCode::kUnavailable);
}

TEST_CASE("faults: stall holds frames until clear_stall") {
  Pair p = insecure_pair();
  auto faults = std::make_shared<FaultInjector>();
  faults->add_rule(parse_fault_rule("stall:type=5:nth=1:dir=send").value());
  auto tx = impair(std::move(p.client), cond(0, 0, 0), nullptr, faults);
  REQUIRE(tx->send(make_frame(4, 4, 0, 1)).is_ok());
  REQUIRE(tx->send(make_frame(5, 4, 0, 2)).is_ok());  // stalls from here
  REQUIRE(tx->send(make_frame(4, 4, 0, 3)).is_ok());  // queued behind it
  CHECK(p.server->receive(2s)->correlation == 1);
  CHECK(p.server->receive(150ms).status().code() == ErrorCode::kDeadlineExceeded);
  CHECK(faults->fired_count() == 1);
  faults->clear_stall();
  CHECK(p.server->receive(2s)->correlation == 2);
  CHECK(p.server->receive(2s)->correlation == 3);
}

TEST_CASE("faults: stall is released by close") {
  Pair p = insecure_pair();
  auto faults = std::make_shared<FaultInjector>();
  faults->add_rule(parse_fault_rule("stall").value());
  auto tx = impair(std::move(p.client), cond(0, 0, 0), nullptr, faults);
  REQUIRE(tx->send(make_frame(1, 4)).is_ok());
  std::this_thread::sleep_for(50ms);
  const auto t0 = Clock::now();
  tx.reset();  // destructor must not hang on the stalled sender thread
  CHECK(Clock::now() - t0 < 2s);
}

TEST_CASE("faults: corrupt payload") {
  Pair p = insecure_pair();
  auto faults = std::make_shared<FaultInjector>();
  faults->add_rule(parse_fault_rule("corrupt:nth=2").value());
  auto tx = impair(std::move(p.client), cond(0, 0, 0), nullptr, faults);
  const Frame original = make_frame(1, 64, 0x5A, 7);
  REQUIRE(tx->send(original).is_ok());
  REQUIRE(tx->send(original).is_ok());
  auto first = p.server->receive(2s);
  auto second = p.server->receive(2s);
  REQUIRE(first.is_ok());
  REQUIRE(second.is_ok());
  CHECK(first->payload == original.payload);
  CHECK(second->payload != original.payload);
  CHECK(second->payload.size() == original.payload.size());
}

TEST_CASE("faults: receive direction") {
  Pair p = insecure_pair();
  auto faults = std::make_shared<FaultInjector>();
  faults->add_rule(parse_fault_rule("close-before:nth=2:dir=recv").value());
  auto rx = impair(std::move(p.server), cond(0, 0, 0), nullptr, faults);
  REQUIRE(p.client->send(make_frame(1, 4, 0, 1)).is_ok());
  REQUIRE(p.client->send(make_frame(1, 4, 0, 2)).is_ok());
  CHECK(rx->receive(2s)->correlation == 1);
  CHECK(rx->receive(2s).status().code() == ErrorCode::kUnavailable);
  CHECK(faults->fired_count() == 1);

  SUBCASE("stall on receive keeps the frame until cleared") {
    Pair q = insecure_pair();
    auto f2 = std::make_shared<FaultInjector>();
    f2->add_rule(parse_fault_rule("stall:dir=recv").value());
    auto rx2 = impair(std::move(q.server), cond(0, 0, 0), nullptr, f2);
    REQUIRE(q.client->send(make_frame(1, 4, 0, 9)).is_ok());
    CHECK(rx2->receive(100ms).status().code() == ErrorCode::kDeadlineExceeded);
    f2->clear_stall();
    auto r = rx2->receive(2s);
    REQUIRE(r.is_ok());
    CHECK(r->correlation == 9);
  }
}
