// Coordinator::interrupt(): a generate() blocked on a stalled Node returns promptly (not after window_timeout), the
// session is invalidated like for any stage failure, and release() still works afterwards (docs/scheduler-design.md D1).
#include <doctest/doctest.h>

#include <thread>

#include "cluster_fixture.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/protocol/messages.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using namespace std::chrono_literals;

TEST_CASE("interrupt() ends a generate blocked on a stalled Node within the wait slice, and release() still works") {
  auto faults = std::make_shared<transport::FaultInjector>();
  transport::FaultRule rule;
  rule.frame_type = static_cast<std::uint16_t>(protocol::MessageType::kStageResult);
  rule.nth = 3;
  rule.dir = transport::FaultDirection::kSend;
  rule.action = transport::FaultAction::kStall;
  faults->add_rule(rule);
  TestCluster cl("interrupt", 2, {NodeOptions{}, NodeOptions::faulty(faults)});
  const auto prompt = test_prompt(8, cl.manifest().geometry.vocab_size, 5);
  auto cfg = cl.config();
  cfg.request_timeout = 30s;
  cfg.window_timeout = 30s;   // without interrupt() the blocked generate would run for this long
  auto f = cl.father(cfg, kSplitPlan);
  auto conv = f->open_conversation();
  REQUIRE(conv.is_ok());

  coordinator::GenerationRequest r;
  r.conversation = conv.value();
  r.prompt = prompt;
  r.max_new_tokens = 20;
  Result<coordinator::GenerationResult> out = Status(ErrorCode::kInternal, "not run");
  std::thread gen([&] { out = f->generate(r); });
  for (int i = 0; i < 500 && faults->fired_count() == 0; ++i) std::this_thread::sleep_for(10ms);
  REQUIRE(faults->fired_count() == 1);   // the Node's third StageResult is being held
  std::this_thread::sleep_for(100ms);
  Stopwatch sw;
  f->interrupt();
  gen.join();
  const double waited_ms = sw.elapsed_ms();
  CHECK(waited_ms < 2000);               // not 30 s
  CHECK(f->interrupted());
  CHECK_FALSE(out.is_ok());
  CHECK_FALSE(conv.value()->valid());    // the distributed session was invalidated, as for any stage failure

  faults->clear_stall();
  auto rel = f->release();               // clears the interrupt and talks to the Nodes normally
  REQUIRE_MESSAGE(rel.is_ok(), rel.status().to_string());
  CHECK_FALSE(f->interrupted());
  for (const auto& n : rel->nodes) CHECK(n.errors.empty());
}
