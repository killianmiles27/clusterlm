// Replay: a RunWindow captured from an earlier lease is rejected by a Node that has since granted a new lease, and
// a RunWindow that names the new lease but replays an old epoch/session/window is refused by the window ledger.
#include <doctest/doctest.h>

#include <algorithm>

#include "../coordinator/cluster_fixture.hpp"
#include "../privacy/chat_fixture.hpp"
#include "../privacy/raw_client.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using namespace clusterlm::protocol;

TEST_CASE("a RunWindow captured under a previous lease is rejected as stale, and replaying old windows changes nothing") {
  TrafficRecorder rec;
  TestCluster cluster("replay", 2);
  auto first_cfg = cluster.config(/*direct=*/true);
  first_cfg.faults = rec.tap("father");

  // First lease: a short generation, capturing what Father sent.
  {
    auto father = cluster.father(first_cfg, kSplitPlan, 32);
    coordinator::GenerationRequest req;
    req.prompt = test_prompt(6, cluster.manifest().geometry.vocab_size);
    req.max_new_tokens = 4;
    REQUIRE(father->generate(req).is_ok());
    REQUIRE(father->release().is_ok());
  }
  // The first RunWindow Father sent to the first Node.
  const auto frames = rec.frames();
  const auto it = std::find_if(frames.begin(), frames.end(), [](const CapturedFrame& f) {
    return f.party == "father" && f.sent && f.type == static_cast<std::uint16_t>(MessageType::kRunWindow);
  });
  REQUIRE(it != frames.end());
  auto decoded = decode(MessageType::kRunWindow, it->payload);
  REQUIRE(decoded.is_ok());
  const RunWindow captured = std::get<RunWindow>(decoded.value());
  const std::uint64_t old_lease = captured.lease.value;
  REQUIRE(old_lease > 0);

  // Second lease: the Nodes now sit at a newer lease generation.
  auto second = cluster.father(cluster.config(true), kSplitPlan, 32);
  node::NodeWorker& n0 = cluster.worker(0);
  const std::uint64_t new_lease = n0.status().lease_generation;
  REQUIRE(new_lease > old_lease);
  const std::uint64_t stale_before = n0.status().stale_rejections;

  auto act = open_stream(n0.endpoint(), insecure_security(), Channel::kActivation, NodeRole::kFather, "replayer",
                         LeaseGeneration{new_lease});
  REQUIRE(act.is_ok());
  REQUIRE(std::holds_alternative<HelloAck>(next_message(**act).value()));

  SUBCASE("verbatim replay: kStaleEpoch, nothing executed") {
    REQUIRE((*act)->send(captured).is_ok());
    auto r = next_message(**act);
    REQUIRE(r.is_ok());
    REQUIRE(std::holds_alternative<StageResult>(r.value()));
    const auto& result = std::get<StageResult>(r.value());
    CHECK(result.status == ErrorCode::kStaleEpoch);
    CHECK(result.activations.data.empty());
    CHECK(n0.status().stale_rejections == stale_before + 1);
  }

  SUBCASE("replay with the lease field patched to the current lease: the window ledger refuses it") {
    RunWindow patched = captured;
    patched.lease = LeaseGeneration{new_lease};
    REQUIRE((*act)->send(patched).is_ok());
    auto r = next_message(**act);
    REQUIRE(r.is_ok());
    REQUIRE(std::holds_alternative<StageResult>(r.value()));
    const auto& result = std::get<StageResult>(r.value());
    CHECK(result.status != ErrorCode::kOk);  // no session of that epoch/id is open under the new lease
    CHECK(result.activations.data.empty());
  }

  SUBCASE("a second control connection cannot be used to inject commits") {
    auto control = open_stream(n0.endpoint(), insecure_security(), Channel::kControl, NodeRole::kFather, "replayer",
                               LeaseGeneration{new_lease});
    // Father's own control connection is already attached: a second one is turned away and, in particular,
    // cannot be used to inject commits.
    REQUIRE(control.is_ok());
    auto m = next_message(**control);
    REQUIRE(m.is_ok());
    CHECK(std::holds_alternative<ErrorMessage>(m.value()));
  }

  (*act)->close();
  REQUIRE(second->release().is_ok());
}
