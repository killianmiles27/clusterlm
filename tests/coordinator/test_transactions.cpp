// Distributed transaction semantics through the real protocol: pipelined prefill, every speculative acceptance
// length, window abort, conversations, cancellation, lost acknowledgements, provisioning resume.
#include <doctest/doctest.h>

#include <thread>

#include "cluster_fixture.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/protocol/messages.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using namespace std::chrono_literals;

TEST_CASE("pipelined prefill matches the reference for every chunk size, in-flight bound and routing mode") {
  TestCluster cl("prefill");
  const auto prompt = test_prompt(37, cl.manifest().geometry.vocab_size);
  const auto ref = cl.reference(prompt, 6);
  for (bool direct : {true, false}) {
    auto f = cl.father(cl.config(direct), kSplitPlan);
    for (std::uint32_t chunk : {1u, 3u, 8u, 16u, 64u}) {
      for (std::uint32_t inflight : {1u, 2u, 4u}) {
        CAPTURE(direct);
        CAPTURE(chunk);
        CAPTURE(inflight);
        coordinator::GenerationRequest r;
        r.prompt = prompt;
        r.max_new_tokens = 6;
        r.prefill_chunk = chunk;
        r.prefill_inflight = inflight;
        auto g = f->generate(r);
        REQUIRE_MESSAGE(g.is_ok(), g.status().to_string());
        CHECK(g->tokens == ref);
        CHECK(g->prefill_tokens == prompt.size());
        CHECK(g->prefill_chunks == (prompt.size() + std::min<std::size_t>(chunk, 64) - 1) / std::min<std::size_t>(chunk, 64));
        CHECK(g->prefill_max_in_flight <= inflight);
        if (inflight > 1 && chunk < prompt.size()) CHECK(g->prefill_max_in_flight > 1);
      }
    }
    REQUIRE(f->release().is_ok());
  }
}

TEST_CASE("every speculative acceptance length commits exactly the accepted prefix") {
  TestCluster cl("accept");
  const auto& g = cl.manifest().geometry;
  const auto prompt = test_prompt(12, g.vocab_size);
  const std::uint32_t max_new = 20;
  const auto ref = cl.reference(prompt, max_new + 8);
  std::vector<std::int32_t> sequence = prompt;
  sequence.insert(sequence.end(), ref.begin(), ref.end());
  auto f = cl.father(cl.config(), kSplitPlan);
  for (std::uint32_t q = 1; q <= 4; ++q) {
    // corrupt index j (1-based draft) => exactly j-1 drafts accepted per round; none => all accepted.
    for (std::uint32_t j = 0; j <= q - 1; ++j) {
      CAPTURE(q);
      CAPTURE(j);
      domain::ScriptedDrafter::Config dc;
      dc.vocab = g.vocab_size;
      if (j > 0) dc.corrupt_draft_index = j;
      coordinator::GenerationRequest r;
      r.prompt = prompt;
      r.max_new_tokens = max_new;
      r.q = q;
      r.drafter = std::make_shared<domain::ScriptedDrafter>(sequence, dc);
      auto out = f->generate(r);
      REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());
      CHECK(out->tokens == std::vector<std::int32_t>(ref.begin(), ref.begin() + max_new));
      const std::uint32_t expect_drafts = j == 0 ? q - 1 : j - 1;
      for (const auto& rd : out->rounds) {
        if (rd.prefill) continue;
        // Rounds capped by the output budget verify fewer positions.
        if (rd.positions == q) CHECK(rd.accepted_drafts == expect_drafts);
        CHECK(rd.accepted == rd.accepted_drafts + 1);
        CHECK(rd.emitted <= rd.accepted);
      }
      CHECK(out->accepted_drafts <= out->proposed_positions);
    }
  }
  REQUIRE(f->release().is_ok());
}

TEST_CASE("a conversation keeps its distributed session across turns") {
  TestCluster cl("conversation");
  const auto& g = cl.manifest().geometry;
  const auto a = test_prompt(9, g.vocab_size, 1), b = test_prompt(7, g.vocab_size, 2);
  auto f = cl.father(cl.config(), kSplitPlan);
  auto conv = f->open_conversation();
  REQUIRE(conv.is_ok());
  coordinator::GenerationRequest r;
  r.conversation = conv.value();
  r.prompt = a;
  r.max_new_tokens = 8;
  auto t1 = f->generate(r);
  REQUIRE(t1.is_ok());
  r.prompt = b;
  auto t2 = f->generate(r);
  REQUIRE(t2.is_ok());
  // Only the new tokens (plus the one pending prediction) were prefilled in turn 2.
  CHECK(t2->prefill_tokens == b.size() + 1);
  CHECK(conv.value()->valid());
  REQUIRE(f->close_conversation(*conv.value()).is_ok());
  REQUIRE(f->release().is_ok());

  // Reference: one-shot over the full history.
  std::vector<std::int32_t> history = a;
  history.insert(history.end(), t1->tokens.begin(), t1->tokens.end());
  history.insert(history.end(), b.begin(), b.end());
  CHECK(cl.reference(history, 8) == t2->tokens);
}

TEST_CASE("cancelling between windows and while a window is outstanding keeps the conversation usable") {
  // Slow links make sure the cancel lands while a decode window is in flight (exercising AbortWindow).
  transport::NetworkConditions slow{"test-slow", 0, 15.0, 0.0, 1};
  TestCluster cl("cancel", 2, {NodeOptions::impaired(slow), NodeOptions::impaired(slow)});
  const auto& g = cl.manifest().geometry;
  const auto prompt = test_prompt(10, g.vocab_size, 3);
  const auto ref = cl.reference(prompt, 30);
  auto f = cl.father(cl.config(), kSplitPlan);

  SUBCASE("cancel from the token callback") {
    auto conv = f->open_conversation();
    REQUIRE(conv.is_ok());
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    coordinator::GenerationRequest r;
    r.conversation = conv.value();
    r.prompt = prompt;
    r.max_new_tokens = 30;
    r.cancel = cancel;
    r.on_tokens = [&](std::span<const std::int32_t>) { cancel->store(true); };
    auto first = f->generate(r);
    REQUIRE(first.is_ok());
    CHECK(first->cancelled);
    CHECK(first->tokens.size() < 30);
    r.prompt.clear();
    r.cancel.reset();
    r.on_tokens = nullptr;
    r.max_new_tokens = static_cast<std::uint32_t>(30 - first->tokens.size());
    auto rest = f->generate(r);
    REQUIRE_MESSAGE(rest.is_ok(), rest.status().to_string());
    // The continuation re-emits nothing: it starts after the pending prediction the user already saw.
    std::vector<std::int32_t> all = first->tokens;
    all.insert(all.end(), rest->tokens.begin(), rest->tokens.end());
    // The pending token is fed first and the model predicts the token after it, so the continuation's first
    // token is ref[first.size()].
    CHECK(std::vector<std::int32_t>(all.begin(), all.end()) == ref);
  }

  SUBCASE("cancel while a verification window is outstanding (AbortWindow)") {
    auto conv = f->open_conversation();
    REQUIRE(conv.is_ok());
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    coordinator::GenerationRequest r;
    r.conversation = conv.value();
    r.prompt = prompt;
    r.max_new_tokens = 30;
    r.cancel = cancel;
    std::thread canceller([&] {
      std::this_thread::sleep_for(400ms);
      cancel->store(true);
    });
    auto first = f->generate(r);
    canceller.join();
    REQUIRE_MESSAGE(first.is_ok(), first.status().to_string());
    REQUIRE(first->cancelled);
    CHECK(conv.value()->valid());
    r.prompt.clear();
    r.cancel.reset();
    r.max_new_tokens = static_cast<std::uint32_t>(30 - first->tokens.size());
    if (r.max_new_tokens > 0) {
      auto rest = f->generate(r);
      REQUIRE_MESSAGE(rest.is_ok(), rest.status().to_string());
      std::vector<std::int32_t> all = first->tokens;
      all.insert(all.end(), rest->tokens.begin(), rest->tokens.end());
      CHECK(all == ref);
    }
  }
  REQUIRE(f->release().is_ok());
}

TEST_CASE("cancelling during prefill drains in-flight chunks and leaves a consistent conversation") {
  transport::NetworkConditions slow{"test-slow", 0, 10.0, 0.0, 1};
  TestCluster cl("cancel-prefill", 2, {NodeOptions::impaired(slow), NodeOptions::impaired(slow)});
  const auto& g = cl.manifest().geometry;
  const auto prompt = test_prompt(200, g.vocab_size, 4);
  const auto ref = cl.reference(prompt, 5);
  auto f = cl.father(cl.config(), kSplitPlan);
  auto conv = f->open_conversation();
  REQUIRE(conv.is_ok());
  auto cancel = std::make_shared<std::atomic<bool>>(false);
  coordinator::GenerationRequest r;
  r.conversation = conv.value();
  r.prompt = prompt;
  r.max_new_tokens = 5;
  r.prefill_chunk = 4;
  r.prefill_inflight = 2;
  r.cancel = cancel;
  std::thread canceller([&] {
    std::this_thread::sleep_for(150ms);
    cancel->store(true);
  });
  Stopwatch sw;
  auto out = f->generate(r);
  const double ms = sw.elapsed_ms();
  canceller.join();
  REQUIRE(out.is_ok());
  CHECK(out->cancelled);
  CHECK(out->tokens.empty());
  // Responsive: only the in-flight chunks drain after the cancel (the full prefill takes far longer). The bound is
  // generous so a loaded CI machine does not make it flaky.
  CHECK(ms < 5000.0);
  const auto done = conv.value()->committed_tokens().size();
  CHECK(done > 0);
  CHECK(done < prompt.size());
  CHECK(conv.value()->committed_positions() == done);
  // Continue with the rest of the prompt: identical to an uninterrupted prefill.
  r.prompt.assign(prompt.begin() + static_cast<std::ptrdiff_t>(done), prompt.end());
  r.cancel.reset();
  auto rest = f->generate(r);
  REQUIRE_MESSAGE(rest.is_ok(), rest.status().to_string());
  CHECK(rest->tokens == ref);
  REQUIRE(f->release().is_ok());
}

TEST_CASE("a lost commit acknowledgement is retried idempotently") {
  auto faults = std::make_shared<transport::FaultInjector>();
  transport::FaultRule rule;
  rule.frame_type = static_cast<std::uint16_t>(protocol::MessageType::kCommitAck);
  rule.nth = 3;
  rule.action = transport::FaultAction::kDelay;
  rule.delay_ms = 1200;
  faults->add_rule(rule);
  TestCluster cl("lost-ack", 2, {NodeOptions{}, NodeOptions::faulty(faults)});
  const auto prompt = test_prompt(8, cl.manifest().geometry.vocab_size, 5);
  const auto ref = cl.reference(prompt, 10);
  auto cfg = cl.config();
  cfg.request_timeout = 400ms;
  auto f = cl.father(cfg, kSplitPlan);
  coordinator::GenerationRequest r;
  r.prompt = prompt;
  r.max_new_tokens = 10;
  auto out = f->generate(r);
  REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());
  CHECK(out->tokens == ref);
  std::uint32_t retries = 0;
  for (const auto& rd : out->rounds) retries += rd.commit_retries;
  CHECK(retries >= 1);
  CHECK(faults->fired_count() == 1);
  REQUIRE(f->release().is_ok());
}

TEST_CASE("a broken provisioning stream resumes within the same lease") {
  TestCluster cl("resume");
  const auto prompt = test_prompt(8, cl.manifest().geometry.vocab_size, 6);
  const auto ref = cl.reference(prompt, 6);
  auto cfg = cl.config();
  cfg.faults = std::make_shared<transport::FaultInjector>();
  transport::FaultRule rule;
  rule.frame_type = static_cast<std::uint16_t>(protocol::MessageType::kProvisionChunk);
  rule.nth = 40;
  rule.action = transport::FaultAction::kCloseBefore;
  cfg.faults->add_rule(rule);
  auto f = cl.father(cfg, kSplitPlan);
  std::uint32_t resumes = 0;
  for (const auto& n : cl.last_prepare().nodes) resumes += n.resumes;
  CHECK(resumes == 1);
  coordinator::GenerationRequest r;
  r.prompt = prompt;
  r.max_new_tokens = 6;
  auto out = f->generate(r);
  REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());
  CHECK(out->tokens == ref);
  auto rel = f->release();
  REQUIRE(rel.is_ok());
  for (const auto& n : rel->nodes) CHECK(n.storage_cleaned);
}

TEST_CASE("cancelling preparation stops provisioning and releases the partial lease") {
  TestCluster cl("cancel-prepare");
  auto cfg = cl.config();
  cfg.impairment = transport::NetworkConditions{"test-slow", 2e6, 1.0, 0.0, 1};  // ~5 MB over 2 MB/s
  auto coord = coordinator::Coordinator::create(cfg);
  REQUIRE(coord.is_ok());
  REQUIRE(coord.value()->connect().is_ok());
  auto plan = coordinator::ClusterPlan::parse(kSplitPlan, cl.manifest().geometry.n_layers);
  REQUIRE(plan.is_ok());
  std::thread canceller([&] {
    std::this_thread::sleep_for(300ms);
    coord.value()->cancel_prepare();
  });
  Stopwatch sw;
  auto prep = coord.value()->prepare(plan.value());
  canceller.join();
  CHECK_FALSE(prep.is_ok());
  CHECK(prep.status().code() == ErrorCode::kCancelled);
  CHECK(sw.elapsed_ms() < 2000.0);
  for (std::size_t i = 0; i < 2; ++i) {
    bool clean = false;
    for (int k = 0; k < 100 && !clean; ++k) {
      clean = cl.worker(i).status().staging_census_bytes == 0;
      if (!clean) std::this_thread::sleep_for(20ms);
    }
    CHECK(clean);
  }
}

TEST_CASE("lease-scoped temporary file backing executes like RAM residency and is deleted on release") {
  TestCluster cl("disk", 2, {NodeOptions::with_disk(64ull << 20), NodeOptions{}});
  const auto& g = cl.manifest().geometry;
  const auto prompt = test_prompt(8, g.vocab_size, 7);
  const auto ref = cl.reference(prompt, 6);
  auto coord = coordinator::Coordinator::create(cl.config());
  REQUIRE(coord.is_ok());
  REQUIRE(coord.value()->connect().is_ok());
  auto plan = coordinator::ClusterPlan::parse(kSplitPlan, g.n_layers);
  REQUIRE(plan.is_ok());
  for (std::uint32_t L = 4; L < 10; ++L)
    for (std::uint32_t e = 0; e < g.n_experts; ++e)
      plan.value().targets.emplace_back(objects::expert_object_name(L, e), objects::AllocationTarget::kTemporaryBacking);
  auto prep = coord.value()->prepare(plan.value());
  REQUIRE_MESSAGE(prep.is_ok(), prep.status().to_string());
  CHECK(cl.worker(0).status().staging_census_bytes > 0);  // the experts live in lease-scoped files
  coordinator::GenerationRequest r;
  r.prompt = prompt;
  r.max_new_tokens = 6;
  auto out = coord.value()->generate(r);
  REQUIRE_MESSAGE(out.is_ok(), out.status().to_string());
  CHECK(out->tokens == ref);
  auto rel = coord.value()->release();
  REQUIRE(rel.is_ok());
  for (const auto& n : rel->nodes) CHECK(n.storage_cleaned);
  CHECK(cl.worker(0).status().staging_census_bytes == 0);
}
