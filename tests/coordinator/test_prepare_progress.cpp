// Preparation progress: the optional sink of Coordinator::prepare() reports per-Node bytes sent / total, objects
// sealed / total and phases, serialized, monotone, ending exactly at the totals the PrepareReport states.
#include <doctest/doctest.h>

#include <atomic>
#include <mutex>
#include <thread>

#include "cluster_fixture.hpp"
#include "clusterlm/protocol/messages.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using coordinator::PreparePhase;
using coordinator::PrepareProgress;

namespace {

struct Recorder {
  std::mutex mu;
  std::vector<PrepareProgress> seen;
  std::atomic<int> inside{0};
  std::atomic<bool> overlapped{false};
  coordinator::PrepareProgressSink sink() {
    return [this](const PrepareProgress& p) {
      if (inside.fetch_add(1) != 0) overlapped = true;  // the contract: invocations are never concurrent
      {
        std::lock_guard lk(mu);
        seen.push_back(p);
      }
      inside.fetch_sub(1);
    };
  }
  std::vector<PrepareProgress> snapshot() {
    std::lock_guard lk(mu);
    return seen;
  }
};

}  // namespace

TEST_CASE("prepare progress reports every Node, ends at the report's totals and never goes backwards") {
  TestCluster cl("progress");
  Recorder rec;
  auto coord = coordinator::Coordinator::create(cl.config());
  REQUIRE(coord.is_ok());
  REQUIRE(coord.value()->connect().is_ok());
  auto plan = coordinator::ClusterPlan::parse(kSplitPlan, cl.manifest().geometry.n_layers);
  REQUIRE(plan.is_ok());
  auto report = coord.value()->prepare(plan.value(), rec.sink());
  REQUIRE_MESSAGE(report.is_ok(), report.status().to_string());
  CHECK_FALSE(rec.overlapped.load());
  const auto seen = rec.snapshot();
  REQUIRE(seen.size() >= 4);

  CHECK(seen.front().phase == PreparePhase::kFatherDomains);
  const auto& last = seen.back();
  CHECK(last.phase == PreparePhase::kDone);
  REQUIRE(last.nodes.size() == report->nodes.size());
  std::uint64_t bytes = 0;
  std::uint32_t objects = 0;
  for (std::size_t i = 0; i < last.nodes.size(); ++i) {
    CAPTURE(i);
    CHECK(last.nodes[i].node == report->nodes[i].node);
    CHECK(last.nodes[i].phase == PreparePhase::kNodeReady);
    CHECK(last.nodes[i].bytes_total > 0);
    CHECK(last.nodes[i].bytes_sent == last.nodes[i].bytes_total);
    CHECK(last.nodes[i].bytes_sent == report->nodes[i].bytes);  // what the report says was sent
    CHECK(last.nodes[i].objects_sealed == last.nodes[i].objects_total);
    CHECK(last.nodes[i].objects_total == report->nodes[i].objects);
    bytes += last.nodes[i].bytes_sent;
    objects += last.nodes[i].objects_sealed;
  }
  CHECK(last.bytes_sent == bytes);
  CHECK(last.bytes_total == bytes);
  CHECK(last.objects_sealed == objects);
  CHECK(last.objects_total == objects);

  // Per Node: bytes and sealed objects never decrease, the phase never moves backwards, sent never exceeds total.
  bool saw_provisioning = false, saw_node_preparing = false, saw_authorizing = false;
  for (std::size_t k = 0; k < seen.size(); ++k) {
    const auto& p = seen[k];
    saw_provisioning = saw_provisioning || p.phase == PreparePhase::kProvisioning;
    saw_authorizing = saw_authorizing || p.phase == PreparePhase::kAuthorizing;
    for (std::size_t i = 0; i < p.nodes.size(); ++i) {
      CHECK(p.nodes[i].bytes_sent <= p.nodes[i].bytes_total);
      saw_node_preparing = saw_node_preparing || p.nodes[i].phase == PreparePhase::kNodePreparing;
      if (k > 0) {
        const auto& q = seen[k - 1].nodes[i];
        CHECK(p.nodes[i].bytes_sent >= q.bytes_sent);
        CHECK(p.nodes[i].objects_sealed >= q.objects_sealed);
        CHECK(static_cast<int>(p.nodes[i].phase) >= static_cast<int>(q.phase));
      }
    }
  }
  CHECK(saw_provisioning);
  CHECK(saw_node_preparing);
  CHECK(saw_authorizing);  // direct_peer is on in the fixture config
  REQUIRE(coord.value()->release().is_ok());
}

TEST_CASE("prepare without a sink behaves exactly as before") {
  TestCluster cl("progress-none");
  auto f = cl.father(cl.config(), kSplitPlan);  // the fixture calls prepare(plan) with no sink
  CHECK(f->ready());
  REQUIRE(f->release().is_ok());
}

TEST_CASE("progress stays consistent when a provisioning stream is resumed") {
  TestCluster cl("progress-resume");
  Recorder rec;
  auto cfg = cl.config();
  cfg.faults = std::make_shared<transport::FaultInjector>();
  transport::FaultRule rule;
  rule.frame_type = static_cast<std::uint16_t>(protocol::MessageType::kProvisionChunk);
  rule.nth = 40;
  rule.action = transport::FaultAction::kCloseBefore;
  cfg.faults->add_rule(rule);
  auto coord = coordinator::Coordinator::create(cfg);
  REQUIRE(coord.is_ok());
  REQUIRE(coord.value()->connect().is_ok());
  auto plan = coordinator::ClusterPlan::parse(kSplitPlan, cl.manifest().geometry.n_layers);
  REQUIRE(plan.is_ok());
  auto report = coord.value()->prepare(plan.value(), rec.sink());
  REQUIRE_MESSAGE(report.is_ok(), report.status().to_string());
  std::uint32_t resumes = 0;
  for (const auto& n : report->nodes) resumes += n.resumes;
  REQUIRE(resumes == 1);
  const auto seen = rec.snapshot();
  for (const auto& n : seen.back().nodes) {
    CHECK(n.bytes_sent == n.bytes_total);  // a resumed object is re-sent whole, but the count ends at the total
    CHECK(n.objects_sealed == n.objects_total);
  }
  for (const auto& p : seen)
    for (const auto& n : p.nodes) CHECK(n.bytes_sent <= n.bytes_total);  // never reports more than assigned
  REQUIRE(coord.value()->release().is_ok());
}

TEST_CASE("a cancelled prepare ends its progress: the sink is not called after prepare returns") {
  TestCluster cl("progress-cancel");
  Recorder rec;
  auto cfg = cl.config();
  cfg.impairment = transport::NetworkConditions{"test-slow", 2e6, 1.0, 0.0, 1};
  auto coord = coordinator::Coordinator::create(cfg);
  REQUIRE(coord.is_ok());
  REQUIRE(coord.value()->connect().is_ok());
  auto plan = coordinator::ClusterPlan::parse(kSplitPlan, cl.manifest().geometry.n_layers);
  REQUIRE(plan.is_ok());
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    coord.value()->cancel_prepare();
  });
  auto report = coord.value()->prepare(plan.value(), rec.sink());
  canceller.join();
  REQUIRE_FALSE(report.is_ok());
  CHECK(report.status().code() == ErrorCode::kCancelled);
  const std::size_t calls = rec.snapshot().size();
  CHECK(calls >= 2);  // it did report while provisioning
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(rec.snapshot().size() == calls);
}
