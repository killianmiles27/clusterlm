// A Node's error replies never echo bytes of the request that provoked them, and hostile requests (including a
// SealObject naming an object outside the plan) leave it alive and still serving.
#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "chat_fixture.hpp"
#include "message_samples.hpp"
#include "raw_client.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using namespace clusterlm::protocol;

namespace {

const std::string kSentinel = "ZZ-SENTINEL-ZZ-7f3a91";

// Everything a reply could echo, as bytes: its encoding plus its human-readable error texts.
struct Replies {
  std::vector<Bytes> encoded;
  std::vector<std::string> texts;
  void add(const Message& m) {
    encoded.push_back(encode(m));
    if (const auto* e = std::get_if<ErrorMessage>(&m)) texts.push_back(e->message);
    if (const auto* s = std::get_if<StageResult>(&m)) texts.push_back(s->error_message);
    if (const auto* r = std::get_if<ReleaseComplete>(&m)) texts.push_back(r->errors);
  }
};

}  // namespace

TEST_CASE("Node error replies never echo request payload bytes, and hostile requests do not stop the Node") {
  const auto dir = privacy_temp_dir("errors");
  node::NodeConfig nc;
  nc.name = "node-under-test";
  nc.security = insecure_security();
  nc.staging_root = dir / "staging";
  nc.ram_allowance = 1ull << 30;
  nc.vram_allowance = 1ull << 30;
  auto started = node::NodeWorker::start(nc);
  REQUIRE(started.is_ok());
  auto worker = std::move(started).value();
  const transport::Endpoint ep = worker->endpoint();
  const LeaseGeneration lease{worker->status().lease_generation};
  Replies replies;

  // ---- control channel: requests that are wrong in every way a peer can get them wrong --------------------
  auto control = open_stream(ep, insecure_security(), Channel::kControl, NodeRole::kFather, kSentinel, lease);
  REQUIRE(control.is_ok());
  auto ack = next_message(**control);
  REQUIRE(ack.is_ok());
  REQUIRE(std::holds_alternative<HelloAck>(ack.value()));
  auto offer = next_message(**control);
  REQUIRE(offer.is_ok());
  REQUIRE(std::holds_alternative<OfferResources>(offer.value()));

  auto ask = [&](const Message& m) {
    REQUIRE((*control)->send(m).is_ok());
    auto r = next_message(**control);
    INFO("sent " << to_string(type_of(m)) << ": " << r.status().to_string());
    REQUIRE(r.is_ok());
    replies.add(r.value());
    return std::move(r).value();
  };

  // No plan yet: peer authorization for a lease that has no plan.
  CHECK(std::holds_alternative<ErrorMessage>(
      ask(AuthorizePeer{lease, Digest256{}, StageId{1}, StageId{2}, kSentinel, kSentinel, lease})));
  // Backend build mismatch (the sentinel is in the request, the Node's own build in the reply).
  {
    PreparePlan p;
    p.lease = lease;
    p.backend_build = kSentinel + "-build";
    p.manifest = sample_manifest();
    auto r = ask(p);
    REQUIRE(std::holds_alternative<ErrorMessage>(r));
    CHECK(std::get<ErrorMessage>(r).code == ErrorCode::kVersionMismatch);
  }
  {  // plan for a lease the Node never granted
    PreparePlan p;
    p.lease = LeaseGeneration{lease.value + 41};
    p.backend_build = kSentinel;
    p.manifest = sample_manifest();
    auto r = ask(p);
    REQUIRE(std::holds_alternative<ErrorMessage>(r));
    CHECK(std::get<ErrorMessage>(r).code == ErrorCode::kStaleEpoch);
  }
  CHECK(std::holds_alternative<ErrorMessage>(ask(CommitWindow{{Epoch{1}, SessionId{1}, WindowId{1}, 1, StateVersion{0}}, StageId{9}})));
  CHECK(std::holds_alternative<ErrorMessage>(ask(OpenSession{Epoch{1}, SessionId{1}})));
  CHECK(std::holds_alternative<ErrorMessage>(ask(AbortWindow{Epoch{1}, SessionId{1}, WindowId{1}, StageId{9}})));
  CHECK(std::holds_alternative<ErrorMessage>(ask(ReleaseLease{LeaseGeneration{lease.value + 7}, ReleaseReason::kFatherRequest})));
  // A message type that belongs to another channel's vocabulary.
  CHECK(std::holds_alternative<ErrorMessage>(ask(HelloAck{kProtocolVersion, kSentinel, lease})));

  // ---- a valid plan, then hostile provisioning --------------------------------------------------------------
  const objects::ModelManifest manifest = sample_manifest();
  PreparePlan plan;
  plan.lease = lease;
  plan.model_root = manifest.root_hash();
  plan.backend_build = "reference-cpu-fp32-v1";
  plan.plan_hash = sample_digest(3);
  plan.stages.push_back(StageAssignment{StageId{1}, domain::StageRole::kMiddle, objects::LayerRange{3, 4}, 64, 8, 4});
  plan.manifest = manifest;
  std::vector<std::uint32_t> assigned;
  for (std::uint32_t i = 0; i < manifest.objects.size(); ++i) {
    const auto& o = manifest.objects[i];
    if (!o.father_only() && o.layer && *o.layer == 3) {
      plan.assignments.push_back(ObjectAssignment{i, objects::AllocationTarget::kCpuResident});
      assigned.push_back(i);
    }
  }
  REQUIRE(!assigned.empty());
  REQUIRE(std::holds_alternative<PlanAccepted>(ask(plan)));
  CHECK(std::holds_alternative<ErrorMessage>(ask(plan)));  // second plan while Preparing

  auto provision = open_stream(ep, insecure_security(), Channel::kProvision, NodeRole::kFather, kSentinel, lease);
  REQUIRE(provision.is_ok());
  REQUIRE(std::holds_alternative<HelloAck>(next_message(**provision).value()));
  REQUIRE(std::holds_alternative<ProvisionStatus>(next_message(**provision).value()));
  auto ask_provision = [&](const Message& m) {
    REQUIRE((*provision)->send(m).is_ok());
    auto r = next_message(**provision);
    REQUIRE(r.is_ok());
    replies.add(r.value());
    return std::move(r).value();
  };
  const Bytes sentinel_bytes(kSentinel.begin(), kSentinel.end());
  {  // wrong chunk digest
    ProvisionChunk c{lease, assigned[0], 0, Digest256{}, sentinel_bytes};
    CHECK(std::holds_alternative<ErrorMessage>(ask_provision(c)));
  }
  {  // correct digest, out of bounds
    ProvisionChunk c{lease, assigned[0], 1ull << 40, Sha256::of(sentinel_bytes), sentinel_bytes};
    CHECK(std::holds_alternative<ErrorMessage>(ask_provision(c)));
  }
  {  // object not in the plan
    ProvisionChunk c{lease, 9999, 0, Sha256::of(sentinel_bytes), sentinel_bytes};
    CHECK(std::holds_alternative<ErrorMessage>(ask_provision(c)));
  }
  {  // seal naming an object outside the manifest: previously an uncaught std::out_of_range
    auto r = ask_provision(SealObject{lease, 9999, 1, Sha256::of(sentinel_bytes)});
    REQUIRE(std::holds_alternative<ErrorMessage>(r));
    CHECK(std::get<ErrorMessage>(r).code == ErrorCode::kOutOfRange);
  }
  {  // seal that does not match the manifest
    auto r = ask_provision(SealObject{lease, assigned[0], 1, Sha256::of(sentinel_bytes)});
    REQUIRE(std::holds_alternative<ErrorMessage>(r));
    CHECK(std::get<ErrorMessage>(r).code == ErrorCode::kDataLoss);
  }
  {  // seal of an object that is in the manifest but not assigned to this Node
    std::uint32_t unassigned = 0;
    while (std::find(assigned.begin(), assigned.end(), unassigned) != assigned.end()) ++unassigned;
    const auto& o = manifest.objects[unassigned];
    auto r = ask_provision(SealObject{lease, unassigned, o.byte_size, o.object_digest});
    CHECK(std::holds_alternative<ErrorMessage>(r));
  }

  // ---- activation channel: unauthorized peer, and a RunWindow naming a stale lease ----------------------------
  {
    auto peer = open_stream(ep, insecure_security(), Channel::kActivation, NodeRole::kNode, kSentinel, lease);
    REQUIRE(peer.is_ok());
    auto r = next_message(**peer);
    REQUIRE(r.is_ok());
    replies.add(r.value());
    CHECK(std::holds_alternative<ErrorMessage>(r.value()));
    CHECK(std::get<ErrorMessage>(r.value()).code == ErrorCode::kPermissionDenied);
  }
  {
    auto act = open_stream(ep, insecure_security(), Channel::kActivation, NodeRole::kFather, kSentinel, lease);
    REQUIRE(act.is_ok());
    REQUIRE(std::holds_alternative<HelloAck>(next_message(**act).value()));
    RunWindow run;
    run.lease = LeaseGeneration{lease.value + 100};
    run.request = domain::WindowRequest{Epoch{1}, SessionId{1}, WindowId{1}, 0, StateVersion{0}, 2};
    run.stage = StageId{1};
    run.activations = sample_activations(2, sample_layout());
    for (auto& v : run.activations.data) v = 3.14159274f;  // 0x40490FDB: a recognizable float image
    REQUIRE((*act)->send(run).is_ok());
    auto r = next_message(**act);
    REQUIRE(r.is_ok());
    replies.add(r.value());
    REQUIRE(std::holds_alternative<StageResult>(r.value()));
    CHECK(std::get<StageResult>(r.value()).status == ErrorCode::kStaleEpoch);
    CHECK(std::get<StageResult>(r.value()).activations.data.empty());  // an error result returns no activations
  }

  // ---- the Node is still alive and serving ----------------------------------------------------------------
  REQUIRE((*control)->send(Ping{77}).is_ok());
  auto pong = next_message(**control);
  REQUIRE(pong.is_ok());
  CHECK(std::holds_alternative<Pong>(pong.value()));
  CHECK(worker->status().state == node::NodeState::kPreparing);

  // ---- nothing echoed -------------------------------------------------------------------------------------
  const Bytes float_image = {0xDB, 0x0F, 0x49, 0x40};
  REQUIRE(replies.encoded.size() > 12);
  for (const auto& e : replies.encoded) {
    CHECK_FALSE(contains(e, kSentinel));
    CHECK_FALSE(contains(e, std::string("SENTINEL")));
    CHECK_FALSE(contains(e, float_image));
  }
  for (const auto& t : replies.texts) {
    CAPTURE(t);
    CHECK(t.find("SENTINEL") == std::string::npos);
    CHECK(t.size() < 200);  // short diagnostic sentences, not dumps
  }

  (*control)->close();
  worker->stop();
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
