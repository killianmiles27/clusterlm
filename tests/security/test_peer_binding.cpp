// Peer identity binding on a mutual-TLS Node:
//   * a direct peer must match the AuthorizePeer identity (a different paired identity is refused, even if it
//     claims the authorized device id in its Hello);
//   * a peer Node is trusted only dynamically and for one lease, so it can never claim the Father role;
//   * the dynamic trust is revoked when the lease ends.
#include <doctest/doctest.h>

#include <filesystem>
#include <memory>

#include "../privacy/message_samples.hpp"
#include "../privacy/raw_client.hpp"

using namespace clusterlm;
using namespace clusterlm::testing;
using namespace clusterlm::protocol;

namespace {

std::shared_ptr<const transport::DeviceIdentity> make_identity(const char* name) {
  auto id = transport::DeviceIdentity::generate(name);
  REQUIRE(id.is_ok());
  return std::make_shared<const transport::DeviceIdentity>(std::move(id).value());
}

transport::SecurityConfig tls(std::shared_ptr<const transport::DeviceIdentity> self, std::vector<std::string> trusted) {
  transport::SecurityConfig c;
  c.mode = transport::SecurityConfig::Mode::kMutualTls;
  c.identity = std::move(self);
  c.trusted_peers = std::move(trusted);
  return c;
}

// True if the Node answered the Hello with HelloAck.
bool admitted(protocol::MessageStream& s) {
  auto m = next_message(s);
  return m.is_ok() && std::holds_alternative<HelloAck>(m.value());
}

}  // namespace

TEST_CASE("direct peers are bound to the AuthorizePeer identity and can never claim to be Father") {
  auto father = make_identity("father");
  auto node = make_identity("node");
  auto peer = make_identity("authorized-peer");
  auto rogue = make_identity("paired-but-unauthorized");

  const auto dir = privacy_temp_dir("peer-binding");
  node::NodeConfig nc;
  nc.name = "node";
  // As clusterlm-node does, the Node trusts Father through the runtime trust store (--trust) and also pairs with
  // `rogue`, e.g. a second paired machine. `peer` is known only through AuthorizePeer.
  nc.security = tls(node, {rogue->fingerprint()});
  nc.security.trust(father->fingerprint());
  nc.staging_root = dir / "staging";
  nc.ram_allowance = 1ull << 30;
  nc.vram_allowance = 1ull << 30;
  auto started = node::NodeWorker::start(nc);
  REQUIRE(started.is_ok());
  auto worker = std::move(started).value();
  const transport::Endpoint ep = worker->endpoint();
  const LeaseGeneration lease{worker->status().lease_generation};
  const auto father_sec = tls(father, {node->fingerprint()});
  const auto peer_sec = tls(peer, {node->fingerprint()});
  const auto rogue_sec = tls(rogue, {node->fingerprint()});

  // Father: control channel, then a plan (so the Node has something to authorize peers for).
  auto control = open_stream(ep, father_sec, Channel::kControl, NodeRole::kFather, father->fingerprint(), lease,
                             node->fingerprint());
  REQUIRE(control.is_ok());
  REQUIRE(admitted(**control));
  REQUIRE(std::holds_alternative<OfferResources>(next_message(**control).value()));

  const objects::ModelManifest manifest = sample_manifest();
  PreparePlan plan;
  plan.lease = lease;
  plan.model_root = manifest.root_hash();
  plan.backend_build = "reference-cpu-fp32-v1";
  plan.plan_hash = sample_digest(5);
  plan.stages.push_back(StageAssignment{StageId{1}, domain::StageRole::kMiddle, objects::LayerRange{3, 4}, 64, 8, 4});
  plan.manifest = manifest;
  for (std::uint32_t i = 0; i < manifest.objects.size(); ++i) {
    const auto& o = manifest.objects[i];
    if (!o.father_only() && o.layer && *o.layer == 3)
      plan.assignments.push_back(ObjectAssignment{i, objects::AllocationTarget::kCpuResident});
  }
  REQUIRE((*control)->send(plan).is_ok());
  REQUIRE(std::holds_alternative<PlanAccepted>(next_message(**control).value()));

  // Father authorizes `peer` as the upstream sender of stage 1.
  REQUIRE((*control)->send(AuthorizePeer{lease, plan.plan_hash, StageId{0}, StageId{1}, peer->fingerprint(),
                                         "127.0.0.1:1", lease}).is_ok());
  auto auth_reply = next_message(**control);
  REQUIRE(auth_reply.is_ok());
  REQUIRE(!std::holds_alternative<ErrorMessage>(auth_reply.value()));

  SUBCASE("the authorized peer is admitted as a Node") {
    auto s = open_stream(ep, peer_sec, Channel::kActivation, NodeRole::kNode, peer->fingerprint(), lease, node->fingerprint());
    REQUIRE(s.is_ok());
    CHECK(admitted(**s));
  }

  SUBCASE("a different paired identity is refused, even when its Hello claims the authorized device id") {
    auto s = open_stream(ep, rogue_sec, Channel::kActivation, NodeRole::kNode, /*claimed=*/peer->fingerprint(), lease,
                         node->fingerprint());
    REQUIRE(s.is_ok());
    auto m = next_message(**s);
    REQUIRE(m.is_ok());
    REQUIRE(std::holds_alternative<ErrorMessage>(m.value()));
    CHECK(std::get<ErrorMessage>(m.value()).code == ErrorCode::kPermissionDenied);
  }

  SUBCASE("Father itself is not an authorized upstream peer either") {
    auto s = open_stream(ep, father_sec, Channel::kActivation, NodeRole::kNode, father->fingerprint(), lease, node->fingerprint());
    REQUIRE(s.is_ok());
    auto m = next_message(**s);
    REQUIRE(m.is_ok());
    CHECK(std::holds_alternative<ErrorMessage>(m.value()));
  }

  SUBCASE("a dynamically trusted peer cannot claim the Father role on any channel") {
    for (Channel ch : {Channel::kControl, Channel::kActivation, Channel::kProvision}) {
      auto s = open_stream(ep, peer_sec, ch, NodeRole::kFather, peer->fingerprint(), lease, node->fingerprint());
      REQUIRE(s.is_ok());
      auto m = next_message(**s, std::chrono::milliseconds(2000));
      // The Node drops the connection: no HelloAck and no state change.
      CHECK_FALSE(m.is_ok());
    }
    // The real Father's control channel is untouched.
    REQUIRE((*control)->send(Ping{9}).is_ok());
    auto pong = next_message(**control);
    REQUIRE(pong.is_ok());
    CHECK(std::holds_alternative<Pong>(pong.value()));
  }

  SUBCASE("an unpaired identity cannot even complete the TLS handshake") {
    auto stranger = make_identity("stranger");
    auto s = open_stream(ep, tls(stranger, {node->fingerprint()}), Channel::kActivation, NodeRole::kNode,
                         stranger->fingerprint(), lease, node->fingerprint());
    if (s.is_ok()) {
      auto m = next_message(**s, std::chrono::milliseconds(2000));
      CHECK_FALSE(m.is_ok());
    }
  }

  SUBCASE("the dynamic trust ends with the lease") {
    REQUIRE((*control)->send(ReleaseLease{lease, ReleaseReason::kFatherRequest}).is_ok());
    auto done = next_message(**control, std::chrono::milliseconds(5000));
    REQUIRE(done.is_ok());
    REQUIRE(std::holds_alternative<ReleaseComplete>(done.value()));
    const LeaseGeneration next{worker->status().lease_generation};
    auto s = open_stream(ep, peer_sec, Channel::kActivation, NodeRole::kNode, peer->fingerprint(), next, node->fingerprint());
    if (s.is_ok()) {
      auto m = next_message(**s, std::chrono::milliseconds(2000));
      CHECK_FALSE(m.is_ok());  // TLS refuses the no-longer-trusted identity
    }
  }

  (*control)->close();
  worker->stop();
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
