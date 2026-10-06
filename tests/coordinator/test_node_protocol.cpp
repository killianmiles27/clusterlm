// A raw protocol client drives a real NodeWorker with stale, malformed and hostile requests. The Node must reject
// each one precisely and stay consistent (and clean) afterwards.
#include <doctest/doctest.h>

#include "cluster_fixture.hpp"
#include "clusterlm/domain/backend_adapter.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/protocol/wire.hpp"

#include <thread>

using namespace clusterlm;
using namespace clusterlm::testing;
using namespace std::chrono_literals;
using protocol::Channel;
using protocol::MessageStream;

namespace {

constexpr auto kT = 3s;

class RawFather {
 public:
  RawFather(node::NodeWorker& w, const std::filesystem::path& model_dir) : worker_(w) {
    auto st = objects::CanonicalModelStore::open(model_dir);
    REQUIRE(st.is_ok());
    store_ = std::move(st).value();
    build_ = domain::make_reference_backend()->info().build_hash;
  }

  std::shared_ptr<MessageStream> open(Channel ch, LeaseGeneration lease, protocol::NodeRole role = protocol::NodeRole::kFather,
                                      std::uint16_t version = protocol::kProtocolVersion, std::string device = "father") {
    transport::SecurityConfig sec;
    sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    auto c = transport::connect(worker_.endpoint(), sec, std::nullopt, kT);
    REQUIRE(c.is_ok());
    auto s = std::make_shared<MessageStream>(std::move(c).value(), ch);
    protocol::Hello h;
    h.protocol_version = version;
    h.role = role;
    h.channel = ch;
    h.device_id = std::move(device);
    h.backend_build = build_;
    h.lease = lease;
    REQUIRE(s->send(h).is_ok());
    return s;
  }

  // Control channel + HelloAck + OfferResources.
  void connect() {
    control_ = open(Channel::kControl, LeaseGeneration{0});
    auto ack = control_->expect<protocol::HelloAck>(kT);
    REQUIRE(ack.is_ok());
    auto offer = control_->expect<protocol::OfferResources>(kT);
    REQUIRE(offer.is_ok());
    lease_ = offer->lease;
  }

  protocol::PreparePlan plan(objects::LayerRange layers, domain::StageRole role = domain::StageRole::kMiddle) {
    const auto& m = store_->manifest();
    protocol::PreparePlan p;
    p.lease = lease_;
    p.model_root = m.root_hash();
    p.backend_build = build_;
    p.plan_hash = Sha256::of(std::string_view("raw-plan"));
    p.stages.push_back({StageId{1}, role, layers, 256, 16, 0});
    p.manifest.artifact_id = m.artifact_id;
    p.manifest.geometry = m.geometry;
    p.manifest.shards = m.shards;
    for (const auto* o : m.layer_objects(layers)) {
      p.assignments.push_back({static_cast<std::uint32_t>(p.manifest.objects.size()), objects::AllocationTarget::kCpuResident});
      p.manifest.objects.push_back(*o);
    }
    return p;
  }

  template <typename Reply>
  Result<Reply> call(const protocol::Message& m) {
    const auto corr = control_->next_correlation();
    CLM_RETURN_IF_ERROR(control_->send(m, corr));
    while (true) {
      CLM_ASSIGN_OR_RETURN(auto r, control_->receive(kT));
      if (r.correlation != corr) continue;  // unsolicited (offers)
      if (auto* v = std::get_if<Reply>(&r.message)) return std::move(*v);
      if (auto* e = std::get_if<protocol::ErrorMessage>(&r.message)) return make_error(e->code, e->message);
      return make_error(ErrorCode::kProtocolError, "unexpected reply");
    }
  }

  // Provision every assigned object; returns the PlanReady (or the first error).
  Result<protocol::PlanReady> provision(const protocol::PreparePlan& p) {
    auto bulk = open(Channel::kProvision, lease_);
    CLM_RETURN_IF_ERROR(bulk->expect<protocol::HelloAck>(kT).status());
    CLM_RETURN_IF_ERROR(bulk->expect<protocol::ProvisionStatus>(kT).status());
    for (const auto& a : p.assignments) {
      const auto& o = p.manifest.objects[a.object_index];
      CLM_ASSIGN_OR_RETURN(auto bytes, store_->read_object_bytes(o.name));
      protocol::ProvisionChunk c{lease_, a.object_index, 0, Sha256::of(bytes), bytes};
      CLM_RETURN_IF_ERROR(bulk->send(c));
      CLM_RETURN_IF_ERROR(bulk->send(protocol::SealObject{lease_, a.object_index, o.byte_size, o.object_digest}, 1));
      CLM_RETURN_IF_ERROR(bulk->expect<protocol::ObjectSealed>(kT).status());
    }
    while (true) {
      CLM_ASSIGN_OR_RETURN(auto r, control_->receive(kT));
      if (auto* v = std::get_if<protocol::PlanReady>(&r.message)) return *v;
      if (auto* e = std::get_if<protocol::ErrorMessage>(&r.message)) return make_error(e->code, e->message);
    }
  }

  std::shared_ptr<MessageStream> control_;
  LeaseGeneration lease_;
  std::unique_ptr<objects::CanonicalModelStore> store_;
  std::string build_;
  node::NodeWorker& worker_;
};

domain::StageActivations zeros(const objects::ModelGeometry& g, std::uint32_t positions, std::uint64_t first) {
  domain::StageActivations a;
  a.layout = domain::BoundaryLayout::for_geometry(g);
  a.positions = positions;
  a.first_position = first;
  a.data.assign(std::size_t{positions} * a.layout.floats_per_position(), 0.25f);
  return a;
}

}  // namespace

TEST_CASE("handshake: wrong protocol version, second Father and unauthorized peers are refused") {
  TestCluster cl("proto-hello", 1);
  RawFather f(cl.worker(0), cl.dir() / "model");
  auto bad = f.open(Channel::kControl, LeaseGeneration{0}, protocol::NodeRole::kFather, protocol::kProtocolVersion + 1);
  CHECK_FALSE(bad->receive(kT).is_ok());  // closed without a reply
  f.connect();
  RawFather second(cl.worker(0), cl.dir() / "model");
  auto s2 = second.open(Channel::kControl, LeaseGeneration{0});
  auto r = s2->expect<protocol::HelloAck>(kT);
  CHECK(r.status().code() == ErrorCode::kFailedPrecondition);
  // A Node (not Father) opening an activation channel for this lease without AuthorizePeer.
  auto peer = f.open(Channel::kActivation, f.lease_, protocol::NodeRole::kNode, protocol::kProtocolVersion, "intruder");
  auto pr = peer->expect<protocol::HelloAck>(kT);
  CHECK_FALSE(pr.is_ok());
}

TEST_CASE("PreparePlan admission: stale lease, wrong backend, Father-only objects, prefix roles and budgets") {
  TestCluster cl("proto-prepare", 1, {NodeOptions{}});
  RawFather f(cl.worker(0), cl.dir() / "model");
  f.connect();
  {
    auto p = f.plan({4, 8});
    p.lease = f.lease_.next();
    CHECK(f.call<protocol::PlanAccepted>(p).status().code() == ErrorCode::kStaleEpoch);
  }
  {
    auto p = f.plan({4, 8});
    p.backend_build = "some-other-build";
    CHECK(f.call<protocol::PlanAccepted>(p).status().code() == ErrorCode::kVersionMismatch);
  }
  {
    auto p = f.plan({4, 8});
    const auto* emb = f.store_->manifest().find(objects::kEmbeddingObjectName);
    REQUIRE(emb != nullptr);
    p.assignments.push_back({static_cast<std::uint32_t>(p.manifest.objects.size()), objects::AllocationTarget::kCpuResident});
    p.manifest.objects.push_back(*emb);
    CHECK(f.call<protocol::PlanAccepted>(p).status().code() == ErrorCode::kPermissionDenied);
  }
  {
    auto p = f.plan({4, 8});
    p.ram_cap_bytes = 1024;  // the Node's live policy cap is smaller than the plan
    CHECK(f.call<protocol::PlanAccepted>(p).status().code() == ErrorCode::kResourceExhausted);
  }
  {
    auto p = f.plan({4, 8});
    p.assignments.front().target = objects::AllocationTarget::kTemporaryBacking;  // no disk staging allowed
    CHECK(f.call<protocol::PlanAccepted>(p).status().code() == ErrorCode::kResourceExhausted);
  }
  CHECK(cl.worker(0).status().state == node::NodeState::kAvailable);
  CHECK(cl.worker(0).status().staging_census_bytes == 0);
  // A plan asking a Node to host the token-consuming prefix (or tail) is refused before any transfer.
  for (auto role : {domain::StageRole::kPrefix, domain::StageRole::kTail}) {
    auto p = f.plan({0, 4}, role);
    CHECK(f.call<protocol::PlanAccepted>(p).status().code() == ErrorCode::kPermissionDenied);
  }
  CHECK(cl.worker(0).status().staging_census_bytes == 0);
}

TEST_CASE("disk full during allocation fails the plan cleanly") {
  // A declared object far larger than the free space: the lease store's preallocation fails like a full disk.
  TestCluster cl("proto-disk", 1, {NodeOptions::with_disk(~std::uint64_t{0} >> 4)});
  RawFather f(cl.worker(0), cl.dir() / "model");
  f.connect();
  auto p = f.plan({4, 6});
  p.assignments.front().target = objects::AllocationTarget::kTemporaryBacking;
  p.manifest.objects[p.assignments.front().object_index].byte_size = 1ull << 50;  // 1 PiB
  p.staging_cap_bytes = ~std::uint64_t{0} >> 4;
  auto r = f.call<protocol::PlanAccepted>(p);
  CHECK_FALSE(r.is_ok());
  CHECK(cl.worker(0).status().state == node::NodeState::kAvailable);
  CHECK(cl.worker(0).status().staging_census_bytes == 0);
}

TEST_CASE("provisioning rejects corrupted chunks and objects that do not match the plan manifest") {
  TestCluster cl("proto-corrupt", 1);
  RawFather f(cl.worker(0), cl.dir() / "model");
  f.connect();
  auto p = f.plan({4, 6});
  REQUIRE(f.call<protocol::PlanAccepted>(p).is_ok());
  auto bulk = f.open(Channel::kProvision, f.lease_);
  REQUIRE(bulk->expect<protocol::HelloAck>(kT).is_ok());
  REQUIRE(bulk->expect<protocol::ProvisionStatus>(kT).is_ok());
  const auto& o = p.manifest.objects[0];
  auto bytes = f.store_->read_object_bytes(o.name).value();
  // Wrong chunk digest.
  REQUIRE(bulk->send(protocol::ProvisionChunk{f.lease_, 0, 0, Sha256::of(std::string_view("x")), bytes}).is_ok());
  CHECK(bulk->expect<protocol::ObjectSealed>(kT).status().code() == ErrorCode::kDataLoss);
  // Valid chunk with a flipped byte (consistent chunk digest): the object digest catches it at seal.
  bytes[bytes.size() / 2] ^= 0x40;
  REQUIRE(bulk->send(protocol::ProvisionChunk{f.lease_, 0, 0, Sha256::of(bytes), bytes}).is_ok());
  REQUIRE(bulk->send(protocol::SealObject{f.lease_, 0, o.byte_size, o.object_digest}, 1).is_ok());
  CHECK(bulk->expect<protocol::ObjectSealed>(kT).status().code() == ErrorCode::kDataLoss);
  // A seal that names a different digest than the plan manifest (model mismatch).
  REQUIRE(bulk->send(protocol::SealObject{f.lease_, 0, o.byte_size, Sha256::of(std::string_view("other"))}, 2).is_ok());
  CHECK(bulk->expect<protocol::ObjectSealed>(kT).status().code() == ErrorCode::kDataLoss);
  // Chunks for a stale lease.
  REQUIRE(bulk->send(protocol::ProvisionChunk{f.lease_.next(), 0, 0, Sha256::of(bytes), bytes}).is_ok());
  CHECK(bulk->expect<protocol::ObjectSealed>(kT).status().code() == ErrorCode::kStaleEpoch);
  auto rel = f.call<protocol::ReleaseComplete>(protocol::ReleaseLease{f.lease_, protocol::ReleaseReason::kFatherRequest});
  REQUIRE(rel.is_ok());
  CHECK(rel->storage_cleaned);
  CHECK(rel->residual_bytes == 0);
}

TEST_CASE("execution: stale leases and epochs, duplicate commits, window abort, release") {
  TestCluster cl("proto-exec", 1);
  const auto& g = cl.manifest().geometry;
  RawFather f(cl.worker(0), cl.dir() / "model");
  f.connect();
  auto p = f.plan({4, 8});
  REQUIRE(f.call<protocol::PlanAccepted>(p).is_ok());
  REQUIRE(f.provision(p).is_ok());
  auto act = f.open(Channel::kActivation, f.lease_);
  REQUIRE(act->expect<protocol::HelloAck>(kT).is_ok());
  const Epoch e5{5};
  const SessionId s{9};
  REQUIRE(f.call<protocol::SessionOpened>(protocol::OpenSession{e5, s}).is_ok());

  auto run = [&](LeaseGeneration lease, Epoch e, WindowId w, std::uint64_t base, StateVersion v, std::uint32_t q) {
    protocol::RunWindow r;
    r.lease = lease;
    r.request = {e, s, w, base, v, q};
    r.stage = StageId{1};
    r.activations = zeros(g, q, base);
    REQUIRE(act->send(r, 1).is_ok());
    auto res = act->expect<protocol::StageResult>(kT);
    REQUIRE(res.is_ok());
    return res.value();
  };
  CHECK(run(f.lease_.next(), e5, WindowId{1}, 0, StateVersion{0}, 2).status == ErrorCode::kStaleEpoch);  // stale lease
  CHECK(run(f.lease_, Epoch{4}, WindowId{1}, 0, StateVersion{0}, 2).status == ErrorCode::kStaleEpoch);   // stale epoch
  CHECK(run(f.lease_, e5, WindowId{1}, 3, StateVersion{0}, 2).status == ErrorCode::kFailedPrecondition);  // wrong base

  // Window 2 then abort it: the session keeps its committed state and the next window starts at position 0.
  auto w2 = run(f.lease_, e5, WindowId{2}, 0, StateVersion{0}, 3);
  REQUIRE(w2.status == ErrorCode::kOk);
  auto ab = f.call<protocol::WindowAborted>(protocol::AbortWindow{e5, s, WindowId{2}, StageId{1}});
  REQUIRE(ab.is_ok());
  CHECK(ab->ack.committed_position == 0);
  CHECK(f.call<protocol::WindowAborted>(protocol::AbortWindow{e5, s, WindowId{2}, StageId{1}}).is_ok());  // idempotent
  CHECK(f.call<protocol::WindowAborted>(protocol::AbortWindow{e5, s, WindowId{7}, StageId{1}}).status().code() ==
        ErrorCode::kFailedPrecondition);  // never admitted
  // Re-running the same window id is not allowed (no blind replay); a new id at the same base is.
  CHECK(run(f.lease_, e5, WindowId{2}, 0, StateVersion{0}, 3).status == ErrorCode::kFailedPrecondition);
  auto w3 = run(f.lease_, e5, WindowId{3}, 0, StateVersion{0}, 3);
  REQUIRE(w3.status == ErrorCode::kOk);
  CHECK(w3.activations.data == w2.activations.data);  // the aborted window left no trace

  // Commit 2 of 3; duplicates replay the identical ack; a different "duplicate" is refused.
  protocol::CommitWindow c{{e5, s, WindowId{3}, 2, StateVersion{0}}, StageId{1}};
  auto a1 = f.call<protocol::CommitAckMessage>(c);
  REQUIRE(a1.is_ok());
  CHECK(a1->ack.committed_position == 2);
  auto a2 = f.call<protocol::CommitAckMessage>(c);
  REQUIRE(a2.is_ok());
  CHECK(a2->ack == a1->ack);
  auto bad = c;
  bad.request.accepted = 3;
  CHECK(f.call<protocol::CommitAckMessage>(bad).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(cl.worker(0).status().stale_rejections >= 2);

  // Release: stale lease refused; current lease releases everything.
  CHECK(f.call<protocol::ReleaseComplete>(protocol::ReleaseLease{f.lease_.next(), protocol::ReleaseReason::kFatherRequest})
            .status()
            .code() == ErrorCode::kStaleEpoch);
  auto rel = f.call<protocol::ReleaseComplete>(protocol::ReleaseLease{f.lease_, protocol::ReleaseReason::kFatherRequest});
  REQUIRE(rel.is_ok());
  CHECK(rel->resources_released);
  CHECK(rel->storage_cleaned);
  // A captured RunWindow from the released lease is stale on the next lease.
  CHECK(cl.worker(0).status().lease_generation == f.lease_.value + 1);
}

TEST_CASE("Father loss releases the lease and deletes staged objects") {
  TestCluster cl("proto-fatherloss", 1);
  {
    RawFather f(cl.worker(0), cl.dir() / "model");
    f.connect();
    auto p = f.plan({4, 8});
    REQUIRE(f.call<protocol::PlanAccepted>(p).is_ok());
    REQUIRE(f.provision(p).is_ok());
    CHECK(cl.worker(0).status().state == node::NodeState::kReady);
    f.control_->close();  // Father disappears
  }
  bool clean = false;
  for (int i = 0; i < 100 && !clean; ++i) {
    const auto st = cl.worker(0).status();
    clean = st.state == node::NodeState::kAvailable && st.staging_census_bytes == 0 && st.leases_released == 1;
    if (!clean) std::this_thread::sleep_for(20ms);
  }
  CHECK(clean);
}
