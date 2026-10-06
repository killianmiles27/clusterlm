// Domain loss, stalls, stale epochs and malformed traffic: errors, never hangs.
#include <doctest/doctest.h>

#include <chrono>

#include "../objects/test_util.hpp"
#include "clusterlm/expert_domains/rig.hpp"

using namespace clusterlm;
using namespace clusterlm::expert_domains;
using namespace clusterlm::testutil;
using namespace std::chrono_literals;

namespace {

RigOptions quick_options(const objects::CanonicalModelStore& store) {
  RigOptions o;
  o.store = &store;
  o.max_context = 32;
  o.max_window = 4;
  o.layer_timeout = 1000ms;
  return o;
}

const std::vector<std::int32_t> kTokens = {3, 9, 27};

}  // namespace

TEST_CASE("a domain that dies mid-window fails the window with an error, quickly") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  RigOptions o = quick_options(*store);
  o.die_after_batches = {0, 5};  // domain2 answers 5 batches then vanishes without replying
  auto rig = GroupedRig::create(o);
  REQUIRE_MESSAGE(rig.is_ok(), rig.status().to_string());
  FatherExecutor& ex = (*rig)->executor();

  const auto t0 = std::chrono::steady_clock::now();
  Result<domain::Logits> r = ex.run_window(kTokens);
  // Window 1 needs more than 5 domain2 batches over 8 layers unless routing avoids domain2; keep going until it dies.
  for (int i = 0; i < 4 && r.is_ok(); ++i) {
    REQUIRE(ex.commit(static_cast<std::uint32_t>(kTokens.size())).is_ok());
    r = ex.run_window(kTokens);
  }
  const auto elapsed = std::chrono::steady_clock::now() - t0;
  REQUIRE_FALSE(r.is_ok());
  CHECK((r.status().code() == ErrorCode::kUnavailable || r.status().code() == ErrorCode::kDeadlineExceeded));
  CHECK(r.status().message().find("domain2") != std::string::npos);  // names the domain, nothing else
  CHECK(elapsed < 10s);
  CHECK(ex.broken());
  // A broken executor refuses further work instead of guessing.
  CHECK(ex.run_window(kTokens).status().code() == ErrorCode::kAborted);
  CHECK(ex.commit(1).code() == ErrorCode::kAborted);
}

TEST_CASE("a domain stopped from outside is detected as lost") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  RigOptions o = quick_options(*store);
  auto rig = GroupedRig::create(o);
  REQUIRE(rig.is_ok());
  REQUIRE((*rig)->executor().run_window(kTokens).is_ok());
  REQUIRE((*rig)->executor().commit(3).is_ok());
  (*rig)->server(0).stop();
  const auto t0 = std::chrono::steady_clock::now();
  auto r = (*rig)->executor().run_window(kTokens);
  REQUIRE_FALSE(r.is_ok());
  CHECK(std::chrono::steady_clock::now() - t0 < 10s);
  CHECK((*rig)->executor().broken());
}

TEST_CASE("a domain that is connected but silent times out instead of hanging") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const auto& g = f.manifest.geometry;
  auto pair = make_loopback_pair();  // nobody serves the domain side
  REQUIRE(pair.is_ok());
  auto assignment = ExpertAssignment::ranges(g.n_experts, {1, 1});
  REQUIRE(assignment.is_ok());
  FatherExecutorConfig cfg;
  cfg.assignment = *assignment;
  cfg.max_context = 32;
  cfg.max_window = 4;
  cfg.layer_timeout = 300ms;
  cfg.epoch = Epoch{1};
  std::vector<RemoteLink> links;
  links.push_back(RemoteLink{"silent", std::move(pair->first)});
  auto ex = FatherExecutor::create(f.manifest, *store, cfg, std::move(links));
  REQUIRE_MESSAGE(ex.is_ok(), ex.status().to_string());
  const auto t0 = std::chrono::steady_clock::now();
  auto r = (*ex)->run_window(kTokens);
  const auto elapsed = std::chrono::steady_clock::now() - t0;
  REQUIRE_FALSE(r.is_ok());
  CHECK(r.status().code() == ErrorCode::kDeadlineExceeded);
  CHECK(elapsed >= 250ms);
  CHECK(elapsed < 5s);
  CHECK((*ex)->broken());
}

TEST_CASE("a stale epoch is rejected by the domain and surfaced as an error") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  RigOptions o = quick_options(*store);
  auto rig = GroupedRig::create(o);
  REQUIRE(rig.is_ok());
  (*rig)->server(0).set_epoch(Epoch{2});  // Father still speaks epoch 1
  (*rig)->server(1).set_epoch(Epoch{2});
  auto r = (*rig)->executor().run_window(kTokens);
  REQUIRE_FALSE(r.is_ok());
  CHECK(r.status().code() == ErrorCode::kStaleEpoch);
  CHECK((*rig)->server(0).metrics().rejected + (*rig)->server(1).metrics().rejected >= 1);
}

TEST_CASE("the domain validates batches before executing them") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const auto& g = f.manifest.geometry;
  RigOptions o = quick_options(*store);
  auto rig = GroupedRig::create(o);
  REQUIRE(rig.is_ok());
  ExpertDomainServer& s = (*rig)->server(0);
  const std::size_t owned = s.config().owned_experts.size();

  ExpertBatch b;
  b.epoch = Epoch{1};
  b.window = WindowId{1};
  b.layer = 0;
  b.positions = 1;
  b.hidden = g.hidden_size;
  b.activations.assign(g.hidden_size, 0.5f);
  b.routes = {{{0, 1.0f}}};
  auto ok = s.execute(b);
  REQUIRE_MESSAGE(ok.is_ok(), ok.status().to_string());
  CHECK(ok->partial.size() == g.hidden_size);
  CHECK(ok->experts_executed == 1);

  ExpertBatch bad = b;
  bad.routes = {{{static_cast<std::uint32_t>(owned), 1.0f}}};  // one past the domain's list
  CHECK(s.execute(bad).status().code() == ErrorCode::kOutOfRange);
  bad = b;
  bad.layer = g.n_layers;
  CHECK(s.execute(bad).status().code() == ErrorCode::kOutOfRange);
  bad = b;
  bad.hidden = g.hidden_size + 1;
  CHECK(s.execute(bad).status().code() == ErrorCode::kInvalidArgument);
  bad = b;
  bad.window = WindowId{0};  // window ids may not move backwards
  CHECK(s.execute(bad).status().code() == ErrorCode::kFailedPrecondition);
  bad = b;
  bad.epoch = Epoch{9};
  CHECK(s.execute(bad).status().code() == ErrorCode::kStaleEpoch);
}

TEST_CASE("garbage on the wire is answered with an error and the domain keeps serving") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const auto& g = f.manifest.geometry;
  auto resolver = provision_expert_objects(*store, {0, 1, 2, 3}, 0, g.n_layers);
  REQUIRE(resolver.is_ok());
  ExpertDomainConfig dc;
  dc.owned_experts = {0, 1, 2, 3};
  dc.epoch = Epoch{1};
  auto server = ExpertDomainServer::create(f.manifest, **resolver, dc);
  REQUIRE_MESSAGE(server.is_ok(), server.status().to_string());
  auto pair = make_loopback_pair();
  REQUIRE(pair.is_ok());
  REQUIRE((*server)->serve(std::move(pair->second)).is_ok());
  transport::Connection& c = *pair->first;

  transport::Frame junk;
  junk.type = kMsgExpertBatch;
  junk.channel = kExpertChannel;
  junk.correlation = 11;
  junk.payload = {1, 2, 3};
  REQUIRE(c.send(junk).is_ok());
  auto reply = c.receive(2000ms);
  REQUIRE(reply.is_ok());
  CHECK(reply->type == kMsgExpertError);
  CHECK(reply->correlation == 11);

  junk.type = kMsgExpertResult;  // a message type the domain never accepts
  REQUIRE(c.send(junk).is_ok());
  reply = c.receive(2000ms);
  REQUIRE(reply.is_ok());
  CHECK(reply->type == kMsgExpertError);

  ExpertBatch b;
  b.epoch = Epoch{1};
  b.window = WindowId{1};
  b.layer = 0;
  b.positions = 1;
  b.hidden = g.hidden_size;
  b.activations.assign(g.hidden_size, 0.25f);
  b.routes = {{{1, 0.5f}, {3, 0.5f}}};
  REQUIRE(c.send(to_frame(b, 12)).is_ok());
  reply = c.receive(2000ms);
  REQUIRE(reply.is_ok());
  CHECK(reply->type == kMsgExpertResult);
  CHECK(reply->correlation == 12);
  CHECK((*server)->metrics().rejected == 2);
  (*server)->stop();
}

TEST_CASE("a domain refuses a plan that does not provision its experts") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const auto& g = f.manifest.geometry;
  auto resolver = provision_expert_objects(*store, {0, 1}, 0, g.n_layers);  // lacks experts 2 and 3
  REQUIRE(resolver.is_ok());
  ExpertDomainConfig dc;
  dc.owned_experts = {0, 1, 2, 3};
  auto server = ExpertDomainServer::create(f.manifest, **resolver, dc);
  REQUIRE_FALSE(server.is_ok());
  CHECK(server.status().code() == ErrorCode::kNotFound);  // never fetches anything outside its plan
}

TEST_CASE("a domain holds only routed experts and none of the dense, embedding or head material") {
  const Fixture& f = tiny_fixture();
  auto store = f.open_store();
  const auto& g = f.manifest.geometry;
  auto resolver = provision_expert_objects(*store, {2, 5}, 0, g.n_layers);
  REQUIRE(resolver.is_ok());
  CHECK((*resolver)->size() == 2u * g.n_layers);
  CHECK_FALSE((*resolver)->resolve(objects::dense_object_name(0)).is_ok());
  CHECK_FALSE((*resolver)->resolve(std::string(objects::kEmbeddingObjectName)).is_ok());
  CHECK_FALSE((*resolver)->resolve(std::string(objects::kHeadObjectName)).is_ok());
  CHECK_FALSE((*resolver)->resolve(objects::expert_object_name(0, 3)).is_ok());
  CHECK((*resolver)->resolve(objects::expert_object_name(0, 5)).is_ok());
}
