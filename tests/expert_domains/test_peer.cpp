// LAN peer mode of the experimental expert-domain prototype (HQ-P0C-02), tested over loopback: domains as separate
// serving loops reached through the production transport (insecure loopback and mutual TLS with pinned fingerprints),
// and the quantized (Strata IQ) expert-kernel mode.
#include <doctest/doctest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <future>
#include <thread>

#include "../objects/test_util.hpp"
#include "clusterlm/expert_domains/peer.hpp"
#include "clusterlm/expert_domains/quant_experts.hpp"
#include "clusterlm/expert_domains/rig.hpp"
#include "clusterlm/expert_domains/simulation.hpp"

using namespace clusterlm;
using namespace clusterlm::expert_domains;
using namespace clusterlm::testutil;

namespace {

std::int32_t tok(std::uint64_t i, std::uint32_t vocab) { return static_cast<std::int32_t>((i * 31u + 7u) % vocab); }

// A domain serving loop on a thread, as a separate process would run it (it loads its own experts).
struct PeerDomain {
  std::atomic<bool> stop{false};
  std::promise<transport::Endpoint> bound;
  std::thread thread;
  Status result;
  std::string device_id;

  PeerDomain(const Fixture& f, std::uint32_t index, std::uint32_t remotes, transport::SecurityConfig sec, std::uint32_t sessions,
             ExpertKernelSpec kernel = {}) {
    PeerServeOptions o;
    o.fixture = f.spec;
    o.model_dir = f.dir.path();
    o.listen = transport::Endpoint{"127.0.0.1", 0};
    o.security = std::move(sec);
    o.remote_domains = remotes;
    o.domain_index = index;
    o.kernel = std::move(kernel);
    o.max_sessions = sessions;
    o.stop = &stop;
    o.on_listening = [this](const transport::Endpoint& e, const std::string& id) {
      device_id = id;
      bound.set_value(e);
    };
    thread = std::thread([this, o] {
      result = serve_peer(o);
      // A failed start (e.g. the kernels are not built) must not leave the test waiting for the endpoint.
      if (!result.is_ok()) try { bound.set_exception(std::make_exception_ptr(std::runtime_error(result.to_string()))); } catch (...) {}
    });
  }
  ~PeerDomain() {
    stop = true;
    if (thread.joinable()) thread.join();
  }
  transport::Endpoint endpoint() { return bound.get_future().get(); }
};

transport::SecurityConfig insecure() {
  transport::SecurityConfig s;
  s.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  return s;
}

transport::SecurityConfig tls(const TempDir& dir, const std::string& name, const std::vector<std::string>& trusted) {
  auto id = transport::DeviceIdentity::load_or_generate(dir.path() / name, name);
  REQUIRE(id.is_ok());
  transport::SecurityConfig s;
  s.mode = transport::SecurityConfig::Mode::kMutualTls;
  s.identity = std::make_shared<const transport::DeviceIdentity>(std::move(id).value());
  s.trusted_peers = trusted;
  return s;
}

std::vector<std::vector<float>> run_schedule(GroupedRig& rig, std::uint32_t vocab) {
  std::vector<std::vector<float>> out;
  std::uint64_t t = 0;
  for (std::uint32_t q : {3u, 1u, 4u, 2u, 4u}) {
    std::vector<std::int32_t> tokens;
    for (std::uint32_t i = 0; i < q; ++i) tokens.push_back(tok(t++, vocab));
    auto lg = rig.executor().run_window(tokens);
    REQUIRE_MESSAGE(lg.is_ok(), lg.status().to_string());
    out.push_back(lg->data);
    REQUIRE(rig.executor().commit(q).is_ok());
  }
  return out;
}

RigOptions rig_options(const objects::CanonicalModelStore& store, std::uint32_t remotes) {
  RigOptions o;
  o.store = &store;
  o.remote_domains = remotes;
  o.max_context = 64;
  o.max_window = 8;
  o.layer_timeout = std::chrono::milliseconds(10'000);
  return o;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("peer specs parse NAME=HOST:PORT[@FINGERPRINT] and reject the rest") {
  auto p = parse_peer("d1=10.0.0.5:7500");
  REQUIRE(p.is_ok());
  CHECK(p->name == "d1");
  CHECK(p->endpoint.host == "10.0.0.5");
  CHECK(p->endpoint.port == 7500);
  CHECK_FALSE(p->expected_id.has_value());
  auto pinned = parse_peer("g14=192.168.1.20:7500@abcdef");
  REQUIRE(pinned.is_ok());
  CHECK(pinned->expected_id.value() == "abcdef");
  CHECK_FALSE(parse_peer("no-equals").is_ok());
  CHECK_FALSE(parse_peer("=1.2.3.4:5").is_ok());
  CHECK_FALSE(parse_peer("d=1.2.3.4:5@").is_ok());
  CHECK_FALSE(parse_peer("d=not-an-endpoint").is_ok());
}

TEST_CASE("domains as separate serving loops over loopback give the same bits as in-process domains") {
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  const std::uint32_t vocab = f.manifest.geometry.vocab_size;

  // Reference: the same ownership with the domains as threads of this process.
  auto local = GroupedRig::create(rig_options(*store, 2));
  REQUIRE_MESSAGE(local.is_ok(), local.status().to_string());
  const auto want = run_schedule(**local, vocab);

  // Peers: each domain loads its own experts from its own copy of the model and serves two Father sessions.
  PeerDomain d1(f, 1, 2, insecure(), 2), d2(f, 2, 2, insecure(), 2);
  const auto e1 = d1.endpoint(), e2 = d2.endpoint();
  for (int session = 0; session < 2; ++session) {  // the second session proves a domain outlives its Fathers
    CAPTURE(session);
    RigOptions o = rig_options(*store, 2);
    o.connect_peers = [&] { return connect_peers({{"d1", e1, std::nullopt}, {"d2", e2, std::nullopt}}, insecure()); };
    auto rig = GroupedRig::create(o);
    REQUIRE_MESSAGE(rig.is_ok(), rig.status().to_string());
    CHECK((*rig)->remote_count() == 0);  // no in-process servers: every domain is a peer
    const auto got = run_schedule(**rig, vocab);
    REQUIRE(got.size() == want.size());
    for (std::size_t w = 0; w < want.size(); ++w) CHECK_MESSAGE(bit_equal(got[w], want[w]), "window " << w);
    (*rig)->shutdown();
  }
}

TEST_CASE("peer mode over mutual TLS with pinned fingerprints; an untrusted Father is refused") {
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  TempDir ids("peer-ids");
  transport::SecurityConfig father = tls(ids, "father", {});
  transport::SecurityConfig d1_sec = tls(ids, "d1", {father.identity->fingerprint()});
  transport::SecurityConfig d2_sec = tls(ids, "d2", {father.identity->fingerprint()});
  father.trusted_peers = {d1_sec.identity->fingerprint(), d2_sec.identity->fingerprint()};

  PeerDomain d1(f, 1, 2, d1_sec, 0), d2(f, 2, 2, d2_sec, 0);
  const auto e1 = d1.endpoint(), e2 = d2.endpoint();
  CHECK(d1.device_id == d1_sec.identity->fingerprint());

  // An identity the domains do not trust cannot even connect (and the domain keeps listening).
  transport::SecurityConfig stranger = tls(ids, "stranger", {d1_sec.identity->fingerprint()});
  auto refused = connect_peers({{"d1", e1, d1_sec.identity->fingerprint()}}, stranger, std::chrono::seconds(3));
  if (refused.is_ok()) {
    // TLS 1.3: the client's handshake can complete before the server rejects the client certificate; the first
    // use of the link then fails.
    transport::Frame probe;
    probe.type = kMsgExpertBatch;
    probe.channel = kExpertChannel;
    (void)refused.value().front().connection->send(probe);
    CHECK_FALSE(refused.value().front().connection->receive(std::chrono::seconds(2)).is_ok());
  }

  // A Father naming the wrong expected fingerprint for a domain is refused.
  auto wrong = connect_peers({{"d1", e1, std::string(64, 'a')}}, father, std::chrono::seconds(3));
  CHECK_FALSE(wrong.is_ok());

  RigOptions o = rig_options(*store, 2);
  o.connect_peers = [&] {
    return connect_peers({{"d1", e1, d1_sec.identity->fingerprint()}, {"d2", e2, d2_sec.identity->fingerprint()}}, father);
  };
  auto rig = GroupedRig::create(o);
  REQUIRE_MESSAGE(rig.is_ok(), rig.status().to_string());
  const auto got = run_schedule(**rig, f.manifest.geometry.vocab_size);
  auto local = GroupedRig::create(rig_options(*store, 2));
  REQUIRE(local.is_ok());
  const auto want = run_schedule(**local, f.manifest.geometry.vocab_size);
  for (std::size_t w = 0; w < want.size(); ++w) CHECK(bit_equal(got[w], want[w]));
  (*rig)->shutdown();
}

TEST_CASE("peer mode refuses a mismatching domain count and non-loopback insecure peers") {
  const Fixture& f = full_fixture();
  auto store = f.open_store();
  PeerDomain d1(f, 1, 2, insecure(), 1);
  const auto e1 = d1.endpoint();
  RigOptions o = rig_options(*store, 2);  // two remote domains configured, one peer connected
  o.connect_peers = [&] { return connect_peers({{"d1", e1, std::nullopt}}, insecure()); };
  auto rig = GroupedRig::create(o);
  REQUIRE_FALSE(rig.is_ok());
  CHECK(rig.status().code() == ErrorCode::kInvalidArgument);

  // Insecure mode is loopback-only by construction: a LAN address is refused before any connection is made.
  auto lan = connect_peers({{"far", transport::Endpoint{"192.0.2.10", 7500}, std::nullopt}}, insecure(), std::chrono::seconds(1));
  CHECK_FALSE(lan.is_ok());
  CHECK(lan.status().message().find("far") != std::string::npos);  // the error names the peer

  PeerServeOptions bad;
  bad.fixture = f.spec;
  bad.model_dir = f.dir.path();
  bad.listen = transport::Endpoint{"127.0.0.1", 0};
  bad.security = insecure();
  bad.remote_domains = 2;
  bad.domain_index = 3;  // out of 1..2
  CHECK(serve_peer(bad).code() == ErrorCode::kInvalidArgument);
}

TEST_CASE("the simulation in peer mode runs against live domains, rejects impairment presets, and stays Synthetic") {
  const Fixture& f = tiny_fixture();
  PeerDomain d1(f, 1, 2, insecure(), 1), d2(f, 2, 2, insecure(), 1);
  const auto e1 = d1.endpoint(), e2 = d2.endpoint();

  SimulationOptions opt;
  opt.fixture = f.spec;
  opt.q_values = {2};
  opt.presets = {"unlimited"};
  opt.windows = 3;
  opt.prompt_len = 4;
  opt.layer_domain = false;
  opt.peers = {{"d1", e1, std::nullopt}, {"d2", e2, std::nullopt}};
  opt.peer_security = insecure();
  opt.work_dir = TempDir("peer-sim").path();
  bench::BenchmarkResult r("dev-peer", bench::probe_host());
  const Status st = run_simulation(opt, r);
  REQUIRE_MESSAGE(st.is_ok(), st.to_string());
  const auto doc = r.finish(0);
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["configuration"]["peer_mode"] == true);
  CHECK(doc["configuration"]["remote_domains"] == 2);
  CHECK(doc["configuration"]["peer_transport"] == "insecure-loopback");
  CHECK(doc["simulated"]["localhost_cluster"] == true);  // both peers are loopback
  CHECK(doc["metrics"].contains("grouped.unlimited.q2.layer_barrier_wait_ms"));
  for (const auto& c : doc["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>());

  SimulationOptions impaired = opt;
  impaired.presets = {"gige-simulated"};
  bench::BenchmarkResult r2("dev-peer", bench::probe_host());
  const Status bad = run_simulation(impaired, r2);
  REQUIRE_FALSE(bad.is_ok());
  CHECK(bad.message().find("unlimited") != std::string::npos);
}

TEST_CASE("quantized expert kernels: parse, and either run (Strata CPU build) or fail with the stub's reason") {
  CHECK(parse_expert_kernel("fixture").value().representation.empty());
  CHECK(parse_expert_kernel("").value().representation.empty());
  CHECK(parse_expert_kernel("iq3_s").value().representation == "iq3_s");
  CHECK(parse_expert_kernel("strata-iq2_xs").value().representation == "iq2_xs");
  CHECK_FALSE(parse_expert_kernel("q4_0").is_ok());

  auto spec = parse_expert_kernel("iq3_s", 7).value();
  auto qe = QuantExperts::create(spec, 256, 256, 0, 2, 4, 1);
  if (!qe.is_ok()) {
    // No Strata CPU kernels in this build: a precise error, never a silent FP32 fallback.
    CHECK(qe.status().code() == ErrorCode::kHardwareUnavailable);
    return;
  }
  // Built with the kernels: deterministic, finite, input-dependent, and weights scale the result.
  std::vector<float> h(256), y1(256, 0.0f), y2(256, 0.0f), y3(256, 0.0f);
  for (std::size_t i = 0; i < h.size(); ++i) h[i] = std::sin(0.1f * static_cast<float>(i)) * 0.5f;
  REQUIRE((*qe)->accumulate(1, 2, h.data(), 1.0f, y1.data()).is_ok());
  REQUIRE((*qe)->accumulate(1, 2, h.data(), 1.0f, y2.data()).is_ok());
  REQUIRE((*qe)->accumulate(1, 2, h.data(), 0.5f, y3.data()).is_ok());
  bool any_nonzero = false;
  for (std::size_t i = 0; i < y1.size(); ++i) {
    CHECK(std::isfinite(y1[i]));
    any_nonzero |= y1[i] != 0.0f;
    CHECK(y1[i] == y2[i]);
    CHECK(y3[i] == doctest::Approx(0.5f * y1[i]).epsilon(1e-6));
  }
  CHECK(any_nonzero);
  CHECK((*qe)->resident_bytes() > 0);
  CHECK_FALSE((*qe)->kernel_path().empty());
  CHECK_FALSE((*qe)->accumulate(5, 0, h.data(), 1.0f, y1.data()).is_ok());  // layer not held
}

TEST_CASE("grouped windows run through the quantized kernels in peer mode (Strata CPU build only)") {
  // Needs hidden and expert_ff multiples of 256: a larger-than-default fixture, so it is skipped quickly otherwise.
  auto probe = QuantExperts::create(parse_expert_kernel("iq3_s").value(), 256, 256, 0, 1, 1, 0);
  if (!probe.is_ok()) {
    CHECK(probe.status().code() == ErrorCode::kHardwareUnavailable);
    return;
  }
  objects::FixtureSpec spec = objects::FixtureSpec::tiny();
  spec.hidden = 256;
  spec.expert_ff = 256;
  spec.shared_expert_ff = 64;
  spec.n_experts = 8;
  spec.n_active = 2;
  Fixture f(spec, "quant-peer");
  auto store = f.open_store();
  const ExpertKernelSpec kernel = parse_expert_kernel("iq3_s", 3).value();
  PeerDomain d1(f, 1, 1, insecure(), 1, kernel);
  const auto e1 = d1.endpoint();
  RigOptions o = rig_options(*store, 1);
  o.kernel = kernel;
  o.connect_peers = [&] { return connect_peers({{"d1", e1, std::nullopt}}, insecure()); };
  auto rig = GroupedRig::create(o);
  REQUIRE_MESSAGE(rig.is_ok(), rig.status().to_string());
  std::vector<std::int32_t> tokens = {1, 2, 3};
  auto lg = (*rig)->executor().run_window(tokens);
  REQUIRE_MESSAGE(lg.is_ok(), lg.status().to_string());
  for (float v : lg->data) CHECK(std::isfinite(v));
}
