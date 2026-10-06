// Pairing: code handling, SPAKE2, and the full protocol over real TLS sockets, including wrong code, lockout,
// relay (MITM) attempts, replay and timeouts.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <set>
#include <thread>

#include "clusterlm/pairing/pairing.hpp"
#include "clusterlm/pairing/spake2.hpp"

using namespace clusterlm;
using namespace clusterlm::pairing;
using namespace std::chrono_literals;

namespace {

std::shared_ptr<const transport::DeviceIdentity> make_identity(const char* name) {
  auto id = transport::DeviceIdentity::generate(name);
  REQUIRE(id.is_ok());
  return std::make_shared<const transport::DeviceIdentity>(std::move(id).value());
}

constexpr const char* kCode = "ABCD-2345";

ResponderOptions node_options(std::shared_ptr<const transport::DeviceIdentity> id, const char* code = kCode) {
  ResponderOptions o;
  o.identity = std::move(id);
  o.listen = {"127.0.0.1", 0};
  o.self = {"G14", "node", 47600};
  o.code = code;
  o.window = 20s;
  o.step_timeout = 2s;
  return o;
}

const DeviceInfo kFatherInfo{"Father", "father", 0};

Status wrong_code_attempt(const transport::Endpoint& ep, const std::shared_ptr<const transport::DeviceIdentity>& id,
                          const char* code) {
  auto r = pair_with(ep, id, code, kFatherInfo, 5s);
  return r.is_ok() ? Status::ok() : r.status();
}

}  // namespace

TEST_CASE("codes: 40 bits of base32, normalized leniently, rejected strictly") {
  std::set<std::string> seen;
  for (int i = 0; i < 64; ++i) {
    const auto c = generate_code();
    REQUIRE(c.size() == 9);
    CHECK(c[4] == '-');
    auto n = normalize_code(c);
    REQUIRE(n.is_ok());
    CHECK(n->size() == kCodeChars);
    seen.insert(c);
  }
  CHECK(seen.size() == 64);  // no repeats in 64 draws from 2^40
  CHECK(normalize_code("abcd 2345").value() == "ABCD2345");
  CHECK_FALSE(normalize_code("ABCD-234").is_ok());     // too short
  CHECK_FALSE(normalize_code("ABCD-23451").is_ok());   // too long
  CHECK_FALSE(normalize_code("ABCD-2341").is_ok());    // '1' is not in the alphabet
  CHECK(short_fingerprint(std::string(64, 'a')) == "aaaa-aaaa-aaaa");
}

TEST_CASE("SPAKE2: equal codes and equal channel bindings agree; anything else does not") {
  const Bytes binding{1, 2, 3};
  auto a = Spake2::create(Spake2::Role::kInitiator, "ABCD2345");
  auto b = Spake2::create(Spake2::Role::kResponder, "ABCD2345");
  REQUIRE(a.is_ok());
  REQUIRE(b.is_ok());
  CHECK(a->share().size() == kSpake2ShareBytes);
  auto ka = a->finish(b->share(), binding);
  auto kb = b->finish(a->share(), binding);
  REQUIRE(ka.is_ok());
  REQUIRE(kb.is_ok());
  CHECK(ka->confirm_initiator == kb->confirm_initiator);
  CHECK(ka->confirm_responder == kb->confirm_responder);
  CHECK(ka->session_key == kb->session_key);
  CHECK(ka->confirm_initiator != ka->confirm_responder);

  auto c = Spake2::create(Spake2::Role::kResponder, "ABCD2346");  // one character off
  REQUIRE(c.is_ok());
  auto kc = c->finish(a->share(), binding);
  auto ka_c = a->finish(c->share(), binding);
  REQUIRE(kc.is_ok());
  REQUIRE(ka_c.is_ok());
  CHECK(kc->confirm_initiator != ka_c->confirm_initiator);

  const Bytes other_binding{9, 9, 9};
  auto kb2 = b->finish(a->share(), other_binding);
  REQUIRE(kb2.is_ok());
  CHECK(kb2->confirm_initiator != ka->confirm_initiator);  // fingerprints / exporter are bound in

  Bytes bad = a->share();
  bad[10] ^= 0x55;  // off the curve (overwhelmingly)
  CHECK_FALSE(b->finish(bad, binding).is_ok());
  CHECK_FALSE(b->finish(Bytes(65, 0), binding).is_ok());
  CHECK_FALSE(b->finish(Bytes(10, 4), binding).is_ok());

  // Fresh ephemeral secrets every run.
  auto a2 = Spake2::create(Spake2::Role::kInitiator, "ABCD2345");
  REQUIRE(a2.is_ok());
  CHECK(a2->share() != a->share());
}

TEST_CASE("pairing succeeds with the right code; each side learns the other's fingerprint and info") {
  auto node_id = make_identity("node");
  auto father_id = make_identity("father");
  auto responder = PairingResponder::start(node_options(node_id));
  REQUIRE(responder.is_ok());
  CHECK(responder.value()->code() == kCode);
  CHECK(responder.value()->state() == PairingState::kListening);

  auto peer = pair_with(responder.value()->endpoint(), father_id, "abcd-2345", kFatherInfo, 5s);
  REQUIRE_MESSAGE(peer.is_ok(), peer.status().to_string());
  CHECK(peer->fingerprint == node_id->fingerprint());
  CHECK(peer->info.name == "G14");
  CHECK(peer->info.role == "node");
  CHECK(peer->info.data_port == 47600);
  CHECK(peer->remote_host == "127.0.0.1");

  auto on_node = responder.value()->wait(2s);
  REQUIRE(on_node.is_ok());
  CHECK(on_node->fingerprint == father_id->fingerprint());
  CHECK(on_node->info.name == "Father");
  CHECK(on_node->info.role == "father");
  CHECK(responder.value()->state() == PairingState::kPaired);
  CHECK(responder.value()->failures() == 0);

  // Single use: pairing mode has ended and the port is closed.
  CHECK_FALSE(wrong_code_attempt(responder.value()->endpoint(), father_id, kCode).is_ok());
}

TEST_CASE("a wrong code is rejected without ending pairing mode; the right code still works afterwards") {
  auto node_id = make_identity("node");
  auto father_id = make_identity("father");
  auto responder = PairingResponder::start(node_options(node_id));
  REQUIRE(responder.is_ok());
  const auto ep = responder.value()->endpoint();

  auto st = wrong_code_attempt(ep, father_id, "ZZZZ-2222");
  CHECK(st.code() == ErrorCode::kUnauthenticated);
  CHECK(responder.value()->failures() == 1);
  CHECK(responder.value()->state() == PairingState::kListening);

  auto peer = pair_with(ep, father_id, kCode, kFatherInfo, 5s);
  REQUIRE_MESSAGE(peer.is_ok(), peer.status().to_string());
  CHECK(responder.value()->wait(2s).is_ok());
}

TEST_CASE("three failures lock pairing mode: even the correct code is refused afterwards") {
  auto node_id = make_identity("node");
  auto father_id = make_identity("father");
  auto responder = PairingResponder::start(node_options(node_id));
  REQUIRE(responder.is_ok());
  const auto ep = responder.value()->endpoint();
  for (int i = 0; i < 3; ++i) CHECK(wrong_code_attempt(ep, father_id, "ZZZZ-2222").code() == ErrorCode::kUnauthenticated);
  auto waited = responder.value()->wait(3s);
  REQUIRE_FALSE(waited.is_ok());
  CHECK(waited.status().code() == ErrorCode::kUnauthenticated);
  CHECK(responder.value()->state() == PairingState::kLocked);
  CHECK(responder.value()->failures() == 3);
  CHECK_FALSE(wrong_code_attempt(ep, father_id, kCode).is_ok());  // listener is gone
}

namespace {

// A relay that terminates TLS with ITS OWN certificate toward both sides and forwards every pairing frame
// unchanged. It does not know the code, and does not need to: it is the strongest passive-looking splice.
class Relay {
 public:
  Relay(transport::Endpoint upstream, std::shared_ptr<const transport::DeviceIdentity> id)
      : upstream_(std::move(upstream)), id_(std::move(id)) {
    transport::SecurityConfig sec;
    sec.identity = id_;
    sec.pairing_channel = true;
    auto l = transport::listen({"127.0.0.1", 0}, sec);
    REQUIRE(l.is_ok());
    listener_ = std::move(l).value();
    thread_ = std::thread([this] { run(); });
  }
  ~Relay() {
    stop_ = true;
    listener_->close();
    if (thread_.joinable()) thread_.join();
  }
  transport::Endpoint endpoint() const { return listener_->local_endpoint(); }
  int forwarded() const { return forwarded_.load(); }

 private:
  void run() {
    auto client = listener_->accept(5s);
    if (!client.is_ok()) return;
    transport::SecurityConfig sec;
    sec.identity = id_;
    sec.pairing_channel = true;
    auto server = transport::connect(upstream_, sec, std::nullopt, 5s);
    if (!server.is_ok()) return;
    auto pump = [this](transport::Connection& from, transport::Connection& to) {
      while (!stop_.load()) {
        auto f = from.receive(100ms);
        if (!f.is_ok()) {
          if (f.status().code() == ErrorCode::kDeadlineExceeded) continue;
          to.close();
          return;
        }
        ++forwarded_;
        if (!to.send(f.value()).is_ok()) return;
      }
    };
    std::thread back([&] { pump(*server.value(), *client.value()); });
    pump(*client.value(), *server.value());
    client.value()->close();
    server.value()->close();
    back.join();
  }

  transport::Endpoint upstream_;
  std::shared_ptr<const transport::DeviceIdentity> id_;
  std::unique_ptr<transport::Listener> listener_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<int> forwarded_{0};
};

}  // namespace

TEST_CASE("a TLS relay with its own certificate cannot pair the two devices, even relaying every message verbatim") {
  auto node_id = make_identity("node");
  auto father_id = make_identity("father");
  auto mallory = make_identity("mallory");
  auto responder = PairingResponder::start(node_options(node_id));
  REQUIRE(responder.is_ok());
  {
    Relay relay(responder.value()->endpoint(), mallory);
    // Father uses the CORRECT code; the relay merely sits in the middle.
    auto r = pair_with(relay.endpoint(), father_id, kCode, kFatherInfo, 5s);
    REQUIRE_FALSE(r.is_ok());
    CHECK(r.status().code() == ErrorCode::kUnauthenticated);
    CHECK(relay.forwarded() >= 2);  // the exchange really went through it
  }
  // The Node saw a failed confirmation (fingerprints/exporter differ per leg) and did not pair with anyone.
  CHECK(responder.value()->failures() == 1);
  CHECK(responder.value()->state() == PairingState::kListening);
  // The direct path still works: the relay cost the user one of the three attempts, nothing more.
  auto peer = pair_with(responder.value()->endpoint(), father_id, kCode, kFatherInfo, 5s);
  REQUIRE_MESSAGE(peer.is_ok(), peer.status().to_string());
  CHECK(peer->fingerprint == node_id->fingerprint());
}

namespace {

struct Recorded {
  transport::Frame hello, confirm;
};

// Plays the initiator by hand against `ep` with `code`, recording its two messages.
Result<Recorded> manual_initiator(const transport::Endpoint& ep, const std::shared_ptr<const transport::DeviceIdentity>& id,
                                  const std::string& code, bool complete) {
  transport::SecurityConfig sec;
  sec.identity = id;
  sec.pairing_channel = true;
  CLM_ASSIGN_OR_RETURN(auto conn, transport::connect(ep, sec, std::nullopt, 5s));
  CLM_ASSIGN_OR_RETURN(auto spake, Spake2::create(Spake2::Role::kInitiator, code));
  Recorded rec;
  ByteWriter hw;
  hw.u16(kPairingProtocolVersion);
  hw.raw(spake.share());
  rec.hello.type = wire::kHello;
  rec.hello.payload = std::move(hw).take();
  CLM_RETURN_IF_ERROR(conn->send(rec.hello));
  CLM_ASSIGN_OR_RETURN(auto share, conn->receive(5s));
  CLM_ASSIGN_OR_RETURN(auto exporter, conn->export_keying_material(wire::kExporterLabel, wire::kExporterBytes));
  CLM_ASSIGN_OR_RETURN(auto keys, spake.finish(share.payload, wire::channel_binding(id->fingerprint(), conn->peer().device_id, exporter)));
  rec.confirm.type = wire::kConfirmA;
  rec.confirm.payload = keys.confirm_initiator;
  CLM_RETURN_IF_ERROR(conn->send(rec.confirm));
  auto reply = conn->receive(5s);
  if (complete && reply.is_ok() && reply->type == wire::kConfirmB) {
    transport::Frame info;
    info.type = wire::kInfo;
    info.payload = wire::encode_info(kFatherInfo);
    CLM_RETURN_IF_ERROR(conn->send(info));
    (void)conn->receive(5s);
  }
  return rec;
}

}  // namespace

TEST_CASE("replaying a recorded legitimate exchange against a new pairing session fails") {
  auto node_id = make_identity("node");
  auto father_id = make_identity("father");
  Recorded recorded;
  {
    auto first = PairingResponder::start(node_options(node_id));
    REQUIRE(first.is_ok());
    auto rec = manual_initiator(first.value()->endpoint(), father_id, "ABCD2345", true);
    REQUIRE_MESSAGE(rec.is_ok(), rec.status().to_string());
    recorded = std::move(rec).value();
    CHECK(first.value()->wait(3s).is_ok());  // the genuine exchange paired
  }
  // A second pairing mode with the SAME code (user retried): the attacker replays the captured messages from
  // their own connection. The responder's fresh share and TLS session make the old confirmation worthless.
  auto second = PairingResponder::start(node_options(node_id));
  REQUIRE(second.is_ok());
  auto attacker = make_identity("attacker");
  transport::SecurityConfig sec;
  sec.identity = attacker;
  sec.pairing_channel = true;
  auto conn = transport::connect(second.value()->endpoint(), sec, std::nullopt, 5s);
  REQUIRE(conn.is_ok());
  REQUIRE(conn.value()->send(recorded.hello).is_ok());
  auto share = conn.value()->receive(5s);
  REQUIRE(share.is_ok());
  REQUIRE(conn.value()->send(recorded.confirm).is_ok());
  auto reply = conn.value()->receive(5s);
  REQUIRE(reply.is_ok());
  CHECK(reply->type == wire::kError);  // no ConfirmB, no pairing
  CHECK(second.value()->failures() == 1);
  CHECK(second.value()->state() == PairingState::kListening);
}

TEST_CASE("pairing mode expires after its window; a stalled client does not block a later legitimate one") {
  auto node_id = make_identity("node");
  auto father_id = make_identity("father");
  {
    auto o = node_options(node_id);
    o.window = 1s;
    auto responder = PairingResponder::start(o);
    REQUIRE(responder.is_ok());
    auto waited = responder.value()->wait(5s);
    REQUIRE_FALSE(waited.is_ok());
    CHECK(waited.status().code() == ErrorCode::kDeadlineExceeded);
    CHECK(responder.value()->state() == PairingState::kExpired);
    CHECK_FALSE(wrong_code_attempt(responder.value()->endpoint(), father_id, kCode).is_ok());
  }
  {
    auto o = node_options(node_id);
    o.step_timeout = 400ms;
    auto responder = PairingResponder::start(o);
    REQUIRE(responder.is_ok());
    // A client that completes TLS and then says nothing.
    transport::SecurityConfig sec;
    sec.identity = make_identity("idle");
    sec.pairing_channel = true;
    auto idle = transport::connect(responder.value()->endpoint(), sec, std::nullopt, 5s);
    REQUIRE(idle.is_ok());
    // The responder drops the silent client after its step timeout; wait for that instead of sleeping a fixed time
    // (a loaded machine stretches the 400 ms arbitrarily).
    CHECK_FALSE(idle.value()->receive(30s).is_ok());
    auto peer = pair_with(responder.value()->endpoint(), father_id, kCode, kFatherInfo, 5s);
    REQUIRE_MESSAGE(peer.is_ok(), peer.status().to_string());
    CHECK(responder.value()->failures() == 0);  // stalling is not a guess
  }
}

TEST_CASE("cancel ends pairing mode; malformed input is a protocol error, not a guess") {
  auto node_id = make_identity("node");
  auto responder = PairingResponder::start(node_options(node_id));
  REQUIRE(responder.is_ok());
  // Garbage hello.
  transport::SecurityConfig sec;
  sec.identity = make_identity("junk");
  sec.pairing_channel = true;
  auto conn = transport::connect(responder.value()->endpoint(), sec, std::nullopt, 5s);
  REQUIRE(conn.is_ok());
  transport::Frame f;
  f.type = wire::kHello;
  f.payload = {1, 2, 3};
  REQUIRE(conn.value()->send(f).is_ok());
  auto reply = conn.value()->receive(3s);
  REQUIRE(reply.is_ok());
  CHECK(reply->type == wire::kError);
  CHECK(responder.value()->failures() == 0);

  responder.value()->cancel();
  auto waited = responder.value()->wait(3s);
  REQUIRE_FALSE(waited.is_ok());
  CHECK(waited.status().code() == ErrorCode::kCancelled);
  CHECK(responder.value()->state() == PairingState::kCancelled);
}

TEST_CASE("a Node cannot pair with a Node: role check on both ends") {
  auto node_id = make_identity("node");
  auto other = make_identity("other-node");
  auto responder = PairingResponder::start(node_options(node_id));
  REQUIRE(responder.is_ok());
  auto r = pair_with(responder.value()->endpoint(), other, kCode, DeviceInfo{"Imposter", "node", 1}, 5s);
  CHECK_FALSE(r.is_ok());
  CHECK(r.status().code() != ErrorCode::kOk);
  CHECK(responder.value()->state() != PairingState::kPaired);
}
