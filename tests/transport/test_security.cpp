#include <filesystem>
#include <thread>

#include "clusterlm/transport/security.hpp"
#include "test_helpers.hpp"

#ifndef _WIN32
#include <sys/stat.h>
#endif

using namespace clm_test;

namespace {

std::shared_ptr<const DeviceIdentity> make_identity(const char* name) {
  auto id = DeviceIdentity::generate(name);
  REQUIRE(id.is_ok());
  return std::make_shared<const DeviceIdentity>(std::move(id).value());
}

SecurityConfig tls(std::shared_ptr<const DeviceIdentity> self, std::vector<std::string> trusted) {
  SecurityConfig c;
  c.mode = SecurityConfig::Mode::kMutualTls;
  c.identity = std::move(self);
  c.trusted_peers = std::move(trusted);
  return c;
}

std::filesystem::path unique_temp_dir(const char* tag) {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / (std::string("clusterlm_") + tag + "_" + std::to_string(ticks));
}

}  // namespace

TEST_CASE("device identity: fingerprint format and uniqueness") {
  auto a = make_identity("father");
  auto b = make_identity("node-a");
  CHECK(a->fingerprint().size() == 64);
  for (char c : a->fingerprint()) CHECK(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
  CHECK(a->fingerprint() != b->fingerprint());
  CHECK(a->common_name() == "father");
  CHECK_FALSE(DeviceIdentity::generate("").is_ok());
}

TEST_CASE("device identity save/load round trip keeps the fingerprint") {
  auto a = make_identity("father");
  const auto dir = unique_temp_dir("identity");
  REQUIRE(a->save(dir).is_ok());
  auto loaded = DeviceIdentity::load(dir);
  REQUIRE(loaded.is_ok());
  CHECK(loaded->fingerprint() == a->fingerprint());
  CHECK(loaded->common_name() == "father");
#ifndef _WIN32
  struct stat st {};
  REQUIRE(::stat((dir / "device_key.pem").c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
#endif

  SUBCASE("a reloaded identity authenticates as the same device") {
    auto server_id = std::make_shared<const DeviceIdentity>(std::move(loaded).value());
    auto client_id = make_identity("client");
    Pair p = connect_pair(tls(server_id, {client_id->fingerprint()}), tls(client_id, {a->fingerprint()}),
                          a->fingerprint());
    REQUIRE(p.client != nullptr);
    CHECK(p.client->peer().device_id == a->fingerprint());
  }

  SUBCASE("a key that does not match the certificate is rejected") {
    auto other = make_identity("other");
    const auto dir2 = unique_temp_dir("identity_other");
    REQUIRE(other->save(dir2).is_ok());
    std::filesystem::copy_file(dir2 / "device_key.pem", dir / "device_key.pem",
                               std::filesystem::copy_options::overwrite_existing);
    CHECK(DeviceIdentity::load(dir).status().code() == ErrorCode::kDataLoss);
    std::filesystem::remove_all(dir2);
  }
  CHECK_FALSE(DeviceIdentity::load(dir / "missing").is_ok());
  std::filesystem::remove_all(dir);
}

TEST_CASE("mutual TLS with pinned identities") {
  auto father = make_identity("father");
  auto node = make_identity("node-a");
  Pair p = connect_pair(tls(node, {father->fingerprint()}), tls(father, {node->fingerprint()}), node->fingerprint());
  REQUIRE(p.client != nullptr);
  REQUIRE(p.server != nullptr);
  CHECK(p.client->peer().authenticated);
  CHECK(p.server->peer().authenticated);
  CHECK(p.client->peer().device_id == node->fingerprint());
  CHECK(p.server->peer().device_id == father->fingerprint());
  CHECK_FALSE(p.server->peer().address.empty());

  SUBCASE("frames flow both ways, concurrently, including a large one") {
    Frame big = make_frame(1, 3u << 20);
    for (std::size_t i = 0; i < big.payload.size(); ++i) big.payload[i] = static_cast<std::uint8_t>(i * 13);
    auto c2s = std::async(std::launch::async, [&] {
      for (int i = 0; i < 20; ++i) CHECK(p.client->send(make_frame(2, 1000, 0x22, static_cast<std::uint64_t>(i))).is_ok());
      return p.client->send(big);
    });
    auto s2c = std::async(std::launch::async, [&] {
      for (int i = 0; i < 20; ++i) CHECK(p.server->send(make_frame(3, 500, 0x33, static_cast<std::uint64_t>(i))).is_ok());
      return Status::ok();
    });
    std::thread server_reader([&] {
      for (int i = 0; i < 20; ++i) {
        auto r = p.server->receive(5s);
        REQUIRE(r.is_ok());
        CHECK(r->type == 2);
        CHECK(r->correlation == static_cast<std::uint64_t>(i));
      }
      auto r = p.server->receive(5s);
      REQUIRE(r.is_ok());
      CHECK(r->payload == big.payload);
    });
    for (int i = 0; i < 20; ++i) {
      auto r = p.client->receive(5s);
      REQUIRE(r.is_ok());
      CHECK(r->type == 3);
      CHECK(r->payload.size() == 500);
    }
    server_reader.join();
    CHECK(c2s.get().is_ok());
    CHECK(s2c.get().is_ok());
  }

  SUBCASE("close() unblocks a TLS receive promptly") {
    std::thread closer([&] {
      std::this_thread::sleep_for(50ms);
      p.client->close();
    });
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(p.client->receive(30s).status().code() == ErrorCode::kUnavailable);
    CHECK(std::chrono::steady_clock::now() - t0 < 2s);
    closer.join();
    CHECK(p.server->receive(2s).status().code() == ErrorCode::kUnavailable);
  }

  SUBCASE("receive timeout leaves a TLS connection usable") {
    CHECK(p.server->receive(50ms).status().code() == ErrorCode::kDeadlineExceeded);
    REQUIRE(p.client->send(make_frame(9, 8)).is_ok());
    CHECK(p.server->receive(2s).is_ok());
  }
}

TEST_CASE("mutual TLS rejects a wrong expected peer on both sides") {
  auto father = make_identity("father");
  auto node = make_identity("node-a");
  auto impostor = make_identity("impostor");
  // Both sides trust each other, but the client expects a different device than the one it reaches.
  Pair p = connect_pair(tls(node, {father->fingerprint()}), tls(father, {node->fingerprint(), impostor->fingerprint()}),
                        impostor->fingerprint());
  CHECK(p.client == nullptr);
  CHECK(p.client_status.code() == ErrorCode::kUnauthenticated);
  CHECK(p.server == nullptr);  // the handshake is dropped on the server side as well
  INFO(p.server_status.to_string());
  CHECK((p.server_status.code() == ErrorCode::kUnauthenticated || p.server_status.code() == ErrorCode::kUnavailable));
}

TEST_CASE("mutual TLS rejects an untrusted client") {
  auto node = make_identity("node-a");
  auto stranger = make_identity("stranger");
  auto father = make_identity("father");
  // The server only trusts `father`; `stranger` connects, and trusts the server.
  Pair p = connect_pair(tls(node, {father->fingerprint()}), tls(stranger, {node->fingerprint()}), node->fingerprint());
  CHECK(p.server == nullptr);
  CHECK(p.server_status.code() == ErrorCode::kUnauthenticated);
  if (p.client) {
    // TLS 1.3 lets the client finish first; it must then observe the drop.
    auto r = p.client->receive(2s);
    INFO(r.status().to_string());
    CHECK_FALSE(r.is_ok());
    CHECK((r.status().code() == ErrorCode::kUnauthenticated || r.status().code() == ErrorCode::kUnavailable));
  } else {
    CHECK(p.client_status.code() == ErrorCode::kUnauthenticated);
  }
}

TEST_CASE("mutual TLS: a client that does not trust the server refuses it") {
  auto node = make_identity("node-a");
  auto father = make_identity("father");
  Pair p = connect_pair(tls(node, {father->fingerprint()}), tls(father, {}), std::nullopt);
  CHECK(p.client == nullptr);
  CHECK(p.client_status.code() == ErrorCode::kUnauthenticated);
  CHECK(p.server == nullptr);
}

TEST_CASE("mutual TLS refuses a plaintext peer and kMutualTls requires an identity") {
  auto node = make_identity("node-a");
  auto l = listen(Endpoint{"127.0.0.1", 0}, tls(node, {}));
  REQUIRE(l.is_ok());
  auto accepted = std::async(std::launch::async, [&] { return (*l)->accept(5s); });
  // An insecure client speaks plain frames at a TLS server: the handshake must fail, not hang or crash.
  auto c = connect((*l)->local_endpoint(), insecure(), std::nullopt, 2s);
  if (c.is_ok()) {
    (void)(*c)->send(make_frame(1, 64));
  }
  auto a = accepted.get();
  CHECK_FALSE(a.is_ok());

  SecurityConfig no_identity;
  no_identity.mode = SecurityConfig::Mode::kMutualTls;
  CHECK(listen(Endpoint{"127.0.0.1", 0}, no_identity).status().code() == ErrorCode::kInvalidArgument);
}
