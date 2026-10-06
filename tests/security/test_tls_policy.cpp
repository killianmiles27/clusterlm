// TLS policy of the mutual-TLS transport, verified from the OUTSIDE with a raw OpenSSL client:
//   * TLS 1.3 only (a TLS 1.2-only client cannot connect),
//   * no session tickets / resumption,
//   * the server demands and pins a client certificate (none, or an unpaired one, is refused).
// The raw client needs POSIX sockets; on Windows these cases are skipped (the policy is set in one place,
// TlsContext::create, which compiles and is exercised identically there through test_transport).
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>

#include "clusterlm/transport/security.hpp"
#include "clusterlm/transport/transport.hpp"

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace clusterlm;
using namespace clusterlm::transport;
using namespace std::chrono_literals;

#ifndef _WIN32
namespace {

std::shared_ptr<const DeviceIdentity> make_identity(const char* name) {
  auto id = DeviceIdentity::generate(name);
  REQUIRE(id.is_ok());
  return std::make_shared<const DeviceIdentity>(std::move(id).value());
}

std::filesystem::path temp_dir(const char* tag) {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  auto p = std::filesystem::temp_directory_path() / (std::string("clm-tlspolicy-") + tag + "-" + std::to_string(ticks));
  std::filesystem::create_directories(p);
  return p;
}

struct SslCtxDeleter {
  void operator()(SSL_CTX* c) const { SSL_CTX_free(c); }
};
struct SslDeleter {
  void operator()(SSL* s) const { SSL_free(s); }
};

// What the raw client observed.
struct ClientResult {
  bool handshake_ok = false;
  std::string version;
  std::string cipher;
  int tickets_received = 0;
  bool read_ok = false;  // could read application data after the handshake (i.e. the server did not drop us)
};

int ticket_count_slot() {
  static const int idx = SSL_get_ex_new_index(0, nullptr, nullptr, nullptr, nullptr);
  return idx;
}

int on_new_session(SSL* ssl, SSL_SESSION*) {
  auto* n = static_cast<int*>(SSL_get_ex_data(ssl, ticket_count_slot()));
  if (n != nullptr) ++*n;
  return 0;  // do not keep the session
}

// Connects a raw TLS client to `port` and reads one byte after the handshake. `identity_dir` empty = no client
// certificate. `max_version` 0 = default (TLS 1.3 capable).
ClientResult raw_client(std::uint16_t port, const std::filesystem::path& identity_dir, int max_version) {
  ClientResult out;
  std::unique_ptr<SSL_CTX, SslCtxDeleter> ctx(SSL_CTX_new(TLS_client_method()));
  REQUIRE(ctx != nullptr);
  SSL_CTX_set_verify(ctx.get(), SSL_VERIFY_NONE, nullptr);  // pinning is the server's concern in this test
  if (max_version != 0) SSL_CTX_set_max_proto_version(ctx.get(), max_version);
  if (!identity_dir.empty()) {
    REQUIRE(SSL_CTX_use_certificate_file(ctx.get(), (identity_dir / "device_cert.pem").c_str(), SSL_FILETYPE_PEM) == 1);
    REQUIRE(SSL_CTX_use_PrivateKey_file(ctx.get(), (identity_dir / "device_key.pem").c_str(), SSL_FILETYPE_PEM) == 1);
  }
  SSL_CTX_set_session_cache_mode(ctx.get(), SSL_SESS_CACHE_CLIENT | SSL_SESS_CACHE_NO_INTERNAL_STORE);
  SSL_CTX_sess_set_new_cb(ctx.get(), on_new_session);

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  REQUIRE(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
  timeval tv{5, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

  std::unique_ptr<SSL, SslDeleter> ssl(SSL_new(ctx.get()));
  REQUIRE(ssl != nullptr);
  int tickets = 0;
  SSL_set_ex_data(ssl.get(), ticket_count_slot(), &tickets);
  SSL_set_fd(ssl.get(), fd);
  if (SSL_connect(ssl.get()) == 1) {
    out.handshake_ok = true;
    out.version = SSL_get_version(ssl.get());
    out.cipher = SSL_get_cipher_name(ssl.get());
    char b;
    out.read_ok = SSL_read(ssl.get(), &b, 1) == 1;  // also processes any post-handshake NewSessionTicket
  }
  ERR_clear_error();
  out.tickets_received = tickets;
  ::close(fd);
  return out;
}

struct Server {
  std::shared_ptr<const DeviceIdentity> identity = make_identity("server");
  std::shared_ptr<const DeviceIdentity> paired = make_identity("paired-client");
  std::unique_ptr<Listener> listener;
  std::filesystem::path paired_dir = temp_dir("paired");

  Server() {
    REQUIRE(paired->save(paired_dir).is_ok());
    SecurityConfig cfg;
    cfg.mode = SecurityConfig::Mode::kMutualTls;
    cfg.identity = identity;
    cfg.trusted_peers = {paired->fingerprint()};
    auto l = listen(Endpoint{"127.0.0.1", 0}, cfg);
    REQUIRE(l.is_ok());
    listener = std::move(l).value();
  }
  ~Server() {
    std::error_code ec;
    std::filesystem::remove_all(paired_dir, ec);
  }
  std::uint16_t port() const { return listener->local_endpoint().port; }
};

}  // namespace

TEST_CASE("mTLS: a TLS 1.3 client with the paired certificate connects, and no session ticket is ever issued") {
  Server s;
  auto accepted = std::async(std::launch::async, [&] { return s.listener->accept(5s); });
  auto client = std::async(std::launch::async, [&] { return raw_client(s.port(), s.paired_dir, 0); });
  auto conn = accepted.get();
  REQUIRE(conn.is_ok());
  // Push one frame so the client's post-handshake read has something to wake up on; tickets (if any) precede it.
  Frame f;
  f.type = 1;
  f.payload = {1};
  REQUIRE(conn.value()->send(f).is_ok());
  const ClientResult r = client.get();
  CHECK(r.handshake_ok);
  CHECK(r.version == "TLSv1.3");
  CHECK(r.cipher.rfind("TLS_", 0) == 0);  // TLS 1.3 suites are the only ones named TLS_*
  CHECK(r.read_ok);
  CHECK(r.tickets_received == 0);
  CHECK(conn.value()->peer().device_id == s.paired->fingerprint());
}

TEST_CASE("mTLS: a TLS 1.2-only client is rejected") {
  Server s;
  auto accepted = std::async(std::launch::async, [&] { return s.listener->accept(5s); });
  auto client = std::async(std::launch::async, [&] { return raw_client(s.port(), s.paired_dir, TLS1_2_VERSION); });
  const ClientResult r = client.get();
  auto conn = accepted.get();
  CHECK_FALSE(r.handshake_ok);
  CHECK_FALSE(conn.is_ok());
}

TEST_CASE("mTLS: a client without a certificate, or with an unpaired one, is rejected by the server") {
  Server s;
  SUBCASE("no client certificate") {
    auto accepted = std::async(std::launch::async, [&] { return s.listener->accept(5s); });
    auto client = std::async(std::launch::async, [&] { return raw_client(s.port(), {}, 0); });
    auto conn = accepted.get();
    (void)client.get();
    CHECK_FALSE(conn.is_ok());
    if (!conn.is_ok()) CHECK(conn.status().code() == ErrorCode::kUnauthenticated);
  }
  SUBCASE("unpaired certificate") {
    auto rogue = make_identity("rogue");
    const auto dir = temp_dir("rogue");
    REQUIRE(rogue->save(dir).is_ok());
    auto accepted = std::async(std::launch::async, [&] { return s.listener->accept(5s); });
    auto client = std::async(std::launch::async, [&] { return raw_client(s.port(), dir, 0); });
    auto conn = accepted.get();
    const ClientResult r = client.get();
    CHECK_FALSE(conn.is_ok());
    if (!conn.is_ok()) CHECK(conn.status().code() == ErrorCode::kUnauthenticated);
    CHECK_FALSE(r.read_ok);  // in TLS 1.3 the client's handshake can finish first; the server then drops it
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
}

#else  // _WIN32

TEST_CASE("mTLS policy probes need the POSIX raw-socket client") {
  MESSAGE("skipped on Windows: the TLS policy is set in TlsContext::create and covered by test_transport");
}

#endif
