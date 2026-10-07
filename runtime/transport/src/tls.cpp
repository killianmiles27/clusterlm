#include "tls.hpp"

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <mutex>
#include <unordered_set>

#ifndef _WIN32
#include <openssl/bio.h>
#include <sys/socket.h>
#endif

#include "security_internal.hpp"

namespace clusterlm::transport {

namespace {

#ifndef _WIN32
// OpenSSL's socket BIO writes with write(2), which raises SIGPIPE when the peer has already closed the connection
// and kills the process. This BIO is the same descriptor I/O with send(MSG_NOSIGNAL), so a vanished peer surfaces as
// an error on the connection (as on the plain-TCP path), never as a signal.
// The BIO's data pointer is the owning stream's socket (kept alive by TlsStream: SSL is freed before the socket).
int bio_fd(BIO* b) { return static_cast<int>(static_cast<const net::Socket*>(BIO_get_data(b))->handle()); }

int nosig_write(BIO* b, const char* buf, int len) {
  const int fd = bio_fd(b);
  BIO_clear_retry_flags(b);
  const ssize_t n = ::send(fd, buf, static_cast<std::size_t>(len), MSG_NOSIGNAL);
  if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) BIO_set_retry_write(b);
  return static_cast<int>(n);
}

int nosig_read(BIO* b, char* buf, int len) {
  const int fd = bio_fd(b);
  BIO_clear_retry_flags(b);
  const ssize_t n = ::recv(fd, buf, static_cast<std::size_t>(len), 0);
  if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) BIO_set_retry_read(b);
  return static_cast<int>(n);
}

long nosig_ctrl(BIO* b, int cmd, long, void* ptr) {
  switch (cmd) {
    case BIO_CTRL_FLUSH:
      return 1;
    case BIO_C_GET_FD:
      if (ptr != nullptr) *static_cast<int*>(ptr) = bio_fd(b);
      return static_cast<long>(bio_fd(b));
    default:
      return 0;
  }
}

int nosig_create(BIO* b) {
  BIO_set_init(b, 1);
  return 1;
}

const BIO_METHOD* nosig_socket_method() {
  static BIO_METHOD* method = [] {
    BIO_METHOD* m = BIO_meth_new(BIO_get_new_index() | BIO_TYPE_SOURCE_SINK | BIO_TYPE_DESCRIPTOR, "clusterlm-socket");
    if (m == nullptr) return m;
    BIO_meth_set_write(m, nosig_write);
    BIO_meth_set_read(m, nosig_read);
    BIO_meth_set_ctrl(m, nosig_ctrl);
    BIO_meth_set_create(m, nosig_create);
    return m;
  }();
  return method;
}
#endif

// Per-connection pinning decision, consulted from the OpenSSL verify callback.
struct PinPolicy {
  std::unordered_set<std::string> trusted;
  std::optional<std::string> expected;
  bool accept_any = false;  // pairing channel: identity is established by the pairing exchange, not by a pin
  bool rejected = false;
  std::string reject_reason;

  bool accept(const std::string& fingerprint) {
    if (accept_any) return !fingerprint.empty();
    if (trusted.find(fingerprint) == trusted.end()) {
      rejected = true;
      reject_reason = "peer certificate is not in the trusted set";
      return false;
    }
    if (expected && *expected != fingerprint) {
      rejected = true;
      reject_reason = "peer certificate does not match the expected device id";
      return false;
    }
    return true;
  }
};

// Chain validation is replaced wholesale by pinning: self-signed certificates have no chain, and trust comes
// from the pairing-time fingerprint, so the only certificate that matters is the leaf (depth 0).
int verify_callback(int /*preverify_ok*/, X509_STORE_CTX* sctx) {
  SSL* ssl = static_cast<SSL*>(X509_STORE_CTX_get_ex_data(sctx, SSL_get_ex_data_X509_STORE_CTX_idx()));
  auto* policy = ssl != nullptr ? static_cast<PinPolicy*>(SSL_get_app_data(ssl)) : nullptr;
  if (policy == nullptr) return 0;
  if (X509_STORE_CTX_get_error_depth(sctx) != 0) {
    policy->rejected = true;
    policy->reject_reason = "unexpected certificate chain";
    return 0;
  }
  X509* cert = X509_STORE_CTX_get_current_cert(sctx);
  const std::string fp = cert != nullptr ? cert_fingerprint(cert) : std::string();
  if (fp.empty() || !policy->accept(fp)) {
    X509_STORE_CTX_set_error(sctx, X509_V_ERR_APPLICATION_VERIFICATION);
    return 0;
  }
  X509_STORE_CTX_set_error(sctx, X509_V_OK);  // discard the self-signed-chain error; the pin is the trust root
  return 1;
}

bool is_auth_reason(int reason) {
  switch (reason) {
    case SSL_R_PEER_DID_NOT_RETURN_A_CERTIFICATE:
    case SSL_R_CERTIFICATE_VERIFY_FAILED:
    case SSL_R_SSLV3_ALERT_BAD_CERTIFICATE:
    case SSL_R_SSLV3_ALERT_UNSUPPORTED_CERTIFICATE:
    case SSL_R_SSLV3_ALERT_CERTIFICATE_UNKNOWN:
    case SSL_R_TLSV1_ALERT_UNKNOWN_CA:
    case SSL_R_TLSV1_ALERT_ACCESS_DENIED:
    case SSL_R_TLSV13_ALERT_CERTIFICATE_REQUIRED:
    // OpenSSL answers an application-level verify rejection (our pin check) with handshake_failure.
    case SSL_R_SSLV3_ALERT_HANDSHAKE_FAILURE:
      return true;
    default:
      return false;
  }
}

std::string openssl_error_text(unsigned long e) {
  char buf[256];
  ERR_error_string_n(e, buf, sizeof buf);
  return buf;
}

constexpr int kMaxSslChunk = 1 << 20;

class TlsStream final : public Stream {
 public:
  TlsStream(net::Socket sock, SSL_CTX* ctx, std::shared_ptr<PinPolicy> policy)
      : sock_(std::move(sock)), policy_(std::move(policy)) {
    ssl_ = SSL_new(ctx);
  }
  ~TlsStream() override {
    // SSL first, then the socket (member order); no close_notify: a closing peer simply disappears.
    SSL_free(ssl_);
  }

  Status init(bool server) {
    if (ssl_ == nullptr) return make_error(ErrorCode::kInternal, "SSL_new failed: " + openssl_errors());
    SSL_set_app_data(ssl_, policy_.get());
#ifdef _WIN32
    if (SSL_set_fd(ssl_, static_cast<int>(sock_.handle())) != 1)
      return make_error(ErrorCode::kInternal, "SSL_set_fd failed");
#else
    const BIO_METHOD* method = nosig_socket_method();
    BIO* bio = method != nullptr ? BIO_new(method) : nullptr;
    if (bio == nullptr) return make_error(ErrorCode::kInternal, "socket BIO allocation failed: " + openssl_errors());
    BIO_set_data(bio, &sock_);
    SSL_set_bio(ssl_, bio, bio);  // the SSL owns the BIO; the socket stays owned by sock_
#endif
    if (server) SSL_set_accept_state(ssl_);
    else SSL_set_connect_state(ssl_);
    return Status::ok();
  }

  Status handshake(net::Deadline deadline, const std::atomic<bool>* external_cancel) {
    cancel_ = external_cancel != nullptr ? external_cancel : &closed_;
    int rc = 0;
    Status s = run([&] { return SSL_do_handshake(ssl_); }, rc, deadline);
    cancel_ = &closed_;
    return s;
  }

  Status read_some(void* buf, std::size_t n, std::size_t& got, net::Deadline deadline) override {
    const int want = static_cast<int>(std::min<std::size_t>(n, kMaxSslChunk));
    int rc = 0;
    CLM_RETURN_IF_ERROR(run([&] { return SSL_read(ssl_, buf, want); }, rc, deadline));
    got = static_cast<std::size_t>(rc);
    return Status::ok();
  }

  Status write_all(const void* buf, std::size_t n) override {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    while (n > 0) {
      const int want = static_cast<int>(std::min<std::size_t>(n, kMaxSslChunk));
      int rc = 0;
      CLM_RETURN_IF_ERROR(run([&] { return SSL_write(ssl_, p, want); }, rc, net::no_deadline()));
      p += rc;
      n -= static_cast<std::size_t>(rc);
    }
    return Status::ok();
  }

  void shutdown() override {
    closed_.store(true, std::memory_order_release);
    sock_.shutdown_both();
  }

  X509* peer_cert() { return SSL_get1_peer_certificate(ssl_); }

  Result<Bytes> export_keying_material(std::string_view label, std::size_t length) const override {
    if (length == 0 || length > 1024) return make_error(ErrorCode::kInvalidArgument, "bad exporter length");
    Bytes out(length);
    std::lock_guard<std::mutex> lk(mu_);
    if (SSL_export_keying_material(ssl_, out.data(), out.size(), label.data(), label.size(), nullptr, 0, 0) != 1) {
      ERR_clear_error();
      return make_error(ErrorCode::kInternal, "SSL_export_keying_material failed");
    }
    return out;
  }
  const PinPolicy& policy() const { return *policy_; }

 private:
  // SSL objects tolerate one reader and one writer only if calls are serialized, so every SSL call runs under
  // mu_; waiting for the socket happens outside the lock so a blocked reader never stalls a sender.
  template <typename Op>
  Status run(Op op, int& result, net::Deadline deadline) {
    for (;;) {
      if (cancel_->load(std::memory_order_acquire) || closed_.load(std::memory_order_acquire))
        return make_error(ErrorCode::kUnavailable, "connection closed");
      int rc;
      int err = SSL_ERROR_NONE;
      unsigned long first_error = 0;
      {
        std::lock_guard<std::mutex> lk(mu_);
        ERR_clear_error();
        rc = op();
        if (rc <= 0) {
          err = SSL_get_error(ssl_, rc);
          if (err == SSL_ERROR_SSL || err == SSL_ERROR_SYSCALL) first_error = ERR_get_error();
          ERR_clear_error();
        }
      }
      if (rc > 0) {
        result = rc;
        return Status::ok();
      }
      switch (err) {
        case SSL_ERROR_WANT_READ:
        case SSL_ERROR_WANT_WRITE: {
          const bool for_write = err == SSL_ERROR_WANT_WRITE;
          switch (net::wait_ready(sock_.handle(), for_write, deadline, cancel_)) {
            case net::WaitResult::kReady: continue;
            case net::WaitResult::kTimeout: return make_error(ErrorCode::kDeadlineExceeded, "TLS I/O timed out");
            default: return make_error(ErrorCode::kUnavailable, "connection closed");
          }
        }
        case SSL_ERROR_SSL:
          if (policy_->rejected) return make_error(ErrorCode::kUnauthenticated, policy_->reject_reason);
          if (is_auth_reason(ERR_GET_REASON(first_error)))
            return make_error(ErrorCode::kUnauthenticated, "TLS peer authentication failed");
          // OpenSSL 3 reports a peer that vanishes without close_notify as an SSL error, not a syscall error.
          if (ERR_GET_REASON(first_error) == SSL_R_UNEXPECTED_EOF_WHILE_READING)
            return make_error(ErrorCode::kUnavailable, "TLS peer closed the connection");
          return make_error(ErrorCode::kProtocolError, "TLS protocol error: " + openssl_error_text(first_error));
        default:
          return make_error(ErrorCode::kUnavailable, "TLS peer closed the connection");
      }
    }
  }

  net::Socket sock_;
  std::shared_ptr<PinPolicy> policy_;
  SSL* ssl_ = nullptr;
  mutable std::mutex mu_;
  std::atomic<bool> closed_{false};
  const std::atomic<bool>* cancel_ = &closed_;
};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

struct TlsContext::Impl {
  SSL_CTX* ctx = nullptr;
  bool server = false;
  bool pairing_channel = false;
  std::unordered_set<std::string> trusted;
  std::shared_ptr<TrustStore> dynamic_trust;
  ~Impl() { SSL_CTX_free(ctx); }
};

TlsContext::TlsContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TlsContext::~TlsContext() = default;

Result<std::shared_ptr<TlsContext>> TlsContext::create(const SecurityConfig& security, bool server) {
  if (security.mode != SecurityConfig::Mode::kMutualTls)
    return make_error(ErrorCode::kInvalidArgument, "TLS context requires kMutualTls");
  if (!security.identity) return make_error(ErrorCode::kInvalidArgument, "kMutualTls requires a device identity");

  auto impl = std::make_unique<Impl>();
  impl->server = server;
  impl->pairing_channel = security.pairing_channel;
  for (const auto& id : security.trusted_peers) impl->trusted.insert(lower(id));
  impl->dynamic_trust = security.dynamic_trust;
  impl->ctx = SSL_CTX_new(server ? TLS_server_method() : TLS_client_method());
  if (impl->ctx == nullptr) return make_error(ErrorCode::kInternal, "SSL_CTX_new: " + openssl_errors());
  SSL_CTX* ctx = impl->ctx;
  const auto& ident = security.identity->native();
  if (SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) != 1 ||
      SSL_CTX_set_max_proto_version(ctx, TLS1_3_VERSION) != 1 || SSL_CTX_use_certificate(ctx, ident.cert) != 1 ||
      SSL_CTX_use_PrivateKey(ctx, ident.key) != 1 || SSL_CTX_check_private_key(ctx) != 1)
    return make_error(ErrorCode::kInternal, "TLS context setup: " + openssl_errors());
  SSL_CTX_set_mode(ctx, SSL_MODE_ENABLE_PARTIAL_WRITE | SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
  SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_OFF);
  SSL_CTX_set_num_tickets(ctx, 0);  // no resumption: every connection re-proves its identity
  int mode = SSL_VERIFY_PEER;
  if (server) mode |= SSL_VERIFY_FAIL_IF_NO_PEER_CERT;
  SSL_CTX_set_verify(ctx, mode, verify_callback);
  return std::shared_ptr<TlsContext>(new TlsContext(std::move(impl)));
}

Result<TlsEstablished> TlsContext::establish(net::Socket socket, const std::string& peer_address,
                                             std::optional<std::string> expected_peer, net::Deadline deadline,
                                             const std::atomic<bool>* cancel) {
  auto policy = std::make_shared<PinPolicy>();
  policy->trusted = impl_->trusted;
  policy->accept_any = impl_->pairing_channel;
  if (impl_->dynamic_trust)
    for (const auto& id : impl_->dynamic_trust->snapshot()) policy->trusted.insert(lower(id));
  if (expected_peer) policy->expected = lower(*expected_peer);
  auto stream = std::make_unique<TlsStream>(std::move(socket), impl_->ctx, policy);
  CLM_RETURN_IF_ERROR(stream->init(impl_->server));
  CLM_RETURN_IF_ERROR(stream->handshake(deadline, cancel));

  X509* peer = stream->peer_cert();
  if (peer == nullptr) return make_error(ErrorCode::kUnauthenticated, "peer presented no certificate");
  const std::string fp = cert_fingerprint(peer);
  X509_free(peer);
  // The callback already enforced this; re-check so a wiring mistake cannot silently disable pinning.
  PinPolicy check = *policy;
  if (fp.empty() || !check.accept(fp)) return make_error(ErrorCode::kUnauthenticated, check.reject_reason);

  TlsEstablished out;
  out.peer = PeerIdentity{fp, true, peer_address};
  out.stream = std::move(stream);
  return out;
}

}  // namespace clusterlm::transport
