#pragma once
// Internal: mutual-TLS 1.3 streams with certificate-fingerprint pinning.
#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "clusterlm/transport/security.hpp"
#include "clusterlm/transport/transport.hpp"
#include "stream.hpp"

namespace clusterlm::transport {

struct TlsEstablished {
  std::unique_ptr<Stream> stream;
  PeerIdentity peer;
};

class TlsContext {
 public:
  // `server` selects the role of every connection established through this context.
  static Result<std::shared_ptr<TlsContext>> create(const SecurityConfig& security, bool server);
  ~TlsContext();
  TlsContext(const TlsContext&) = delete;
  TlsContext& operator=(const TlsContext&) = delete;

  // Runs the handshake on `socket` and verifies the peer against the pin set (and `expected_peer`, if any).
  // Failures: kUnauthenticated (peer untrusted / no certificate / peer rejected us), kDeadlineExceeded,
  // kUnavailable (peer vanished or `cancel` set), kProtocolError (not TLS 1.3).
  Result<TlsEstablished> establish(net::Socket socket, const std::string& peer_address,
                                   std::optional<std::string> expected_peer, net::Deadline deadline,
                                   const std::atomic<bool>* cancel);

  struct Impl;

 private:
  explicit TlsContext(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace clusterlm::transport
