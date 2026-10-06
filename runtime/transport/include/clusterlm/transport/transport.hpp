#pragma once
// Framed, authenticated TCP transport. Message-agnostic: it moves typed frames; protocol codecs live in
// runtime/protocol.
//
// Wire header (24 bytes, explicit little-endian):
//   u32 magic 0x464D4C43 ("CLMF") | u16 frame_version=1 | u16 type | u8 channel | u8 flags | u16 reserved=0
//   u64 correlation | u32 payload_len
// The receiver validates magic/version/reserved and payload_len <= max_payload BEFORE allocating; a violation
// is kProtocolError and the connection is closed.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/transport/security.hpp"

namespace clusterlm::transport {

inline constexpr std::uint32_t kFrameMagic = 0x464D4C43u;
inline constexpr std::uint16_t kFrameVersion = 1;
inline constexpr std::size_t kFrameHeaderSize = 24;

struct Endpoint {
  std::string host;
  std::uint16_t port = 0;

  // "host:port", "[v6]:port". Port 0 is allowed (ephemeral, for listen).
  static Result<Endpoint> parse(std::string_view text);
  std::string str() const;
  // Literal loopback only ("localhost", 127.0.0.0/8, ::1) — no DNS lookup.
  bool is_loopback() const;
};

struct Frame {
  std::uint16_t type = 0;
  std::uint8_t channel = 0;
  std::uint8_t flags = 0;
  std::uint64_t correlation = 0;
  Bytes payload;
};

// Snapshot of wire-level counters (header + payload bytes).
struct ConnectionStats {
  std::uint64_t bytes_sent = 0;
  std::uint64_t bytes_received = 0;
  std::uint64_t frames_sent = 0;
  std::uint64_t frames_received = 0;
};

struct PeerIdentity {
  std::string device_id;   // peer certificate fingerprint, or "insecure-loopback"
  bool authenticated = false;
  std::string address;     // remote "ip:port"
};

class Connection {
 public:
  virtual ~Connection() = default;

  // Thread-safe for concurrent senders; frames never interleave on the wire.
  virtual Status send(const Frame& frame) = 0;
  // Single reader. kDeadlineExceeded on timeout (connection stays usable, partial frames are resumed on the
  // next call); kUnavailable when closed or the peer is gone.
  virtual Result<Frame> receive(std::chrono::milliseconds timeout) = 0;
  // Safe from any thread; promptly unblocks a receive() (and send()) blocked in another thread.
  virtual void close() = 0;

  virtual PeerIdentity peer() const = 0;
  virtual ConnectionStats stats() const = 0;
  virtual void set_max_payload(std::uint32_t max_payload) = 0;
};

class Listener {
 public:
  virtual ~Listener() = default;
  // kDeadlineExceeded on timeout. A peer that fails authentication yields kUnauthenticated (the socket is
  // dropped); the caller may simply call accept() again.
  virtual Result<std::unique_ptr<Connection>> accept(std::chrono::milliseconds timeout) = 0;
  // Actual bound endpoint (port 0 resolved to the ephemeral port).
  virtual Endpoint local_endpoint() const = 0;
  // Safe from any thread; unblocks accept() and releases the port.
  virtual void close() = 0;
};

// kInsecureLoopbackOnly: non-loopback endpoints are refused with kPermissionDenied.
Result<std::unique_ptr<Listener>> listen(const Endpoint& endpoint, const SecurityConfig& security);
// `timeout` bounds TCP connect plus TLS handshake. In kMutualTls the peer fingerprint must be trusted and, if
// given, equal `expected_peer_device_id` (else kUnauthenticated). The expected id is ignored in insecure mode.
Result<std::unique_ptr<Connection>> connect(const Endpoint& endpoint, const SecurityConfig& security,
                                            std::optional<std::string> expected_peer_device_id,
                                            std::chrono::milliseconds timeout);

}  // namespace clusterlm::transport
