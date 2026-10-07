#pragma once
// Authenticated pairing of a Father and a Node over TLS when neither side knows the other's fingerprint yet.
//
// Roles. The NODE is put into pairing mode locally (clusterlm-node-service --pair). It opens a time-boxed
// listener, shows a one-time code and its short fingerprint. The FATHER user enters the Node's address and the
// code; Father is the initiator.
//
// Transport. TLS 1.3 with both certificates but NO pin (SecurityConfig::pairing_channel): the channel carries
// nothing except this exchange. Each side learns the other's certificate fingerprint from the handshake.
//
// Authentication (details and residual risk in docs/pairing.md and ADR 0270):
//   1. SPAKE2 over P-256 (spake2.hpp) turns the 8-character code (40 bits) into a shared key. Passive and
//      active network attackers cannot test guesses offline; an active attacker gets ONE guess per run.
//   2. The key-derivation transcript includes BOTH certificate fingerprints and the TLS exporter secret of the
//      session, so confirmations are valid only on the exact TLS session between the two certificates. A relay
//      (a machine terminating TLS with its own certificate on each leg, or splicing sessions) produces
//      different fingerprints/exporters on the two legs and neither confirmation verifies, whatever it knows.
//   3. Key confirmation: initiator MAC first, verified by the responder; the responder only sends its MAC after
//      the initiator's verified. A failed run reveals nothing testable.
//   4. Only after both confirmations each side sends its DeviceInfo (name, role, data port) over the TLS channel.
// Rate limiting: the responder serves one connection at a time, counts failed confirmations and LOCKS pairing
// mode after `max_failures` (default 3): the listener closes and only a fresh pairing-mode start re-opens it.
// The code is single use and the window is bounded (default 5 minutes).
//
// NOT resisted: an attacker who learns the code (shoulder surfing) before the user uses it can complete a real
// pairing of their own device; the Node's fingerprint short form displayed by the Node is what the Father user
// compares afterwards to notice a substituted device. Availability: any LAN host can burn the 3 attempts.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "clusterlm/common/status.hpp"
#include "clusterlm/transport/security.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::pairing {

// ---- one-time code ---------------------------------------------------------------------------------------
inline constexpr std::size_t kCodeChars = 8;  // RFC 4648 base32 alphabet: 40 bits
// "ABCD-EFGH" (CSPRNG).
std::string generate_code();
// Accepts lower case and "-"/space separators; rejects anything else or a wrong length. Returns the 8 bare
// characters.
Result<std::string> normalize_code(std::string_view user_text);
// "ab12-cd34-ef56": the first 12 hex digits of a fingerprint, for humans to compare.
std::string short_fingerprint(std::string_view fingerprint);

// ---- exchanged data --------------------------------------------------------------------------------------
struct DeviceInfo {
  std::string name;               // label shown to the user on the other side
  std::string role;               // "father" or "node"
  std::uint16_t data_port = 0;    // Node: port of the worker (data/control) listener; 0 for Father
};

struct PairedPeer {
  std::string fingerprint;  // the peer's TLS certificate fingerprint = its device id
  DeviceInfo info;
  std::string remote_host;  // peer address as seen on the pairing connection (no port)
};

inline constexpr std::uint16_t kPairingProtocolVersion = 1;

// ---- Node side -------------------------------------------------------------------------------------------
enum class PairingState : std::uint8_t { kListening, kPaired, kLocked, kExpired, kCancelled };
std::string_view to_string(PairingState s) noexcept;

struct ResponderOptions {
  std::shared_ptr<const transport::DeviceIdentity> identity;  // the Node's device identity (same as its worker's)
  transport::Endpoint listen{"0.0.0.0", 0};
  DeviceInfo self;
  std::chrono::seconds window{300};           // pairing mode lasts at most this long
  std::uint32_t max_failures = 3;             // failed confirmations before the mode locks
  std::chrono::milliseconds step_timeout{10000};  // each message of the exchange
  std::string code;                           // empty = generate (tests may force one)
};

class PairingResponder {
 public:
  static Result<std::unique_ptr<PairingResponder>> start(ResponderOptions options);
  ~PairingResponder();
  PairingResponder(const PairingResponder&) = delete;
  PairingResponder& operator=(const PairingResponder&) = delete;

  const std::string& code() const { return code_display_; }   // "ABCD-EFGH"
  transport::Endpoint endpoint() const { return endpoint_; }
  const std::string& fingerprint() const { return fingerprint_; }
  PairingState state() const;
  std::uint32_t failures() const { return failures_.load(); }
  // Blocks until pairing mode ends. Returns the peer on success; kUnauthenticated when locked,
  // kDeadlineExceeded when `timeout` elapses first (mode stays active) or the window expired, kCancelled.
  Result<PairedPeer> wait(std::chrono::milliseconds timeout);
  void cancel();

 private:
  PairingResponder() = default;
  void run();
  // Returns the peer on success; an error Status otherwise (failure counting happens inside).
  Result<PairedPeer> serve_one(transport::Connection& conn);
  void finish(PairingState s);

  ResponderOptions opts_;
  std::string code_, code_display_, fingerprint_;
  transport::Endpoint endpoint_;
  std::unique_ptr<transport::Listener> listener_;
  std::thread thread_;
  std::atomic<std::uint32_t> failures_{0};
  std::atomic<bool> cancel_{false};
  mutable std::mutex mu_;
  std::condition_variable cv_;
  PairingState state_ = PairingState::kListening;
  PairedPeer result_;
};

// ---- Father side -----------------------------------------------------------------------------------------
// Connects to a Node in pairing mode and runs the exchange. Errors: kUnauthenticated (wrong code, relay
// detected, Node locked its mode), kUnavailable (nothing listening / closed), kDeadlineExceeded, kInvalidArgument
// (malformed code). Nothing about the code or the keys is logged.
Result<PairedPeer> pair_with(const transport::Endpoint& node, const std::shared_ptr<const transport::DeviceIdentity>& identity,
                             std::string_view code, const DeviceInfo& self,
                             std::chrono::milliseconds timeout = std::chrono::milliseconds(15000));

// ---- wire (exposed for tests that play an attacker) -----------------------------------------------------
namespace wire {
inline constexpr std::uint16_t kHello = 0x5001;      // initiator -> responder: u16 version, 65-byte share
inline constexpr std::uint16_t kShare = 0x5002;      // responder -> initiator: 65-byte share
inline constexpr std::uint16_t kConfirmA = 0x5003;   // initiator -> responder: 32-byte MAC
inline constexpr std::uint16_t kConfirmB = 0x5004;   // responder -> initiator: 32-byte MAC
inline constexpr std::uint16_t kInfo = 0x5005;       // both: name, role, data port
inline constexpr std::uint16_t kError = 0x5006;      // either: u16 ErrorCode (no free text)
inline constexpr std::string_view kExporterLabel = "EXPORTER-clusterlm-pairing-v1";
inline constexpr std::size_t kExporterBytes = 32;
// The channel binding both sides feed into the SPAKE2 transcript.
Bytes channel_binding(std::string_view initiator_fingerprint, std::string_view responder_fingerprint, ByteSpan exporter);
Bytes encode_info(const DeviceInfo& info);
Result<DeviceInfo> decode_info(ByteSpan payload);
}  // namespace wire

}  // namespace clusterlm::pairing
