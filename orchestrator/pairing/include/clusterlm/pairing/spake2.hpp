#pragma once
// SPAKE2 over NIST P-256 (the symmetric-password form of RFC 9382, without the augmented verifier), built on
// OpenSSL's EC and BN primitives. It turns the short one-time pairing code into a shared key that a network
// attacker cannot test offline: an active attacker gets exactly ONE password guess per protocol run and learns
// nothing from a failed run beyond "that guess was wrong".
//
//   initiator (Father):  X = x*G + w*M          responder (Node):  Y = y*G + w*N
//   initiator:  T = Y - w*N,  Z = x*T,  V = w*T
//   responder:  T = X - w*M,  Z = y*T,  V = y*w*G          (both obtain the same Z and V iff w matches)
//
// M and N are fixed public points derived by hashing fixed strings to the curve ("nothing up my sleeve":
// nobody knows their discrete logarithms). w is derived from the code with PBKDF2-HMAC-SHA256.
// The session keys and the two key-confirmation MACs are derived from a transcript that ALSO binds the two
// TLS certificate fingerprints and the TLS exporter secret; see pairing.hpp for how that defeats a relay.
#include <memory>
#include <string_view>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"

namespace clusterlm::pairing {

inline constexpr std::size_t kSpake2ShareBytes = 65;  // uncompressed SEC1 point
inline constexpr std::size_t kSpake2ConfirmBytes = 32;

class Spake2 {
 public:
  enum class Role { kInitiator, kResponder };

  // `code` is the normalized pairing code. Fresh ephemeral scalar from the OpenSSL CSPRNG.
  static Result<Spake2> create(Role role, std::string_view code);
  Spake2(Spake2&&) noexcept;
  Spake2& operator=(Spake2&&) noexcept;
  ~Spake2();

  // This side's public share (65 bytes).
  const Bytes& share() const;

  struct Keys {
    Bytes confirm_initiator;  // 32-byte MAC the initiator sends
    Bytes confirm_responder;  // 32-byte MAC the responder sends
    Bytes session_key;        // 32 bytes; available to callers that want an application key
  };
  // Validates the peer share (on the curve, not the point at infinity), computes Z and V and binds
  // `channel_binding` (an opaque, already length-delimited encoding of both fingerprints and the exporter
  // secret) into the transcript. Fails with kUnauthenticated on a malformed share.
  Result<Keys> finish(ByteSpan peer_share, ByteSpan channel_binding) const;

  struct Impl;

 private:
  explicit Spake2(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Constant-time equality for MACs.
bool constant_time_equal(ByteSpan a, ByteSpan b);

}  // namespace clusterlm::pairing
