#pragma once
// Device identity and transport security configuration.
//
// Trust model (spec: identities pinned at pairing): there is no CA. A device is identified by the SHA-256 of
// its self-signed certificate (DER), lowercase hex — the `device_id`. A peer is accepted iff its certificate
// fingerprint is in `trusted_peers` (and equals the expected id when the caller names one). Possession of the
// matching private key is proven by the TLS 1.3 CertificateVerify exchange.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::transport {

class DeviceIdentity {
 public:
  struct Impl;  // OpenSSL key + certificate; defined in the implementation only.

  // EC P-256 key and a self-signed certificate (valid ~10 years; expiry is not part of the pinning decision).
  static Result<DeviceIdentity> generate(std::string_view common_name);
  // Reads `device_key.pem` and `device_cert.pem` from `dir`; rejects a key that does not match the certificate.
  static Result<DeviceIdentity> load(const std::filesystem::path& dir);
  // Loads the identity from `dir`, or generates and saves one if `dir` holds none (first run / pairing).
  static Result<DeviceIdentity> load_or_generate(const std::filesystem::path& dir, std::string_view common_name);
  // Writes both PEM files (creating `dir`). The key file is mode 0600 on POSIX; on Windows it inherits the
  // ACL of the (per-user) directory, which the caller must choose accordingly.
  Status save(const std::filesystem::path& dir) const;

  // Lowercase hex SHA-256 of the DER certificate: the device_id.
  const std::string& fingerprint() const;
  const std::string& common_name() const;

  // Implementation detail for the TLS layer.
  const Impl& native() const { return *impl_; }

 private:
  std::shared_ptr<const Impl> impl_;
};

// Fingerprints trusted at runtime in addition to the static paired set. Father authorizes a direct
// Node->Node channel for one plan/lease by naming the peer's paired identity; the Node trusts it only for that
// lease and revokes it on release. Shared by every listener/connection created from the same SecurityConfig.
class TrustStore {
 public:
  void add(std::string fingerprint);
  void remove(const std::string& fingerprint);
  bool contains(const std::string& fingerprint) const;
  std::vector<std::string> snapshot() const;

 private:
  mutable std::mutex mu_;
  std::vector<std::string> ids_;
};

struct SecurityConfig {
  enum class Mode : std::uint8_t {
    kMutualTls,            // TLS 1.3, both sides present certificates, fingerprint pinning
    kInsecureLoopbackOnly  // plain TCP for unit tests / early numerical harnesses; refuses non-loopback
  };
  Mode mode = Mode::kMutualTls;
  std::shared_ptr<const DeviceIdentity> identity;  // required for kMutualTls
  std::vector<std::string> trusted_peers;          // device_ids (fingerprints) accepted as peers
  std::uint32_t max_payload = 64u * 1024u * 1024u; // initial per-connection frame payload limit
  std::shared_ptr<TrustStore> dynamic_trust = std::make_shared<TrustStore>();

  void trust(std::string fingerprint) const { dynamic_trust->add(std::move(fingerprint)); }
  void revoke(const std::string& fingerprint) const { dynamic_trust->remove(fingerprint); }
};

inline constexpr std::string_view kInsecureLoopbackDeviceId = "insecure-loopback";

}  // namespace clusterlm::transport
