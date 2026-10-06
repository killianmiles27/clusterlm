#pragma once
#include <openssl/evp.h>
#include <openssl/x509.h>

#include <string>

#include "clusterlm/transport/security.hpp"

namespace clusterlm::transport {

struct DeviceIdentity::Impl {
  EVP_PKEY* key = nullptr;
  X509* cert = nullptr;
  std::string fingerprint;
  std::string common_name;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl() {
    EVP_PKEY_free(key);
    X509_free(cert);
  }
};

// Drains the thread's OpenSSL error queue into one line.
std::string openssl_errors();
// Lowercase hex SHA-256 of the DER encoding; empty on failure.
std::string cert_fingerprint(X509* cert);

}  // namespace clusterlm::transport
