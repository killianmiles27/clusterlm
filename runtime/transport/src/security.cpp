#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/x509.h>

#include <cstdio>
#include <memory>
#include <system_error>

#include "clusterlm/common/digest.hpp"
#include "security_internal.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace clusterlm::transport {
namespace {

constexpr const char* kKeyFile = "device_key.pem";
constexpr const char* kCertFile = "device_cert.pem";

struct BioDeleter {
  void operator()(BIO* b) const { BIO_free_all(b); }
};
using BioPtr = std::unique_ptr<BIO, BioDeleter>;

Status openssl_error(ErrorCode code, const std::string& what) {
  return make_error(code, what + ": " + openssl_errors());
}

Result<std::shared_ptr<const DeviceIdentity::Impl>> finish_impl(std::unique_ptr<DeviceIdentity::Impl> impl) {
  impl->fingerprint = cert_fingerprint(impl->cert);
  if (impl->fingerprint.empty()) return openssl_error(ErrorCode::kInternal, "certificate fingerprint");
  X509_NAME* subject = X509_get_subject_name(impl->cert);
  char cn[256] = {0};
  if (subject != nullptr) X509_NAME_get_text_by_NID(subject, NID_commonName, cn, sizeof cn);
  impl->common_name = cn;
  return std::shared_ptr<const DeviceIdentity::Impl>(std::move(impl));
}

}  // namespace

std::string openssl_errors() {
  std::string out;
  unsigned long e;
  while ((e = ERR_get_error()) != 0) {
    char buf[256];
    ERR_error_string_n(e, buf, sizeof buf);
    if (!out.empty()) out += "; ";
    out += buf;
  }
  return out.empty() ? "unknown OpenSSL error" : out;
}

std::string cert_fingerprint(X509* cert) {
  unsigned char* der = nullptr;
  const int len = i2d_X509(cert, &der);
  if (len <= 0 || der == nullptr) return {};
  const Digest256 d = Sha256::of(ByteSpan(der, static_cast<std::size_t>(len)));
  OPENSSL_free(der);
  return d.hex();
}

Result<DeviceIdentity> DeviceIdentity::generate(std::string_view common_name) {
  if (common_name.empty() || common_name.size() > 64)
    return make_error(ErrorCode::kInvalidArgument, "common name must be 1..64 characters");
  auto impl = std::make_unique<Impl>();
  impl->key = EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256");
  if (impl->key == nullptr) return openssl_error(ErrorCode::kInternal, "EC P-256 key generation");

  impl->cert = X509_new();
  if (impl->cert == nullptr) return openssl_error(ErrorCode::kInternal, "X509_new");
  X509* x = impl->cert;
  X509_set_version(x, 2);  // v3

  BIGNUM* serial = BN_new();
  if (serial == nullptr || BN_rand(serial, 63, BN_RAND_TOP_ANY, BN_RAND_BOTTOM_ANY) != 1 ||
      BN_to_ASN1_INTEGER(serial, X509_get_serialNumber(x)) == nullptr) {
    BN_free(serial);
    return openssl_error(ErrorCode::kInternal, "certificate serial");
  }
  BN_free(serial);

  if (X509_gmtime_adj(X509_getm_notBefore(x), -24L * 3600) == nullptr ||
      X509_gmtime_adj(X509_getm_notAfter(x), 3650L * 24 * 3600) == nullptr)
    return openssl_error(ErrorCode::kInternal, "certificate validity");

  X509_NAME* name = X509_get_subject_name(x);
  const std::string cn(common_name);
  if (X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_UTF8, reinterpret_cast<const unsigned char*>(cn.data()),
                                 static_cast<int>(cn.size()), -1, 0) != 1 ||
      X509_set_issuer_name(x, name) != 1 || X509_set_pubkey(x, impl->key) != 1 ||
      X509_sign(x, impl->key, EVP_sha256()) <= 0)
    return openssl_error(ErrorCode::kInternal, "certificate construction");

  CLM_ASSIGN_OR_RETURN(auto shared, finish_impl(std::move(impl)));
  DeviceIdentity id;
  id.impl_ = std::move(shared);
  return id;
}

Result<DeviceIdentity> DeviceIdentity::load(const std::filesystem::path& dir) {
  const auto key_path = (dir / kKeyFile).string();
  const auto cert_path = (dir / kCertFile).string();
  auto impl = std::make_unique<Impl>();
  {
    BioPtr kb(BIO_new_file(key_path.c_str(), "rb"));
    if (!kb) return make_error(ErrorCode::kNotFound, "cannot open " + key_path);
    impl->key = PEM_read_bio_PrivateKey(kb.get(), nullptr, nullptr, nullptr);
    if (impl->key == nullptr) return openssl_error(ErrorCode::kDataLoss, "parse " + key_path);
  }
  {
    BioPtr cb(BIO_new_file(cert_path.c_str(), "rb"));
    if (!cb) return make_error(ErrorCode::kNotFound, "cannot open " + cert_path);
    impl->cert = PEM_read_bio_X509(cb.get(), nullptr, nullptr, nullptr);
    if (impl->cert == nullptr) return openssl_error(ErrorCode::kDataLoss, "parse " + cert_path);
  }
  if (X509_check_private_key(impl->cert, impl->key) != 1)
    return make_error(ErrorCode::kDataLoss, "device key does not match device certificate");
  CLM_ASSIGN_OR_RETURN(auto shared, finish_impl(std::move(impl)));
  DeviceIdentity id;
  id.impl_ = std::move(shared);
  return id;
}

Status DeviceIdentity::save(const std::filesystem::path& dir) const {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return make_error(ErrorCode::kUnavailable, "cannot create " + dir.string() + ": " + ec.message());

  const auto cert_path = (dir / kCertFile).string();
  {
    BioPtr cb(BIO_new_file(cert_path.c_str(), "wb"));
    if (!cb || PEM_write_bio_X509(cb.get(), impl_->cert) != 1)
      return openssl_error(ErrorCode::kUnavailable, "write " + cert_path);
  }

  const auto key_path = (dir / kKeyFile).string();
#ifndef _WIN32
  // Create with 0600 from the start so the key is never world-readable, even briefly.
  const int fd = ::open(key_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) return make_error(ErrorCode::kUnavailable, "cannot create " + key_path);
  (void)::fchmod(fd, 0600);  // tighten a pre-existing file too
  FILE* fp = ::fdopen(fd, "wb");
  if (fp == nullptr) {
    ::close(fd);
    return make_error(ErrorCode::kUnavailable, "fdopen " + key_path);
  }
  const int ok = PEM_write_PrivateKey(fp, impl_->key, nullptr, nullptr, 0, nullptr, nullptr);
  const int cl = std::fclose(fp);
  if (ok != 1 || cl != 0) return openssl_error(ErrorCode::kUnavailable, "write " + key_path);
#else
  BioPtr kb(BIO_new_file(key_path.c_str(), "wb"));
  if (!kb || PEM_write_bio_PrivateKey(kb.get(), impl_->key, nullptr, nullptr, 0, nullptr, nullptr) != 1)
    return openssl_error(ErrorCode::kUnavailable, "write " + key_path);
#endif
  return Status::ok();
}

const std::string& DeviceIdentity::fingerprint() const { return impl_->fingerprint; }
const std::string& DeviceIdentity::common_name() const { return impl_->common_name; }

}  // namespace clusterlm::transport
