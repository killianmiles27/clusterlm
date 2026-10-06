#include "clusterlm/common/digest.hpp"

#include <openssl/evp.h>

#include <stdexcept>

namespace clusterlm {

Result<Digest256> Digest256::from_hex(std::string_view hex) {
  CLM_ASSIGN_OR_RETURN(Bytes raw, clusterlm::from_hex(hex));
  if (raw.size() != 32) return make_error(ErrorCode::kInvalidArgument, "digest must be 32 bytes");
  Digest256 d;
  std::copy(raw.begin(), raw.end(), d.bytes.begin());
  return d;
}

bool Digest256::is_zero() const noexcept {
  for (auto b : bytes)
    if (b != 0) return false;
  return true;
}

struct Sha256::Impl {
  EVP_MD_CTX* ctx = nullptr;
};

Sha256::Sha256() : impl_(std::make_unique<Impl>()) {
  impl_->ctx = EVP_MD_CTX_new();
  if (impl_->ctx == nullptr || EVP_DigestInit_ex(impl_->ctx, EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA-256 initialization failed");
}

Sha256::~Sha256() {
  if (impl_ && impl_->ctx) EVP_MD_CTX_free(impl_->ctx);
}

Sha256::Sha256(Sha256&&) noexcept = default;
Sha256& Sha256::operator=(Sha256&&) noexcept = default;

void Sha256::update(ByteSpan data) {
  if (!data.empty()) EVP_DigestUpdate(impl_->ctx, data.data(), data.size());
}

Digest256 Sha256::finish() {
  Digest256 d;
  unsigned int len = 0;
  EVP_DigestFinal_ex(impl_->ctx, d.bytes.data(), &len);
  EVP_DigestInit_ex(impl_->ctx, EVP_sha256(), nullptr);
  return d;
}

Digest256 Sha256::of(ByteSpan data) {
  Sha256 h;
  h.update(data);
  return h.finish();
}

Digest256 Sha256::of(std::string_view s) {
  Sha256 h;
  h.update(s);
  return h.finish();
}

}  // namespace clusterlm
