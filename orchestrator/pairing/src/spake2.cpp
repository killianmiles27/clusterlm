#include "clusterlm/pairing/spake2.hpp"

#include <openssl/bn.h>
#include <openssl/crypto.h>
#include <openssl/ec.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/obj_mac.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include <array>
#include <cstring>

namespace clusterlm::pairing {

namespace {

template <typename T, void (*Free)(T*)>
struct Deleter {
  void operator()(T* p) const { Free(p); }
};
using BnPtr = std::unique_ptr<BIGNUM, Deleter<BIGNUM, BN_clear_free>>;
using PointPtr = std::unique_ptr<EC_POINT, Deleter<EC_POINT, EC_POINT_free>>;
using CtxPtr = std::unique_ptr<BN_CTX, Deleter<BN_CTX, BN_CTX_free>>;

constexpr std::string_view kSalt = "ClusterLM pairing v1 / SPAKE2 password salt";
constexpr int kPbkdf2Iterations = 100000;  // a speed bump only: online guessing is what is rate limited

Status crypto_error(const char* what) {
  ERR_clear_error();
  return make_error(ErrorCode::kInternal, std::string("pairing crypto: ") + what);
}

// The fixed generators M (initiator side) and N (responder side): hash a public label with a counter to an x
// coordinate until it lies on the curve. Nobody knows log_G(M) or log_G(N).
struct Constants {
  EC_GROUP* group = nullptr;
  EC_POINT* m = nullptr;
  EC_POINT* n = nullptr;
  BIGNUM* order = nullptr;
  bool ok = false;

  static EC_POINT* derive(const EC_GROUP* g, std::string_view label, BN_CTX* ctx) {
    BnPtr p(BN_new());
    if (!p || EC_GROUP_get_curve(g, p.get(), nullptr, nullptr, ctx) != 1) return nullptr;
    for (std::uint32_t counter = 0; counter < 1000; ++counter) {
      std::array<unsigned char, 32> digest{};
      EVP_MD_CTX* md = EVP_MD_CTX_new();
      if (md == nullptr) return nullptr;
      unsigned int len = 0;
      const unsigned char c[4] = {static_cast<unsigned char>(counter), static_cast<unsigned char>(counter >> 8),
                                  static_cast<unsigned char>(counter >> 16), static_cast<unsigned char>(counter >> 24)};
      const bool hashed = EVP_DigestInit_ex(md, EVP_sha256(), nullptr) == 1 &&
                          EVP_DigestUpdate(md, label.data(), label.size()) == 1 &&
                          EVP_DigestUpdate(md, c, sizeof c) == 1 && EVP_DigestFinal_ex(md, digest.data(), &len) == 1;
      EVP_MD_CTX_free(md);
      if (!hashed) return nullptr;
      BnPtr x(BN_bin2bn(digest.data(), static_cast<int>(digest.size()), nullptr));
      if (!x || BN_cmp(x.get(), p.get()) >= 0) continue;
      EC_POINT* pt = EC_POINT_new(g);
      if (pt == nullptr) return nullptr;
      if (EC_POINT_set_compressed_coordinates(g, pt, x.get(), 0, ctx) == 1 && EC_POINT_is_on_curve(g, pt, ctx) == 1)
        return pt;
      ERR_clear_error();
      EC_POINT_free(pt);
    }
    return nullptr;
  }

  Constants() {
    CtxPtr ctx(BN_CTX_new());
    group = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
    if (!ctx || group == nullptr) return;
    order = BN_new();
    if (order == nullptr || EC_GROUP_get_order(group, order, ctx.get()) != 1) return;
    m = derive(group, "ClusterLM pairing v1 / SPAKE2 M (initiator)", ctx.get());
    n = derive(group, "ClusterLM pairing v1 / SPAKE2 N (responder)", ctx.get());
    ok = m != nullptr && n != nullptr;
  }
  ~Constants() {
    EC_POINT_free(m);
    EC_POINT_free(n);
    BN_free(order);
    EC_GROUP_free(group);
  }
};

const Constants& constants() {
  static const Constants c;  // initialisation is thread-safe; the group is only read afterwards
  return c;
}

Bytes point_bytes(const EC_GROUP* g, const EC_POINT* p, BN_CTX* ctx) {
  Bytes out(kSpake2ShareBytes);
  const std::size_t n = EC_POINT_point2oct(g, p, POINT_CONVERSION_UNCOMPRESSED, out.data(), out.size(), ctx);
  if (n != kSpake2ShareBytes) out.clear();
  return out;
}

void put_lp(ByteWriter& w, ByteSpan data) { w.blob(data); }

Bytes hmac_sha256(ByteSpan key, ByteSpan data) {
  Bytes out(32);
  unsigned int len = 0;
  if (HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()), data.data(), data.size(), out.data(), &len) == nullptr ||
      len != 32)
    out.clear();
  return out;
}

ByteSpan span_of(std::string_view s) { return ByteSpan(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }

}  // namespace

struct Spake2::Impl {
  Role role = Role::kInitiator;
  BnPtr w, scalar;
  Bytes w_bytes;   // 32-byte big-endian, in the transcript
  Bytes share;     // this side's point
  ~Impl() { OPENSSL_cleanse(w_bytes.data(), w_bytes.size()); }
};

Spake2::Spake2(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Spake2::Spake2(Spake2&&) noexcept = default;
Spake2& Spake2::operator=(Spake2&&) noexcept = default;
Spake2::~Spake2() = default;

const Bytes& Spake2::share() const { return impl_->share; }

Result<Spake2> Spake2::create(Role role, std::string_view code) {
  const Constants& k = constants();
  if (!k.ok) return crypto_error("curve constants");
  if (code.empty() || code.size() > 64) return make_error(ErrorCode::kInvalidArgument, "bad pairing code length");
  CtxPtr ctx(BN_CTX_new());
  auto impl = std::make_unique<Impl>();
  impl->role = role;

  std::array<unsigned char, 48> okm{};
  if (PKCS5_PBKDF2_HMAC(code.data(), static_cast<int>(code.size()), reinterpret_cast<const unsigned char*>(kSalt.data()),
                        static_cast<int>(kSalt.size()), kPbkdf2Iterations, EVP_sha256(), static_cast<int>(okm.size()),
                        okm.data()) != 1)
    return crypto_error("PBKDF2");
  impl->w.reset(BN_bin2bn(okm.data(), static_cast<int>(okm.size()), nullptr));
  OPENSSL_cleanse(okm.data(), okm.size());
  if (!impl->w || BN_nnmod(impl->w.get(), impl->w.get(), k.order, ctx.get()) != 1 || BN_is_zero(impl->w.get()))
    return crypto_error("password scalar");
  impl->w_bytes.resize(32);
  if (BN_bn2binpad(impl->w.get(), impl->w_bytes.data(), 32) != 32) return crypto_error("password scalar encoding");

  impl->scalar.reset(BN_new());
  if (!impl->scalar || BN_rand_range(impl->scalar.get(), k.order) != 1) return crypto_error("random scalar");
  if (BN_is_zero(impl->scalar.get())) BN_one(impl->scalar.get());

  PointPtr pt(EC_POINT_new(k.group));
  if (!pt) return crypto_error("point alloc");
  const EC_POINT* blind = role == Role::kInitiator ? k.m : k.n;
  // share = scalar*G + w*blind
  if (EC_POINT_mul(k.group, pt.get(), impl->scalar.get(), blind, impl->w.get(), ctx.get()) != 1)
    return crypto_error("share computation");
  impl->share = point_bytes(k.group, pt.get(), ctx.get());
  if (impl->share.empty()) return crypto_error("share encoding");
  return Spake2(std::move(impl));
}

Result<Spake2::Keys> Spake2::finish(ByteSpan peer_share, ByteSpan channel_binding) const {
  const Constants& k = constants();
  if (!k.ok) return crypto_error("curve constants");
  if (peer_share.size() != kSpake2ShareBytes || peer_share[0] != 0x04)
    return make_error(ErrorCode::kUnauthenticated, "malformed pairing share");
  CtxPtr ctx(BN_CTX_new());
  PointPtr peer(EC_POINT_new(k.group)), t(EC_POINT_new(k.group)), z(EC_POINT_new(k.group)), v(EC_POINT_new(k.group)),
      unblind(EC_POINT_new(k.group));
  if (!ctx || !peer || !t || !z || !v || !unblind) return crypto_error("alloc");
  if (EC_POINT_oct2point(k.group, peer.get(), peer_share.data(), peer_share.size(), ctx.get()) != 1 ||
      EC_POINT_is_on_curve(k.group, peer.get(), ctx.get()) != 1 || EC_POINT_is_at_infinity(k.group, peer.get()) == 1) {
    ERR_clear_error();
    return make_error(ErrorCode::kUnauthenticated, "pairing share is not a valid curve point");
  }
  // T = peer - w * (blind of the PEER's role)
  const EC_POINT* peer_blind = impl_->role == Role::kInitiator ? k.n : k.m;
  if (EC_POINT_mul(k.group, unblind.get(), nullptr, peer_blind, impl_->w.get(), ctx.get()) != 1 ||
      EC_POINT_invert(k.group, unblind.get(), ctx.get()) != 1 ||
      EC_POINT_add(k.group, t.get(), peer.get(), unblind.get(), ctx.get()) != 1)
    return crypto_error("unblinding");
  if (EC_POINT_is_at_infinity(k.group, t.get()) == 1)
    return make_error(ErrorCode::kUnauthenticated, "degenerate pairing share");
  // Z = scalar*T. V: initiator w*T; responder (scalar*w)*G (equal when the passwords match).
  if (EC_POINT_mul(k.group, z.get(), nullptr, t.get(), impl_->scalar.get(), ctx.get()) != 1) return crypto_error("Z");
  if (impl_->role == Role::kInitiator) {
    if (EC_POINT_mul(k.group, v.get(), nullptr, t.get(), impl_->w.get(), ctx.get()) != 1) return crypto_error("V");
  } else {
    BnPtr sw(BN_new());
    if (!sw || BN_mod_mul(sw.get(), impl_->scalar.get(), impl_->w.get(), k.order, ctx.get()) != 1 ||
        EC_POINT_mul(k.group, v.get(), sw.get(), nullptr, nullptr, ctx.get()) != 1)
      return crypto_error("V");
  }
  const Bytes zb = point_bytes(k.group, z.get(), ctx.get());
  const Bytes vb = point_bytes(k.group, v.get(), ctx.get());
  if (zb.empty() || vb.empty()) return make_error(ErrorCode::kUnauthenticated, "degenerate pairing secret");

  const bool init = impl_->role == Role::kInitiator;
  const Bytes peer_copy(peer_share.begin(), peer_share.end());
  const Bytes& xs = init ? impl_->share : peer_copy;
  const Bytes& ys = init ? peer_copy : impl_->share;

  ByteWriter tt;
  put_lp(tt, span_of("ClusterLM pairing v1 / SPAKE2 transcript"));
  put_lp(tt, channel_binding);
  put_lp(tt, xs);
  put_lp(tt, ys);
  put_lp(tt, zb);
  put_lp(tt, vb);
  put_lp(tt, impl_->w_bytes);
  const Bytes tt_bytes = std::move(tt).take();
  Bytes key(32);
  unsigned int len = 0;
  if (EVP_Digest(tt_bytes.data(), tt_bytes.size(), key.data(), &len, EVP_sha256(), nullptr) != 1 || len != 32)
    return crypto_error("transcript hash");

  Keys out;
  out.confirm_initiator = hmac_sha256(key, span_of("ClusterLM pairing v1 / confirm / initiator"));
  out.confirm_responder = hmac_sha256(key, span_of("ClusterLM pairing v1 / confirm / responder"));
  out.session_key = hmac_sha256(key, span_of("ClusterLM pairing v1 / session"));
  OPENSSL_cleanse(key.data(), key.size());
  if (out.confirm_initiator.empty() || out.confirm_responder.empty() || out.session_key.empty())
    return crypto_error("key derivation");
  return out;
}

bool constant_time_equal(ByteSpan a, ByteSpan b) {
  return a.size() == b.size() && CRYPTO_memcmp(a.data(), b.data(), a.size()) == 0;
}

}  // namespace clusterlm::pairing
