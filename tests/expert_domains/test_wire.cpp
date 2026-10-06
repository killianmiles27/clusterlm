#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "clusterlm/expert_domains/wire.hpp"

using namespace clusterlm;
using namespace clusterlm::expert_domains;

namespace {

ExpertBatch sample_batch(std::uint32_t q = 3, std::uint32_t h = 8) {
  ExpertBatch b;
  b.epoch = Epoch{7};
  b.window = WindowId{42};
  b.layer = 5;
  b.positions = q;
  b.hidden = h;
  for (std::uint32_t i = 0; i < q * h; ++i) b.activations.push_back(0.25f * static_cast<float>(i) - 1.0f);
  b.routes.resize(q);
  for (std::uint32_t p = 0; p < q; ++p)
    for (std::uint32_t k = 0; k < 3; ++k) b.routes[p].push_back({k * 2 + p, 0.1f * static_cast<float>(k + 1)});
  if (q > 1) b.routes[1].clear();  // a position may route nothing to this domain
  return b;
}

ExpertDecodeLimits limits() {
  ExpertDecodeLimits l;
  l.max_positions = 4;
  l.max_hidden = 64;
  l.max_routes_per_position = 8;
  l.max_local_experts = 32;
  l.max_layer = 16;
  return l;
}

// Patches a little-endian u32 at `offset`.
void put_u32(Bytes& b, std::size_t offset, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) b[offset + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v >> (8 * i));
}
constexpr std::size_t kLayerOff = 16, kPositionsOff = 20, kHiddenOff = 24, kBodyOff = 28;

}  // namespace

TEST_CASE("batch, result and error round-trip exactly") {
  const ExpertBatch b = sample_batch();
  auto back = decode_batch(encode(b), limits());
  REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
  CHECK(*back == b);

  ExpertResult r;
  r.epoch = Epoch{7};
  r.window = WindowId{42};
  r.layer = 5;
  r.positions = 3;
  r.hidden = 8;
  r.partial.assign(24, 1.5f);
  r.compute_ns = 12345;
  r.experts_executed = 6;
  auto rb = decode_result(encode(r), limits());
  REQUIRE(rb.is_ok());
  CHECK(*rb == r);

  ExpertError e;
  e.code = ErrorCode::kStaleEpoch;
  e.message = "stale";
  auto eb = decode_error(encode(e), limits());
  REQUIRE(eb.is_ok());
  CHECK(eb->code == ErrorCode::kStaleEpoch);
  CHECK(eb->message == "stale");
}

TEST_CASE("frames carry the expert message types on the expert channel") {
  CHECK(to_frame(sample_batch(), 9).type == kMsgExpertBatch);
  CHECK(to_frame(sample_batch(), 9).channel == kExpertChannel);
  CHECK(to_frame(sample_batch(), 9).correlation == 9);
  CHECK(to_frame(ExpertResult{}, 1).type == kMsgExpertResult);
  CHECK(to_frame(ExpertError{}, 1).type == kMsgExpertError);
}

TEST_CASE("decode rejects every truncation and any trailing byte") {
  const Bytes full = encode(sample_batch());
  for (std::size_t n = 0; n < full.size(); ++n) {
    Bytes cut(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(n));
    CHECK_FALSE(decode_batch(cut, limits()).is_ok());
  }
  Bytes extra = full;
  extra.push_back(0);
  CHECK_FALSE(decode_batch(extra, limits()).is_ok());

  ExpertResult r;
  r.positions = 1;
  r.hidden = 4;
  r.partial.assign(4, 0.0f);
  const Bytes rfull = encode(r);
  for (std::size_t n = 0; n < rfull.size(); ++n)
    CHECK_FALSE(decode_result(Bytes(rfull.begin(), rfull.begin() + static_cast<std::ptrdiff_t>(n)), limits()).is_ok());
}

TEST_CASE("every dimension is bounded before allocation") {
  const Bytes good = encode(sample_batch());
  {  // a header that promises far more than the payload holds must fail without allocating that much
    Bytes b = good;
    put_u32(b, kPositionsOff, 4);
    put_u32(b, kHiddenOff, 64);  // legal under the limits but the payload is 8 floats wide
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
  }
  {
    Bytes b = good;
    put_u32(b, kPositionsOff, 0xFFFFFFFFu);
    auto r = decode_batch(b, limits());
    REQUIRE_FALSE(r.is_ok());
    CHECK(r.status().code() == ErrorCode::kProtocolError);
  }
  {
    Bytes b = good;
    put_u32(b, kPositionsOff, 5);  // > max_positions (4)
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
    put_u32(b, kPositionsOff, 0);
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
  }
  {
    Bytes b = good;
    put_u32(b, kHiddenOff, 65);
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
    put_u32(b, kHiddenOff, 0);
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
    put_u32(b, kHiddenOff, 0xFFFFFFFFu);
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
  }
  {
    Bytes b = good;
    put_u32(b, kLayerOff, 16);  // == max_layer
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
  }
  {  // the per-position route count is bounded
    ExpertBatch m = sample_batch(1, 8);
    m.routes[0].clear();
    for (std::uint32_t k = 0; k < 9; ++k) m.routes[0].push_back({k, 0.1f});  // 9 > max_routes_per_position (8)
    CHECK_FALSE(decode_batch(encode(m), limits()).is_ok());
    m.routes[0].resize(8);
    CHECK(decode_batch(encode(m), limits()).is_ok());
  }
  {  // a route count that exceeds the remaining bytes
    ExpertBatch m = sample_batch(1, 8);
    Bytes b = encode(m);
    const std::size_t counts_at = kBodyOff + 8 * 4;
    b[counts_at] = 8;
    b.resize(b.size() - 8);  // keep fewer routes than the count claims
    CHECK_FALSE(decode_batch(b, limits()).is_ok());
  }
}

TEST_CASE("route and activation contents are validated") {
  {  // local expert id beyond the limit
    ExpertBatch m = sample_batch();
    m.routes[0].back().local_expert = 32;
    CHECK_FALSE(decode_batch(encode(m), limits()).is_ok());
  }
  {  // not strictly ascending: duplicate, then descending
    ExpertBatch m = sample_batch();
    m.routes[0] = {{1, 0.5f}, {1, 0.5f}};
    CHECK_FALSE(decode_batch(encode(m), limits()).is_ok());
    m.routes[0] = {{3, 0.5f}, {2, 0.5f}};
    CHECK_FALSE(decode_batch(encode(m), limits()).is_ok());
  }
  {  // non-finite weight / activation
    ExpertBatch m = sample_batch();
    m.routes[0][0].weight = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(decode_batch(encode(m), limits()).is_ok());
    m = sample_batch();
    m.activations[3] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(decode_batch(encode(m), limits()).is_ok());
  }
  {  // error message length is bounded
    ExpertError e;
    e.message = std::string(10000, 'x');
    CHECK_FALSE(decode_error(encode(e), limits()).is_ok());
  }
}

TEST_CASE("the payload bound covers the largest legal message") {
  const ExpertDecodeLimits l = limits();
  ExpertBatch m = sample_batch(l.max_positions, l.max_hidden);
  for (auto& per : m.routes) {
    per.clear();
    for (std::uint32_t k = 0; k < l.max_routes_per_position; ++k) per.push_back({k, 0.01f});
  }
  CHECK(encode(m).size() <= l.max_payload_bytes());
  CHECK(decode_batch(encode(m), l).is_ok());
}

TEST_CASE("the wire format carries no token ids, only activations and route metadata") {
  // The batch layout is fully described by its dimensions: header + q*H floats + routes. There is no field
  // that could hold a vocabulary id, and the encoded size is a pure function of the dimensions.
  const ExpertBatch b = sample_batch(3, 8);
  std::size_t route_count = 0;
  for (const auto& per : b.routes) route_count += per.size();
  CHECK(encode(b).size() == 28 + 3 * 8 * 4 + 3 * 2 + route_count * 8);
}
