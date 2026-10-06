#include <doctest/doctest.h>

#include "clusterlm/domain/boundary.hpp"
#include "clusterlm/domain/execution_domain.hpp"
#include "clusterlm/objects/fixture_model.hpp"

using namespace clusterlm;
using namespace clusterlm::domain;

namespace {
StageActivations sample(std::uint32_t positions) {
  StageActivations a;
  a.layout = BoundaryLayout::for_geometry(objects::FixtureSpec::tiny().geometry());
  a.first_position = 40;
  a.positions = positions;
  a.data.resize(std::size_t{positions} * a.layout.floats_per_position());
  for (std::size_t i = 0; i < a.data.size(); ++i) a.data[i] = static_cast<float>(i) * 0.25f - 3.0f;
  if (!a.data.empty()) a.data[0] = -0.0f;
  return a;
}
}  // namespace

TEST_CASE("boundary layout follows the geometry") {
  const BoundaryLayout l = BoundaryLayout::for_geometry(objects::FixtureSpec{}.geometry());
  CHECK(l.residual_streams == 4);
  CHECK(l.hidden_size == 64);
  CHECK(l.floats_per_position() == 4 * 64 + 64 + 4);
  CHECK(l.bytes_per_position() == l.floats_per_position() * 4u);
  CHECK(l.pending_offset() == 256);
  CHECK(l.injection_offset() == 320);
  // Flash-Next dimensions from the contract header comment: 51,216 bytes per position.
  BoundaryLayout fn;
  fn.residual_streams = 4;
  fn.hidden_size = 2560;
  CHECK(fn.bytes_per_position() == 51216);
}

TEST_CASE("StageActivations round trip is bit exact") {
  const StageActivations a = sample(3);
  REQUIRE(a.validate().is_ok());
  ByteWriter w;
  a.encode(w);
  CHECK(w.size() == 4 + 4 + 4 + 8 + 4 + a.data.size() * 4);
  ByteReader r(w.bytes());
  auto b = StageActivations::decode(r, 8);
  REQUIRE(b.is_ok());
  CHECK(r.at_end());
  CHECK(b->layout == a.layout);
  CHECK(b->first_position == 40);
  CHECK(b->positions == 3);
  REQUIRE(b->data.size() == a.data.size());
  CHECK(std::memcmp(b->data.data(), a.data.data(), a.data.size() * 4) == 0);  // including the -0.0
  CHECK(b->position(2).size() == a.layout.floats_per_position());
}

TEST_CASE("decode rejects more positions than the receiver allows, before allocating") {
  const StageActivations a = sample(4);
  ByteWriter w;
  a.encode(w);
  ByteReader r(w.bytes());
  CHECK(StageActivations::decode(r, 3).status().code() == ErrorCode::kResourceExhausted);

  // A header claiming a gigantic payload with no bytes behind it must fail cleanly.
  ByteWriter h;
  h.u32(static_cast<std::uint32_t>(BoundaryAbi::kResidualHandoffF32V1));
  h.u32(4);
  h.u32(1u << 20);
  h.u64(0);
  h.u32(0xFFFFFFFFu);
  ByteReader hr(h.bytes());
  CHECK(StageActivations::decode(hr, 0xFFFFFFFFu).status().code() == ErrorCode::kProtocolError);
}

TEST_CASE("decode rejects a wrong ABI, bad dimensions and truncation") {
  const StageActivations a = sample(2);
  ByteWriter w;
  a.encode(w);
  Bytes bad = w.bytes();
  bad[0] ^= 0x7F;  // ABI word
  ByteReader r1(bad);
  CHECK(StageActivations::decode(r1, 8).status().code() == ErrorCode::kVersionMismatch);

  Bytes zero_h = w.bytes();
  zero_h[4] = zero_h[5] = zero_h[6] = zero_h[7] = 0;  // residual_streams = 0
  ByteReader r2(zero_h);
  CHECK_FALSE(StageActivations::decode(r2, 8).is_ok());

  for (std::size_t cut : {std::size_t{0}, std::size_t{3}, std::size_t{20}, w.size() - 1}) {
    ByteReader r(ByteSpan(w.bytes()).subspan(0, cut));
    CHECK_FALSE(StageActivations::decode(r, 8).is_ok());
  }
}

TEST_CASE("validate catches inconsistent payload sizes and unsupported ABI") {
  StageActivations a = sample(2);
  a.data.pop_back();
  CHECK(a.validate().code() == ErrorCode::kInvalidArgument);
  a = sample(2);
  a.positions = 3;
  CHECK_FALSE(a.validate().is_ok());
  a = sample(2);
  a.layout.abi = static_cast<BoundaryAbi>(0x00020001);
  CHECK(a.validate().code() == ErrorCode::kVersionMismatch);
  CHECK(sample(0).validate().is_ok());
}

TEST_CASE("stage roles have stable names") {
  CHECK(to_string(StageRole::kPrefix) == "prefix");
  CHECK(to_string(StageRole::kMiddle) == "middle");
  CHECK(to_string(StageRole::kTail) == "tail");
}
