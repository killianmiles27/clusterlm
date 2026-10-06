#include <doctest/doctest.h>

#include <cstring>
#include <vector>

#include "clusterlm/backends/strata_handoff.hpp"

using namespace clusterlm;

TEST_CASE("Strata field-major handoff <-> ClusterLM position-major wire is an exact permutation") {
  domain::BoundaryLayout layout;
  layout.residual_streams = 4;
  layout.hidden_size = 8;
  const std::uint32_t T = 3;
  const std::size_t per = layout.floats_per_position();
  std::vector<float> wire(T * per);
  for (std::size_t i = 0; i < wire.size(); ++i) wire[i] = static_cast<float>(i) * 0.5f - 7.25f;
  wire[5] = -0.0f;  // signed zero must survive

  std::vector<float> strata(wire.size()), back(wire.size());
  REQUIRE(backends::strata::wire_to_strata(layout, T, wire, strata).is_ok());
  // Field-major: all residual rows first, then all pending vectors, then all injections.
  const std::size_t r = 4 * 8;
  CHECK(strata[1 * r + 0] == wire[1 * per + layout.streams_offset()]);
  CHECK(strata[T * r + 2 * 8 + 3] == wire[2 * per + layout.pending_offset() + 3]);
  CHECK(strata[T * r + T * 8 + 1 * 4 + 2] == wire[1 * per + layout.injection_offset() + 2]);
  REQUIRE(backends::strata::strata_to_wire(layout, T, strata, back).is_ok());
  CHECK(std::memcmp(back.data(), wire.data(), wire.size() * sizeof(float)) == 0);
}

TEST_CASE("handoff conversion rejects size mismatches") {
  domain::BoundaryLayout layout;
  layout.residual_streams = 4;
  layout.hidden_size = 8;
  std::vector<float> a(layout.floats_per_position() * 2), b(layout.floats_per_position());
  CHECK_FALSE(backends::strata::wire_to_strata(layout, 2, a, b).is_ok());
}
