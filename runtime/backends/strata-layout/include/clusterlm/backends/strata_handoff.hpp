#pragma once
// Conversion between ClusterLM's wire boundary ABI and Strata's native verifier handoff buffer.
//
// Both carry the same (hc·H + H + hc) FP32 values per position in the same field order — residual streams,
// pending block output, injections — but the layouts differ (pinned Strata 1735d64, src/core/verify.cpp
// 682-688 reader, 1335-1340 writer):
//   ClusterLM wire  (position-major): for each position t: [R_t: hc·H][bo_t: H][inj_t: hc]
//   Strata handoff  (field-major):    [R: T × hc·H][bo: T × H][inj: T × hc]
// The wire stays position-major so its layout is independent of window length. These functions are exact
// copies (no arithmetic), so a round trip is bit-identical. Built unconditionally: the conversion is
// platform-independent and tested without CUDA.
#include <cstdint>
#include <span>

#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/boundary.hpp"

namespace clusterlm::backends::strata {

// `wire` holds positions × layout.floats_per_position() floats; `strata` receives the field-major buffer of the
// same size.
Status wire_to_strata(const domain::BoundaryLayout& layout, std::uint32_t positions, std::span<const float> wire,
                      std::span<float> strata);
Status strata_to_wire(const domain::BoundaryLayout& layout, std::uint32_t positions, std::span<const float> strata,
                      std::span<float> wire);

}  // namespace clusterlm::backends::strata
