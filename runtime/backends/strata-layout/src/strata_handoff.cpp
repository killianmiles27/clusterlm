#include "clusterlm/backends/strata_handoff.hpp"

#include <algorithm>

namespace clusterlm::backends::strata {
namespace {

Status check(const domain::BoundaryLayout& layout, std::uint32_t positions, std::size_t a, std::size_t b) {
  if (layout.abi != domain::BoundaryAbi::kResidualHandoffF32V1)
    return make_error(ErrorCode::kVersionMismatch, "unsupported boundary ABI for the Strata handoff");
  const std::size_t need = std::size_t{positions} * layout.floats_per_position();
  if (a != need || b != need) return make_error(ErrorCode::kInvalidArgument, "handoff buffer size mismatch");
  return Status::ok();
}

}  // namespace

Status wire_to_strata(const domain::BoundaryLayout& layout, std::uint32_t positions, std::span<const float> wire,
                      std::span<float> strata) {
  CLM_RETURN_IF_ERROR(check(layout, positions, wire.size(), strata.size()));
  const std::size_t r = std::size_t{layout.residual_streams} * layout.hidden_size, h = layout.hidden_size,
                    c = layout.residual_streams, per = layout.floats_per_position(), T = positions;
  float* R = strata.data();
  float* bo = R + T * r;
  float* inj = bo + T * h;
  for (std::size_t t = 0; t < T; ++t) {
    const float* src = wire.data() + t * per;
    std::copy_n(src + layout.streams_offset(), r, R + t * r);
    std::copy_n(src + layout.pending_offset(), h, bo + t * h);
    std::copy_n(src + layout.injection_offset(), c, inj + t * c);
  }
  return Status::ok();
}

Status strata_to_wire(const domain::BoundaryLayout& layout, std::uint32_t positions, std::span<const float> strata,
                      std::span<float> wire) {
  CLM_RETURN_IF_ERROR(check(layout, positions, strata.size(), wire.size()));
  const std::size_t r = std::size_t{layout.residual_streams} * layout.hidden_size, h = layout.hidden_size,
                    c = layout.residual_streams, per = layout.floats_per_position(), T = positions;
  const float* R = strata.data();
  const float* bo = R + T * r;
  const float* inj = bo + T * h;
  for (std::size_t t = 0; t < T; ++t) {
    float* dst = wire.data() + t * per;
    std::copy_n(R + t * r, r, dst + layout.streams_offset());
    std::copy_n(bo + t * h, h, dst + layout.pending_offset());
    std::copy_n(inj + t * c, c, dst + layout.injection_offset());
  }
  return Status::ok();
}

}  // namespace clusterlm::backends::strata
