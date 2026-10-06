#include "clusterlm/domain/boundary.hpp"

#include "clusterlm/domain/execution_domain.hpp"

namespace clusterlm::domain {

namespace {
// Sanity bound on dimensions read from the wire, before any allocation sized by them.
constexpr std::uint32_t kMaxDim = 1u << 20;
// floats_per_position() is a 32-bit expression (hc*H + H + hc); dimensions that individually pass kMaxDim can
// still wrap it. Bound the true 64-bit value so the layout arithmetic everywhere else cannot overflow.
constexpr std::uint64_t kMaxFloatsPerPosition = 1u << 28;
}  // namespace

BoundaryLayout BoundaryLayout::for_geometry(const objects::ModelGeometry& g) {
  BoundaryLayout l;
  l.abi = BoundaryAbi::kResidualHandoffF32V1;
  l.residual_streams = g.residual_streams;
  l.hidden_size = g.hidden_size;
  return l;
}

std::string_view to_string(StageRole role) {
  switch (role) {
    case StageRole::kPrefix: return "prefix";
    case StageRole::kMiddle: return "middle";
    case StageRole::kTail: return "tail";
  }
  return "unknown";
}

Status StageActivations::validate() const {
  if (layout.abi != BoundaryAbi::kResidualHandoffF32V1)
    return make_error(ErrorCode::kVersionMismatch, "unsupported boundary ABI");
  if (layout.residual_streams == 0 || layout.hidden_size == 0)
    return make_error(ErrorCode::kInvalidArgument, "boundary layout has a zero dimension");
  if (std::uint64_t{layout.residual_streams} * layout.hidden_size + layout.hidden_size + layout.residual_streams >
      kMaxFloatsPerPosition)
    return make_error(ErrorCode::kInvalidArgument, "boundary layout exceeds the maximum record size");
  if (data.size() != std::uint64_t{positions} * layout.floats_per_position())
    return make_error(ErrorCode::kInvalidArgument, "activation payload size does not match positions * layout");
  return Status::ok();
}

void StageActivations::encode(ByteWriter& w) const {
  w.u32(static_cast<std::uint32_t>(layout.abi));
  w.u32(layout.residual_streams);
  w.u32(layout.hidden_size);
  w.u64(first_position);
  w.u32(positions);
  for (float v : data) w.f32(v);
}

Result<StageActivations> StageActivations::decode(ByteReader& r, std::uint32_t max_positions) {
  std::uint32_t abi = 0;
  StageActivations a;
  r.u32(abi);
  r.u32(a.layout.residual_streams);
  r.u32(a.layout.hidden_size);
  r.u64(a.first_position);
  r.u32(a.positions);
  if (!r.ok()) return make_error(ErrorCode::kProtocolError, "activations: truncated header");
  if (abi != static_cast<std::uint32_t>(BoundaryAbi::kResidualHandoffF32V1))
    return make_error(ErrorCode::kVersionMismatch, "activations: unsupported boundary ABI");
  a.layout.abi = BoundaryAbi::kResidualHandoffF32V1;
  if (a.layout.residual_streams == 0 || a.layout.residual_streams > kMaxDim || a.layout.hidden_size == 0 ||
      a.layout.hidden_size > kMaxDim)
    return make_error(ErrorCode::kProtocolError, "activations: implausible dimensions");
  if (std::uint64_t{a.layout.residual_streams} * a.layout.hidden_size + a.layout.hidden_size +
          a.layout.residual_streams >
      kMaxFloatsPerPosition)
    return make_error(ErrorCode::kProtocolError, "activations: implausible dimensions");
  if (a.positions > max_positions)
    return make_error(ErrorCode::kResourceExhausted, "activations: positions exceed the receiver's maximum");
  // Check the bytes are really there before allocating anything sized by header fields.
  const std::uint64_t floats = std::uint64_t{a.positions} * a.layout.floats_per_position();
  if (r.remaining() < floats * 4) return make_error(ErrorCode::kProtocolError, "activations: truncated payload");
  a.data.resize(static_cast<std::size_t>(floats));
  for (float& v : a.data) r.f32(v);
  if (!r.ok()) return make_error(ErrorCode::kProtocolError, "activations: truncated payload");
  return a;
}

}  // namespace clusterlm::domain
