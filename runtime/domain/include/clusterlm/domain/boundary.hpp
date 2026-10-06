#pragma once
// StageBoundary: the activation representation exchanged between pipeline stages.
//
// ABI kResidualHandoffF32V1 mirrors Strata's verifier handoff (verify.hpp, `handoff_floats`):
//   hc residual streams of width H, one pending block-output vector of width H, and hc injection values,
// i.e. (hc*H + H + hc) FP32 values per token position, explicit little-endian, contiguous, position-major.
// For Flash-Next (hc=4, H=2560) that is 51,216 bytes per position. The next stage folds the pending write
// into its first operation. Changing precision or dropping the pending vector is a different ABI, never a
// silent change.
//
// A StageBoundary carries positions, dimensions and identities — never vocabulary token IDs.
#include <cstdint>
#include <span>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/objects/geometry.hpp"

namespace clusterlm::domain {

enum class BoundaryAbi : std::uint32_t {
  kResidualHandoffF32V1 = 0x0001'0001,
};

struct BoundaryLayout {
  BoundaryAbi abi = BoundaryAbi::kResidualHandoffF32V1;
  std::uint32_t residual_streams = 0;  // hc
  std::uint32_t hidden_size = 0;       // H

  static BoundaryLayout for_geometry(const objects::ModelGeometry& g);
  std::uint32_t floats_per_position() const { return residual_streams * hidden_size + hidden_size + residual_streams; }
  std::uint64_t bytes_per_position() const { return std::uint64_t{floats_per_position()} * 4; }
  // Offsets (in floats) within one position's record.
  std::uint32_t streams_offset() const { return 0; }
  std::uint32_t pending_offset() const { return residual_streams * hidden_size; }
  std::uint32_t injection_offset() const { return residual_streams * hidden_size + hidden_size; }
  friend bool operator==(const BoundaryLayout&, const BoundaryLayout&) = default;
};

// Activations for `positions` consecutive token positions starting at `first_position`.
struct StageActivations {
  BoundaryLayout layout;
  std::uint64_t first_position = 0;
  std::uint32_t positions = 0;
  std::vector<float> data;  // positions * layout.floats_per_position()

  std::span<float> position(std::uint32_t i) {
    return std::span<float>(data).subspan(std::size_t{i} * layout.floats_per_position(), layout.floats_per_position());
  }
  std::span<const float> position(std::uint32_t i) const {
    return std::span<const float>(data).subspan(std::size_t{i} * layout.floats_per_position(),
                                                layout.floats_per_position());
  }
  Status validate() const;

  // Wire encoding: header (abi, hc, H, first_position, positions) + raw little-endian FP32 payload.
  void encode(ByteWriter& w) const;
  // `max_positions` bounds receive memory before any allocation happens.
  static Result<StageActivations> decode(ByteReader& r, std::uint32_t max_positions);
};

}  // namespace clusterlm::domain
