#pragma once
// Wire messages of the EXPERIMENTAL grouped expert-domain topology (P0-C).
//
// They are deliberately NOT part of runtime/protocol: the production protocol has no message that carries
// routing metadata. They reuse the production transport (framing, TLS pinning, impairment, fault injection)
// through transport::Frame, with their own type range 0x0E00.. on channel kExpertChannel.
//
// PRIVACY: this topology sends selected expert IDs and weights to the domain that owns them (the addendum allows
// that for this experimental topology only). It still never sends token IDs, logits or text. Route IDs are
// Father-local derived data; nothing here may be written to a log.
//
//   ExpertBatch  Father -> domain, ONE per participating domain per layer:
//       u64 epoch | u64 window | u32 layer | u32 positions q | u32 hidden H | f32 act[q*H]
//       then per position: u16 n | n x (u32 local_expert, f32 weight)   (local ids strictly ascending)
//   ExpertResult domain -> Father, ONE per batch: the weighted partial sum per position
//       u64 epoch | u64 window | u32 layer | u32 q | u32 H | f32 partial[q*H] | u64 compute_ns | u32 experts
//   ExpertError  domain -> Father instead of a result: u16 code | str message
//
// Every count is bounded by ExpertDecodeLimits BEFORE anything is allocated.
#include <cstdint>
#include <string>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/ids.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::expert_domains {

inline constexpr std::uint16_t kMsgExpertBatch = 0x0E01;
inline constexpr std::uint16_t kMsgExpertResult = 0x0E02;
inline constexpr std::uint16_t kMsgExpertError = 0x0E03;
inline constexpr std::uint8_t kExpertChannel = 1;  // same lane as activation traffic in the layer-domain design

struct ExpertDecodeLimits {
  std::uint32_t max_positions = 16;
  std::uint32_t max_hidden = 16384;
  std::uint32_t max_routes_per_position = 64;
  std::uint32_t max_local_experts = 8192;
  std::uint32_t max_layer = 1024;
  std::uint32_t max_error_message = 512;

  // Upper bound on any ExpertBatch/ExpertResult payload under these limits (for transport set_max_payload).
  std::uint32_t max_payload_bytes() const {
    const std::uint64_t acts = std::uint64_t{max_positions} * max_hidden * 4;
    const std::uint64_t routes = std::uint64_t{max_positions} * (2 + std::uint64_t{max_routes_per_position} * 8);
    return static_cast<std::uint32_t>(acts + routes + 128 + max_error_message);
  }
};

struct ExpertRoute {
  std::uint32_t local_expert = 0;  // index into the receiving domain's own expert list
  float weight = 0.0f;             // normalized router weight, applied by the domain
  friend bool operator==(const ExpertRoute&, const ExpertRoute&) = default;
};

struct ExpertBatch {
  Epoch epoch;
  WindowId window;
  std::uint32_t layer = 0;
  std::uint32_t positions = 0;
  std::uint32_t hidden = 0;
  std::vector<float> activations;                 // positions * hidden
  std::vector<std::vector<ExpertRoute>> routes;   // per position; may be empty for a position
  friend bool operator==(const ExpertBatch&, const ExpertBatch&) = default;
};

struct ExpertResult {
  Epoch epoch;
  WindowId window;
  std::uint32_t layer = 0;
  std::uint32_t positions = 0;
  std::uint32_t hidden = 0;
  std::vector<float> partial;          // positions * hidden: sum_k weight_k * expert_k(act) over this domain
  std::uint64_t compute_ns = 0;        // domain-side compute time (timing only)
  std::uint32_t experts_executed = 0;  // (position, expert) executions, a count only
  friend bool operator==(const ExpertResult&, const ExpertResult&) = default;
};

struct ExpertError {
  ErrorCode code = ErrorCode::kInternal;
  std::string message;
};

Bytes encode(const ExpertBatch& m);
Bytes encode(const ExpertResult& m);
Bytes encode(const ExpertError& m);

Result<ExpertBatch> decode_batch(ByteSpan payload, const ExpertDecodeLimits& limits);
Result<ExpertResult> decode_result(ByteSpan payload, const ExpertDecodeLimits& limits);
Result<ExpertError> decode_error(ByteSpan payload, const ExpertDecodeLimits& limits);

transport::Frame to_frame(const ExpertBatch& m, std::uint64_t correlation);
transport::Frame to_frame(const ExpertResult& m, std::uint64_t correlation);
transport::Frame to_frame(const ExpertError& m, std::uint64_t correlation);

}  // namespace clusterlm::expert_domains
