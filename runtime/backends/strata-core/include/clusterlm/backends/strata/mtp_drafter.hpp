#pragma once
// StrataMtpDrafter: ClusterLM's Drafter over Strata's MTP head (strata::core::MtpDrafter), bound to the Father tail.
//
// The MTP layer consumes token IDs and the main model's FINAL residual rows (Verifier::final_R_all of the tail's last
// window), so it can only live beside the tail domain on Father; nothing it reads or produces crosses the network.
// The tail engine implements MtpEngine (the CUDA engine does, with Strata's own MtpDrafter, its draft head and
// coupled draft sampling); this class adapts it to the domain::Drafter contract:
//   * draft()/propose() return exactly `count` tokens (the coordinator verifies [next_token, drafts...]);
//   * for stochastic speculative sampling, propose() returns the drafter's own per-draft distributions (`probs`,
//     vocabulary-sized), so verify_speculative keeps the target distribution exact;
//   * an engine failure never fails generation: the drafter falls back to a deterministic proposal (one-hot Q, valid
//     for exact speculative sampling - it only lowers acceptance) and records the error for diagnostics.
#include <cstdint>
#include <span>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/domain/sampling.hpp"

namespace clusterlm::backends::strata {

class MtpEngine {
 public:
  virtual ~MtpEngine() = default;
  virtual std::uint32_t mtp_vocab() const = 0;
  // `committed`: every committed token (Father's history); `next_token`: the token after them, not yet committed.
  // Fills `drafts` with up to `count` tokens expected to follow next_token; with `sampling` (temperature > 0) the
  // drafts are sampled and `probs` (if non-null) receives one distribution per draft.
  virtual Status mtp_draft(std::span<const std::int32_t> committed, std::int32_t next_token, std::uint32_t count,
                           const domain::SamplingParams* sampling, std::vector<std::int32_t>& drafts,
                           std::vector<std::vector<float>>* probs) = 0;
};

class StrataMtpDrafter final : public domain::Drafter {
 public:
  explicit StrataMtpDrafter(MtpEngine& engine) : engine_(engine) {}

  std::vector<std::int32_t> draft(std::span<const std::int32_t> committed_tokens, std::int32_t next_token,
                                  std::uint32_t count) override;
  domain::DraftProposal propose(std::span<const std::int32_t> committed_tokens, std::int32_t next_token,
                                std::uint32_t count, const domain::SamplingParams& params, domain::Rng& rng) override;

  // The last engine failure (OK when the last call drafted normally) and how many calls fell back.
  const Status& last_status() const { return last_; }
  std::uint64_t fallbacks() const { return fallbacks_; }

 private:
  // Pads (repeating the last token) or truncates to exactly `count`.
  static void fit(std::vector<std::int32_t>& drafts, std::int32_t next_token, std::uint32_t count);

  MtpEngine& engine_;
  Status last_;
  std::uint64_t fallbacks_ = 0;
};

}  // namespace clusterlm::backends::strata
