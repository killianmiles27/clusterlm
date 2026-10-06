#pragma once
// Father-side draft-token producers for speculative windows. Drafters see token IDs, so they live with
// Father and never cross a domain boundary.
//
// draft() contract: `committed_tokens` are the tokens whose positions are already committed; `next_token`
// is the next token to feed (the model's own prediction, not yet committed). The result is the `count`
// tokens expected to FOLLOW next_token, so a verification window is [next_token, draft...] (q = count + 1).
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "clusterlm/objects/provisioned.hpp"

namespace clusterlm::domain {

class Drafter {
 public:
  virtual ~Drafter() = default;
  virtual std::vector<std::int32_t> draft(std::span<const std::int32_t> committed_tokens, std::int32_t next_token,
                                          std::uint32_t count) = 0;
};

// Chained one-step draft using the fixture `mtp_drafter` object: d = argmax(W * rmsnorm(embd[prev])).
class MtpFixtureDrafter final : public Drafter {
 public:
  static Result<std::unique_ptr<MtpFixtureDrafter>> create(const objects::ModelManifest& manifest,
                                                           const objects::ObjectResolver& resolver);
  std::vector<std::int32_t> draft(std::span<const std::int32_t> committed_tokens, std::int32_t next_token,
                                  std::uint32_t count) override;
  // One draft step (exposed for tests).
  std::int32_t step(std::int32_t prev) const;

 private:
  MtpFixtureDrafter() = default;
  std::uint32_t hidden_ = 0, vocab_ = 0;
  std::vector<float> embd_, norm_, head_;
};

// Test drafter: replays a known reference sequence (prompt + continuation) with deterministic corruption so
// that every acceptance length 0..q-1 can be exercised.
class ScriptedDrafter final : public Drafter {
 public:
  struct Config {
    std::uint32_t vocab = 2;
    std::uint64_t seed = 1;
    double corruption_rate = 0.0;                       // per drafted token, deterministic in (seed, position)
    std::vector<std::uint64_t> corrupt_positions;       // absolute positions that are always corrupted
    std::optional<std::uint32_t> corrupt_draft_index;   // 1-based draft index corrupted in every window
  };
  ScriptedDrafter(std::vector<std::int32_t> reference_sequence, Config config)
      : ref_(std::move(reference_sequence)), cfg_(std::move(config)) {}

  std::vector<std::int32_t> draft(std::span<const std::int32_t> committed_tokens, std::int32_t next_token,
                                  std::uint32_t count) override;

 private:
  std::vector<std::int32_t> ref_;
  Config cfg_;
};

}  // namespace clusterlm::domain
