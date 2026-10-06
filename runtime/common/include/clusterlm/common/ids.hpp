#pragma once
// Strongly typed identifiers that appear on the wire. Mixing them up (e.g. passing a window ID where an
// epoch is expected) must be a compile error, because every one of them gates stale-message rejection.
#include <compare>
#include <cstdint>
#include <functional>
#include <string>

namespace clusterlm {

template <typename Tag, typename Rep = std::uint64_t>
struct StrongId {
  Rep value = 0;
  constexpr StrongId() = default;
  constexpr explicit StrongId(Rep v) : value(v) {}
  friend constexpr bool operator==(StrongId, StrongId) = default;
  friend constexpr auto operator<=>(StrongId, StrongId) = default;
  constexpr StrongId next() const { return StrongId(value + 1); }
  std::string str() const { return std::to_string(value); }
};

// Lease generation: incremented by a Node every time it grants a new lease. A message naming an older lease
// generation is stale and rejected even if it arrives on a still-open connection.
using LeaseGeneration = StrongId<struct LeaseGenerationTag>;
// Execution epoch: incremented by Father when a distributed session is (re)established. Any failure that
// invalidates a session advances the epoch; stragglers from the old epoch are rejected.
using Epoch = StrongId<struct EpochTag>;
// Inference session (one conversation's sequence state across all domains).
using SessionId = StrongId<struct SessionIdTag>;
// Monotonic speculative-window ID within a session.
using WindowId = StrongId<struct WindowIdTag>;
// Logical pipeline stage index (Father prefix = 0, ..., Father tail = last).
using StageId = StrongId<struct StageIdTag, std::uint32_t>;
// Version of a domain's committed sequence state; advanced by every successful commit.
using StateVersion = StrongId<struct StateVersionTag>;

}  // namespace clusterlm

template <typename Tag, typename Rep>
struct std::hash<clusterlm::StrongId<Tag, Rep>> {
  std::size_t operator()(clusterlm::StrongId<Tag, Rep> id) const noexcept { return std::hash<Rep>{}(id.value); }
};
