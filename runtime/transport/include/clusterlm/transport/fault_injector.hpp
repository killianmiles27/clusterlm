#pragma once
// Deterministic fault injection for transport tests and the failure-matrix benchmarks.
//
// A rule fires exactly once, on the `nth` (1-based) frame that matches its type/channel filter in its
// direction. Counters are per rule and shared by every connection using the same injector.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include "clusterlm/common/status.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::transport {

enum class FaultDirection : std::uint8_t { kSend, kReceive };
enum class FaultAction : std::uint8_t {
  kCloseBefore,    // close the connection instead of passing the matching frame
  kCloseAfter,     // pass the matching frame, then close the connection
  kDelay,          // hold the matching frame for delay_ms
  kStall,          // hold the matching frame and everything behind it until clear_stall() or close
  kCorruptPayload  // flip bits in the payload (or the correlation id when the payload is empty)
};

struct FaultRule {
  std::optional<std::uint16_t> frame_type;
  std::optional<std::uint8_t> channel;
  std::uint32_t nth = 1;
  FaultDirection dir = FaultDirection::kSend;
  FaultAction action = FaultAction::kCloseBefore;
  double delay_ms = 0;
};

// "close-before:type=22:nth=3:dir=send". Actions: close-before, close-after, delay (needs ms=), stall,
// corrupt. Keys: type, channel, nth, dir (send|recv), ms.
Result<FaultRule> parse_fault_rule(std::string_view text);

class FaultInjector {
 public:
  void add_rule(const FaultRule& rule);
  // Removes every rule and releases any stall.
  void clear();
  // Number of rule firings so far.
  std::uint64_t fired_count() const;
  // Releases stalls in both directions (a stall already released does not re-arm for the same rule).
  void clear_stall();

  // Observer of every frame the decorator counts (both directions), called with no injector lock held. For
  // tests that capture traffic (privacy assertions); frames are seen exactly as the decorator sees them,
  // before any corruption rule is applied. An empty function clears it.
  using Tap = std::function<void(FaultDirection, const Frame&)>;
  void set_tap(Tap tap);

  // --- used by the impairment decorator ---
  struct Decision {
    bool close_before = false;
    bool close_after = false;
    bool corrupt = false;
    bool stall = false;
    double delay_ms = 0;
  };
  // Counts the frame against all rules for `dir` and returns the actions that fire now. A kStall decision
  // arms the stall for `dir`.
  Decision on_frame(FaultDirection dir, const Frame& frame);
  // Blocks while `dir` is stalled; returns false if `abort` became true or the deadline passed first.
  bool wait_unstalled(FaultDirection dir, std::chrono::steady_clock::time_point deadline,
                      const std::atomic<bool>& abort);

 private:
  struct Slot {
    FaultRule rule;
    std::uint32_t seen = 0;
    bool fired = false;
  };
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Slot> rules_;
  Tap tap_;
  std::uint64_t fired_ = 0;
  bool stalled_[2] = {false, false};
};

}  // namespace clusterlm::transport
