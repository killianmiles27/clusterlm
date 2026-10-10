#pragma once
// Injectable time source: the scheduler never reads a clock directly, so every timeout test is deterministic.
#include <atomic>

#include "clusterlm/scheduler/types.hpp"

namespace clusterlm::scheduler {

using Duration = Clock::duration;

class ClockSource {
 public:
  virtual ~ClockSource() = default;
  virtual TimePoint now() const = 0;
};

class SteadyClockSource final : public ClockSource {
 public:
  TimePoint now() const override { return Clock::now(); }
};

// Test clock: moves only when told, in either direction (`set` may go backwards to exercise clamping).
class ManualClock final : public ClockSource {
 public:
  TimePoint now() const override { return TimePoint(Duration(ns_.load())); }
  void advance(Duration d) { ns_ += d.count(); }
  void set(TimePoint t) { ns_ = t.time_since_epoch().count(); }

 private:
  std::atomic<Duration::rep> ns_{1'000'000'000LL};  // start away from the epoch so "now - X" never underflows
};

}  // namespace clusterlm::scheduler
