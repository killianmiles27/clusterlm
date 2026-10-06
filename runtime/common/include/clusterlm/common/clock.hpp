#pragma once
#include <chrono>
#include <cstdint>

namespace clusterlm {

using SteadyClock = std::chrono::steady_clock;
using Nanos = std::chrono::nanoseconds;

inline std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<Nanos>(SteadyClock::now().time_since_epoch()).count());
}

class Stopwatch {
 public:
  Stopwatch() : start_(SteadyClock::now()) {}
  void reset() { start_ = SteadyClock::now(); }
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start_).count();
  }
  std::uint64_t elapsed_ns() const {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<Nanos>(SteadyClock::now() - start_).count());
  }

 private:
  SteadyClock::time_point start_;
};

}  // namespace clusterlm
