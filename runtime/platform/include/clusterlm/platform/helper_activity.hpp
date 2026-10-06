#pragma once
// HelperActivityMonitor: the Node service's ActivityMonitor, fed by reports from per-session helpers over IPC.
//
// The service runs in session 0 and cannot read user input itself, so local activity reaches it only as helper
// reports. This monitor fails closed: with no report, or any tracked session whose last report is older than
// `stale_after`, it reports the machine as IN USE (idle 0, not locked). A Node must never offer resources on the
// strength of data it can no longer refresh.
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/platform/adapters.hpp"
#include "clusterlm/platform/ipc_messages.hpp"

namespace clusterlm::platform {

class HelperActivityMonitor final : public ActivityMonitor {
 public:
  using Clock = std::function<SteadyClock::time_point()>;

  explicit HelperActivityMonitor(std::chrono::milliseconds stale_after = std::chrono::milliseconds(5000),
                                 Clock clock = nullptr);

  // Thread-safe. Records the latest report for report.session_id.
  void submit(const ipc::ActivityReport& report);
  // Logoff/disconnect: the session no longer counts (otherwise its stale report would pin the Node to Busy).
  void session_ended(std::uint32_t session_id);
  // The service knows (via WTS) that no user session exists, e.g. the machine sits at the logon screen: nobody is
  // using it. Ignored while any session is tracked.
  void set_no_interactive_sessions(bool headless);
  // After resume from suspend: previous reports describe a world that no longer exists. Discard them so the
  // monitor reports "in use" until fresh reports arrive.
  void invalidate();
  // User pause from the tray: in use for `duration` (nullopt = until resume()).
  void pause(std::optional<std::chrono::seconds> duration);
  void resume();
  bool paused() const;
  // True when at least one session is tracked and none is stale.
  bool reports_fresh() const;

  Result<ActivitySample> sample() override;

 private:
  struct Entry {
    ActivitySample sample;
    SteadyClock::time_point at;
  };
  bool paused_locked(SteadyClock::time_point now) const;

  mutable std::mutex mu_;
  std::chrono::milliseconds stale_after_;
  Clock clock_;
  std::map<std::uint32_t, Entry> sessions_;
  bool headless_ = false;
  bool paused_ = false;
  std::optional<SteadyClock::time_point> pause_until_;
};

}  // namespace clusterlm::platform
