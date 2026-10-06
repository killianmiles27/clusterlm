#include "clusterlm/platform/helper_activity.hpp"

#include <algorithm>
#include <limits>

namespace clusterlm::platform {

namespace {
// "Idle for ages": what an unattended machine with no user session reports.
constexpr std::uint32_t kHeadlessIdleSeconds = 24u * 3600u;
}  // namespace

HelperActivityMonitor::HelperActivityMonitor(std::chrono::milliseconds stale_after, Clock clock)
    : stale_after_(stale_after), clock_(clock ? std::move(clock) : Clock([] { return SteadyClock::now(); })) {}

void HelperActivityMonitor::submit(const ipc::ActivityReport& r) {
  std::lock_guard lock(mu_);
  sessions_[r.session_id] = Entry{{r.idle_seconds, r.session_locked}, clock_()};
}

void HelperActivityMonitor::session_ended(std::uint32_t session_id) {
  std::lock_guard lock(mu_);
  sessions_.erase(session_id);
}

void HelperActivityMonitor::set_no_interactive_sessions(bool headless) {
  std::lock_guard lock(mu_);
  headless_ = headless;
}

void HelperActivityMonitor::invalidate() {
  std::lock_guard lock(mu_);
  sessions_.clear();
  headless_ = false;  // must be re-asserted by the service after it re-enumerates sessions
}

void HelperActivityMonitor::pause(std::optional<std::chrono::seconds> duration) {
  std::lock_guard lock(mu_);
  paused_ = true;
  pause_until_ = duration ? std::optional(clock_() + *duration) : std::nullopt;
}

void HelperActivityMonitor::resume() {
  std::lock_guard lock(mu_);
  paused_ = false;
  pause_until_.reset();
}

bool HelperActivityMonitor::paused_locked(SteadyClock::time_point now) const {
  if (!paused_) return false;
  return !pause_until_ || now < *pause_until_;
}

bool HelperActivityMonitor::paused() const {
  std::lock_guard lock(mu_);
  return paused_locked(clock_());
}

bool HelperActivityMonitor::reports_fresh() const {
  std::lock_guard lock(mu_);
  if (sessions_.empty()) return false;
  const auto now = clock_();
  return std::all_of(sessions_.begin(), sessions_.end(),
                     [&](const auto& kv) { return now - kv.second.at <= stale_after_; });
}

Result<ActivitySample> HelperActivityMonitor::sample() {
  std::lock_guard lock(mu_);
  const auto now = clock_();
  const ActivitySample in_use{0, false};
  if (paused_locked(now)) return in_use;
  if (sessions_.empty()) {
    if (headless_) return ActivitySample{kHeadlessIdleSeconds, true};
    return in_use;  // no report yet: fail closed
  }
  ActivitySample out{std::numeric_limits<std::uint32_t>::max(), true};
  for (const auto& [id, e] : sessions_) {
    (void)id;
    if (now - e.at > stale_after_) return in_use;  // a silent helper might be hiding a user at the keyboard
    out.idle_seconds = std::min(out.idle_seconds, e.sample.idle_seconds);
    out.session_locked = out.session_locked && e.sample.session_locked;
  }
  return out;
}

}  // namespace clusterlm::platform
