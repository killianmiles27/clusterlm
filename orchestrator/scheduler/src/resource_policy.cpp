#include "clusterlm/scheduler/resource_policy.hpp"

#include <algorithm>

namespace clusterlm::scheduler {

namespace {
constexpr std::uint32_t kDay = 24 * 60;
constexpr std::uint32_t kWeek = 7 * kDay;

bool window_contains(const ScheduleWindow& w, std::uint32_t minute_of_week) {
  const std::uint32_t day = minute_of_week / kDay;
  const std::uint32_t m = minute_of_week % kDay;
  const auto has = [&](std::uint32_t d) { return (w.days >> d) & 1U; };
  if (w.end_minute > w.start_minute) return has(day) && m >= w.start_minute && m < w.end_minute;
  // Wrapping window: [start, 24h) on the start day, [0, end) on the following day.
  if (has(day) && m >= w.start_minute) return true;
  return has((day + 6) % 7) && m < w.end_minute;
}

std::string gib(std::uint64_t b) {
  const auto tenths = (b * 10) / (1ULL << 30);
  return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " GiB";
}
}  // namespace

std::string_view to_string(PolicyBlock b) noexcept {
  switch (b) {
    case PolicyBlock::kNone: return "none";
    case PolicyBlock::kPaused: return "paused";
    case PolicyBlock::kLocalUserActive: return "local_user_active";
    case PolicyBlock::kNotIdleLongEnough: return "not_idle_long_enough";
    case PolicyBlock::kIdleNotAllowed: return "idle_not_allowed";
    case PolicyBlock::kOnBattery: return "on_battery";
    case PolicyBlock::kBatterySaver: return "battery_saver";
    case PolicyBlock::kBatteryLow: return "battery_low";
    case PolicyBlock::kOutsideSchedule: return "outside_schedule";
    case PolicyBlock::kStaleObservation: return "stale_observation";
    case PolicyBlock::kOffline: return "offline";
  }
  return "unknown";
}

bool schedule_allows(const std::vector<ScheduleWindow>& windows, std::uint32_t minute_of_week) {
  if (windows.empty()) return true;
  minute_of_week %= kWeek;
  const std::size_t n = std::min(windows.size(), kMaxScheduleWindows);
  for (std::size_t i = 0; i < n; ++i)
    if (window_contains(windows[i], minute_of_week)) return true;
  return false;
}

std::string validate_policy(const WorkerResourcePolicy& p) {
  if (p.schedule.size() > kMaxScheduleWindows) return "too many schedule windows (limit 28)";
  for (const auto& w : p.schedule) {
    if (w.days == 0 || (w.days & 0x80U)) return "schedule window needs at least one valid day";
    if (w.start_minute >= kDay || w.end_minute > kDay) return "schedule window time is out of range";
    if (w.start_minute == w.end_minute) return "schedule window is empty";
  }
  if (p.min_battery_percent > 100) return "battery floor must be 0..100";
  if (p.idle_seconds_required > 7 * 24 * 3600) return "idle time is out of range";
  return {};
}

PolicyDecision evaluate_worker_policy(const WorkerResourcePolicy& policy, const WorkerObservation& obs,
                                      const PolicyContext& ctx, const std::string& name) {
  const auto deny = [&](PolicyBlock b, std::string why) {
    PolicyDecision d;
    d.block = b;
    d.reason = std::move(why);
    return d;
  };
  if (!obs.connected) return deny(PolicyBlock::kOffline, name + " is offline");
  // Unsigned subtraction guarded: a report "from the future" (clock skew) is treated as fresh, never as stale.
  const bool stale = ctx.now_ms > obs.observed_at_ms && ctx.now_ms - obs.observed_at_ms > ctx.staleness_ttl_ms;
  if (stale) return deny(PolicyBlock::kStaleObservation, "no recent status from " + name);
  if (policy.paused) return deny(PolicyBlock::kPaused, "sharing is paused on " + name);
  if (obs.local_user_active) return deny(PolicyBlock::kLocalUserActive, name + " is in use");
  if (!policy.allow_when_idle) return deny(PolicyBlock::kIdleNotAllowed, name + " is not set to share when idle");
  if (obs.idle_seconds < policy.idle_seconds_required)
    return deny(PolicyBlock::kNotIdleLongEnough, name + " has not been idle long enough");
  if (!obs.on_ac) {
    if (policy.require_ac) return deny(PolicyBlock::kOnBattery, name + " is on battery");
    if (policy.min_battery_percent > 0 && obs.battery_percent && *obs.battery_percent < policy.min_battery_percent)
      return deny(PolicyBlock::kBatteryLow, "battery on " + name + " is below " +
                                                std::to_string(policy.min_battery_percent) + "%");
  }
  if (obs.battery_saver && !policy.allow_battery_saver)
    return deny(PolicyBlock::kBatterySaver, "battery saver is on for " + name);
  if (!schedule_allows(policy.schedule, ctx.minute_of_week % kWeek)) {
    PolicyDecision d = deny(PolicyBlock::kOutsideSchedule, name + " is outside its sharing schedule");
    // Scan forward minute by minute (at most one week) for the next opening; bounded and deterministic.
    for (std::uint32_t k = 1; k <= kWeek; ++k) {
      if (schedule_allows(policy.schedule, (ctx.minute_of_week + k) % kWeek)) {
        d.schedule_opens_in_minutes = k;
        break;
      }
    }
    return d;
  }
  PolicyDecision ok;
  ok.allowed = true;
  return ok;
}

CapCheck check_caps(const WorkerResourcePolicy& p, const CapRequest& want, const std::string& name) {
  if (p.ram_cap_bytes && want.ram_bytes > p.ram_cap_bytes)
    return {false, "the plan needs " + gib(want.ram_bytes) + " of RAM on " + name + " but sharing is capped at " +
                       gib(p.ram_cap_bytes)};
  if (p.vram_cap_bytes && want.vram_bytes > p.vram_cap_bytes)
    return {false, "the plan needs " + gib(want.vram_bytes) + " of VRAM on " + name + " but sharing is capped at " +
                       gib(p.vram_cap_bytes)};
  if (p.thread_cap && want.threads > p.thread_cap)
    return {false, "the plan needs " + std::to_string(want.threads) + " threads on " + name +
                       " but sharing is capped at " + std::to_string(p.thread_cap)};
  return {};
}

}  // namespace clusterlm::scheduler
