#pragma once
// Worker resource policy: may this Worker's resources be used right now, and for how much?
//
// A pure function of (policy, observation, wall-clock minute-of-week). It is the Host-side half of "a busy machine is
// not available compute": the Node enforces the same rules locally and its report is authoritative; the Host evaluates
// them again so that readiness, admission and the status APIs can explain *why* a Worker is not usable and *when it
// might become usable* without asking the Worker. Nothing here reads a clock, a file or the network.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace clusterlm::scheduler {

// Stable machine codes; the sentence is for people and never contains an address or fingerprint.
enum class PolicyBlock : std::uint8_t {
  kNone = 0,
  kPaused,             // the owner paused sharing (Host-side or Worker-side)
  kLocalUserActive,    // a local user is at the machine (immediate release)
  kNotIdleLongEnough,  // idle, but shorter than the required idle time
  kIdleNotAllowed,     // policy says never share, even when idle (`allow_when_idle == false`)
  kOnBattery,          // `require_ac` and the machine is on battery
  kBatterySaver,       // battery saver is on and not allowed
  kBatteryLow,         // battery percentage below the configured floor
  kOutsideSchedule,    // schedule windows exist and none contains "now"
  kStaleObservation,   // the last report is too old to trust
  kOffline,            // not connected
};
std::string_view to_string(PolicyBlock b) noexcept;

// A weekly allow window. `days` is a bit mask (bit 0 = Monday ... bit 6 = Sunday). Minutes are local time of day,
// [start, end); `end <= start` wraps past midnight and then belongs to the day it *starts* on.
struct ScheduleWindow {
  std::uint8_t days = 0x7F;
  std::uint16_t start_minute = 0;
  std::uint16_t end_minute = 24 * 60;
};

struct WorkerResourcePolicy {
  bool paused = false;
  bool allow_when_idle = true;
  std::uint32_t idle_seconds_required = 300;
  bool require_ac = true;
  bool allow_battery_saver = false;
  std::uint8_t min_battery_percent = 0;         // consulted only when running on battery; 0 = no floor
  std::vector<ScheduleWindow> schedule;         // empty = always allowed; at most 28 windows
  // Caps on what a plan may take from this Worker; 0 = no cap beyond what the Worker advertises.
  std::uint64_t ram_cap_bytes = 0;
  std::uint64_t vram_cap_bytes = 0;
  std::uint32_t thread_cap = 0;
  // Continuous-idle retention: keep a lease while the Worker stays idle. 0 = release as soon as the Host is done.
  std::uint32_t retain_idle_seconds = 0;
};

// Hard limits so a hostile or corrupt settings file cannot make the evaluator unbounded (G-13).
inline constexpr std::size_t kMaxScheduleWindows = 28;

struct WorkerObservation {
  bool connected = false;
  bool local_user_active = false;
  std::uint32_t idle_seconds = 0;               // continuous time without local activity
  bool on_ac = true;
  bool battery_saver = false;
  std::optional<std::uint8_t> battery_percent;  // unknown on desktops
  std::uint64_t ram_free_bytes = 0, vram_free_bytes = 0;
  std::uint64_t observed_at_ms = 0;             // monotonic ms on the Host clock when the report arrived
};

struct PolicyContext {
  std::uint64_t now_ms = 0;                     // same clock as observed_at_ms
  std::uint32_t minute_of_week = 0;             // 0 = Monday 00:00 local, < 7*1440
  std::uint32_t staleness_ttl_ms = 15'000;
};

struct PolicyDecision {
  bool allowed = false;
  PolicyBlock block = PolicyBlock::kNone;
  std::string reason;                           // user-readable; empty when allowed
  // Estimated minutes until the *schedule* (only) next permits use; nullopt when unknown or not schedule-bound.
  std::optional<std::uint32_t> schedule_opens_in_minutes;
};

// Checks in priority order (first failing wins): offline, stale, paused, local user, idle policy, AC/battery, schedule.
// `display_name` is the user's name for the Worker.
PolicyDecision evaluate_worker_policy(const WorkerResourcePolicy& policy, const WorkerObservation& obs,
                                      const PolicyContext& ctx, const std::string& display_name);

// True when the schedule has no windows or one contains `minute_of_week`.
bool schedule_allows(const std::vector<ScheduleWindow>& windows, std::uint32_t minute_of_week);

// Structural validation of a policy (window counts/ranges, battery percent); empty string = valid.
std::string validate_policy(const WorkerResourcePolicy& policy);

// Caps. A plan asks for `want_*`; the Worker advertises `have_*`. The result is the allowance (min of non-zero caps) or
// nullopt with `why` filled when the request does not fit the *cap* (never silently shrunk; no truncation).
struct CapRequest { std::uint64_t ram_bytes = 0, vram_bytes = 0; std::uint32_t threads = 0; };
struct CapCheck { bool fits = true; std::string why; };
CapCheck check_caps(const WorkerResourcePolicy& policy, const CapRequest& want, const std::string& display_name);

}  // namespace clusterlm::scheduler
