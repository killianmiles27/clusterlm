// Worker resource policy: every blocker has a stable code and a readable reason; schedule windows incl. wrap-around.
#include <doctest/doctest.h>

#include "clusterlm/scheduler/resource_policy.hpp"

using namespace clusterlm::scheduler;

namespace {
WorkerObservation idle_obs() {
  WorkerObservation o;
  o.connected = true;
  o.idle_seconds = 3600;
  o.on_ac = true;
  o.observed_at_ms = 1000;
  return o;
}
PolicyContext ctx(std::uint32_t minute = 0) {
  PolicyContext c;
  c.now_ms = 2000;
  c.minute_of_week = minute;
  return c;
}
constexpr std::uint32_t at(unsigned day, unsigned hour, unsigned min = 0) { return day * 1440 + hour * 60 + min; }
}  // namespace

TEST_CASE("idle AC Worker with default policy is allowed") {
  auto d = evaluate_worker_policy({}, idle_obs(), ctx(), "G14");
  CHECK(d.allowed);
  CHECK(d.block == PolicyBlock::kNone);
  CHECK(d.reason.empty());
}

TEST_CASE("each blocker reports its own code, in priority order") {
  WorkerResourcePolicy p;
  auto o = idle_obs();
  SUBCASE("offline") { o.connected = false; CHECK(evaluate_worker_policy(p, o, ctx(), "G14").block == PolicyBlock::kOffline); }
  SUBCASE("stale") {
    PolicyContext c = ctx();
    c.now_ms = 1000 + 15'001;
    CHECK(evaluate_worker_policy(p, o, c, "G14").block == PolicyBlock::kStaleObservation);
    c.now_ms = 1000 + 15'000;  // boundary: exactly the ttl is still fresh
    CHECK(evaluate_worker_policy(p, o, c, "G14").allowed);
  }
  SUBCASE("a report from the future is fresh, not stale") {
    PolicyContext c = ctx();
    c.now_ms = 10;
    CHECK(evaluate_worker_policy(p, o, c, "G14").allowed);
  }
  SUBCASE("paused wins over local user") {
    p.paused = true;
    o.local_user_active = true;
    CHECK(evaluate_worker_policy(p, o, ctx(), "G14").block == PolicyBlock::kPaused);
  }
  SUBCASE("local user") {
    o.local_user_active = true;
    auto d = evaluate_worker_policy(p, o, ctx(), "G14");
    CHECK(d.block == PolicyBlock::kLocalUserActive);
    CHECK(d.reason == "G14 is in use");
  }
  SUBCASE("idle not long enough") { o.idle_seconds = 299; CHECK(evaluate_worker_policy(p, o, ctx(), "A").block == PolicyBlock::kNotIdleLongEnough); }
  SUBCASE("idle exactly long enough") { o.idle_seconds = 300; CHECK(evaluate_worker_policy(p, o, ctx(), "A").allowed); }
  SUBCASE("never share when idle") { p.allow_when_idle = false; CHECK(evaluate_worker_policy(p, o, ctx(), "A").block == PolicyBlock::kIdleNotAllowed); }
  SUBCASE("battery with require_ac") { o.on_ac = false; CHECK(evaluate_worker_policy(p, o, ctx(), "A").block == PolicyBlock::kOnBattery); }
  SUBCASE("battery allowed above floor") {
    p.require_ac = false; p.min_battery_percent = 40; o.on_ac = false; o.battery_percent = 41;
    CHECK(evaluate_worker_policy(p, o, ctx(), "A").allowed);
    o.battery_percent = 39;
    CHECK(evaluate_worker_policy(p, o, ctx(), "A").block == PolicyBlock::kBatteryLow);
    o.battery_percent.reset();  // unknown percentage never blocks by itself
    CHECK(evaluate_worker_policy(p, o, ctx(), "A").allowed);
  }
  SUBCASE("battery saver") { o.battery_saver = true; CHECK(evaluate_worker_policy(p, o, ctx(), "A").block == PolicyBlock::kBatterySaver); p.allow_battery_saver = true; CHECK(evaluate_worker_policy(p, o, ctx(), "A").allowed); }
}

TEST_CASE("schedule windows") {
  WorkerResourcePolicy p;
  p.schedule = {{0x1F, 22 * 60, 24 * 60}};  // Mon-Fri 22:00-24:00
  CHECK(schedule_allows(p.schedule, at(0, 22)));
  CHECK(schedule_allows(p.schedule, at(4, 23, 59)));
  CHECK_FALSE(schedule_allows(p.schedule, at(0, 21, 59)));
  CHECK_FALSE(schedule_allows(p.schedule, at(5, 23)));  // Saturday not listed
  auto d = evaluate_worker_policy(p, idle_obs(), ctx(at(0, 20)), "G14");
  CHECK(d.block == PolicyBlock::kOutsideSchedule);
  REQUIRE(d.schedule_opens_in_minutes);
  CHECK(*d.schedule_opens_in_minutes == 120);
}

TEST_CASE("a window that wraps midnight belongs to the day it starts on") {
  const std::vector<ScheduleWindow> w = {{0x01, 22 * 60, 6 * 60}};  // Monday 22:00 -> Tuesday 06:00
  CHECK(schedule_allows(w, at(0, 23)));
  CHECK(schedule_allows(w, at(1, 5, 59)));
  CHECK_FALSE(schedule_allows(w, at(1, 6)));
  CHECK_FALSE(schedule_allows(w, at(0, 5)));   // Monday early morning is not covered
  CHECK_FALSE(schedule_allows(w, at(1, 23)));  // Tuesday evening is not
  const std::vector<ScheduleWindow> sunday = {{0x40, 22 * 60, 6 * 60}};  // Sunday night wraps into Monday
  CHECK(schedule_allows(sunday, at(0, 3)));
}

TEST_CASE("validate_policy rejects malformed policies") {
  WorkerResourcePolicy p;
  CHECK(validate_policy(p).empty());
  p.schedule = {{0, 0, 60}};
  CHECK_FALSE(validate_policy(p).empty());
  p.schedule = {{0x01, 10, 10}};
  CHECK_FALSE(validate_policy(p).empty());
  p.schedule = {{0x01, 1500, 60}};
  CHECK_FALSE(validate_policy(p).empty());
  p.schedule.assign(kMaxScheduleWindows + 1, ScheduleWindow{});
  CHECK_FALSE(validate_policy(p).empty());
  p.schedule.clear();
  p.min_battery_percent = 101;
  CHECK_FALSE(validate_policy(p).empty());
}

TEST_CASE("caps are never silently shrunk") {
  WorkerResourcePolicy p;
  p.ram_cap_bytes = 8ULL << 30;
  p.vram_cap_bytes = 4ULL << 30;
  p.thread_cap = 6;
  CHECK(check_caps(p, {8ULL << 30, 4ULL << 30, 6}, "G14").fits);
  auto r = check_caps(p, {9ULL << 30, 0, 0}, "G14");
  CHECK_FALSE(r.fits);
  CHECK(r.why.find("RAM") != std::string::npos);
  CHECK_FALSE(check_caps(p, {0, 5ULL << 30, 0}, "G14").fits);
  CHECK_FALSE(check_caps(p, {0, 0, 7}, "G14").fits);
  CHECK(check_caps({}, {1ULL << 40, 1ULL << 40, 999}, "G14").fits);  // 0 = no cap
}
