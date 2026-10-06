// HelperActivityMonitor: helper reports with staleness -> fail closed.
#include <doctest/doctest.h>

#include "clusterlm/platform/helper_activity.hpp"

using namespace clusterlm;
using namespace clusterlm::platform;
using namespace std::chrono_literals;

namespace {
struct Fixture {
  SteadyClock::time_point now = SteadyClock::now();
  HelperActivityMonitor mon{5000ms, [this] { return now; }};
  void advance(std::chrono::milliseconds d) { now += d; }
  ActivitySample sample() {
    auto s = mon.sample();
    REQUIRE(s.is_ok());
    return s.value();
  }
};
}  // namespace

TEST_CASE("no report yet means in use (fail closed)") {
  Fixture f;
  const auto s = f.sample();
  CHECK(s.idle_seconds == 0);
  CHECK_FALSE(s.session_locked);
  CHECK_FALSE(f.mon.reports_fresh());
}

TEST_CASE("a fresh report is passed through and a stale one fails closed") {
  Fixture f;
  f.mon.submit({600, false, 1});
  CHECK(f.mon.reports_fresh());
  auto s = f.sample();
  CHECK(s.idle_seconds == 600);
  CHECK_FALSE(s.session_locked);

  f.advance(4900ms);
  CHECK(f.sample().idle_seconds == 600);  // still within the staleness window
  f.advance(200ms);                       // 5.1 s without a report: the helper may have died with the user present
  s = f.sample();
  CHECK(s.idle_seconds == 0);
  CHECK_FALSE(s.session_locked);
  CHECK_FALSE(f.mon.reports_fresh());

  f.mon.submit({700, true, 1});  // helper recovers
  s = f.sample();
  CHECK(s.idle_seconds == 700);
  CHECK(s.session_locked);
}

TEST_CASE("with several sessions the machine is idle only if every session is, and any stale one fails closed") {
  Fixture f;
  f.mon.submit({900, false, 1});
  f.mon.submit({30, false, 2});
  CHECK(f.sample().idle_seconds == 30);  // the busier session decides
  f.mon.submit({900, true, 1});
  f.mon.submit({900, false, 2});
  CHECK_FALSE(f.sample().session_locked);  // locked only when all are locked
  f.mon.submit({900, true, 2});
  CHECK(f.sample().session_locked);

  f.advance(3000ms);
  f.mon.submit({900, true, 1});  // session 2 stops reporting
  f.advance(3000ms);
  f.mon.submit({900, true, 1});
  const auto s = f.sample();
  CHECK(s.idle_seconds == 0);  // session 2 is stale
  CHECK_FALSE(s.session_locked);

  f.mon.session_ended(2);  // logoff: it no longer counts
  CHECK(f.sample().idle_seconds == 900);
}

TEST_CASE("headless machine (no user session) counts as idle, but only while nothing is tracked") {
  Fixture f;
  f.mon.set_no_interactive_sessions(true);
  auto s = f.sample();
  CHECK(s.idle_seconds >= 3600);
  CHECK(s.session_locked);
  f.mon.submit({1, false, 1});  // someone logs on and the helper reports
  CHECK(f.sample().idle_seconds == 1);
}

TEST_CASE("invalidate discards reports after resume; headless must be re-asserted") {
  Fixture f;
  f.mon.submit({900, false, 1});
  f.mon.set_no_interactive_sessions(true);
  f.mon.invalidate();
  CHECK(f.sample().idle_seconds == 0);
  f.mon.submit({900, false, 1});
  CHECK(f.sample().idle_seconds == 900);
}

TEST_CASE("pause (timed and indefinite) forces in-use until it expires or is resumed") {
  Fixture f;
  f.mon.submit({900, false, 1});
  f.mon.pause(60s);
  CHECK(f.mon.paused());
  CHECK(f.sample().idle_seconds == 0);
  f.advance(4000ms);
  f.mon.submit({900, false, 1});
  f.advance(30s);
  f.mon.submit({900, false, 1});
  CHECK(f.mon.paused());
  f.advance(30s);
  f.mon.submit({900, false, 1});
  CHECK_FALSE(f.mon.paused());  // 64 s later the timed pause is over
  CHECK(f.sample().idle_seconds == 900);

  f.mon.pause(std::nullopt);
  f.advance(3600s);
  f.mon.submit({900, false, 1});
  CHECK(f.mon.paused());
  CHECK(f.sample().idle_seconds == 0);
  f.mon.resume();
  CHECK(f.sample().idle_seconds == 900);
}
