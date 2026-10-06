#include <doctest/doctest.h>

#include "clusterlm/platform/mock_adapters.hpp"

using namespace clusterlm;
using namespace clusterlm::platform;

TEST_CASE("mock activity and power monitors return settable values or injected errors") {
  MockActivityMonitor act;
  act.current = {42, true};
  auto a = act.sample();
  REQUIRE(a.is_ok());
  CHECK(a->idle_seconds == 42);
  CHECK(a->session_locked);
  act.fail_with = make_error(ErrorCode::kUnavailable, "x");
  CHECK_FALSE(act.sample().is_ok());

  MockPowerMonitor pw;
  pw.current.on_ac_power = false;
  pw.current.battery_percent = 55;
  auto p = pw.sample();
  REQUIRE(p.is_ok());
  CHECK_FALSE(p->on_ac_power);
  CHECK(*p->battery_percent == 55);
}

TEST_CASE("mock process job kills assigned processes on close when kill_on_close") {
  MockProcessJob job;
  CHECK(job.set_limits({1ull << 30, true}).is_ok());
  CHECK(job.assign_process(100).is_ok());
  CHECK(job.assign_process(101).is_ok());
  CHECK(*job.active_process_count() == 2);
  CHECK(job.close().is_ok());
  CHECK(job.killed_pids.size() == 2);
  CHECK(*job.active_process_count() == 0);
  CHECK_FALSE(job.assign_process(102).is_ok());
}

TEST_CASE("mock gpu probe reports configured synthetic adapters") {
  MockGpuBudgetProbe probe;
  GpuAdapterBudget b;
  b.name = "synthetic-adapter";  // synthetic fixture, not a measurement
  b.budget_bytes = 1000;
  b.current_usage_bytes = 100;
  probe.adapters.push_back(b);
  auto r = probe.query();
  REQUIRE(r.is_ok());
  REQUIRE(r->size() == 1);
  CHECK((*r)[0].budget_bytes - (*r)[0].current_usage_bytes == 900);
}
