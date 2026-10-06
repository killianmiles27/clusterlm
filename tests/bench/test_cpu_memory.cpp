#include <doctest/doctest.h>

#include <atomic>
#include <cmath>

#include "bench_common.hpp"
#include "clusterlm/platform/mock_adapters.hpp"
#include "cpu_bench.hpp"
#include "memory_probe.hpp"
#include "thread_team.hpp"

using namespace clusterlm;
using namespace clusterlm::bench;

namespace {

CpuBenchOptions tiny_options() {
  CpuBenchOptions o = quick_cpu_options();
  o.shape = {64, 32};
  o.active = 3;
  o.qs = {1, 2};
  o.threads = {1, 2};
  o.max_threads = 2;
  o.reps = 5;
  o.target_rep_seconds = 0.004;
  o.bank_bytes = 4ull << 20;
  return o;
}

SustainedReport series(const std::vector<double>& values, double dt) {
  SustainedReport r;
  double t = 0;
  for (double v : values) {
    t += dt;
    r.samples.push_back({t, v, std::nullopt});
  }
  summarize_sustained(r);
  return r;
}

}  // namespace

TEST_CASE("ThreadTeam runs every task exactly once on distinct workers") {
  for (unsigned size : {1u, 2u, 4u}) {
    ThreadTeam team(size);
    std::vector<std::atomic<int>> hits(37);
    std::atomic<unsigned> max_worker{0};
    for (int round = 0; round < 3; ++round)
      team.run(hits.size(), [&](std::size_t t, unsigned w) {
        hits[t].fetch_add(1);
        unsigned seen = max_worker.load();
        while (w > seen && !max_worker.compare_exchange_weak(seen, w)) {}
      });
    for (auto& h : hits) CHECK(h.load() == 3);
    CHECK(max_worker.load() < size);
  }
}

TEST_CASE("thread_sweep covers 1..max and always includes max") {
  CHECK(thread_sweep(1) == std::vector<unsigned>{1});
  CHECK(thread_sweep(2) == std::vector<unsigned>{1, 2});
  CHECK(thread_sweep(6) == std::vector<unsigned>{1, 2, 3, 4, 6});
  const auto s = thread_sweep(14);
  CHECK(s.front() == 1);
  CHECK(s.back() == 14);
}

TEST_CASE("CPU bench reports throughput, thread scaling, q scaling and the dequant split for the reference provider") {
  register_builtin_expert_providers();
  auto rep = run_cpu_bench(tiny_options());
  REQUIRE_MESSAGE(rep.is_ok(), rep.status().to_string());
  CHECK(rep->provider_id == "reference");
  REQUIRE(rep->representations.size() == 2);
  for (const auto& rr : rep->representations) {
    CAPTURE(rr.representation);
    CHECK(rr.best_bytes_per_s > 0);
    CHECK(rr.single_thread_bytes_per_s > 0);
    REQUIRE(rr.thread_bytes_per_s.count(1) == 1);
    REQUIRE(rr.thread_bytes_per_s.count(2) == 1);
    CHECK(rr.thread_bytes_per_s.at(1).samples.size() == 5);  // >= 5 repetitions, warm-up excluded
    CHECK((rr.best_threads == 1 || rr.best_threads == 2));
    CHECK(rr.usable_threads >= 1);
    CHECK(rr.usable_threads <= rr.best_threads);
    // q=1 same + q=2 {same, overlap, disjoint}
    CHECK(rr.q_points.size() == 4);
    for (const auto& pt : rr.q_points) {
      CHECK(pt.bytes_per_s.samples.size() == 5);
      if (pt.pattern == SelectionPattern::kSame) CHECK(pt.union_mean == doctest::Approx(3.0));
      if (pt.pattern == SelectionPattern::kDisjoint && pt.q == 2) {
        CHECK(pt.union_mean > 3.0);
        CHECK(pt.union_mean <= 6.0);
      }
      if (pt.pattern == SelectionPattern::kOverlap && pt.q == 2) {
        CHECK(pt.union_mean > 3.0);
        CHECK(pt.union_mean < 6.0);  // half the experts are shared with the previous position
      }
    }
    CHECK(rr.q_scaling_valid);
    CHECK(rr.q_scaling >= 0);
    if (rr.representation == "f32") {
      CHECK(rr.dequant_fraction == 0);
    } else {
      CHECK(rr.dequant_fraction > 0);
      CHECK(rr.dequant_fraction < 1);
      CHECK(rr.dequant_bytes_per_s > 0);
    }
    CHECK(rr.gemv_bytes_per_s > 0);
  }
}

TEST_CASE("CPU bench against the strata-cpu provider: the stub fails with kHardwareUnavailable, never a crash") {
  register_builtin_expert_providers();
  CpuBenchOptions o = tiny_options();
  o.provider = "strata-cpu";
  auto rep = run_cpu_bench(o);
  REQUIRE_FALSE(rep.is_ok());
#ifndef CLUSTERLM_BENCH_HAS_STRATA_CPU
  CHECK(rep.status().code() == ErrorCode::kHardwareUnavailable);
#else
  CHECK(rep.status().code() == ErrorCode::kInvalidArgument);  // the real kernels refuse the tiny 64 x 32 shape
#endif
  o.provider = "does-not-exist";
  CHECK(run_cpu_bench(o).status().code() == ErrorCode::kNotFound);
}

TEST_CASE("sustained summary: derate, trend and equilibrium") {
  SUBCASE("flat series: no derate") {
    const auto r = series(std::vector<double>(20, 100.0), 10);
    CHECK(r.sustained_factor == doctest::Approx(1.0));
    CHECK(std::fabs(r.slope_pct_per_min) < 1e-6);
    CHECK(r.time_to_equilibrium_s == doctest::Approx(10.0));  // stable from the first sample
  }
  SUBCASE("thermal throttle: drops then plateaus") {
    std::vector<double> v;
    for (int i = 0; i < 6; ++i) v.push_back(100.0 - 5.0 * i);  // 100 .. 75
    for (int i = 0; i < 14; ++i) v.push_back(75.0);
    const auto r = series(v, 10);
    CHECK(r.initial_bytes_per_s >= 90.0);
    CHECK(r.final_bytes_per_s == doctest::Approx(75.0));
    CHECK(r.sustained_factor == doctest::Approx(75.0 / 90.0).epsilon(0.02));
    CHECK(r.slope_pct_per_min < 0);
    CHECK(r.time_to_equilibrium_s >= 60.0);  // not before the plateau begins
    CHECK(r.time_to_equilibrium_s <= 70.0);
  }
  SUBCASE("a speed-up is not headroom") {
    std::vector<double> v;
    for (int i = 0; i < 12; ++i) v.push_back(100.0 + i);
    CHECK(series(v, 5).sustained_factor == doctest::Approx(1.0));
  }
  SUBCASE("a series that never settles reports no equilibrium") {
    std::vector<double> v;
    for (int i = 0; i < 12; ++i) v.push_back(100.0 - 4.0 * i);
    CHECK(series(v, 5).time_to_equilibrium_s == doctest::Approx(0.0));
  }
  SUBCASE("power source changes are flagged") {
    SustainedReport r;
    r.samples = {{1, 10, true}, {2, 10, true}, {3, 10, false}};
    summarize_sustained(r);
    CHECK(r.power_source_changed);
  }
}

TEST_CASE("sustained run samples throughput over time and records the power source from the adapter") {
  register_builtin_expert_providers();
  platform::MockPowerMonitor power;
  power.current.on_ac_power = true;
  CpuBenchOptions o = tiny_options();
  auto rep = run_cpu_sustained(o, "f32", 2, 0.012, 0.1, &power);  // 0.72 s
  REQUIRE_MESSAGE(rep.is_ok(), rep.status().to_string());
  CHECK(rep->samples.size() >= 3);
  CHECK(rep->threads == 2);
  for (const auto& s : rep->samples) {
    CHECK(s.bytes_per_s > 0);
    REQUIRE(s.on_ac_power.has_value());
    CHECK(*s.on_ac_power);
  }
  CHECK(rep->sustained_factor > 0);
  CHECK(rep->sustained_factor <= 1.0);
  CHECK_FALSE(run_cpu_sustained(o, "f32", 1, 0, 1, nullptr).is_ok());
}

TEST_CASE("sustained run stops early on request") {
  register_builtin_expert_providers();
  int calls = 0;
  auto rep = run_cpu_sustained(tiny_options(), "f32", 1, 5.0, 0.05, nullptr, [&] { return ++calls > 2; });
  REQUIRE(rep.is_ok());
  CHECK(rep->samples.size() == 2);
}

TEST_CASE("meminfo parsing: totals, available, commit limit") {
  const std::string text =
      "MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    8192000 kB\nCached: 2000000 kB\n"
      "CommitLimit:    12000000 kB\nCommitted_AS:    4000000 kB\n";
  const auto m = parse_meminfo(text);
  CHECK(m.total_physical == 16384000ull * 1024);
  CHECK(m.available_physical == 8192000ull * 1024);
  REQUIRE(m.commit_limit.has_value());
  CHECK(*m.commit_limit == 12000000ull * 1024);
  REQUIRE(m.commit_available.has_value());
  CHECK(*m.commit_available == 8000000ull * 1024);
  const auto sparse = parse_meminfo("MemTotal: 100 kB\nMemFree: 40 kB\nCached: 10 kB\n");
  CHECK(sparse.available_physical == 50ull * 1024);  // fallback when MemAvailable is absent
  CHECK_FALSE(sparse.commit_limit.has_value());
}

TEST_CASE("allocation ceiling never exceeds options, hard cap or available-minus-reserve") {
  MemoryInfo info;
  info.available_physical = 10ull << 30;
  AllocProbeOptions o;
  o.max_bytes = 4ull << 30;
  o.reserve_bytes = 2ull << 30;
  CHECK(allocation_ceiling(info, o) == (4ull << 30));
  o.max_bytes = 100ull << 30;
  CHECK(allocation_ceiling(info, o) == (8ull << 30));  // available - reserve
  info.available_physical = 1ull << 30;
  CHECK(allocation_ceiling(info, o) == 0);  // under the reserve: nothing may be allocated
  info.available_physical = 1000ull << 30;
  CHECK(allocation_ceiling(info, o) == kHardCapBytes);
}

TEST_CASE("allocation probe stays inside policy and touches pages") {
  AllocProbeOptions o;
  o.max_bytes = 64ull << 20;
  o.step_bytes = 16ull << 20;
  o.reserve_bytes = 64ull << 20;
  auto r = probe_largest_allocation(o);
  REQUIRE(r.is_ok());
  CHECK(r->largest_ok_bytes <= r->ceiling_bytes);
  CHECK(r->ceiling_bytes <= o.max_bytes);
  CHECK(r->largest_ok_bytes % o.step_bytes == 0);
  if (r->ceiling_bytes >= o.step_bytes) {
    CHECK(r->largest_ok_bytes >= o.step_bytes);
    CHECK(r->touch_bytes_per_s > 0);
    CHECK(r->stopped_by_policy);
  }
  o.step_bytes = 0;
  CHECK_FALSE(probe_largest_allocation(o).is_ok());
}

TEST_CASE("multi-threaded read bandwidth reports a distribution per thread count") {
  for (unsigned t : {1u, 2u}) {
    auto bw = measure_read_bandwidth_mt(16ull << 20, t, 5);
    REQUIRE(bw.is_ok());
    CHECK(bw->threads == t);
    CHECK(bw->bytes_per_s.samples.size() == 5);
    CHECK(bw->bytes_per_s.median() > 0);
  }
  CHECK_FALSE(measure_read_bandwidth_mt(64, 8, 1).is_ok());
}

TEST_CASE("option parsing helpers") {
  CHECK(parse_size("64M").value() == (64ull << 20));
  CHECK(parse_size("1G").value() == (1ull << 30));
  CHECK(parse_size("512k").value() == (512ull << 10));
  CHECK(parse_size("1000").value() == 1000);
  CHECK_FALSE(parse_size("").is_ok());
  CHECK_FALSE(parse_size("12X").is_ok());
  CHECK_FALSE(parse_size("abc").is_ok());
  CHECK(parse_u32_list("1,2,4") == std::vector<std::uint32_t>{1, 2, 4});

  const char* argv[] = {"clusterlm-bench", "cpu", "--machine-id", "g14", "--on-target", "--role", "node", "--run-id", "run-x"};
  const cli::Args on_target(9, const_cast<char**>(argv), 2);
  const RunContext ctx = make_run_context(on_target);
  CHECK(ctx.machine_id == "g14");
  CHECK(ctx.run_id == "run-x");
  CHECK(ctx.on_target);
  const char* argv2[] = {"clusterlm-bench", "cpu"};
  const cli::Args dev(2, const_cast<char**>(argv2), 2);
  const RunContext d = make_run_context(dev);
  CHECK(d.machine_id.rfind("dev-", 0) == 0);  // never mistaken for a target name
  CHECK_FALSE(d.on_target);
  BenchmarkResult r("dev-x", HostInfo{});
  apply_run_context(r, d);
  CHECK(r.finish(0)["environment"]["host_role"] == "development-host");
  BenchmarkResult r2("x", HostInfo{});
  apply_run_context(r2, ctx);
  CHECK(r2.finish(0)["environment"]["host_role"] == "node");
}
