#include "cpu_bench.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>

#include "clusterlm/common/clock.hpp"
#include "thread_team.hpp"

namespace clusterlm::bench {

const char* to_string(SelectionPattern p) {
  switch (p) {
    case SelectionPattern::kSame: return "same";
    case SelectionPattern::kOverlap: return "overlap";
    case SelectionPattern::kDisjoint: return "disjoint";
  }
  return "same";
}

CpuBenchOptions quick_cpu_options() {
  CpuBenchOptions o;
  o.shape = {256, 64};
  o.active = 4;
  o.qs = {1, 2, 4};
  o.reps = 5;
  o.target_rep_seconds = 0.02;
  o.bank_bytes = 24ull << 20;
  o.max_threads = 2;
  o.quick = true;
  return o;
}

std::vector<unsigned> thread_sweep(unsigned max_threads) {
  std::vector<unsigned> out;
  max_threads = std::max(1u, max_threads);
  for (unsigned t : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 24u, 32u, 48u, 64u})
    if (t < max_threads) out.push_back(t);
  out.push_back(max_threads);
  return out;
}

namespace {

std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// `k` distinct experts out of `n`, excluding `keep` (already chosen), appended to `out`.
void draw_distinct(std::uint64_t& rng, std::size_t n, std::size_t k, std::vector<std::size_t>& out) {
  std::set<std::size_t> have(out.begin(), out.end());
  const std::size_t target = std::min(n, out.size() + k);
  while (out.size() < target) {
    const std::size_t e = static_cast<std::size_t>(splitmix(rng) % n);
    if (have.insert(e).second) out.push_back(e);
  }
}

struct Task {
  std::size_t expert = 0;
  std::vector<std::uint32_t> positions;
};

// The union of the experts selected by q positions, as tasks.
std::vector<Task> plan_round(std::uint64_t& rng, std::size_t bank_experts, std::uint32_t active, std::uint32_t q,
                             SelectionPattern pattern) {
  std::vector<std::vector<std::size_t>> per_pos(q);
  for (std::uint32_t p = 0; p < q; ++p) {
    if (p == 0 || pattern == SelectionPattern::kDisjoint) {
      draw_distinct(rng, bank_experts, active, per_pos[p]);
    } else if (pattern == SelectionPattern::kSame) {
      per_pos[p] = per_pos[0];
    } else {
      const std::size_t keep = std::min<std::size_t>(active / 2, per_pos[p - 1].size());
      per_pos[p].assign(per_pos[p - 1].begin(), per_pos[p - 1].begin() + static_cast<std::ptrdiff_t>(keep));
      draw_distinct(rng, bank_experts, active - keep, per_pos[p]);
    }
  }
  std::vector<Task> tasks;
  for (std::uint32_t p = 0; p < q; ++p)
    for (std::size_t e : per_pos[p]) {
      auto it = std::find_if(tasks.begin(), tasks.end(), [&](const Task& t) { return t.expert == e; });
      if (it == tasks.end()) {
        tasks.push_back({e, {p}});
      } else {
        it->positions.push_back(p);
      }
    }
  return tasks;
}

struct alignas(64) WorkerState {
  std::unique_ptr<ExpertContext> ctx;
  ExpertTiming timing;
  std::vector<float> in, out;
  float sink = 0;
};

// An expert bank plus the machinery to execute rounds on it with a given team.
class Runner {
 public:
  Runner(const ExpertBank& bank, ExpertShape shape, unsigned threads, std::uint64_t seed)
      : bank_(bank), shape_(shape), team_(threads), workers_(threads), seed_(seed) {
    for (auto& w : workers_) w.ctx = bank.make_context();
    // Deterministic inputs for up to 8 positions.
    std::uint64_t s = seed ^ 0xA5A5A5A5u;
    inputs_.resize(std::size_t{8} * shape.hidden);
    for (float& x : inputs_) x = static_cast<float>(static_cast<double>(splitmix(s) >> 40) / 16777216.0 - 0.5);
  }

  struct Outcome {
    std::vector<double> round_ns;  // wall time of each round
    std::uint64_t experts_run = 0;
    std::uint64_t bytes = 0;       // stored bytes of the executed experts
    ExpertTiming timing;           // summed over all workers
    Status status;
  };

  Outcome run(std::uint32_t rounds, std::uint32_t active, std::uint32_t q, SelectionPattern pattern, std::uint64_t salt) {
    Outcome o;
    std::uint64_t rng = seed_ ^ (salt * 0x9E3779B97F4A7C15ull);
    std::vector<std::vector<Task>> plans;
    plans.reserve(rounds);
    for (std::uint32_t r = 0; r < rounds; ++r) plans.push_back(plan_round(rng, bank_.experts(), active, q, pattern));
    for (auto& w : workers_) w.timing = {};
    std::atomic<bool> failed{false};
    std::mutex err_mu;
    const std::uint64_t bpe = bank_.bytes_per_expert();
    for (const auto& tasks : plans) {
      Stopwatch sw;
      team_.run(tasks.size(), [&](std::size_t ti, unsigned wi) {
        if (failed.load(std::memory_order_relaxed)) return;
        WorkerState& w = workers_[wi];
        const Task& t = tasks[ti];
        const std::size_t np = t.positions.size();
        w.in.resize(np * shape_.hidden);
        w.out.resize(np * shape_.hidden);
        for (std::size_t i = 0; i < np; ++i)
          std::copy_n(inputs_.data() + std::size_t{t.positions[i] % 8} * shape_.hidden, shape_.hidden,
                      w.in.data() + i * shape_.hidden);
        Status st = bank_.run(t.expert, np, w.in.data(), w.out.data(), *w.ctx, &w.timing);
        w.sink += w.out[0];
        if (!st.is_ok() && !failed.exchange(true)) {
          std::lock_guard<std::mutex> lock(err_mu);
          o.status = st;
        }
      });
      o.round_ns.push_back(static_cast<double>(sw.elapsed_ns()));
      o.experts_run += tasks.size();
      o.bytes += tasks.size() * bpe;
    }
    for (auto& w : workers_) {
      o.timing.dequant_ns += w.timing.dequant_ns;
      o.timing.gemv_ns += w.timing.gemv_ns;
    }
    volatile float keep = 0;
    for (auto& w : workers_) keep = keep + w.sink;
    return o;
  }

  unsigned threads() const { return team_.size(); }

 private:
  const ExpertBank& bank_;
  ExpertShape shape_;
  ThreadTeam team_;
  std::vector<WorkerState> workers_;
  std::vector<float> inputs_;
  std::uint64_t seed_;
};

double sum(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }

// Rounds per repetition so that one repetition lasts about `target_s`.
std::uint32_t calibrate_rounds(Runner& runner, const CpuBenchOptions& o, std::uint32_t q, SelectionPattern pattern) {
  const auto probe = runner.run(1, o.active, q, pattern, 0xCA11);
  if (!probe.status.is_ok() || probe.round_ns.empty()) return 1;
  const double est_s = std::max(1e-6, probe.round_ns[0] * 1e-9);
  return static_cast<std::uint32_t>(std::clamp(std::ceil(o.target_rep_seconds / est_s), 1.0, 400.0));
}

struct Config {
  std::uint32_t q = 1;
  SelectionPattern pattern = SelectionPattern::kSame;
  unsigned threads = 1;
  std::uint32_t rounds = 1;
  Runner* runner = nullptr;
  // results
  Distribution bytes_per_s, round_ms;
  double union_sum = 0, rounds_sum = 0;
  ExpertTiming timing;
  std::uint64_t timed_bytes = 0;
};

Status run_interleaved(std::vector<Config*>& configs, const CpuBenchOptions& o, std::uint32_t active) {
  // Repetitions interleave across configurations so drift (thermal, background load) hits all of them alike.
  const unsigned total = o.warmup_reps + o.reps;
  for (unsigned rep = 0; rep < total; ++rep)
    for (Config* c : configs) {
      auto out = c->runner->run(c->rounds, active, c->q, c->pattern, rep + 1);
      CLM_RETURN_IF_ERROR(out.status);
      if (rep < o.warmup_reps) continue;
      const double secs = sum(out.round_ns) * 1e-9;
      c->bytes_per_s.add(static_cast<double>(out.bytes) / secs);
      for (double ns : out.round_ns) c->round_ms.add(ns * 1e-6);
      c->union_sum += static_cast<double>(out.experts_run);
      c->rounds_sum += static_cast<double>(out.round_ns.size());
      c->timing.dequant_ns += out.timing.dequant_ns;
      c->timing.gemv_ns += out.timing.gemv_ns;
      c->timed_bytes += out.bytes;
    }
  return Status::ok();
}

}  // namespace

Result<CpuBenchReport> run_cpu_bench(const CpuBenchOptions& opt, const std::function<void(const std::string&)>& progress) {
  auto say = [&](const std::string& s) {
    if (progress) progress(s);
  };
  CLM_ASSIGN_OR_RETURN(auto provider, ExpertKernelRegistry::instance().create(opt.provider));
  CpuBenchReport report;
  report.provider_id = provider->id();
  report.provider_description = provider->description();
  report.isa = provider->isa();
  report.shape = opt.shape;
  report.active = opt.active;
  report.max_threads = opt.max_threads ? opt.max_threads : std::max(1u, std::thread::hardware_concurrency());
  if (opt.reps == 0 || opt.active == 0 || opt.qs.empty())
    return make_error(ErrorCode::kInvalidArgument, "reps, active and q must be positive");
  const std::vector<unsigned> sweep = opt.threads.empty() ? thread_sweep(report.max_threads) : opt.threads;
  std::vector<std::string> reps = opt.representations.empty() ? provider->representations() : opt.representations;

  for (const std::string& rep : reps) {
    // Working-set size: enough experts that one round does not fit in cache, never fewer than 2*active.
    // The bank is built once to learn the per-expert size, then rebuilt at the target count.
    CLM_ASSIGN_OR_RETURN(auto probe_bank, provider->make_bank(rep, opt.shape, 1, opt.seed));
    const std::uint64_t per = std::max<std::uint64_t>(1, probe_bank->bytes_per_expert());
    const std::size_t n = static_cast<std::size_t>(
        std::clamp<std::uint64_t>(opt.bank_bytes / per, std::uint64_t{opt.active} * 2, 512));
    probe_bank.reset();
    say("building " + std::to_string(n) + " experts of " + rep);
    CLM_ASSIGN_OR_RETURN(auto bank, provider->make_bank(rep, opt.shape, n, opt.seed));

    RepresentationReport rr;
    rr.representation = rep;
    rr.bytes_per_expert = bank->bytes_per_expert();
    rr.bank_experts = bank->experts();

    std::map<unsigned, std::unique_ptr<Runner>> runners;
    auto runner_for = [&](unsigned t) -> Runner& {
      auto& r = runners[t];
      if (!r) r = std::make_unique<Runner>(*bank, opt.shape, t, opt.seed);
      return *r;
    };

    // Phase 1: thread scaling at q=1 (every thread works on different experts of the same round).
    say(rep + ": thread sweep");
    std::vector<Config> sweep_cfg(sweep.size());
    std::vector<Config*> ptrs;
    for (std::size_t i = 0; i < sweep.size(); ++i) {
      sweep_cfg[i].threads = sweep[i];
      sweep_cfg[i].runner = &runner_for(sweep[i]);
      sweep_cfg[i].rounds = calibrate_rounds(*sweep_cfg[i].runner, opt, 1, SelectionPattern::kSame);
      ptrs.push_back(&sweep_cfg[i]);
    }
    CLM_RETURN_IF_ERROR(run_interleaved(ptrs, opt, opt.active));
    double best = 0;
    for (const auto& c : sweep_cfg) {
      rr.thread_bytes_per_s[c.threads] = c.bytes_per_s;
      if (c.bytes_per_s.median() > best) {
        best = c.bytes_per_s.median();
        rr.best_threads = c.threads;
      }
      if (c.threads == 1) rr.single_thread_bytes_per_s = c.bytes_per_s.median();
    }
    rr.best_bytes_per_s = best;
    rr.usable_threads = rr.best_threads;
    for (const auto& c : sweep_cfg)
      if (c.bytes_per_s.median() >= 0.95 * best) {
        rr.usable_threads = c.threads;
        break;
      }
    // Keep one logical CPU free for the service and the user's session when the machine has room for that.
    if (report.max_threads >= 4 && rr.usable_threads >= report.max_threads) rr.usable_threads = report.max_threads - 1;

    // Phase 2: q and selection patterns at the best thread count.
    say(rep + ": q sweep at " + std::to_string(rr.best_threads) + " threads");
    Runner& best_runner = runner_for(rr.best_threads);
    std::vector<Config> qcfg;
    for (std::uint32_t q : opt.qs)
      for (auto pat : {SelectionPattern::kSame, SelectionPattern::kOverlap, SelectionPattern::kDisjoint}) {
        if (q == 1 && pat != SelectionPattern::kSame) continue;
        Config c;
        c.q = q;
        c.pattern = pat;
        c.threads = rr.best_threads;
        c.runner = &best_runner;
        qcfg.push_back(std::move(c));
      }
    ptrs.clear();
    for (auto& c : qcfg) {
      c.rounds = calibrate_rounds(*c.runner, opt, c.q, c.pattern);
      ptrs.push_back(&c);
    }
    CLM_RETURN_IF_ERROR(run_interleaved(ptrs, opt, opt.active));

    double t1 = 0;
    for (const auto& c : qcfg)
      if (c.q == 1) t1 = c.round_ms.median();
    double scaling_sum = 0;
    int scaling_n = 0;
    for (auto& c : qcfg) {
      QPoint pt;
      pt.q = c.q;
      pt.pattern = c.pattern;
      pt.union_mean = c.rounds_sum > 0 ? c.union_sum / c.rounds_sum : 0;
      pt.round_ms = c.round_ms;
      pt.bytes_per_s = c.bytes_per_s;
      rr.q_points.push_back(pt);
      if (c.q == 1) {
        const double dq = static_cast<double>(c.timing.dequant_ns), gv = static_cast<double>(c.timing.gemv_ns);
        if (dq + gv > 0) rr.dequant_fraction = dq / (dq + gv);
        if (dq > 0) rr.dequant_bytes_per_s = static_cast<double>(c.timed_bytes) / (dq * 1e-9);
        if (gv > 0) rr.gemv_bytes_per_s = static_cast<double>(c.timed_bytes) / (gv * 1e-9);
      } else if (c.pattern == SelectionPattern::kSame && t1 > 0) {
        scaling_sum += (c.round_ms.median() / t1 - 1.0) / static_cast<double>(c.q - 1);
        ++scaling_n;
      }
    }
    if (scaling_n > 0) {
      rr.q_scaling = std::max(0.0, scaling_sum / scaling_n);
      rr.q_scaling_valid = true;
    }
    report.representations.push_back(std::move(rr));
  }
  return report;
}

// ---- sustained ----------------------------------------------------------------------------------------------

void summarize_sustained(SustainedReport& r) {
  const std::size_t n = r.samples.size();
  if (n == 0) return;
  auto median_of = [&](std::size_t from, std::size_t to) {
    Distribution d;
    for (std::size_t i = from; i < to; ++i) d.add(r.samples[i].bytes_per_s);
    return d.median();
  };
  const std::size_t quarter = std::max<std::size_t>(1, n / 4);
  r.initial_bytes_per_s = median_of(0, std::min<std::size_t>(5, quarter));
  r.final_bytes_per_s = median_of(n - quarter, n);
  r.sustained_factor = r.initial_bytes_per_s > 0 ? std::min(1.0, r.final_bytes_per_s / r.initial_bytes_per_s) : 1.0;
  // Least-squares slope of throughput over time, as a percentage of the initial rate per minute.
  if (n >= 3 && r.initial_bytes_per_s > 0) {
    double mt = 0, my = 0;
    for (const auto& s : r.samples) {
      mt += s.t_s;
      my += s.bytes_per_s;
    }
    mt /= static_cast<double>(n);
    my /= static_cast<double>(n);
    double num = 0, den = 0;
    for (const auto& s : r.samples) {
      num += (s.t_s - mt) * (s.bytes_per_s - my);
      den += (s.t_s - mt) * (s.t_s - mt);
    }
    if (den > 0) r.slope_pct_per_min = 100.0 * (num / den) * 60.0 / r.initial_bytes_per_s;
  }
  // Equilibrium: first sample from which at least 3 trailing samples all stay within 3% of the final rate.
  r.time_to_equilibrium_s = 0;
  for (std::size_t i = 0; i + 3 <= n; ++i) {
    bool stable = true;
    for (std::size_t j = i; j < n && stable; ++j)
      stable = std::fabs(r.samples[j].bytes_per_s - r.final_bytes_per_s) <= 0.03 * r.final_bytes_per_s;
    if (stable) {
      r.time_to_equilibrium_s = r.samples[i].t_s;
      break;
    }
  }
  bool any_ac = false, any_batt = false;
  for (const auto& s : r.samples)
    if (s.on_ac_power) (*s.on_ac_power ? any_ac : any_batt) = true;
  r.power_source_changed = any_ac && any_batt;
}

Result<SustainedReport> run_cpu_sustained(const CpuBenchOptions& opt, const std::string& representation, unsigned threads,
                                          double minutes, double sample_interval_s, platform::PowerMonitor* power,
                                          const std::function<bool()>& should_stop) {
  if (minutes <= 0 || sample_interval_s <= 0) return make_error(ErrorCode::kInvalidArgument, "minutes and sample interval must be positive");
  CLM_ASSIGN_OR_RETURN(auto provider, ExpertKernelRegistry::instance().create(opt.provider));
  CLM_ASSIGN_OR_RETURN(auto probe_bank, provider->make_bank(representation, opt.shape, 1, opt.seed));
  const std::uint64_t per = std::max<std::uint64_t>(1, probe_bank->bytes_per_expert());
  const std::size_t n = static_cast<std::size_t>(std::clamp<std::uint64_t>(opt.bank_bytes / per, std::uint64_t{opt.active} * 2, 512));
  probe_bank.reset();
  CLM_ASSIGN_OR_RETURN(auto bank, provider->make_bank(representation, opt.shape, n, opt.seed));
  Runner runner(*bank, opt.shape, std::max(1u, threads), opt.seed);

  SustainedReport rep;
  rep.representation = representation;
  rep.threads = runner.threads();
  rep.minutes = minutes;
  rep.sample_interval_s = sample_interval_s;
  Stopwatch total;
  const double limit_s = minutes * 60.0;
  std::uint64_t salt = 1;
  while (total.elapsed_ms() / 1000.0 < limit_s) {
    if (should_stop && should_stop()) break;
    Stopwatch window;
    std::uint64_t bytes = 0;
    do {
      auto out = runner.run(1, opt.active, 1, SelectionPattern::kSame, salt++);
      CLM_RETURN_IF_ERROR(out.status);
      bytes += out.bytes;
    } while (window.elapsed_ms() / 1000.0 < sample_interval_s && total.elapsed_ms() / 1000.0 < limit_s);
    SustainedSample s;
    s.t_s = total.elapsed_ms() / 1000.0;
    s.bytes_per_s = static_cast<double>(bytes) / (window.elapsed_ms() / 1000.0);
    if (power) {
      auto p = power->sample();
      if (p.is_ok()) s.on_ac_power = p->on_ac_power;
    }
    rep.samples.push_back(s);
  }
  summarize_sustained(rep);
  return rep;
}

std::unique_ptr<platform::PowerMonitor> make_host_power_monitor() {
#ifdef _WIN32
  return platform::make_windows_power_monitor();
#else
  return nullptr;
#endif
}

// ---- result emission ----------------------------------------------------------------------------------------

void emit_cpu_metrics(BenchmarkResult& r, const CpuBenchReport& rep) {
  r.metric("cpu.provider", rep.provider_id);
  r.metric("cpu.provider_description", rep.provider_description);
  r.metric("cpu.isa_selected", rep.isa);
  r.metric("cpu.expert_shape.hidden", rep.shape.hidden);
  r.metric("cpu.expert_shape.ff", rep.shape.ff);
  r.metric("cpu.active_experts_per_position", rep.active);
  double q_scaling = 0;
  bool have_q = false;
  unsigned best = 1, usable = 1;
  for (const auto& rr : rep.representations) {
    const std::string p = "cpu." + rr.representation;
    // Effective expert throughput at q=1 on the best thread count.
    for (const auto& pt : rr.q_points)
      if (pt.q == 1) r.metric("cpu.expert_bytes_per_s." + rr.representation, pt.bytes_per_s, "bytes/s");
    r.metric(p + ".bytes_per_expert", rr.bytes_per_expert);
    r.metric(p + ".bank_experts", rr.bank_experts);
    for (const auto& [t, d] : rr.thread_bytes_per_s) r.metric(p + ".thread_scaling.t" + std::to_string(t), d, "bytes/s");
    r.metric(p + ".best_threads", rr.best_threads);
    r.metric(p + ".usable_threads", rr.usable_threads);
    r.metric(p + ".single_thread_bytes_per_s", rr.single_thread_bytes_per_s);
    for (const auto& pt : rr.q_points) {
      const std::string k = p + ".q" + std::to_string(pt.q) + "." + to_string(pt.pattern);
      r.metric(k + ".round_ms", pt.round_ms, "ms");
      r.metric(k + ".bytes_per_s", pt.bytes_per_s, "bytes/s");
      r.metric(k + ".union_experts_mean", pt.union_mean);
    }
    r.metric(p + ".dequant_fraction", rr.dequant_fraction);
    r.metric(p + ".dequant_bytes_per_s_per_thread", rr.dequant_bytes_per_s);
    r.metric(p + ".gemv_bytes_per_s_per_thread", rr.gemv_bytes_per_s);
    if (rr.q_scaling_valid) {
      r.metric(p + ".q_scaling", rr.q_scaling);
      q_scaling = std::max(q_scaling, rr.q_scaling);
      have_q = true;
    }
    best = std::max(best, rr.best_threads);
    usable = std::max(usable, rr.usable_threads);
  }
  // Placement carries one q_scaling and one thread budget per domain: the conservative (largest) of the
  // representations measured, which for a single-representation model is simply that representation's.
  if (have_q) r.metric("cpu.q_scaling", q_scaling);
  r.metric("cpu.best_threads", best);
  r.metric("cpu.usable_threads", usable);
}

void emit_sustained_metrics(BenchmarkResult& r, const SustainedReport& s) {
  r.metric("cpu.sustained.representation", s.representation);
  r.metric("cpu.sustained.threads", s.threads);
  r.metric("cpu.sustained.minutes", s.minutes);
  r.metric("cpu.sustained.samples", s.samples.size());
  r.metric("cpu.sustained.initial_bytes_per_s", s.initial_bytes_per_s);
  r.metric("cpu.sustained.final_bytes_per_s", s.final_bytes_per_s);
  r.metric("cpu.sustained_factor", s.sustained_factor);
  r.metric("cpu.sustained.slope_pct_per_min", s.slope_pct_per_min);
  r.metric("thermal.time_to_equilibrium_s", s.time_to_equilibrium_s);
  r.metric("cpu.sustained.power_source_changed", s.power_source_changed);
  for (const auto& sm : s.samples) {
    nlohmann::json e = {{"kind", "sustained_sample"}, {"t_s", sm.t_s}, {"bytes_per_s", sm.bytes_per_s}};
    if (sm.on_ac_power) e["on_ac_power"] = *sm.on_ac_power;
    r.trace(std::move(e));
  }
}

}  // namespace clusterlm::bench
