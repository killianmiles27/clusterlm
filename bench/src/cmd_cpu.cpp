// clusterlm-bench cpu / memory: CPU routed-expert throughput, thread scaling, q scaling, sustained behaviour and
// host memory measurements. Results are Measured on THIS host; host_role says whether that host is a target.
#include <csignal>
#include <cstdio>

#include "bench_common.hpp"
#include "clusterlm/common/clock.hpp"
#include "commands.hpp"
#include "nvml_probe.hpp"
#include "system_probe.hpp"

namespace clusterlm::bench {

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }
}  // namespace

int cmd_cpu(const cli::Args& args) {
  Stopwatch sw;
  const auto host = probe_host();
  const RunContext ctx = make_run_context(args);
  BenchmarkResult r(experiment_id(args, ctx, "HQ-CPU-01", "dev-cpu"), host);
  apply_run_context(r, ctx);

  if (args.has("list-providers")) {
    for (const auto& id : ExpertKernelRegistry::instance().ids()) {
      auto p = ExpertKernelRegistry::instance().create(id);
      std::printf("%-12s %s\n", id.c_str(), p.is_ok() ? p.value()->description().c_str() : p.status().message().c_str());
    }
    return 0;
  }

  CpuBenchOptions opt = cpu_options_from_args(args);
  r.config("provider", opt.provider);
  r.config("quick", opt.quick);
  r.config("reps", opt.reps);
  r.config("active_experts", opt.active);
  r.config("q", opt.qs);
  r.config("expert_hidden", opt.shape.hidden);
  r.config("expert_ff", opt.shape.ff);
  r.config("bank_bytes", opt.bank_bytes);
  if (args.has("isa")) r.config("isa_requested", args.get("isa"));
  r.metric("cpu.features", host.cpu_features);
  r.metric("cpu.logical_cpus", host.logical_cpus);

  auto progress = [](const std::string& s) { std::fprintf(stderr, "cpu: %s\n", s.c_str()); };
  auto report = run_cpu_bench(opt, args.has("quiet") ? std::function<void(const std::string&)>{} : progress);
  if (!report.is_ok()) {
    const bool unavailable = report.status().code() == ErrorCode::kHardwareUnavailable;
    r.check("provider_available", false, report.status().to_string());
    r.pending("HQ-CPU-01");
    (void)emit(args, r, sw.elapsed_ms() / 1000.0);
    return unavailable ? 3 : 1;
  }
  r.set_measured();
  emit_cpu_metrics(r, report.value());
  r.check("provider_available", true, report->provider_id + " (" + report->isa + ")");
  r.config("meets_min_repetitions", opt.reps >= 5);  // qualification runs need >= 5
  if (args.has("isa")) {
    bool match = false;
    for (const auto& isa : parse_string_list(args.get("isa"))) match |= isa == report->isa;
    r.check("requested_isa_selected", match, "requested " + args.get("isa") + ", provider selected " + report->isa);
  }
  for (const auto& rr : report->representations) {
    r.check("throughput_positive_" + rr.representation, rr.best_bytes_per_s > 0);
    r.check("q_scaling_measured_" + rr.representation, rr.q_scaling_valid || opt.qs.size() < 2);
  }
  if (report->provider_id == "reference") r.pending("HQ-CPU-01");  // reference kernels are not the production CPU path

  if (args.has("sustained")) {
    const double minutes = args.number("minutes", 1.0);
    const double sample_s = args.number("sample-s", 10.0);
    const std::string rep = args.get("representation", report->representations.empty() ? "" : report->representations.back().representation);
    unsigned threads = static_cast<unsigned>(args.integer("sustained-threads", 0));
    if (threads == 0)
      for (const auto& rr : report->representations)
        if (rr.representation == rep) threads = rr.usable_threads;
    std::signal(SIGINT, on_signal);
    auto power = make_host_power_monitor();
    r.config("sustained_minutes", minutes);
    r.config("sustained_sample_s", sample_s);
    r.metric("power.source_sampled", power != nullptr);
    // HQ-CPU-02: GPU clocks/power/temperature beside the CPU run (NVML, when the driver library is present).
    auto nvml = NvmlTelemetry::open();
    std::unique_ptr<TelemetryRecorder> telemetry;
    if (nvml.is_ok()) {
      telemetry = std::make_unique<TelemetryRecorder>(*nvml.value(), sample_s);
      telemetry->start();
    } else {
      emit_gpu_telemetry_unavailable(r, nvml.status().message(), "sustained.nvml.");
    }
    emit_power_environment(r, probe_power_environment());
    auto sus = run_cpu_sustained(opt, rep, std::max(1u, threads), minutes, sample_s, power.get(), [] { return g_stop != 0; });
    if (telemetry) emit_gpu_telemetry(r, *nvml.value(), telemetry->stop(), "sustained.nvml.");
    if (!sus.is_ok()) {
      r.check("sustained_run", false, sus.status().to_string());
    } else {
      emit_sustained_metrics(r, sus.value());
      r.check("sustained_samples_recorded", !sus->samples.empty());
      if (minutes < kMinSustainedMinutes) r.pending("HQ-CPU-02");  // thermal equilibrium needs the full 30 minutes
    }
  } else {
    r.pending("HQ-CPU-02");
  }
  return emit(args, r, sw.elapsed_ms() / 1000.0);
}

int cmd_memory(const cli::Args& args) {
  Stopwatch sw;
  const auto host = probe_host();
  const RunContext ctx = make_run_context(args);
  BenchmarkResult r(experiment_id(args, ctx, "HQ-PROF-01", "dev-memory"), host);
  apply_run_context(r, ctx);
  r.set_measured();
  const MemoryBenchOptions opt = memory_options_from_args(args);
  r.config("max_bytes", opt.max_bytes);
  r.config("step_bytes", opt.step_bytes);
  r.config("bandwidth_bytes", opt.bandwidth_bytes);
  r.config("reps", opt.reps);
  r.config("quick", opt.quick);
  auto m = measure_memory(opt);
  if (!m.is_ok()) {
    r.check("memory_measured", false, m.status().to_string());
    return emit(args, r, sw.elapsed_ms() / 1000.0);
  }
  emit_memory_metrics(r, m.value());
  r.check("ram_total_positive", m->info.total_physical > 0);
  r.check("allocation_probe_within_policy", m->alloc && m->alloc->largest_ok_bytes <= m->alloc->ceiling_bytes);
  r.check("bandwidth_measured", !m->bandwidth.empty() && m->bandwidth.front().bytes_per_s.median() > 0);
  r.pending("HQ-PCIE-01");  // the pinned-allocation probe needs the CUDA build (see `gpu`)
  return emit(args, r, sw.elapsed_ms() / 1000.0);
}

}  // namespace clusterlm::bench
