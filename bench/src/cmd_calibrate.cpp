// clusterlm-bench calibrate: measure every placement input that can be measured on this machine and write a
// HardwareProfile (and, with --peer, a NetworkProfile) that placement can load directly.
//
// Measured fields carry provenance Measured with source "bench:<run-id>@<machine-id>". Fields the run did not
// measure stay as they were in the existing/base profile, or become Synthetic placeholders. The tool never
// writes Qualified and never promotes beyond Measured (merge rules in bench_profile.hpp).
//
//   clusterlm-bench calibrate --machine-id g14 --role node --on-target [--peer father=HOST:7400 ...]
//   clusterlm-bench calibrate --quick                     (development host: smoke run, dev-<hostname> ids)
#include <cstdio>
#include <filesystem>

#include "bench_common.hpp"
#include "clusterlm/common/clock.hpp"
#include "commands.hpp"
#include "gpu_probe.hpp"
#include "nvml_probe.hpp"

namespace clusterlm::bench {

namespace fs = std::filesystem;
using placement::Provenance;
using placement::Quantity;

namespace {

void log_step(const cli::Args& args, const std::string& s) {
  if (!args.has("quiet")) std::fprintf(stderr, "calibrate: %s\n", s.c_str());
}

GpuBenchOptions calibrate_gpu_options(const cli::Args& args) {
  GpuBenchOptions o;
  o.quick = args.has("quick");
  if (o.quick) {
    o.ring_bytes = {16ull << 20};
    o.alloc_sizes = {1ull << 20, 16ull << 20};
    o.pinned_cap_bytes = 256ull << 20;
    o.pinned_step_bytes = 64ull << 20;
  }
  if (args.has("reps")) o.reps = static_cast<unsigned>(args.integer("reps", o.reps));
  return o;
}

}  // namespace

int cmd_calibrate(const cli::Args& args) {
  Stopwatch sw;
  const auto host = probe_host();
  const RunContext ctx = make_run_context(args);
  if (ctx.role != "father" && ctx.role != "node") {
    std::fprintf(stderr, "--role must be father or node\n");
    return 2;
  }
  BenchmarkResult r(experiment_id(args, ctx, "HQ-PROF-01", "dev-calibrate"), host);
  apply_run_context(r, ctx);
  r.set_measured();
  r.config("machine_id", ctx.machine_id);
  r.config("role", ctx.role);
  r.config("run_id", ctx.run_id);
  r.config("on_target", ctx.on_target);
  r.config("quick", args.has("quick"));

  HardwareMeasurements hm;
  hm.machine_id = ctx.machine_id;
  hm.role = ctx.role == "father" ? placement::DomainRole::kFather : placement::DomainRole::kNode;
  hm.run_id = ctx.run_id;
  hm.host = host;
  // HQ-PROF-01: the power plan (Windows) / cpufreq governor (Linux) the numbers were measured under.
  hm.power = probe_power_environment();
  emit_power_environment(r, *hm.power);

  // ---- CPU ----
  if (!args.has("skip-cpu")) {
    CpuBenchOptions opt = cpu_options_from_args(args);
    log_step(args, "cpu expert kernels (" + opt.provider + ")");
    auto rep = run_cpu_bench(opt);
    if (rep.is_ok()) {
      emit_cpu_metrics(r, rep.value());
      r.check("cpu_measured", true, rep->provider_id + " (" + rep->isa + ")");
      hm.cpu = std::move(rep).value();
    } else {
      r.check("cpu_measured", false, rep.status().to_string());
    }
    if (hm.cpu && args.has("sustained-minutes")) {
      const double minutes = args.number("sustained-minutes", 0);
      const double sample_s = args.number("sample-s", 10.0);
      const auto& last = hm.cpu->representations.back();
      log_step(args, "sustained run, " + std::to_string(minutes) + " min");
      auto power = make_host_power_monitor();
      auto nvml = NvmlTelemetry::open();
      std::unique_ptr<TelemetryRecorder> telemetry;
      if (nvml.is_ok()) {
        telemetry = std::make_unique<TelemetryRecorder>(*nvml.value(), sample_s);
        telemetry->start();
      } else {
        emit_gpu_telemetry_unavailable(r, nvml.status().message(), "sustained.nvml.");
      }
      auto sus = run_cpu_sustained(opt, last.representation, last.usable_threads, minutes, sample_s, power.get());
      if (telemetry) emit_gpu_telemetry(r, *nvml.value(), telemetry->stop(), "sustained.nvml.");
      if (sus.is_ok()) {
        emit_sustained_metrics(r, sus.value());
        hm.sustained = std::move(sus).value();
      } else {
        r.check("sustained_run", false, sus.status().to_string());
      }
    }
  }
  if (!hm.sustained || hm.sustained->minutes < kMinSustainedMinutes) r.pending("HQ-CPU-02");
  if (!hm.cpu || hm.cpu->provider_id == "reference") r.pending("HQ-CPU-01");

  // ---- memory ----
  if (!args.has("skip-memory")) {
    log_step(args, "memory");
    auto mem = measure_memory(memory_options_from_args(args));
    if (mem.is_ok()) {
      emit_memory_metrics(r, mem.value());
      r.check("memory_measured", true);
      hm.memory = std::move(mem).value();
    } else {
      r.check("memory_measured", false, mem.status().to_string());
    }
  }

  // ---- GPU (optional) ----
  bool gpu_missing = false;
  if (!args.has("skip-gpu")) {
    log_step(args, "gpu");
    auto probe = make_gpu_probe();
    GpuBenchReport g = run_gpu_bench(*probe, calibrate_gpu_options(args));
    emit_gpu_metrics(r, g);
    if (g.available && !g.devices.empty()) {
      r.set_gpu(g.devices.front().info.name, "CUDA driver " + std::to_string(g.devices.front().info.driver_version));
      if (auto budget = make_host_gpu_budget_probe()) {
        auto q = budget->query();
        if (q.is_ok())
          for (const auto& a : q.value())
            if (a.name == g.devices.front().info.name) hm.dxgi_budget_bytes.push_back(a.budget_bytes);
      }
      hm.gpu = std::move(g);
    } else {
      gpu_missing = true;
      r.metric("gpu.unavailable_reason", g.unavailable_reason);
      for (const char* id : {"HQ-GPU-01", "HQ-GPU-02", "HQ-PCIE-01"}) r.pending(id);
    }
  }
  if (hm.gpu) {
    r.pending("HQ-GPU-01");  // layer-kind timings need the Strata kernels; cuBLAS shapes are not them
    r.pending("HQ-GPU-02");
  }

  // ---- network (optional) ----
  std::vector<PeerLink> links;
  std::optional<ConcurrentMeasure> concurrent;
  if (args.has("peer")) {
    log_step(args, "network");
    auto st = measure_remote_peers(args, r, links, concurrent);
    r.check("network_measured", st.is_ok(), st.to_string());
  }

  // ---- hardware profile ----
  const fs::path out_dir = args.get("out-dir", "bench-results");
  std::error_code ec;
  fs::create_directories(out_dir, ec);
  const std::string profile_path = args.get("profile-out", (out_dir / ("profile-" + ctx.machine_id + ".json")).string());
  placement::HardwareProfile measured = build_measured_profile(hm);
  placement::HardwareProfile merged = measured;
  MergeReport mr;
  bool merged_into_existing = false;
  std::string base_path;
  if (!args.has("no-merge") && fs::exists(profile_path)) base_path = profile_path;
  else if (args.has("base")) base_path = args.get("base");
  if (!base_path.empty()) {
    auto base = placement::load_hardware_profile(base_path);
    if (!base.is_ok()) {
      std::fprintf(stderr, "cannot load base profile %s: %s\n", base_path.c_str(), base.status().to_string().c_str());
      return 1;
    }
    merged = std::move(base).value();
    mr = merge_hardware_profile(merged, measured);
    merged_into_existing = true;
  }
  const Status valid = placement::validate(merged);
  r.check("profile_valid", valid.is_ok(), valid.to_string());
  bool no_overpromotion = true;
  placement::for_each_quantity(merged, [&](const std::string&, Quantity& q) {
    const auto src = parse_bench_source(q.source);
    if (src && src->run_id == ctx.run_id && q.provenance != Provenance::kMeasured) no_overpromotion = false;
  });
  r.check("measured_fields_from_this_run_are_measured_not_qualified", no_overpromotion);
  if (valid.is_ok()) {
    if (auto st = placement::save_hardware_profile(merged, profile_path); !st.is_ok()) {
      r.check("profile_written", false, st.to_string());
    } else {
      r.check("profile_written", true, profile_path);
    }
  }
  r.metric("profile.path", profile_path);
  r.metric("profile.merged_into_existing", merged_into_existing);
  r.metric("profile.measured_field_count", measured_paths(merged).size());
  r.metric("profile.measured_fields", nlohmann::json(measured_paths(merged)));
  r.metric("profile.replaced_fields", mr.replaced.size());
  r.metric("profile.kept_newer_fields", mr.kept_newer.size());
  r.metric("profile.kept_qualified_fields", mr.kept_qualified.size());
  r.metric("profile.weakest_provenance", std::string(placement::to_string(placement::weakest_provenance(merged))));

  // ---- network profile ----
  if (!links.empty()) {
    const std::string net_path = args.get("network-out", (out_dir / ("network-" + ctx.machine_id + ".json")).string());
    placement::NetworkProfile net = build_network_profile(ctx.machine_id, ctx.run_id, links, concurrent ? &*concurrent : nullptr);
    if (!args.has("no-merge") && fs::exists(net_path)) {
      auto existing = placement::load_network_profile(net_path);
      if (existing.is_ok()) {
        placement::NetworkProfile base = std::move(existing).value();
        merge_network_profile(base, net);
        net = std::move(base);
      }
    }
    const Status nv = placement::validate(net);
    r.check("network_profile_valid", nv.is_ok(), nv.to_string());
    if (nv.is_ok()) {
      auto st = placement::save_network_profile(net, net_path);
      r.check("network_profile_written", st.is_ok(), st.to_string());
    }
    r.metric("network_profile.path", net_path);
  } else if (!args.has("peer")) {
    r.pending("HQ-NET-01");
  }

  const int code = emit(args, r, sw.elapsed_ms() / 1000.0);
  if (code == 0 && gpu_missing && args.has("require-gpu")) return 3;
  return code;
}

}  // namespace clusterlm::bench
