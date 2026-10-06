// clusterlm-bench gpu / pcie: GPU enumeration, VRAM, allocation overhead, host<->device bandwidth (pageable vs
// pinned), pinned-memory limit and cuBLAS GEMV/GEMM throughput for model-relevant shapes.
//
// Without a CUDA build or without a CUDA device the command reports exactly why, lists the qualification items it
// could not answer and exits with code 3 — it never crashes and never fabricates numbers.
#include <cstdio>

#include "bench_common.hpp"
#include "clusterlm/common/clock.hpp"
#include "commands.hpp"
#include "gpu_probe.hpp"

namespace clusterlm::bench {

namespace {

GpuBenchOptions gpu_options_from_args(const cli::Args& args) {
  GpuBenchOptions o;
  o.quick = args.has("quick");
  if (o.quick) {
    o.ring_bytes = {16ull << 20};
    o.alloc_sizes = {1ull << 20, 16ull << 20};
    o.pinned_cap_bytes = 256ull << 20;
    o.pinned_step_bytes = 64ull << 20;
  }
  if (args.has("ring-bytes")) {
    o.ring_bytes.clear();
    for (const auto& s : parse_string_list(args.get("ring-bytes"))) {
      auto v = parse_size(s);
      if (v.is_ok()) o.ring_bytes.push_back(v.value());
    }
  }
  if (args.has("pinned-cap") && args.get("pinned-cap") != "sweep") {
    auto v = parse_size(args.get("pinned-cap"));
    if (v.is_ok()) o.pinned_cap_bytes = v.value();
  }
  if (args.has("pinned-step")) {
    auto v = parse_size(args.get("pinned-step"));
    if (v.is_ok() && v.value() > 0) o.pinned_step_bytes = v.value();
  }
  if (args.has("reps")) o.reps = static_cast<unsigned>(args.integer("reps", o.reps));
  return o;
}

void record_dxgi(BenchmarkResult& r) {
  auto probe = make_host_gpu_budget_probe();
  if (!probe) {
    r.metric("gpu.dxgi_available", false);
    return;
  }
  auto q = probe->query();
  r.metric("gpu.dxgi_available", q.is_ok());
  if (!q.is_ok()) return;
  for (const auto& a : q.value()) {
    const std::string p = "gpu.dxgi." + std::to_string(a.adapter_index);
    r.metric(p + ".name", a.name);
    r.metric(p + ".budget_bytes", a.budget_bytes);
    r.metric(p + ".current_usage_bytes", a.current_usage_bytes);
    r.metric(p + ".available_for_reservation_bytes", a.available_for_reservation_bytes);
    r.metric(p + ".dedicated_video_memory_bytes", a.dedicated_video_memory_bytes);
  }
}

}  // namespace

int cmd_gpu(const cli::Args& args, bool pcie_only) {
  Stopwatch sw;
  const auto host = probe_host();
  const RunContext ctx = make_run_context(args);
  BenchmarkResult r(experiment_id(args, ctx, pcie_only ? "HQ-PCIE-01" : "HQ-GPU-01", pcie_only ? "dev-pcie" : "dev-gpu"), host);
  apply_run_context(r, ctx);
  GpuBenchOptions opt = gpu_options_from_args(args);
  r.config("reps", opt.reps);
  r.config("quick", opt.quick);
  r.config("ring_bytes", opt.ring_bytes);
  r.config("pinned_cap_bytes", opt.pinned_cap_bytes);
  r.config("built_with_cuda", gpu_probe_built_with_cuda());
  record_dxgi(r);

  auto probe = make_gpu_probe();
  if (pcie_only) {
    opt.alloc_sizes.clear();
    opt.include_registered_kernels = false;
  }
  GpuBenchReport report = run_gpu_bench(*probe, opt);
  emit_gpu_metrics(r, report);
  if (!report.available) {
    r.check("cuda_device_available", false, report.unavailable_reason);
    r.pending("HQ-PCIE-01");
    if (!pcie_only) {
      r.pending("HQ-GPU-01");
      r.pending("HQ-GPU-02");
    }
    (void)emit(args, r, sw.elapsed_ms() / 1000.0);
    return 3;
  }
  r.set_measured();
  r.check("cuda_device_available", true, std::to_string(report.devices.size()) + " device(s)");
  for (const auto& d : report.devices) {
    const std::string tag = "device" + std::to_string(d.info.index);
    r.check(tag + "_all_sub_measurements_completed", d.notes.empty(), d.notes.empty() ? "" : d.notes.front());
    r.check(tag + "_pinned_transfer_measured", [&] {
      for (const auto& t : d.transfers)
        if (t.kind != HostMemoryKind::kPageable && !t.h2d_bytes_per_s.empty()) return true;
      return false;
    }());
  }
  r.config("meets_min_repetitions", opt.reps >= 5);  // qualification runs need >= 5
  r.set_gpu(report.devices.front().info.name, "CUDA driver " + std::to_string(report.devices.front().info.driver_version));
  if (!ctx.on_target || !pcie_only) r.pending("HQ-PCIE-01");
  if (!pcie_only) {
    // cuBLAS shapes bound what a GPU can do; the layer-kind timings (dense, QSA, GDN, experts) need the Strata kernels.
    r.pending("HQ-GPU-01");
    r.pending("HQ-GPU-02");
  }
  return emit(args, r, sw.elapsed_ms() / 1000.0);
}

}  // namespace clusterlm::bench
