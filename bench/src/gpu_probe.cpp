// Probe-independent GPU bench logic: orchestration over a GpuProbe, metric emission, the kernel registry and the
// always-unavailable probe. The CUDA implementation lives in gpu_probe_cuda.cpp.
#include "gpu_probe.hpp"

namespace clusterlm::bench {

const char* to_string(HostMemoryKind k) {
  switch (k) {
    case HostMemoryKind::kPageable: return "pageable";
    case HostMemoryKind::kPinnedAlloc: return "pinned";
    case HostMemoryKind::kPinnedRegister: return "registered";
  }
  return "pageable";
}

const char* to_string(GemmPrecision p) { return p == GemmPrecision::kF16 ? "f16" : "f32"; }

GpuKernelRegistry& GpuKernelRegistry::instance() {
  static GpuKernelRegistry r;
  return r;
}

void GpuKernelRegistry::register_case(GpuKernelCase c) { cases_.push_back(std::move(c)); }

std::vector<GemmShape> model_gemm_shapes(bool quick) {
  std::vector<GemmShape> out;
  struct Base {
    const char* name;
    std::uint32_t rows, cols;
  };
  const std::vector<Base> bases = quick ? std::vector<Base>{{"expert_gate_up", 640, 2560}, {"expert_down", 2560, 640}}
                                        : std::vector<Base>{{"expert_gate_up", 640, 2560},
                                                            {"expert_down", 2560, 640},
                                                            {"hidden_square", 2560, 2560},
                                                            {"attn_q", 6144, 2560},
                                                            {"attn_kv", 512, 2560},
                                                            {"attn_o", 2560, 6144}};
  const std::vector<std::uint32_t> batches = quick ? std::vector<std::uint32_t>{1, 4} : std::vector<std::uint32_t>{1, 2, 4, 512};
  for (const auto& b : bases)
    for (auto n : batches) out.push_back({b.name, b.rows, b.cols, n});
  return out;
}

namespace {

class UnavailableProbe final : public GpuProbe {
 public:
  explicit UnavailableProbe(std::string reason) : reason_(std::move(reason)) {}
  bool available() const override { return false; }
  std::string unavailable_reason() const override { return reason_; }
  Result<std::vector<GpuDeviceInfo>> enumerate() override { return fail(); }
  Result<std::vector<AllocOverheadPoint>> allocation_overhead(int, const std::vector<std::uint64_t>&, unsigned) override {
    return fail();
  }
  Result<std::vector<TransferPoint>> transfers(int, const std::vector<std::uint64_t>&, const std::vector<HostMemoryKind>&,
                                               unsigned) override {
    return fail();
  }
  Result<PinnedProbeResult> pinned_limit(int, std::uint64_t, std::uint64_t) override { return fail(); }
  Result<std::vector<GemmPoint>> gemm(int, const std::vector<GemmShape>&, const std::vector<GemmPrecision>&, unsigned) override {
    return fail();
  }
  Result<std::vector<KernelTimePoint>> time_kernels(int, const std::vector<GpuKernelCase>&, unsigned) override { return fail(); }

 private:
  Status fail() const { return make_error(ErrorCode::kHardwareUnavailable, reason_); }
  std::string reason_;
};

}  // namespace

#ifndef CLUSTERLM_BENCH_HAS_CUDA
std::unique_ptr<GpuProbe> make_gpu_probe() {
  return std::make_unique<UnavailableProbe>(
      "this clusterlm-bench was built without CUDA (configure with -DCLUSTERLM_BENCH_CUDA=ON on a host with the "
      "CUDA toolkit)");
}
bool gpu_probe_built_with_cuda() { return false; }
#endif

std::unique_ptr<platform::GpuBudgetProbe> make_host_gpu_budget_probe() {
#ifdef _WIN32
  return platform::make_windows_gpu_budget_probe();
#else
  return nullptr;
#endif
}

std::unique_ptr<GpuProbe> make_unavailable_gpu_probe(std::string reason) {
  return std::make_unique<UnavailableProbe>(std::move(reason));
}

GpuBenchReport run_gpu_bench(GpuProbe& probe, const GpuBenchOptions& o) {
  GpuBenchReport rep;
  if (!probe.available()) {
    rep.unavailable_reason = probe.unavailable_reason();
    return rep;
  }
  auto devices = probe.enumerate();
  if (!devices.is_ok() || devices->empty()) {
    rep.unavailable_reason = devices.is_ok() ? "no CUDA device" : devices.status().message();
    return rep;
  }
  rep.available = true;
  for (const auto& info : devices.value()) {
    GpuDeviceReport d;
    d.info = info;
    const int dev = info.index;
    auto note = [&](const char* what, const Status& st) { d.notes.push_back(std::string(what) + ": " + st.to_string()); };
    if (auto r = probe.allocation_overhead(dev, o.alloc_sizes, o.reps); r.is_ok()) d.alloc = std::move(r).value();
    else note("allocation_overhead", r.status());
    if (auto r = probe.transfers(dev, o.ring_bytes,
                                 {HostMemoryKind::kPageable, HostMemoryKind::kPinnedAlloc, HostMemoryKind::kPinnedRegister}, o.reps);
        r.is_ok())
      d.transfers = std::move(r).value();
    else
      note("transfers", r.status());
    if (auto r = probe.pinned_limit(dev, o.pinned_cap_bytes, o.pinned_step_bytes); r.is_ok()) d.pinned = std::move(r).value();
    else note("pinned_limit", r.status());
    std::vector<GemmPrecision> precisions = {GemmPrecision::kF32, GemmPrecision::kF16};
    if (auto r = probe.gemm(dev, model_gemm_shapes(o.quick), precisions, o.reps); r.is_ok()) d.gemm = std::move(r).value();
    else note("gemm", r.status());
    if (o.include_registered_kernels && !GpuKernelRegistry::instance().cases().empty()) {
      if (auto r = probe.time_kernels(dev, GpuKernelRegistry::instance().cases(), o.reps); r.is_ok()) d.kernels = std::move(r).value();
      else note("kernels", r.status());
    }
    rep.devices.push_back(std::move(d));
  }
  return rep;
}

void emit_gpu_metrics(BenchmarkResult& r, const GpuBenchReport& rep) {
  r.metric("gpu.present", rep.available);
  r.metric("gpu.device_count", rep.devices.size());
  if (!rep.available) r.metric("gpu.unavailable_reason", rep.unavailable_reason);
  for (const auto& d : rep.devices) {
    const std::string p = "gpu." + std::to_string(d.info.index);
    r.metric(p + ".name", d.info.name);
    r.metric(p + ".vram_total", d.info.total_bytes);
    r.metric(p + ".vram_free", d.info.free_bytes);
    r.metric(p + ".compute_capability", std::to_string(d.info.compute_major) + "." + std::to_string(d.info.compute_minor));
    r.metric(p + ".multiprocessors", d.info.multiprocessors);
    r.metric(p + ".driver_version", d.info.driver_version);
    r.metric(p + ".runtime_version", d.info.runtime_version);
    for (const auto& a : d.alloc) {
      r.metric(p + ".alloc_us." + std::to_string(a.bytes), a.malloc_us, "us");
      r.metric(p + ".free_us." + std::to_string(a.bytes), a.free_us, "us");
    }
    for (const auto& t : d.transfers) {
      const std::string k = p + "." + to_string(t.kind) + "." + std::to_string(t.bytes);
      r.metric(k + ".h2d_bytes_per_s", t.h2d_bytes_per_s, "bytes/s");
      r.metric(k + ".d2h_bytes_per_s", t.d2h_bytes_per_s, "bytes/s");
    }
    r.metric(p + ".pinned_limit_bytes", d.pinned.largest_ok_bytes);
    r.metric(p + ".pinned_probe_cap_bytes", d.pinned.cap_bytes);
    r.metric(p + ".pinned_probe_stopped_by_cap", d.pinned.stopped_by_cap);
    for (const auto& g : d.gemm) {
      const std::string k = p + ".gemm." + to_string(g.precision) + "." + g.shape.name + ".n" + std::to_string(g.shape.batch);
      r.metric(k + ".ms", g.ms, "ms");
      r.metric(k + ".weight_bytes_per_s", g.weight_bytes_per_s);
      r.metric(k + ".tflops", g.tflops);
    }
    for (const auto& k : d.kernels) r.metric(p + ".kernel." + k.name + ".ms", k.ms, "ms");
    for (const auto& n : d.notes) r.trace({{"kind", "gpu_note"}, {"device", d.info.index}, {"note", n}});
  }
}

}  // namespace clusterlm::bench
