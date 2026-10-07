// clusterlm-bench nvml / storage-census: OS- and driver-level observers that run beside a workload.
//
//   nvml            samples NVML (clocks, power, temperature, memory used, throttle reasons) every --sample-s for
//                   --minutes. Run it next to a sustained workload (HQ-GPU-03, HQ-CPU-02). Without NVML it records
//                   "unavailable: <reason>" and exits 3; it never writes zeros.
//   storage-census  --before | --after | --diff snapshots of the known backend/driver cache and temp locations
//                   (HQ-STORE-01). Names and sizes only.
#include <csignal>
#include <cstdio>
#include <fstream>
#include <thread>

#include "bench_common.hpp"
#include "clusterlm/common/clock.hpp"
#include "commands.hpp"
#include "nvml_probe.hpp"
#include "storage_census.hpp"

namespace clusterlm::bench {

namespace fs = std::filesystem;

namespace {
volatile std::sig_atomic_t g_nvml_stop = 0;
void on_nvml_signal(int) { g_nvml_stop = 1; }

Result<CensusSnapshot> load_snapshot(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return make_error(ErrorCode::kNotFound, "cannot read snapshot " + path);
  auto j = nlohmann::json::parse(f, nullptr, false);
  if (j.is_discarded()) return make_error(ErrorCode::kInvalidArgument, path + " is not JSON");
  return census_from_json(j);
}

Status save_snapshot(const std::string& path, const CensusSnapshot& s) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return make_error(ErrorCode::kUnavailable, "cannot write snapshot " + path);
  f << census_to_json(s).dump() << "\n";
  return f ? Status::ok() : make_error(ErrorCode::kUnavailable, "write failed: " + path);
}

std::vector<CensusRoot> roots_from_args(const cli::Args& args) {
  std::vector<CensusRoot> extra;
  for (const auto& spec : args.all("root")) {  // LABEL=PATH
    const auto eq = spec.find('=');
    CensusRoot root;
    root.label = eq == std::string::npos ? spec : spec.substr(0, eq);
    root.path = eq == std::string::npos ? spec : spec.substr(eq + 1);
    extra.push_back(std::move(root));
  }
  for (const auto& p : args.all("staging-root")) {
    CensusRoot root;
    root.label = "node_staging_" + std::to_string(extra.size());
    root.path = p;
    extra.push_back(std::move(root));
  }
  std::vector<fs::path> exclude;
  for (const auto& p : args.all("exclude")) exclude.emplace_back(p);
  return default_census_roots(extra, exclude);
}
}  // namespace

int cmd_nvml(const cli::Args& args) {
  Stopwatch sw;
  const auto host = probe_host();
  const RunContext ctx = make_run_context(args);
  BenchmarkResult r(experiment_id(args, ctx, "HQ-GPU-03", "dev-nvml"), host);
  apply_run_context(r, ctx);
  const double minutes = args.number("minutes", 0.05);
  const double sample_s = args.number("sample-s", 1.0);
  r.config("minutes", minutes);
  r.config("sample_s", sample_s);
  auto nvml = NvmlTelemetry::open();
  if (!nvml.is_ok()) {
    emit_gpu_telemetry_unavailable(r, nvml.status().message(), "nvml.");
    r.check("nvml_available", false, "unavailable: " + nvml.status().message());
    r.pending("HQ-GPU-03");
    r.pending("HQ-CPU-02");
    (void)emit(args, r, sw.elapsed_ms() / 1000.0);
    return 3;
  }
  r.set_measured();
  r.set_gpu(nvml.value()->device_names().front(), nvml.value()->driver_version());
  std::signal(SIGINT, on_nvml_signal);
  TelemetryRecorder rec(*nvml.value(), sample_s);
  rec.start();
  while (!g_nvml_stop && sw.elapsed_ms() < minutes * 60'000.0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const auto samples = rec.stop();
  emit_gpu_telemetry(r, *nvml.value(), samples, "nvml.");
  r.check("nvml_samples_recorded", !samples.empty());
  if (minutes < kMinSustainedMinutes) {  // thermal equilibrium needs the full 30 minutes
    r.pending("HQ-GPU-03");
    r.pending("HQ-CPU-02");
  }
  return emit(args, r, sw.elapsed_ms() / 1000.0);
}

int cmd_storage_census(const cli::Args& args) {
  Stopwatch sw;
  const auto host = probe_host();
  const RunContext ctx = make_run_context(args);
  BenchmarkResult r(experiment_id(args, ctx, "HQ-STORE-01", "dev-storage-census"), host);
  apply_run_context(r, ctx);
  r.set_measured();
  const int modes = int(args.has("before")) + int(args.has("after")) + int(args.has("diff"));
  if (modes != 1) {
    std::fprintf(stderr, "storage-census: give exactly one of --before, --after, --diff\n");
    return 2;
  }
  const std::string snap_path = args.get("snapshot", "storage-census.json");
  r.config("snapshot", snap_path);
  r.config("mode", args.has("before") ? "before" : args.has("after") ? "after" : "diff");

  if (args.has("before")) {
    const auto roots = roots_from_args(args);
    const auto snap = take_census(roots);
    if (auto st = save_snapshot(snap_path, snap); !st.is_ok()) {
      r.check("snapshot_written", false, st.to_string());
      return emit(args, r, sw.elapsed_ms() / 1000.0);
    }
    r.check("snapshot_written", true, snap_path);
    for (const auto& rs : snap.roots) {
      r.metric("census.root." + rs.label + ".exists", rs.exists);
      r.metric("census.root." + rs.label + ".files", rs.files.size());
      r.metric("census.root." + rs.label + ".truncated", rs.truncated);
    }
    r.pending("HQ-STORE-01");
    return emit(args, r, sw.elapsed_ms() / 1000.0);
  }

  auto before = load_snapshot(snap_path);
  if (!before.is_ok()) {
    std::fprintf(stderr, "storage-census: %s\n", before.status().message().c_str());
    return 1;
  }
  CensusSnapshot after;
  if (args.has("after")) {
    // The roots of the --before snapshot are re-scanned so the two are comparable.
    std::vector<CensusRoot> roots;
    for (const auto& rs : before->roots) {
      CensusRoot root;
      root.label = rs.label;
      root.path = rs.path;
      for (const auto& p : args.all("exclude")) root.exclude.emplace_back(p);
      roots.push_back(std::move(root));
    }
    after = take_census(roots);
  } else {
    auto other = load_snapshot(args.get("against"));
    if (!other.is_ok()) {
      std::fprintf(stderr, "storage-census: --diff needs --against FILE: %s\n", other.status().message().c_str());
      return 2;
    }
    after = std::move(other).value();
  }
  const auto diff = diff_census(before.value(), after, before->taken_ns, after.taken_ns);
  emit_census_diff(r, diff);
  r.check("census_lists_complete", diff.truncated_roots.empty(),
          diff.truncated_roots.empty() ? "" : "a root hit its entry cap; the diff is a lower bound");
  r.pending("HQ-STORE-01");  // Synthetic here unless run on the target with --on-target; ETW/procmon is still manual
  return emit(args, r, sw.elapsed_ms() / 1000.0);
}

}  // namespace clusterlm::bench
