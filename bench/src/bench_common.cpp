#include "bench_common.hpp"

#include <algorithm>
#include <cstdlib>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace clusterlm::bench {

namespace {
std::string hostname() {
#ifdef _WIN32
  char buf[256] = {};
  DWORD n = sizeof buf;
  if (GetComputerNameA(buf, &n)) return std::string(buf, n);
  return "unknown";
#else
  char buf[256] = {};
  if (gethostname(buf, sizeof buf - 1) == 0 && buf[0]) return buf;
  return "unknown";
#endif
}
}  // namespace

RunContext make_run_context(const cli::Args& args) {
  RunContext c;
  c.run_id = args.get("run-id", make_run_id());
  c.on_target = args.has("on-target");
  c.machine_id = args.get("machine-id", c.on_target ? "" : "dev-" + hostname());
  if (c.machine_id.empty()) c.machine_id = hostname();
  c.role = args.get("role", "node");
  return c;
}

void apply_run_context(BenchmarkResult& r, const RunContext& ctx) {
  r.set_host_role(ctx.on_target ? ctx.role : "development-host", ctx.machine_id);
}

std::string experiment_id(const cli::Args& args, const RunContext& ctx, const char* hq_id, const char* dev_id) {
  if (args.has("experiment")) return args.get("experiment");
  return ctx.on_target ? hq_id : dev_id;
}

std::vector<std::uint32_t> parse_u32_list(const std::string& csv) {
  std::vector<std::uint32_t> out;
  for (const auto& s : cli::split(csv, ',')) out.push_back(static_cast<std::uint32_t>(std::stoul(s)));
  return out;
}

std::vector<std::string> parse_string_list(const std::string& csv) { return cli::split(csv, ','); }

Result<std::uint64_t> parse_size(const std::string& text) {
  if (text.empty()) return make_error(ErrorCode::kInvalidArgument, "empty size");
  std::size_t used = 0;
  std::uint64_t v = 0;
  try {
    v = std::stoull(text, &used);
  } catch (const std::exception&) {
    return make_error(ErrorCode::kInvalidArgument, "bad size '" + text + "'");
  }
  std::string suffix = text.substr(used);
  std::uint64_t mult = 1;
  if (suffix == "K" || suffix == "k") mult = 1ull << 10;
  else if (suffix == "M" || suffix == "m") mult = 1ull << 20;
  else if (suffix == "G" || suffix == "g") mult = 1ull << 30;
  else if (!suffix.empty()) return make_error(ErrorCode::kInvalidArgument, "bad size suffix in '" + text + "'");
  return v * mult;
}

CpuBenchOptions cpu_options_from_args(const cli::Args& args) {
  CpuBenchOptions o = args.has("quick") ? quick_cpu_options() : CpuBenchOptions{};
  o.quick = args.has("quick");
  o.provider = args.get("provider", o.provider);
  if (args.has("representations")) o.representations = parse_string_list(args.get("representations"));
  if (args.has("q")) o.qs = parse_u32_list(args.get("q"));
  if (args.has("threads") && args.get("threads") != "sweep") {
    o.threads.clear();
    for (auto t : parse_u32_list(args.get("threads"))) o.threads.push_back(t);
  }
  if (args.has("max-threads")) o.max_threads = static_cast<unsigned>(args.integer("max-threads", 0));
  if (args.has("active")) o.active = static_cast<std::uint32_t>(args.integer("active", o.active));
  if (args.has("hidden")) o.shape.hidden = static_cast<std::uint32_t>(args.integer("hidden", o.shape.hidden));
  if (args.has("ff")) o.shape.ff = static_cast<std::uint32_t>(args.integer("ff", o.shape.ff));
  if (args.has("reps")) o.reps = static_cast<unsigned>(args.integer("reps", o.reps));
  if (args.has("target-s")) o.target_rep_seconds = args.number("target-s", o.target_rep_seconds);
  if (args.has("bank-mib")) o.bank_bytes = args.integer("bank-mib", 1024) << 20;
  if (args.has("seed")) o.seed = args.integer("seed", o.seed);
  return o;
}

MemoryBenchOptions memory_options_from_args(const cli::Args& args) {
  MemoryBenchOptions o;
  o.quick = args.has("quick");
  if (o.quick) {
    o.max_bytes = 128ull << 20;
    o.step_bytes = 16ull << 20;
    o.bandwidth_bytes = 32ull << 20;
    o.max_threads = 2;
  }
  if (args.has("max-gib")) o.max_bytes = static_cast<std::uint64_t>(args.number("max-gib", 4) * static_cast<double>(1ull << 30));
  if (args.has("step-mib")) o.step_bytes = args.integer("step-mib", 256) << 20;
  if (args.has("bandwidth-mib")) o.bandwidth_bytes = args.integer("bandwidth-mib", 512) << 20;
  if (args.has("max-threads")) o.max_threads = static_cast<unsigned>(args.integer("max-threads", 0));
  if (args.has("reps")) o.reps = static_cast<unsigned>(args.integer("reps", o.reps));
  return o;
}

Result<MemoryMeasurement> measure_memory(const MemoryBenchOptions& o) {
  MemoryMeasurement m;
  m.info = probe_memory();
  AllocProbeOptions ap;
  ap.max_bytes = o.max_bytes;
  ap.step_bytes = std::max<std::uint64_t>(1, o.step_bytes);
  // Reserve at least 2 GiB, or 10% of RAM on large machines, so the probe can never starve the OS.
  ap.reserve_bytes = std::max<std::uint64_t>(o.quick ? (256ull << 20) : (2ull << 30), m.info.total_physical / 10);
  CLM_ASSIGN_OR_RETURN(auto alloc, probe_largest_allocation(ap));
  m.alloc = std::move(alloc);

  // Bandwidth buffer: never more than a quarter of what is available.
  std::uint64_t bytes = std::min(o.bandwidth_bytes, m.info.available_physical / 4);
  bytes = std::max<std::uint64_t>(bytes, 8ull << 20);
  const unsigned max_threads = o.max_threads ? o.max_threads : std::max(1u, std::thread::hardware_concurrency());
  for (unsigned t : thread_sweep(max_threads)) {
    CLM_ASSIGN_OR_RETURN(auto bw, measure_read_bandwidth_mt(bytes, t, o.reps));
    m.bandwidth.push_back(std::move(bw));
  }
  return m;
}

void emit_memory_metrics(BenchmarkResult& r, const MemoryMeasurement& m) {
  r.metric("memory.ram_total", m.info.total_physical);
  r.metric("memory.ram_available", m.info.available_physical);
  if (m.info.commit_limit) r.metric("memory.commit_limit", *m.info.commit_limit);
  if (m.info.commit_available) r.metric("memory.commit_available", *m.info.commit_available);
  r.metric("memory.page_size", m.info.page_size);
  if (m.alloc) {
    r.metric("memory.alloc.largest_ok_bytes", m.alloc->largest_ok_bytes);
    r.metric("memory.alloc.policy_ceiling_bytes", m.alloc->ceiling_bytes);
    r.metric("memory.alloc.stopped_by_policy", m.alloc->stopped_by_policy);
    r.metric("memory.alloc.first_touch_bytes_per_s", m.alloc->touch_bytes_per_s);
    r.metric("memory.peak_before_bytes", m.alloc->peak_before);
    r.metric("memory.peak_after_bytes", m.alloc->peak_after);
    for (const auto& s : m.alloc->steps) r.trace({{"kind", "alloc_step"}, {"bytes", s.bytes}, {"ok", s.ok}, {"touch_s", s.touch_seconds}});
  }
  double best = 0;
  for (const auto& b : m.bandwidth) {
    r.metric("memory.read_bandwidth.t" + std::to_string(b.threads), b.bytes_per_s, "bytes/s");
    best = std::max(best, b.bytes_per_s.median());
  }
  r.metric("memory.ram_bandwidth", best);
}

}  // namespace clusterlm::bench
