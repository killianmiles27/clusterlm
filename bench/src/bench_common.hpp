#pragma once
// Shared plumbing for the measurement commands: run context (machine id, role, run id), option parsing and the
// memory measurement driver.
#include <functional>
#include <string>
#include <vector>

#include "bench_profile.hpp"
#include "cli.hpp"
#include "cpu_bench.hpp"
#include "host_probe.hpp"
#include "memory_probe.hpp"
#include "result.hpp"

namespace clusterlm::bench {

struct RunContext {
  std::string run_id;
  std::string machine_id;  // --machine-id, else "dev-<hostname>" (never a target name unless the operator says so)
  std::string role = "node";  // profile role: father | node
  bool on_target = false;     // --on-target: the operator asserts this machine is the named target
};

RunContext make_run_context(const cli::Args& args);
// Sets host_role / machine_id on the result: "development-host" unless --on-target was given.
void apply_run_context(BenchmarkResult& r, const RunContext& ctx);
// --experiment, else `hq_id` when --on-target, else `dev_id`.
std::string experiment_id(const cli::Args& args, const RunContext& ctx, const char* hq_id, const char* dev_id);

std::vector<std::uint32_t> parse_u32_list(const std::string& csv);
std::vector<std::string> parse_string_list(const std::string& csv);
// Parses "512M", "1G", "64K" or plain bytes.
Result<std::uint64_t> parse_size(const std::string& text);

CpuBenchOptions cpu_options_from_args(const cli::Args& args);

struct MemoryBenchOptions {
  std::uint64_t max_bytes = 4ull << 30;    // --max-gib
  std::uint64_t step_bytes = 256ull << 20;
  std::uint64_t bandwidth_bytes = 512ull << 20;
  unsigned max_threads = 0;                // 0 = logical CPUs
  unsigned reps = 5;
  bool quick = false;
};
MemoryBenchOptions memory_options_from_args(const cli::Args& args);
Result<MemoryMeasurement> measure_memory(const MemoryBenchOptions& options);
void emit_memory_metrics(BenchmarkResult& r, const MemoryMeasurement& m);

}  // namespace clusterlm::bench
