#pragma once
#include <nlohmann/json.hpp>

#include <string>

#include "cli.hpp"
#include "result.hpp"

namespace clusterlm::bench {

// Every command returns a process exit code: 0 success, 1 failed checks/errors, 2 usage, 3 requires
// hardware/backends not present in this environment (the result still records what is pending).
int cmd_profile(const cli::Args& args);
int cmd_transport(const cli::Args& args);
int cmd_placement(const cli::Args& args);
int cmd_qualification(const cli::Args& args);
int cmd_cluster(const cli::Args& args);
int cmd_faults(const cli::Args& args);
int cmd_hardware_only(const std::string& command, const cli::Args& args);

// Writes the result to --out (or stdout) and returns the conventional exit code.
int emit(const cli::Args& args, BenchmarkResult& result, double duration_s, int code_if_ok = 0);

std::string source_dir();  // repository root (for fixtures/registry), overridable with --source-dir

}  // namespace clusterlm::bench
