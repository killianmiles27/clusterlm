#pragma once
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

#include "bench_profile.hpp"
#include "cli.hpp"
#include "result.hpp"

namespace clusterlm::bench {

// Every command returns a process exit code: 0 success, 1 failed checks/errors, 2 usage, 3 requires
// hardware/backends not present in this environment (the result still records what is pending).
int cmd_profile(const cli::Args& args);
int cmd_cpu(const cli::Args& args);
int cmd_memory(const cli::Args& args);
int cmd_gpu(const cli::Args& args, bool pcie_only);
int cmd_calibrate(const cli::Args& args);
int cmd_transport(const cli::Args& args);
int cmd_placement(const cli::Args& args);
int cmd_qualification(const cli::Args& args);
int cmd_cluster(const cli::Args& args);
int cmd_faults(const cli::Args& args);
int cmd_hardware_only(const std::string& command, const cli::Args& args);
// `baseline llama-rpc` (P0-A, HQ-P0A-01): real with -DCLUSTERLM_ENABLE_LLAMA=ON, otherwise reports (exit 3) that it needs it.
int cmd_baseline_llama(const cli::Args& args);

// Connects to every --peer over TLS (unless --no-tls), measures each link and, with two or more peers, the
// concurrent egress/ingress. Used by `calibrate`.
Status measure_remote_peers(const cli::Args& args, BenchmarkResult& result, std::vector<PeerLink>& links,
                            std::optional<ConcurrentMeasure>& concurrent);

// Writes the result to --out (or stdout) and returns the conventional exit code.
int emit(const cli::Args& args, BenchmarkResult& result, double duration_s, int code_if_ok = 0);

std::string source_dir();  // repository root (for fixtures/registry), overridable with --source-dir

// The command-line surface: which command words and --flags each command accepts. `main` rejects anything else
// (exit 2), and the test suite checks that every command line in bench/qualification/experiments.json parses
// against it, so the registry cannot drift from the implemented tool.
struct CommandSpec {
  std::string name;                    // e.g. "cpu"
  std::vector<std::string> subcommands;  // accepted first positional words (may be empty)
  std::vector<std::string> flags;      // without the leading "--"
};
const std::vector<CommandSpec>& command_specs();
const CommandSpec* find_command_spec(const std::string& name);
// `tokens` are the words after the command name. Empty result: every --flag is accepted by the command and a
// leading positional word (if any) is one of its subcommands. Otherwise a message naming the offending token.
std::string check_command_tokens(const CommandSpec& spec, const std::vector<std::string>& tokens);
// Splits a registry command line ("clusterlm-bench cpu --q 1,2,4 --out results/x.json") into tokens and validates
// it. Placeholder values such as <machine> are accepted as values. Returns an error text, or empty when valid.
std::string validate_registry_command(const std::string& command_line);

}  // namespace clusterlm::bench
