// The accepted command-line surface of clusterlm-bench (see commands.hpp).
#include <algorithm>
#include <set>
#include <sstream>

#include "commands.hpp"

namespace clusterlm::bench {

namespace {

std::vector<std::string> join(std::initializer_list<std::vector<std::string>> lists) {
  std::vector<std::string> out;
  for (const auto& l : lists) out.insert(out.end(), l.begin(), l.end());
  return out;
}

const std::vector<std::string> kCommon = {"out", "log", "experiment", "machine-id", "on-target", "role", "run-id", "quiet",
                                          "source-dir", "registry", "json"};
const std::vector<std::string> kCpu = {"provider", "representations", "q", "threads", "max-threads", "active", "hidden", "ff",
                                       "reps", "target-s", "bank-mib", "seed", "quick", "isa"};
const std::vector<std::string> kMemory = {"quick", "max-gib", "step-mib", "bandwidth-mib", "max-threads", "reps"};
const std::vector<std::string> kLink = {"sizes", "iterations", "warmup", "burst-mib", "duration", "peer", "peer-id", "identity",
                                        "trust", "no-tls", "tls"};

std::vector<CommandSpec> build() {
  std::vector<CommandSpec> s;
  s.push_back({"profile", {}, join({kCommon, {"bandwidth-mib", "repeats"}})});
  s.push_back({"cpu", {}, join({kCommon, kCpu, {"sustained", "minutes", "sample-s", "representation", "sustained-threads", "list-providers"}})});
  s.push_back({"memory", {}, join({kCommon, kMemory})});
  const std::vector<std::string> gpu = {"quick", "ring-bytes", "pinned-cap", "pinned-step", "reps"};
  s.push_back({"gpu", {}, join({kCommon, gpu})});
  s.push_back({"pcie", {}, join({kCommon, gpu})});
  s.push_back({"calibrate", {}, join({kCommon, kCpu, kMemory, kLink,
                                      {"out-dir", "profile-out", "network-out", "base", "no-merge", "skip-cpu", "skip-memory", "skip-gpu",
                                       "sustained-minutes", "sample-s", "require-gpu"}})});
  s.push_back({"transport", {}, join({kCommon, kLink, {"serve", "impair", "simulate-nodes", "node-to-node", "exit-on-stdin-eof", "network-out"}})});
  s.push_back({"cluster", {}, join({kCommon, {"plan", "tier", "model", "seed", "insecure", "impair", "relay", "compare-routing",
                                              "window-timeout-ms", "tokens", "max-new", "prompt-len", "context", "repeat", "q",
                                              "drafter", "prefill-chunk", "interleave", "phase", "corpus", "minutes", "work", "keep-work",
                                              // sweep, OS observers (NIC counters, process memory, NVML, cache census)
                                              "contexts", "nic", "no-resources", "sample-ms", "sample-s", "census"}})});
  s.push_back({"faults", {}, join({kCommon, {"plan", "model", "seed", "insecure", "window-timeout-ms", "max-new", "release-cycles",
                                             "work", "keep-work", "impair", "only", "no-resources", "no-census"}})});
  s.push_back({"nvml", {}, join({kCommon, {"minutes", "sample-s"}})});
  s.push_back({"storage-census", {}, join({kCommon, {"before", "after", "diff", "snapshot", "against", "root", "staging-root", "exclude"}})});
  s.push_back({"placement", {}, join({kCommon, {"profiles", "father", "node", "network", "context", "q"}})});
  s.push_back({"placement-inputs", {}, join({kCommon, {"model", "corpus", "q", "max-new", "routing-out", "work"}})});
  s.push_back({"qualification", {}, kCommon});
  s.push_back({"domain",
               {"cpu-experts", "sustained", "gpu-layers", "vram-ledger", "grouped-experts"},
               join({kCommon, {"backend", "model", "layers", "q", "threads", "isa", "minutes", "prefill-chunk", "plan", "nodes"}})});
  s.push_back({"baseline",
               {"llama-rpc", "strata", "fast"},
               join({kCommon, {"pin", "model", "nodes", "partial-domains", "repeat",
                                 // baseline llama-rpc: workload, local proof run and filesystem inspection
                                 "n-predict", "prompt-tokens", "threads", "ngl", "spawn-local", "rpc-server", "server-threads",
                                 "node-cache-dirs", "node-fs-report", "control-with-cache", "verify-local"}})});
  s.push_back({"numerics", {}, join({kCommon, {"backend", "model", "plan", "reference", "q"}})});
  return s;
}

}  // namespace

namespace {
std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}
}  // namespace

// ---- faults --only ---------------------------------------------------------------------------------------------
namespace {
const std::vector<std::string> kCrashPhases = {"transfer", "hashing", "mapping", "allocation", "ready",
                                               "prefill", "inference", "commit", "cleanup"};
}  // namespace

std::vector<std::string> fault_scenario_names() {
  std::vector<std::string> out;
  for (const auto& p : kCrashPhases) out.push_back("crash_" + p);
  for (const char* n : {"father_lost", "local_activity", "link_loss", "stall", "release_cycles"}) out.emplace_back(n);
  return out;
}

Result<std::vector<std::string>> select_fault_scenarios(const std::string& only) {
  const auto all = fault_scenario_names();
  if (only.empty()) return all;
  std::set<std::string> want;
  for (const auto& tok : split_csv(only)) {
    if (tok == "crash") {  // every crash_<phase>
      for (const auto& p : kCrashPhases) want.insert("crash_" + p);
    } else if (std::find(all.begin(), all.end(), tok) != all.end()) {
      want.insert(tok);
    } else if (std::find(kCrashPhases.begin(), kCrashPhases.end(), tok) != kCrashPhases.end()) {
      want.insert("crash_" + tok);  // a bare lifecycle phase selects its crash scenario
    } else {
      std::string valid;
      for (const auto& n : all) valid += (valid.empty() ? "" : ", ") + n;
      return make_error(ErrorCode::kInvalidArgument,
                        "unknown fault scenario or phase '" + tok + "' (scenarios: " + valid + "; groups: crash; phases: " +
                            [&] {
                              std::string ph;
                              for (const auto& p : kCrashPhases) ph += (ph.empty() ? "" : ", ") + p;
                              return ph;
                            }() + ")");
    }
  }
  if (want.empty()) return make_error(ErrorCode::kInvalidArgument, "--only names no scenario");
  std::vector<std::string> out;
  for (const auto& n : all)
    if (want.count(n)) out.push_back(n);  // canonical order
  return out;
}


const std::vector<CommandSpec>& command_specs() {
  static const std::vector<CommandSpec> specs = build();
  return specs;
}

const CommandSpec* find_command_spec(const std::string& name) {
  for (const auto& s : command_specs())
    if (s.name == name) return &s;
  return nullptr;
}

std::string check_command_tokens(const CommandSpec& spec, const std::vector<std::string>& tokens) {
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const std::string& t = tokens[i];
    if (t.rfind("--", 0) == 0) {
      std::string flag = t.substr(2);
      if (const auto eq = flag.find('='); eq != std::string::npos) flag.resize(eq);
      if (std::find(spec.flags.begin(), spec.flags.end(), flag) == spec.flags.end())
        return "'" + spec.name + "' does not accept --" + flag;
    } else if (i == 0) {
      if (std::find(spec.subcommands.begin(), spec.subcommands.end(), t) == spec.subcommands.end())
        return "'" + spec.name + "' has no subcommand '" + t + "'";
    }
  }
  return {};
}

std::string validate_registry_command(const std::string& command_line) {
  // Procedures that only a person at the machine can perform are written "manual: docs/<file>#<anchor> (...)".
  if (command_line.rfind("manual:", 0) == 0) {
    return command_line.find("docs/") != std::string::npos ? std::string{}
                                                            : "manual procedure must reference its docs/ section";
  }
  // Other shipped ClusterLM executables: their own tests own their flag surface.
  static const std::vector<std::string> kOtherTools = {
      "clusterlm-expert-domain-bench", "clusterlm-model-inspect", "clusterlm-node-service", "clusterlm-father",
      "clusterlm-node", "clusterlm-father-agent", "clusterlm-node-helper", "clusterlm-strata", "clusterlm-llama-rpc"};
  // A procedure may build the tools it needs first (pinned upstream checkout + CMake configure/build).
  auto is_setup = [](const std::vector<std::string>& tok) {
    if (tok[0] == "cmake") return true;
    return tok[0] == "python3" && tok.size() >= 2 && tok[1].rfind("scripts/", 0) == 0;
  };
  // Segments are separated by "&&", ";" or a trailing "&" (background); "for ... ; do ... ; done" loops are
  // accepted with each body command validated on its own.
  std::vector<std::string> segments;
  {
    std::string cur;
    for (std::size_t i = 0; i < command_line.size(); ++i) {
      const char c = command_line[i];
      if (c == '&' && i + 1 < command_line.size() && command_line[i + 1] == '&') {
        segments.push_back(cur), cur.clear(), ++i;
      } else if (c == ';' || c == '&') {
        segments.push_back(cur), cur.clear();
      } else {
        cur.push_back(c);
      }
    }
    segments.push_back(cur);
  }
  bool any = false;
  for (const auto& segment : segments) {
    std::istringstream in(segment);
    std::vector<std::string> tok;
    for (std::string t; in >> t;) tok.push_back(t);
    if (!tok.empty() && tok[0] == "do") tok.erase(tok.begin());
    if (tok.empty() || tok[0] == "done") continue;
    if (tok[0] == "for") continue;  // loop header: "for <var> in <values...>"
    while (!tok.empty() && tok[0].find('=') != std::string::npos && tok[0].find('/') == std::string::npos &&
           tok[0][0] >= 'A' && tok[0][0] <= 'Z') {
      tok.erase(tok.begin());  // NAME=value environment prefix
    }
    if (tok.empty()) return "environment assignment without a command";
    if (tok[0].rfind("build/bin/", 0) == 0) tok[0] = tok[0].substr(10);
    any = true;
    if (is_setup(tok)) continue;
    if (std::find(kOtherTools.begin(), kOtherTools.end(), tok[0]) != kOtherTools.end()) continue;
    if (tok.size() < 2 || tok[0] != "clusterlm-bench") return "must start with 'clusterlm-bench <command>'";
    const CommandSpec* spec = find_command_spec(tok[1]);
    if (!spec) return "unknown command '" + tok[1] + "'";
    auto err = check_command_tokens(*spec, std::vector<std::string>(tok.begin() + 2, tok.end()));
    if (!err.empty()) return err;
  }
  if (!any) return "empty command segment";
  return {};
}

}  // namespace clusterlm::bench
