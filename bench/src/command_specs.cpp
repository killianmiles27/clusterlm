// The accepted command-line surface of clusterlm-bench (see commands.hpp).
#include <algorithm>
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
                                              "drafter", "prefill-chunk", "interleave", "phase", "corpus", "minutes", "work", "keep-work"}})});
  s.push_back({"faults", {}, join({kCommon, {"plan", "model", "seed", "insecure", "window-timeout-ms", "max-new", "release-cycles",
                                             "work", "keep-work", "impair"}})});
  s.push_back({"placement", {}, join({kCommon, {"profiles", "father", "node", "network", "context", "q"}})});
  s.push_back({"placement-inputs", {}, join({kCommon, {"model", "corpus", "q", "max-new", "routing-out", "work"}})});
  s.push_back({"qualification", {}, kCommon});
  s.push_back({"domain",
               {"cpu-experts", "sustained", "gpu-layers", "vram-ledger", "grouped-experts"},
               join({kCommon, {"backend", "model", "layers", "q", "threads", "isa", "minutes", "prefill-chunk", "plan", "nodes"}})});
  s.push_back({"baseline",
               {"llama-rpc", "strata", "fast"},
               join({kCommon, {"pin", "model", "nodes", "partial-domains", "repeat"}})});
  s.push_back({"numerics", {}, join({kCommon, {"backend", "model", "plan", "reference", "q"}})});
  return s;
}

}  // namespace

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
  static const std::vector<std::string> kOtherTools = {"clusterlm-expert-domain-bench", "clusterlm-model-inspect",
                                                       "clusterlm-node-service", "clusterlm-father", "clusterlm-node",
                                                       "clusterlm-father-agent", "clusterlm-node-helper"};
  std::size_t start = 0;
  while (start <= command_line.size()) {
    const auto amp = command_line.find("&&", start);
    const std::string segment = command_line.substr(start, amp == std::string::npos ? std::string::npos : amp - start);
    std::istringstream in(segment);
    std::vector<std::string> tok;
    for (std::string t; in >> t;) tok.push_back(t);
    if (tok.empty()) return "empty command segment";
    if (std::find(kOtherTools.begin(), kOtherTools.end(), tok[0]) == kOtherTools.end()) {
      if (tok.size() < 2 || tok[0] != "clusterlm-bench") return "must start with 'clusterlm-bench <command>'";
      const CommandSpec* spec = find_command_spec(tok[1]);
      if (!spec) return "unknown command '" + tok[1] + "'";
      auto err = check_command_tokens(*spec, std::vector<std::string>(tok.begin() + 2, tok.end()));
      if (!err.empty()) return err;
    }
    if (amp == std::string::npos) break;
    start = amp + 2;
  }
  return {};
}

}  // namespace clusterlm::bench
