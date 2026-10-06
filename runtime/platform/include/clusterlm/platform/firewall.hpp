#pragma once
// Windows Defender Firewall rules for ClusterLM, as declarative data plus an applier interface.
//
// The rule SPECS are plain data (firewall_rule_specs) so the installer reuses exactly what the service applies at
// runtime. The applier (FirewallRules) has a real INetFwPolicy2 implementation on Windows and a recording mock.
//
// Policy: inbound TCP only, scoped to the ClusterLM program path, Private + Domain profiles (Public stays closed
// unless the user opts in), remote addresses limited to the local subnet or, when known, the paired peers.
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

enum FirewallProfile : std::uint32_t {
  kProfileDomain = 1,   // NET_FW_PROFILE2_DOMAIN
  kProfilePrivate = 2,  // NET_FW_PROFILE2_PRIVATE
  kProfilePublic = 4,   // NET_FW_PROFILE2_PUBLIC
};

struct FirewallRuleSpec {
  std::string name;         // unique, stable: used for ensure/remove
  std::string description;
  std::string grouping = "ClusterLM";
  std::string program_path;  // absolute path of the executable the rule applies to (mandatory)
  std::uint16_t local_port = 0;
  std::uint32_t profiles = kProfilePrivate | kProfileDomain;
  // Comma-separated NetFW syntax: "LocalSubnet", or IPs/CIDR/ranges. Never empty: an empty restriction would
  // mean "any", which the validator refuses.
  std::vector<std::string> remote_addresses = {"LocalSubnet"};
  bool enabled = true;
  bool operator==(const FirewallRuleSpec&) const = default;
};

struct FirewallSpecOptions {
  std::string node_program_path;     // e.g. C:\Program Files\ClusterLM\clusterlm-node.exe
  std::uint16_t node_data_port = 0;  // 0 = no Node rule
  std::string father_program_path;
  std::uint16_t father_listen_port = 0;  // 0 = no Father rule (Father only dials out unless peers return to it)
  // Restrict to specific peers (IPs/CIDR). Empty = LocalSubnet.
  std::vector<std::string> remote_addresses;
  bool allow_public_profile = false;     // explicit opt-in; the default keeps public networks closed
};

// The rules the installer and the services want present. Deterministic; names are stable identifiers.
std::vector<FirewallRuleSpec> firewall_rule_specs(const FirewallSpecOptions& options);
constexpr const char* kNodeRuleName = "ClusterLM Node data (TCP-In)";
constexpr const char* kFatherRuleName = "ClusterLM Father (TCP-In)";

// Rejects specs that would be broader than intended: relative/empty program path, port 0, no profile, an
// empty/"Any"/"*" remote address, or the public profile without `allow_public`.
Status validate_firewall_rule(const FirewallRuleSpec& spec, bool allow_public = false);
// Machine-readable form for the installer (stable key order).
std::string firewall_specs_to_json(const std::vector<FirewallRuleSpec>& specs);

class FirewallRules {
 public:
  virtual ~FirewallRules() = default;
  // Creates the rule or updates an existing rule of the same name to match `spec` exactly (idempotent).
  virtual Status ensure(const FirewallRuleSpec& spec) = 0;
  // Removes the rule; a missing rule is success.
  virtual Status remove(const std::string& name) = 0;
  virtual Result<std::optional<FirewallRuleSpec>> query(const std::string& name) = 0;
};

// Validates every spec first, then ensures them all; stops at the first error.
Status apply_firewall_specs(FirewallRules& rules, const std::vector<FirewallRuleSpec>& specs, bool allow_public = false);

class MockFirewallRules final : public FirewallRules {
 public:
  std::map<std::string, FirewallRuleSpec> rules;
  std::optional<Status> fail_with;
  Status ensure(const FirewallRuleSpec& spec) override {
    if (fail_with) return *fail_with;
    rules[spec.name] = spec;
    return Status::ok();
  }
  Status remove(const std::string& name) override {
    if (fail_with) return *fail_with;
    rules.erase(name);
    return Status::ok();
  }
  Result<std::optional<FirewallRuleSpec>> query(const std::string& name) override {
    if (fail_with) return *fail_with;
    auto it = rules.find(name);
    if (it == rules.end()) return std::optional<FirewallRuleSpec>{};
    return std::optional<FirewallRuleSpec>{it->second};
  }
};

#ifdef _WIN32
// COM (INetFwPolicy2 / INetFwRule). CoInitializeEx must have been called on the calling thread; needs elevation
// (the service account or the installer) to modify rules.
std::unique_ptr<FirewallRules> make_windows_firewall_rules();
#endif

}  // namespace clusterlm::platform
