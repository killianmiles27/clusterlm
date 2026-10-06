#include "clusterlm/platform/firewall.hpp"

#include <algorithm>
#include <cctype>

namespace clusterlm::platform {

namespace {
bool absolute_path_like(const std::string& p) {
  if (p.size() >= 3 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':' && (p[2] == '\\' || p[2] == '/'))
    return true;                                       // C:\...
  if (p.size() >= 3 && p[0] == '\\' && p[1] == '\\') return true;  // UNC
  return !p.empty() && p[0] == '/';                    // POSIX absolute (dev/tests)
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      default: out.push_back(c);
    }
  }
  return out;
}
}  // namespace

std::vector<FirewallRuleSpec> firewall_rule_specs(const FirewallSpecOptions& o) {
  std::vector<FirewallRuleSpec> out;
  const std::uint32_t profiles =
      kProfilePrivate | kProfileDomain | (o.allow_public_profile ? static_cast<std::uint32_t>(kProfilePublic) : 0u);
  const std::vector<std::string> remote = o.remote_addresses.empty() ? std::vector<std::string>{"LocalSubnet"} : o.remote_addresses;
  if (o.node_data_port != 0) {
    FirewallRuleSpec r;
    r.name = kNodeRuleName;
    r.description = "Allows the paired ClusterLM Father and peer Nodes to reach the ClusterLM Node worker.";
    r.program_path = o.node_program_path;
    r.local_port = o.node_data_port;
    r.profiles = profiles;
    r.remote_addresses = remote;
    out.push_back(std::move(r));
  }
  if (o.father_listen_port != 0) {
    FirewallRuleSpec r;
    r.name = kFatherRuleName;
    r.description = "Allows ClusterLM Nodes to return stage output to the ClusterLM Father.";
    r.program_path = o.father_program_path;
    r.local_port = o.father_listen_port;
    r.profiles = profiles;
    r.remote_addresses = remote;
    out.push_back(std::move(r));
  }
  return out;
}

Status validate_firewall_rule(const FirewallRuleSpec& s, bool allow_public) {
  auto bad = [](const std::string& why) { return make_error(ErrorCode::kInvalidArgument, "firewall rule: " + why); };
  if (s.name.empty() || s.name.size() > 255) return bad("name must be 1..255 characters");
  if (!absolute_path_like(s.program_path)) return bad("program_path must be an absolute path");
  if (s.local_port == 0) return bad("local_port must be nonzero");
  if (s.profiles == 0 || (s.profiles & ~7u) != 0) return bad("profiles must be a non-empty subset of domain|private|public");
  if ((s.profiles & kProfilePublic) != 0 && !allow_public) return bad("public profile requires explicit opt-in");
  if (s.remote_addresses.empty()) return bad("remote_addresses must not be empty (empty means any)");
  for (const auto& a : s.remote_addresses) {
    const std::string l = lower(a);
    if (l.empty() || l == "any" || l == "*" || l == "0.0.0.0/0" || l == "::/0" || a.find(',') != std::string::npos ||
        a.find(' ') != std::string::npos)
      return bad("remote address '" + a + "' is empty, unrestricted or malformed");
  }
  return Status::ok();
}

std::string firewall_specs_to_json(const std::vector<FirewallRuleSpec>& specs) {
  std::string j = "[";
  for (std::size_t i = 0; i < specs.size(); ++i) {
    const auto& s = specs[i];
    if (i != 0) j += ",";
    j += "\n  {\"name\": \"" + json_escape(s.name) + "\", \"description\": \"" + json_escape(s.description) +
         "\", \"grouping\": \"" + json_escape(s.grouping) + "\", \"direction\": \"in\", \"protocol\": \"tcp\"" +
         ", \"program_path\": \"" + json_escape(s.program_path) + "\", \"local_port\": " + std::to_string(s.local_port) +
         ", \"profiles\": [";
    bool first = true;
    auto add = [&](bool on, const char* n) {
      if (!on) return;
      j += std::string(first ? "" : ", ") + "\"" + n + "\"";
      first = false;
    };
    add((s.profiles & kProfileDomain) != 0, "domain");
    add((s.profiles & kProfilePrivate) != 0, "private");
    add((s.profiles & kProfilePublic) != 0, "public");
    j += "], \"remote_addresses\": [";
    for (std::size_t k = 0; k < s.remote_addresses.size(); ++k)
      j += std::string(k == 0 ? "" : ", ") + "\"" + json_escape(s.remote_addresses[k]) + "\"";
    j += std::string("], \"enabled\": ") + (s.enabled ? "true" : "false") + "}";
  }
  j += specs.empty() ? "]\n" : "\n]\n";
  return j;
}

Status apply_firewall_specs(FirewallRules& rules, const std::vector<FirewallRuleSpec>& specs, bool allow_public) {
  for (const auto& s : specs) CLM_RETURN_IF_ERROR(validate_firewall_rule(s, allow_public));
  for (const auto& s : specs) CLM_RETURN_IF_ERROR(rules.ensure(s));
  return Status::ok();
}

}  // namespace clusterlm::platform
