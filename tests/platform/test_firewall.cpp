#include <doctest/doctest.h>

#include "clusterlm/platform/firewall.hpp"

using namespace clusterlm;
using namespace clusterlm::platform;

namespace {
FirewallSpecOptions options() {
  FirewallSpecOptions o;
  o.node_program_path = "C:\\Program Files\\ClusterLM\\clusterlm-node.exe";
  o.node_data_port = 47600;
  return o;
}
}  // namespace

TEST_CASE("default firewall specs: one inbound TCP rule scoped to the program, private+domain, local subnet") {
  const auto specs = firewall_rule_specs(options());
  REQUIRE(specs.size() == 1);
  const auto& r = specs[0];
  CHECK(r.name == kNodeRuleName);
  CHECK(r.program_path == "C:\\Program Files\\ClusterLM\\clusterlm-node.exe");
  CHECK(r.local_port == 47600);
  CHECK((r.profiles & kProfilePrivate) != 0);
  CHECK((r.profiles & kProfileDomain) != 0);
  CHECK((r.profiles & kProfilePublic) == 0);
  REQUIRE(r.remote_addresses.size() == 1);
  CHECK(r.remote_addresses[0] == "LocalSubnet");
  CHECK(validate_firewall_rule(r).is_ok());
}

TEST_CASE("firewall specs honour peer restriction, optional Father rule and the explicit public opt-in") {
  auto o = options();
  o.remote_addresses = {"192.168.1.10", "192.168.1.20"};
  o.father_program_path = "C:\\Program Files\\ClusterLM\\clusterlm-father.exe";
  o.father_listen_port = 47601;
  auto specs = firewall_rule_specs(o);
  REQUIRE(specs.size() == 2);
  CHECK(specs[0].remote_addresses == o.remote_addresses);
  CHECK(specs[1].name == kFatherRuleName);
  CHECK(specs[1].local_port == 47601);

  o.allow_public_profile = true;
  specs = firewall_rule_specs(o);
  CHECK((specs[0].profiles & kProfilePublic) != 0);
  CHECK_FALSE(validate_firewall_rule(specs[0]).is_ok());       // public needs the validator's opt-in too
  CHECK(validate_firewall_rule(specs[0], /*allow_public=*/true).is_ok());

  o.node_data_port = 0;
  o.father_listen_port = 0;
  CHECK(firewall_rule_specs(o).empty());
}

TEST_CASE("validation rejects rules broader than intended") {
  const FirewallRuleSpec good = firewall_rule_specs(options())[0];
  auto with = [&](auto mutate) {
    FirewallRuleSpec s = good;
    mutate(s);
    return validate_firewall_rule(s);
  };
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.program_path = "clusterlm-node.exe"; }).is_ok());  // not absolute
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.program_path.clear(); }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.local_port = 0; }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.profiles = 0; }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.remote_addresses.clear(); }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.remote_addresses = {"Any"}; }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.remote_addresses = {"*"}; }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.remote_addresses = {"0.0.0.0/0"}; }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.remote_addresses = {"10.0.0.1,Any"}; }).is_ok());
  CHECK_FALSE(with([](FirewallRuleSpec& s) { s.name.clear(); }).is_ok());
}

TEST_CASE("apply_firewall_specs ensures rules idempotently, validates first and removes by name") {
  MockFirewallRules fw;
  auto specs = firewall_rule_specs(options());
  CHECK(apply_firewall_specs(fw, specs).is_ok());
  CHECK(apply_firewall_specs(fw, specs).is_ok());
  CHECK(fw.rules.size() == 1);
  auto q = fw.query(kNodeRuleName);
  REQUIRE(q.is_ok());
  REQUIRE(q->has_value());
  CHECK(**q == specs[0]);

  specs[0].local_port = 47700;  // the port changed: ensure updates in place
  CHECK(apply_firewall_specs(fw, specs).is_ok());
  CHECK(fw.rules.size() == 1);
  CHECK(fw.rules.at(kNodeRuleName).local_port == 47700);

  auto bad = specs;
  bad[0].remote_addresses = {"Any"};
  MockFirewallRules fresh;
  CHECK_FALSE(apply_firewall_specs(fresh, bad).is_ok());
  CHECK(fresh.rules.empty());  // nothing was applied

  CHECK(fw.remove(kNodeRuleName).is_ok());
  CHECK(fw.remove(kNodeRuleName).is_ok());  // removing a missing rule is success
  auto gone = fw.query(kNodeRuleName);
  REQUIRE(gone.is_ok());
  CHECK_FALSE(gone->has_value());
}

TEST_CASE("firewall specs export as stable JSON for the installer") {
  const auto json = firewall_specs_to_json(firewall_rule_specs(options()));
  CHECK(json.find("\"name\": \"ClusterLM Node data (TCP-In)\"") != std::string::npos);
  CHECK(json.find("\"direction\": \"in\"") != std::string::npos);
  CHECK(json.find("\"protocol\": \"tcp\"") != std::string::npos);
  CHECK(json.find("\"local_port\": 47600") != std::string::npos);
  CHECK(json.find("\"profiles\": [\"domain\", \"private\"]") != std::string::npos);
  CHECK(json.find("\"remote_addresses\": [\"LocalSubnet\"]") != std::string::npos);
  CHECK(json.find("C:\\\\Program Files\\\\ClusterLM\\\\clusterlm-node.exe") != std::string::npos);  // escaped
  CHECK(firewall_specs_to_json({}) == "[]\n");
}
