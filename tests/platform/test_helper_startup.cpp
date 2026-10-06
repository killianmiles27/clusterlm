#include <doctest/doctest.h>

#include "clusterlm/platform/helper_startup.hpp"

using namespace clusterlm;
using namespace clusterlm::platform;

namespace {
HelperStartupConfig config(bool tcb, HelperLaunchStrategy pref = HelperLaunchStrategy::kAuto) {
  HelperStartupConfig c;
  c.helper_exe = "C:\\Program Files\\ClusterLM\\clusterlm-node-helper.exe";
  c.service_has_tcb_privilege = tcb;
  c.preference = pref;
  return c;
}
}  // namespace

TEST_CASE("strategy: LocalService (no SeTcbPrivilege) resolves to the machine Run key") {
  using S = HelperLaunchStrategy;
  CHECK(choose_helper_strategy(S::kAuto, false) == S::kMachineRunKey);
  CHECK(choose_helper_strategy(S::kAuto, true) == S::kServiceLaunchedInSession);
  CHECK(choose_helper_strategy(S::kServiceLaunchedInSession, true) == S::kServiceLaunchedInSession);
  // Asking for service launch without the privilege cannot work: fall back rather than fail every logon.
  CHECK(choose_helper_strategy(S::kServiceLaunchedInSession, false) == S::kMachineRunKey);
  CHECK(choose_helper_strategy(S::kMachineRunKey, true) == S::kMachineRunKey);
}

TEST_CASE("run-key strategy ensures the value and launches nothing") {
  MockHelperHost host;
  host.session_list = {{1, true}};
  HelperStartupState state;
  REQUIRE(reconcile_helpers(host, config(false), state).is_ok());
  CHECK(host.run_keys.count("ClusterLMNodeHelper") == 1);
  CHECK(host.launched.empty());
  host.run_key_failure = make_error(ErrorCode::kPermissionDenied, "not elevated");
  CHECK_FALSE(reconcile_helpers(host, config(false), state).is_ok());  // reported; the service treats it as non-fatal
}

TEST_CASE("service-launch strategy starts one helper per active user session, once, and forgets ended sessions") {
  MockHelperHost host;
  host.run_keys.insert("ClusterLMNodeHelper");  // left over from an earlier configuration
  host.session_list = {{0, true}, {1, true}, {2, false}, {3, true}};  // 0 = services, 2 = disconnected
  HelperStartupState state;
  REQUIRE(reconcile_helpers(host, config(true), state).is_ok());
  CHECK(host.launched == std::vector<std::uint32_t>{1, 3});
  CHECK(host.run_keys.empty());  // no double start

  REQUIRE(reconcile_helpers(host, config(true), state).is_ok());
  CHECK(host.launched.size() == 2);  // already served

  host.session_list = {{0, true}, {3, true}, {4, true}};  // session 1 logged off, 4 logged on
  REQUIRE(reconcile_helpers(host, config(true), state).is_ok());
  CHECK(host.launched == std::vector<std::uint32_t>{1, 3, 4});
  host.session_list = {{0, true}, {1, true}};  // 1 logs on again: it is a new helper
  REQUIRE(reconcile_helpers(host, config(true), state).is_ok());
  CHECK(host.launched.back() == 1);
}

TEST_CASE("helper command line is quoted") {
  CHECK(helper_command_line("C:\\x\\h.exe", {"--session", "a b"}) == "\"C:\\x\\h.exe\" \"--session\" \"a b\"");
}
