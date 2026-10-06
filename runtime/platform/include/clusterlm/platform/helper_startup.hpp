#pragma once
// How the per-user session helper gets started (ADR 0132).
//
// Two mechanisms exist:
//   kServiceLaunchedInSession  the service calls WTSQueryUserToken + CreateProcessAsUserW into each interactive
//                              session. Needs SeTcbPrivilege, which only LocalSystem holds.
//   kMachineRunKey             an HKLM\...\Run value written by the installer starts the helper at every logon.
// The product service runs as LocalService (no SeTcbPrivilege), so the shipped default resolves to the Run key.
// A missing helper is never unsafe: HelperActivityMonitor fails closed and the Node stays Busy.
//
// The decision and reconciliation logic here is OS-independent and tested with MockHelperHost; WindowsHelperHost
// (windows/helper_launch_win.cpp) performs the real calls.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

enum class HelperLaunchStrategy : std::uint8_t { kAuto, kServiceLaunchedInSession, kMachineRunKey };

// Resolves kAuto and downgrades kServiceLaunchedInSession when the service lacks SeTcbPrivilege.
HelperLaunchStrategy choose_helper_strategy(HelperLaunchStrategy preference, bool service_has_tcb_privilege);

struct UserSession {
  std::uint32_t id = 0;
  bool active = false;  // an interactive user is logged on and connected (WTSActive)
};

struct HelperStartupConfig {
  std::filesystem::path helper_exe;
  std::vector<std::string> helper_args;
  HelperLaunchStrategy preference = HelperLaunchStrategy::kAuto;
  bool service_has_tcb_privilege = false;
  std::string run_key_value_name = "ClusterLMNodeHelper";
};

// Sessions the service has already started a helper in; entries are dropped when the session disappears.
struct HelperStartupState {
  std::set<std::uint32_t> launched;
};

class HelperHost {
 public:
  virtual ~HelperHost() = default;
  virtual Result<std::vector<UserSession>> sessions() = 0;
  virtual Status launch_in_session(std::uint32_t session_id, const std::filesystem::path& exe,
                                   const std::vector<std::string>& args) = 0;
  // HKLM Run value (machine-wide; needs elevation, so normally done by the installer).
  virtual Status ensure_run_key(const std::string& value_name, const std::string& command_line) = 0;
  virtual Status remove_run_key(const std::string& value_name) = 0;
};

// Brings helpers in line with the chosen strategy; call at service start and on every session event.
// Strategy run key: ensures the value (a failure is returned but is non-fatal for the service).
// Strategy service launch: launches a helper into each active, not yet served, non-zero session and removes the
// Run value so a helper is not started twice (a named mutex in the helper also guarantees one per session).
Status reconcile_helpers(HelperHost& host, const HelperStartupConfig& config, HelperStartupState& state);

// The command line stored in the Run value: "<exe>" arg1 arg2 (quoted).
std::string helper_command_line(const std::filesystem::path& exe, const std::vector<std::string>& args);

class MockHelperHost final : public HelperHost {
 public:
  std::vector<UserSession> session_list;
  std::vector<std::uint32_t> launched;
  std::set<std::string> run_keys;
  std::optional<Status> run_key_failure;
  Result<std::vector<UserSession>> sessions() override { return session_list; }
  Status launch_in_session(std::uint32_t id, const std::filesystem::path&, const std::vector<std::string>&) override {
    launched.push_back(id);
    return Status::ok();
  }
  Status ensure_run_key(const std::string& n, const std::string&) override {
    if (run_key_failure) return *run_key_failure;
    run_keys.insert(n);
    return Status::ok();
  }
  Status remove_run_key(const std::string& n) override {
    run_keys.erase(n);
    return Status::ok();
  }
};

#ifdef _WIN32
std::unique_ptr<HelperHost> make_windows_helper_host();
// True if this process holds SeTcbPrivilege (enabled or not).
bool current_process_has_tcb_privilege();
#endif

}  // namespace clusterlm::platform
