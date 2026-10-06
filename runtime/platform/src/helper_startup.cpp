#include "clusterlm/platform/helper_startup.hpp"

#include <iterator>
#include <optional>

namespace clusterlm::platform {

HelperLaunchStrategy choose_helper_strategy(HelperLaunchStrategy preference, bool service_has_tcb_privilege) {
  switch (preference) {
    case HelperLaunchStrategy::kServiceLaunchedInSession:
      return service_has_tcb_privilege ? HelperLaunchStrategy::kServiceLaunchedInSession
                                       : HelperLaunchStrategy::kMachineRunKey;
    case HelperLaunchStrategy::kMachineRunKey: return HelperLaunchStrategy::kMachineRunKey;
    case HelperLaunchStrategy::kAuto: break;
  }
  return service_has_tcb_privilege ? HelperLaunchStrategy::kServiceLaunchedInSession
                                   : HelperLaunchStrategy::kMachineRunKey;
}

std::string helper_command_line(const std::filesystem::path& exe, const std::vector<std::string>& args) {
  std::string out = "\"" + exe.string() + "\"";
  for (const auto& a : args) out += " \"" + a + "\"";
  return out;
}

Status reconcile_helpers(HelperHost& host, const HelperStartupConfig& cfg, HelperStartupState& state) {
  const auto strategy = choose_helper_strategy(cfg.preference, cfg.service_has_tcb_privilege);
  if (strategy == HelperLaunchStrategy::kMachineRunKey) {
    state.launched.clear();
    return host.ensure_run_key(cfg.run_key_value_name, helper_command_line(cfg.helper_exe, cfg.helper_args));
  }
  (void)host.remove_run_key(cfg.run_key_value_name);
  CLM_ASSIGN_OR_RETURN(auto sessions, host.sessions());
  std::set<std::uint32_t> present;
  for (const auto& s : sessions)
    if (s.id != 0 && s.active) present.insert(s.id);
  for (auto it = state.launched.begin(); it != state.launched.end();)
    it = present.count(*it) != 0 ? std::next(it) : state.launched.erase(it);
  std::optional<Status> first_error;
  for (std::uint32_t id : present) {
    if (state.launched.count(id) != 0) continue;
    Status st = host.launch_in_session(id, cfg.helper_exe, cfg.helper_args);
    if (st.is_ok()) {
      state.launched.insert(id);
    } else if (!first_error) {
      first_error = st;  // keep going: one broken session must not block the others
    }
  }
  return first_error ? *first_error : Status::ok();
}

}  // namespace clusterlm::platform
