// Windows implementation of HelperHost: enumerate sessions, start a process in a user session, manage the
// machine-wide Run value (ADR 0132). Compiled only on Windows.
#ifdef _WIN32

#include <windows.h>
#include <userenv.h>
#include <wtsapi32.h>

#include <vector>

#include "../internal_errors.hpp"
#include "clusterlm/platform/helper_startup.hpp"
#include "win_strings.hpp"

namespace clusterlm::platform {

namespace {

using detail::to_wide;
using detail::win_status;

constexpr wchar_t kRunKey[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";

class WindowsHelperHost final : public HelperHost {
 public:
  Result<std::vector<UserSession>> sessions() override {
    WTS_SESSION_INFOW* info = nullptr;
    DWORD count = 0;
    if (!::WTSEnumerateSessionsW(WTS_CURRENT_SERVER_HANDLE, 0, 1, &info, &count)) return win_status("WTSEnumerateSessionsW");
    std::vector<UserSession> out;
    for (DWORD i = 0; i < count; ++i) out.push_back({info[i].SessionId, info[i].State == WTSActive});
    ::WTSFreeMemory(info);
    return out;
  }

  // Needs SeTcbPrivilege (LocalSystem). The helper runs with the user's own token on the interactive desktop.
  Status launch_in_session(std::uint32_t session_id, const std::filesystem::path& exe,
                           const std::vector<std::string>& args) override {
    HANDLE raw = nullptr;
    if (!::WTSQueryUserToken(session_id, &raw)) return win_status("WTSQueryUserToken");
    HANDLE primary = nullptr;
    const BOOL dup = ::DuplicateTokenEx(raw, MAXIMUM_ALLOWED, nullptr, SecurityImpersonation, TokenPrimary, &primary);
    ::CloseHandle(raw);
    if (!dup) return win_status("DuplicateTokenEx");
    LPVOID env = nullptr;
    if (!::CreateEnvironmentBlock(&env, primary, FALSE)) {
      const Status s = win_status("CreateEnvironmentBlock");
      ::CloseHandle(primary);
      return s;
    }
    std::wstring cmd = to_wide(helper_command_line(exe, args));
    std::wstring desktop = L"winsta0\\default";
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.lpDesktop = desktop.data();
    PROCESS_INFORMATION pi{};
    const BOOL ok = ::CreateProcessAsUserW(primary, exe.c_str(), cmd.data(), nullptr, nullptr, FALSE,
                                           CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, env, nullptr, &si, &pi);
    const Status result = ok ? Status::ok() : win_status("CreateProcessAsUserW");
    if (ok) {
      ::CloseHandle(pi.hThread);
      ::CloseHandle(pi.hProcess);
    }
    ::DestroyEnvironmentBlock(env);
    ::CloseHandle(primary);
    return result;
  }

  Status ensure_run_key(const std::string& value_name, const std::string& command_line) override {
    const std::wstring data = to_wide(command_line);
    const LSTATUS rc = ::RegSetKeyValueW(HKEY_LOCAL_MACHINE, kRunKey, to_wide(value_name).c_str(), REG_SZ, data.c_str(),
                                         static_cast<DWORD>((data.size() + 1) * sizeof(wchar_t)));
    if (rc != ERROR_SUCCESS) return win_status("RegSetKeyValueW(Run)", static_cast<DWORD>(rc));
    return Status::ok();
  }

  Status remove_run_key(const std::string& value_name) override {
    const LSTATUS rc = ::RegDeleteKeyValueW(HKEY_LOCAL_MACHINE, kRunKey, to_wide(value_name).c_str());
    if (rc != ERROR_SUCCESS && rc != ERROR_FILE_NOT_FOUND) return win_status("RegDeleteKeyValueW(Run)", static_cast<DWORD>(rc));
    return Status::ok();
  }
};

}  // namespace

std::unique_ptr<HelperHost> make_windows_helper_host() { return std::make_unique<WindowsHelperHost>(); }

bool current_process_has_tcb_privilege() {
  HANDLE tok = nullptr;
  if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &tok)) return false;
  LUID tcb{};
  bool found = false;
  DWORD need = 0;
  ::GetTokenInformation(tok, TokenPrivileges, nullptr, 0, &need);
  if (need != 0 && ::LookupPrivilegeValueW(nullptr, L"SeTcbPrivilege", &tcb)) {
    std::vector<std::uint8_t> buf(need);
    if (::GetTokenInformation(tok, TokenPrivileges, buf.data(), need, &need)) {
      const auto* priv = reinterpret_cast<const TOKEN_PRIVILEGES*>(buf.data());
      for (DWORD i = 0; i < priv->PrivilegeCount; ++i)
        if (priv->Privileges[i].Luid.LowPart == tcb.LowPart && priv->Privileges[i].Luid.HighPart == tcb.HighPart) found = true;
    }
  }
  ::CloseHandle(tok);
  return found;
}

}  // namespace clusterlm::platform

#endif  // _WIN32
