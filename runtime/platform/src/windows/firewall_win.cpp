// Windows Defender Firewall rules through the INetFwPolicy2 / INetFwRule COM API. Compiled only on Windows.
#ifdef _WIN32

#include <netfw.h>
#include <windows.h>
#include <wrl/client.h>

#include <string>

#include "../internal_errors.hpp"
#include "clusterlm/platform/firewall.hpp"
#include "win_strings.hpp"

namespace clusterlm::platform {

namespace {

using Microsoft::WRL::ComPtr;
using detail::to_utf8;
using detail::to_wide;

Status hr_status(const char* what, HRESULT hr) {
  ErrorCode code = ErrorCode::kInternal;
  if (hr == E_ACCESSDENIED) code = ErrorCode::kPermissionDenied;
  return make_error(code, std::string(what) + " failed: HRESULT 0x" + [&] {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%08lx", static_cast<unsigned long>(hr));
    return std::string(buf);
  }());
}

class Bstr {
 public:
  explicit Bstr(const std::wstring& s) : b_(::SysAllocStringLen(s.data(), static_cast<UINT>(s.size()))) {}
  ~Bstr() { ::SysFreeString(b_); }
  Bstr(const Bstr&) = delete;
  Bstr& operator=(const Bstr&) = delete;
  BSTR get() const { return b_; }

 private:
  BSTR b_;
};

std::string take_bstr(BSTR b) {
  std::string out = b != nullptr ? to_utf8(std::wstring(b, ::SysStringLen(b))) : std::string();
  ::SysFreeString(b);
  return out;
}

std::string join(const std::vector<std::string>& v) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) out += (i == 0 ? "" : ",") + v[i];
  return out;
}

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

class WindowsFirewallRules final : public FirewallRules {
 public:
  Status ensure(const FirewallRuleSpec& spec) override {
    ComPtr<INetFwRules> rules;
    CLM_RETURN_IF_ERROR(get_rules(rules));
    ComPtr<INetFwRule> rule;
    const HRESULT found = rules->Item(Bstr(to_wide(spec.name)).get(), &rule);
    const bool exists = SUCCEEDED(found) && rule != nullptr;
    if (!exists) {
      HRESULT hr = ::CoCreateInstance(__uuidof(NetFwRule), nullptr, CLSCTX_INPROC_SERVER, __uuidof(INetFwRule),
                                      reinterpret_cast<void**>(rule.ReleaseAndGetAddressOf()));
      if (FAILED(hr)) return hr_status("CoCreateInstance(NetFwRule)", hr);
    }
    CLM_RETURN_IF_ERROR(fill(*rule.Get(), spec));
    if (!exists) {
      const HRESULT hr = rules->Add(rule.Get());
      if (FAILED(hr)) return hr_status("INetFwRules::Add", hr);
    }
    return Status::ok();
  }

  Status remove(const std::string& name) override {
    ComPtr<INetFwRules> rules;
    CLM_RETURN_IF_ERROR(get_rules(rules));
    const HRESULT hr = rules->Remove(Bstr(to_wide(name)).get());
    if (FAILED(hr)) return hr_status("INetFwRules::Remove", hr);
    return Status::ok();
  }

  Result<std::optional<FirewallRuleSpec>> query(const std::string& name) override {
    ComPtr<INetFwRules> rules;
    CLM_RETURN_IF_ERROR(get_rules(rules));
    ComPtr<INetFwRule> rule;
    const HRESULT hr = rules->Item(Bstr(to_wide(name)).get(), &rule);
    if (FAILED(hr) || rule == nullptr) return std::optional<FirewallRuleSpec>{};
    FirewallRuleSpec s;
    BSTR b = nullptr;
    s.name = name;
    if (SUCCEEDED(rule->get_Description(&b))) s.description = take_bstr(b);
    b = nullptr;
    if (SUCCEEDED(rule->get_Grouping(&b))) s.grouping = take_bstr(b);
    b = nullptr;
    if (SUCCEEDED(rule->get_ApplicationName(&b))) s.program_path = take_bstr(b);
    b = nullptr;
    if (SUCCEEDED(rule->get_LocalPorts(&b))) s.local_port = static_cast<std::uint16_t>(std::strtoul(take_bstr(b).c_str(), nullptr, 10));
    b = nullptr;
    if (SUCCEEDED(rule->get_RemoteAddresses(&b))) s.remote_addresses = split_commas(take_bstr(b));
    long profiles = 0;
    if (SUCCEEDED(rule->get_Profiles(&profiles))) s.profiles = static_cast<std::uint32_t>(profiles);
    VARIANT_BOOL enabled = VARIANT_FALSE;
    if (SUCCEEDED(rule->get_Enabled(&enabled))) s.enabled = enabled != VARIANT_FALSE;
    return std::optional<FirewallRuleSpec>{std::move(s)};
  }

 private:
  static Status get_rules(ComPtr<INetFwRules>& rules) {
    ComPtr<INetFwPolicy2> policy;
    HRESULT hr = ::CoCreateInstance(__uuidof(NetFwPolicy2), nullptr, CLSCTX_INPROC_SERVER, __uuidof(INetFwPolicy2),
                                    reinterpret_cast<void**>(policy.ReleaseAndGetAddressOf()));
    if (FAILED(hr)) return hr_status("CoCreateInstance(NetFwPolicy2)", hr);
    hr = policy->get_Rules(&rules);
    if (FAILED(hr)) return hr_status("INetFwPolicy2::get_Rules", hr);
    return Status::ok();
  }

  static Status fill(INetFwRule& r, const FirewallRuleSpec& s) {
    auto check = [](const char* what, HRESULT hr) { return FAILED(hr) ? hr_status(what, hr) : Status::ok(); };
    CLM_RETURN_IF_ERROR(check("put_Name", r.put_Name(Bstr(to_wide(s.name)).get())));
    CLM_RETURN_IF_ERROR(check("put_Description", r.put_Description(Bstr(to_wide(s.description)).get())));
    CLM_RETURN_IF_ERROR(check("put_Grouping", r.put_Grouping(Bstr(to_wide(s.grouping)).get())));
    CLM_RETURN_IF_ERROR(check("put_ApplicationName", r.put_ApplicationName(Bstr(to_wide(s.program_path)).get())));
    CLM_RETURN_IF_ERROR(check("put_Protocol", r.put_Protocol(NET_FW_IP_PROTOCOL_TCP)));
    CLM_RETURN_IF_ERROR(check("put_LocalPorts", r.put_LocalPorts(Bstr(to_wide(std::to_string(s.local_port))).get())));
    CLM_RETURN_IF_ERROR(check("put_RemoteAddresses", r.put_RemoteAddresses(Bstr(to_wide(join(s.remote_addresses))).get())));
    CLM_RETURN_IF_ERROR(check("put_Direction", r.put_Direction(NET_FW_RULE_DIR_IN)));
    CLM_RETURN_IF_ERROR(check("put_Action", r.put_Action(NET_FW_ACTION_ALLOW)));
    CLM_RETURN_IF_ERROR(check("put_Profiles", r.put_Profiles(static_cast<long>(s.profiles))));
    CLM_RETURN_IF_ERROR(check("put_EdgeTraversal", r.put_EdgeTraversal(VARIANT_FALSE)));
    CLM_RETURN_IF_ERROR(check("put_Enabled", r.put_Enabled(s.enabled ? VARIANT_TRUE : VARIANT_FALSE)));
    return Status::ok();
  }
};

}  // namespace

std::unique_ptr<FirewallRules> make_windows_firewall_rules() { return std::make_unique<WindowsFirewallRules>(); }

}  // namespace clusterlm::platform

#endif  // _WIN32
