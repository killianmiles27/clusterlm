// Windows known-folder paths and owner-only ACLs (ADR 0133). Compiled only on Windows.
#ifdef _WIN32

#include <accctrl.h>
#include <aclapi.h>
#include <shlobj.h>
#include <windows.h>

#include <cstdio>
#include <vector>

#include "../internal_errors.hpp"
#include "clusterlm/platform/paths.hpp"
#include "win_strings.hpp"

namespace clusterlm::platform {

namespace fs = std::filesystem;

namespace {

using detail::win_status;

// FOLDERID_ProgramData / FOLDERID_LocalAppData, spelled out so no object file has to define the SDK's GUIDs.
const GUID kFolderProgramData = {0x62ab5d82, 0xfdc1, 0x4dc3, {0xa9, 0xdd, 0x07, 0x0d, 0x1d, 0x49, 0x5d, 0x97}};
const GUID kFolderLocalAppData = {0xf1b32785, 0x6fba, 0x4fcf, {0x9d, 0x55, 0x7b, 0x8e, 0x7f, 0x15, 0x70, 0x91}};

Result<fs::path> known_folder(const GUID& id) {
  PWSTR raw = nullptr;
  const HRESULT hr = ::SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, nullptr, &raw);
  if (FAILED(hr) || raw == nullptr) {
    if (raw != nullptr) ::CoTaskMemFree(raw);
    return make_error(ErrorCode::kUnavailable, "SHGetKnownFolderPath failed");
  }
  fs::path p(raw);
  ::CoTaskMemFree(raw);
  return p;
}

// The two principals allowed on owner-only objects. Buffers own the SID storage.
struct Principals {
  std::vector<std::uint8_t> user_token;  // TOKEN_USER + SID
  std::vector<std::uint8_t> system_sid;
  PSID user() const { return reinterpret_cast<const TOKEN_USER*>(user_token.data())->User.Sid; }
  PSID system_sid_ptr() const { return const_cast<std::uint8_t*>(system_sid.data()); }
};

Result<Principals> load_principals() {
  Principals p;
  HANDLE tok = nullptr;
  if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &tok)) return win_status("OpenProcessToken");
  DWORD need = 0;
  ::GetTokenInformation(tok, TokenUser, nullptr, 0, &need);
  p.user_token.resize(need);
  const BOOL ok = ::GetTokenInformation(tok, TokenUser, p.user_token.data(), need, &need);
  ::CloseHandle(tok);
  if (!ok) return win_status("GetTokenInformation(TokenUser)");
  DWORD sz = SECURITY_MAX_SID_SIZE;
  p.system_sid.resize(sz);
  if (!::CreateWellKnownSid(WinLocalSystemSid, nullptr, p.system_sid.data(), &sz)) return win_status("CreateWellKnownSid");
  return p;
}

// DACL: full control for the process account and SYSTEM, nothing else. Containers pass it to children.
Result<PACL> build_dacl(const Principals& p, bool inherit) {
  EXPLICIT_ACCESSW ea[2] = {};
  const DWORD flags = inherit ? (CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE) : NO_INHERITANCE;
  PSID sids[2] = {p.user(), p.system_sid_ptr()};
  for (int i = 0; i < 2; ++i) {
    ea[i].grfAccessPermissions = FILE_ALL_ACCESS;
    ea[i].grfAccessMode = SET_ACCESS;
    ea[i].grfInheritance = flags;
    ea[i].Trustee.TrusteeForm = TRUSTEE_IS_SID;
    ea[i].Trustee.TrusteeType = i == 0 ? TRUSTEE_IS_USER : TRUSTEE_IS_WELL_KNOWN_GROUP;
    ea[i].Trustee.ptstrName = reinterpret_cast<LPWSTR>(sids[i]);
  }
  PACL acl = nullptr;
  const DWORD rc = ::SetEntriesInAclW(2, ea, nullptr, &acl);
  if (rc != ERROR_SUCCESS) return win_status("SetEntriesInAclW", rc);
  return acl;
}

// A SECURITY_ATTRIBUTES carrying that DACL as a PROTECTED descriptor, so no parent ACE is merged in at creation.
struct OwnerOnlyAttributes {
  Principals principals;
  PACL acl = nullptr;
  SECURITY_DESCRIPTOR sd{};
  SECURITY_ATTRIBUTES sa{};
  ~OwnerOnlyAttributes() {
    if (acl != nullptr) ::LocalFree(acl);
  }
};

Status init_attributes(OwnerOnlyAttributes& a, bool inherit) {
  CLM_ASSIGN_OR_RETURN(a.principals, load_principals());
  CLM_ASSIGN_OR_RETURN(a.acl, build_dacl(a.principals, inherit));
  if (!::InitializeSecurityDescriptor(&a.sd, SECURITY_DESCRIPTOR_REVISION)) return win_status("InitializeSecurityDescriptor");
  if (!::SetSecurityDescriptorDacl(&a.sd, TRUE, a.acl, FALSE)) return win_status("SetSecurityDescriptorDacl");
  if (!::SetSecurityDescriptorControl(&a.sd, SE_DACL_PROTECTED, SE_DACL_PROTECTED))
    return win_status("SetSecurityDescriptorControl");
  a.sa.nLength = sizeof a.sa;
  a.sa.lpSecurityDescriptor = &a.sd;
  a.sa.bInheritHandle = FALSE;
  return Status::ok();
}

bool is_dir_attr(const fs::path& p) {
  const DWORD attrs = ::GetFileAttributesW(p.c_str());
  return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

}  // namespace

Result<PathRoots> system_path_roots() {
  PathRoots r;
  CLM_ASSIGN_OR_RETURN(r.program_data, known_folder(kFolderProgramData));
  CLM_ASSIGN_OR_RETURN(r.local_app_data, known_folder(kFolderLocalAppData));
  return r;
}

Status restrict_to_owner(const fs::path& path) {
  CLM_ASSIGN_OR_RETURN(auto principals, load_principals());
  CLM_ASSIGN_OR_RETURN(PACL acl, build_dacl(principals, is_dir_attr(path)));
  const DWORD rc = ::SetNamedSecurityInfoW(const_cast<LPWSTR>(path.c_str()), SE_FILE_OBJECT,
                                           DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr,
                                           nullptr, acl, nullptr);
  ::LocalFree(acl);
  if (rc != ERROR_SUCCESS) return win_status("SetNamedSecurityInfoW(" + path.string() + ")", rc);
  return Status::ok();
}

Result<bool> is_owner_only(const fs::path& path) {
  CLM_ASSIGN_OR_RETURN(auto principals, load_principals());
  PACL dacl = nullptr;
  PSECURITY_DESCRIPTOR sd = nullptr;
  const DWORD rc = ::GetNamedSecurityInfoW(path.c_str(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr,
                                           &dacl, nullptr, &sd);
  if (rc != ERROR_SUCCESS) return win_status("GetNamedSecurityInfoW(" + path.string() + ")", rc);
  bool only = false;
  SECURITY_DESCRIPTOR_CONTROL control = 0;
  DWORD revision = 0;
  if (dacl != nullptr && ::GetSecurityDescriptorControl(sd, &control, &revision) && (control & SE_DACL_PROTECTED) != 0) {
    only = true;  // a protected DACL: parents cannot add entries; every explicit entry is inspected
    for (WORD i = 0; i < dacl->AceCount && only; ++i) {
      void* ace = nullptr;
      if (!::GetAce(dacl, i, &ace)) {
        only = false;
        break;
      }
      const auto* header = static_cast<const ACE_HEADER*>(ace);
      if (header->AceType != ACCESS_ALLOWED_ACE_TYPE) {
        only = false;  // anything unusual is not "owner only"
        break;
      }
      PSID sid = reinterpret_cast<PSID>(&static_cast<ACCESS_ALLOWED_ACE*>(ace)->SidStart);
      only = ::EqualSid(sid, principals.user()) || ::EqualSid(sid, principals.system_sid_ptr());
    }
  }
  ::LocalFree(sd);
  return only;
}

Status create_owner_only_directory(const fs::path& dir) {
  std::error_code ec;
  if (dir.has_parent_path()) fs::create_directories(dir.parent_path(), ec);
  if (ec) return make_error(ErrorCode::kUnavailable, "cannot create " + dir.parent_path().string() + ": " + ec.message());
  OwnerOnlyAttributes attrs;
  CLM_RETURN_IF_ERROR(init_attributes(attrs, /*inherit=*/true));
  if (::CreateDirectoryW(dir.c_str(), &attrs.sa)) return Status::ok();  // created with the DACL from the start
  if (::GetLastError() != ERROR_ALREADY_EXISTS) return win_status("CreateDirectoryW(" + dir.string() + ")");
  return restrict_to_owner(dir);
}

Status write_owner_only_file(const fs::path& path, ByteSpan data) {
  std::error_code ec;
  if (fs::exists(path, ec)) CLM_RETURN_IF_ERROR(restrict_to_owner(path));  // CREATE_ALWAYS keeps the old DACL
  OwnerOnlyAttributes attrs;
  CLM_RETURN_IF_ERROR(init_attributes(attrs, /*inherit=*/false));
  HANDLE h = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, &attrs.sa, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
  if (h == INVALID_HANDLE_VALUE) return win_status("CreateFileW(" + path.string() + ")");
  std::size_t done = 0;
  Status result = Status::ok();
  while (done < data.size()) {
    DWORD put = 0;
    const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(data.size() - done, 1u << 30));
    if (!::WriteFile(h, data.data() + done, chunk, &put, nullptr)) {
      result = win_status("WriteFile(" + path.string() + ")");
      break;
    }
    done += put;
  }
  if (result.is_ok() && !::FlushFileBuffers(h)) result = win_status("FlushFileBuffers");
  ::CloseHandle(h);
  return result;
}

}  // namespace clusterlm::platform

#endif  // _WIN32
