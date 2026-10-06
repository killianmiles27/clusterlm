#pragma once
// Maps OS error numbers onto ClusterLM error codes. Internal to clusterlm_platform.
#include <cerrno>
#include <string>
#include <string_view>
#include <system_error>

#include "clusterlm/common/status.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace clusterlm::platform::detail {

inline Status errno_status(std::string_view what, int err = errno) {
  ErrorCode code = ErrorCode::kInternal;
  switch (err) {
    case ENOENT: code = ErrorCode::kNotFound; break;
    case EEXIST: code = ErrorCode::kAlreadyExists; break;
    case ENOSPC:
    case ENOMEM: code = ErrorCode::kResourceExhausted; break;
#ifdef EDQUOT
    case EDQUOT: code = ErrorCode::kResourceExhausted; break;
#endif
    case EACCES:
    case EPERM:
    case ELOOP: code = ErrorCode::kPermissionDenied; break;
    case EINVAL: code = ErrorCode::kInvalidArgument; break;
    default: break;
  }
  return Status(code, std::string(what) + ": " + std::error_code(err, std::generic_category()).message());
}

#ifdef _WIN32
inline Status win_status(std::string_view what, unsigned long err = ::GetLastError()) {
  ErrorCode code = ErrorCode::kInternal;
  switch (err) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND: code = ErrorCode::kNotFound; break;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS: code = ErrorCode::kAlreadyExists; break;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_COMMITMENT_LIMIT: code = ErrorCode::kResourceExhausted; break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION: code = ErrorCode::kPermissionDenied; break;
    case ERROR_INVALID_PARAMETER: code = ErrorCode::kInvalidArgument; break;
    default: break;
  }
  return Status(code, std::string(what) + ": " + std::system_category().message(static_cast<int>(err)));
}
#endif

}  // namespace clusterlm::platform::detail
