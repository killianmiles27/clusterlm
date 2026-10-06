#include "clusterlm/common/status.hpp"

namespace clusterlm {

std::string_view to_string(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kOk: return "OK";
    case ErrorCode::kInvalidArgument: return "INVALID_ARGUMENT";
    case ErrorCode::kOutOfRange: return "OUT_OF_RANGE";
    case ErrorCode::kNotFound: return "NOT_FOUND";
    case ErrorCode::kAlreadyExists: return "ALREADY_EXISTS";
    case ErrorCode::kFailedPrecondition: return "FAILED_PRECONDITION";
    case ErrorCode::kResourceExhausted: return "RESOURCE_EXHAUSTED";
    case ErrorCode::kDataLoss: return "DATA_LOSS";
    case ErrorCode::kUnauthenticated: return "UNAUTHENTICATED";
    case ErrorCode::kPermissionDenied: return "PERMISSION_DENIED";
    case ErrorCode::kStaleEpoch: return "STALE_EPOCH";
    case ErrorCode::kAborted: return "ABORTED";
    case ErrorCode::kCancelled: return "CANCELLED";
    case ErrorCode::kDeadlineExceeded: return "DEADLINE_EXCEEDED";
    case ErrorCode::kUnavailable: return "UNAVAILABLE";
    case ErrorCode::kVersionMismatch: return "VERSION_MISMATCH";
    case ErrorCode::kProtocolError: return "PROTOCOL_ERROR";
    case ErrorCode::kInternal: return "INTERNAL";
    case ErrorCode::kUnimplemented: return "UNIMPLEMENTED";
    case ErrorCode::kHardwareUnavailable: return "HARDWARE_UNAVAILABLE";
  }
  return "UNKNOWN";
}

std::string Status::to_string() const {
  if (is_ok()) return "OK";
  std::string out(clusterlm::to_string(code_));
  out += ": ";
  out += message_;
  return out;
}

}  // namespace clusterlm
