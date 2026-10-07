#pragma once
// Status / Result: the error model used across ClusterLM module boundaries.
//
// Errors carry a stable machine code (for protocol replies, telemetry and tests) and a human-readable
// message. Exceptions are not thrown across module boundaries; Result<T> is returned instead.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace clusterlm {

enum class ErrorCode : std::uint16_t {
  kOk = 0,
  kInvalidArgument = 1,
  kOutOfRange = 2,
  kNotFound = 3,
  kAlreadyExists = 4,
  kFailedPrecondition = 5,   // operation not legal in the current state (e.g. RunWindow before PlanReady)
  kResourceExhausted = 6,    // memory/VRAM/staging budget would be exceeded
  kDataLoss = 7,             // digest mismatch, truncated object
  kUnauthenticated = 8,      // peer identity / pairing failure
  kPermissionDenied = 9,     // authenticated but not authorized for this plan/lease
  kStaleEpoch = 10,          // message names an epoch/lease generation that is no longer current
  kAborted = 11,             // session/window aborted or invalidated
  kCancelled = 12,           // cooperative cancellation (local activity, user)
  kDeadlineExceeded = 13,
  kUnavailable = 14,         // transport closed, peer lost
  kVersionMismatch = 15,     // protocol / boundary ABI / backend build mismatch
  kProtocolError = 16,       // malformed frame or message
  kInternal = 17,
  kUnimplemented = 18,
  kHardwareUnavailable = 19, // code path requires hardware absent in this environment (e.g. CUDA)
};

std::string_view to_string(ErrorCode code) noexcept;

class [[nodiscard]] Status {
 public:
  Status() = default;
  Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

  static Status ok() { return {}; }

  bool is_ok() const noexcept { return code_ == ErrorCode::kOk; }
  explicit operator bool() const noexcept { return is_ok(); }
  ErrorCode code() const noexcept { return code_; }
  const std::string& message() const noexcept { return message_; }
  std::string to_string() const;

 private:
  ErrorCode code_ = ErrorCode::kOk;
  std::string message_;
};

inline Status make_error(ErrorCode code, std::string message) { return Status(code, std::move(message)); }

template <typename T>
class [[nodiscard]] Result {
 public:
  Result(T value) : value_(std::move(value)) {}  // NOLINT(google-explicit-constructor)
  Result(Status status) : status_(std::move(status)) {  // NOLINT(google-explicit-constructor)
    if (status_.is_ok()) status_ = Status(ErrorCode::kInternal, "Result constructed from OK status without a value");
  }

  bool is_ok() const noexcept { return value_.has_value(); }
  explicit operator bool() const noexcept { return is_ok(); }
  const Status& status() const noexcept { return status_; }

  T& value() & { return *value_; }
  const T& value() const& { return *value_; }
  T&& value() && { return std::move(*value_); }
  T* operator->() { return &*value_; }
  const T* operator->() const { return &*value_; }
  T& operator*() & { return *value_; }
  const T& operator*() const& { return *value_; }

 private:
  std::optional<T> value_;
  Status status_;
};

}  // namespace clusterlm

#define CLM_CONCAT_INNER(a, b) a##b
#define CLM_CONCAT(a, b) CLM_CONCAT_INNER(a, b)

// Propagate a non-OK Status.
#define CLM_RETURN_IF_ERROR(expr)                 \
  do {                                            \
    ::clusterlm::Status clm_status_ = (expr);     \
    if (!clm_status_.is_ok()) return clm_status_; \
  } while (0)

// Assign the value of a Result<T> or propagate its Status.
#define CLM_ASSIGN_OR_RETURN(lhs, expr) CLM_ASSIGN_OR_RETURN_IMPL(CLM_CONCAT(clm_result_, __LINE__), lhs, expr)
// NOLINTBEGIN(bugprone-macro-parentheses): `lhs` may be a declaration ("auto x") and cannot be parenthesized.
#define CLM_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr) \
  auto tmp = (expr);                              \
  if (!tmp.is_ok()) return tmp.status();          \
  lhs = std::move(tmp).value()
// NOLINTEND(bugprone-macro-parentheses)
