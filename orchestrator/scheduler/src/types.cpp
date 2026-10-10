#include "clusterlm/scheduler/types.hpp"

namespace clusterlm::scheduler {

std::string_view to_string(Code c) noexcept {
  switch (c) {
    case Code::kInvalidApiKey: return "invalid_api_key";
    case Code::kInsufficientScope: return "insufficient_scope";
    case Code::kModelNotFound: return "model_not_found";
    case Code::kRequestTooLarge: return "request_too_large";
    case Code::kUnsupportedParameter: return "unsupported_parameter";
    case Code::kContextLengthExceeded: return "context_length_exceeded";
    case Code::kInvalidRequest: return "invalid_request";
    case Code::kConflict: return "conflict";
    case Code::kRateLimitExceeded: return "rate_limit_exceeded";
    case Code::kQueueFull: return "queue_full";
    case Code::kModelNotReady: return "model_not_ready";
    case Code::kWorkersUnavailable: return "workers_unavailable";
    case Code::kInsufficientMemory: return "insufficient_memory";
    case Code::kWorkerLost: return "worker_lost";
    case Code::kShuttingDown: return "shutting_down";
    case Code::kRequestCancelled: return "request_cancelled";
    case Code::kQueueTimeout: return "queue_timeout";
    case Code::kInternalError: return "internal_error";
  }
  return "internal_error";
}

int http_status(Code c) noexcept {
  switch (c) {
    case Code::kInvalidApiKey: return 401;
    case Code::kInsufficientScope: return 403;
    case Code::kModelNotFound: return 404;
    case Code::kRequestTooLarge: return 413;
    case Code::kUnsupportedParameter:
    case Code::kContextLengthExceeded:
    case Code::kInvalidRequest: return 400;
    case Code::kConflict: return 409;
    case Code::kRateLimitExceeded: return 429;
    case Code::kQueueFull:
    case Code::kModelNotReady:
    case Code::kWorkersUnavailable:
    case Code::kInsufficientMemory:
    case Code::kWorkerLost:
    case Code::kShuttingDown:
    case Code::kRequestCancelled: return 503;
    case Code::kQueueTimeout: return 504;
    case Code::kInternalError: return 500;
  }
  return 500;
}

std::string_view error_type(Code c) noexcept {
  switch (c) {
    case Code::kInvalidApiKey: return "authentication_error";
    case Code::kInsufficientScope: return "permission_error";
    case Code::kModelNotFound:
    case Code::kRequestTooLarge:
    case Code::kUnsupportedParameter:
    case Code::kContextLengthExceeded:
    case Code::kInvalidRequest:
    case Code::kConflict: return "invalid_request_error";
    case Code::kRateLimitExceeded: return "rate_limit_error";
    default: return "server_error";
  }
}

std::string_view to_string(ReadyState s) noexcept {
  switch (s) {
    case ReadyState::kUnavailable: return "unavailable";
    case ReadyState::kInstalled: return "installed";
    case ReadyState::kCompatible: return "compatible";
    case ReadyState::kLoadable: return "loadable";
    case ReadyState::kPreparing: return "preparing";
    case ReadyState::kReady: return "ready";
    case ReadyState::kBusy: return "busy";
  }
  return "unavailable";
}

}  // namespace clusterlm::scheduler
