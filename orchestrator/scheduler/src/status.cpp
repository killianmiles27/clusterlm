#include "clusterlm/scheduler/status.hpp"

namespace clusterlm::scheduler {

std::string_view to_string(TicketState s) noexcept {
  switch (s) {
    case TicketState::kQueued: return "queued";
    case TicketState::kStarting: return "starting";
    case TicketState::kRunning: return "running";
    case TicketState::kFinishing: return "finishing";
    case TicketState::kCancelling: return "cancelling";
    case TicketState::kDone: return "done";
  }
  return "done";
}
std::string_view to_string(JobKind k) noexcept { return k == JobKind::kPrepare ? "prepare" : "release"; }
std::string_view to_string(JobState s) noexcept {
  switch (s) {
    case JobState::kQueued: return "queued";
    case JobState::kRunning: return "running";
    case JobState::kSucceeded: return "succeeded";
    case JobState::kFailed: return "failed";
    case JobState::kCancelled: return "cancelled";
  }
  return "failed";
}

}  // namespace clusterlm::scheduler
