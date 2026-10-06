// Shared UI definitions: display helpers, state naming, settings validation.
#include <cmath>
#include <cstdio>

#include "clusterlm/catalog/readiness.hpp"
#include "clusterlm/ui/format.hpp"
#include "clusterlm/ui/father_client.hpp"
#include "clusterlm/ui/node_client.hpp"

namespace clusterlm::ui {

namespace {
struct Fold {
  std::string_view from, to;
};
constexpr Fold kFolds[] = {
    {"\xE2\x80\x94", " - "},  // em dash
    {"\xE2\x80\x93", "-"},    // en dash
    {"\xE2\x80\x98", "'"},  {"\xE2\x80\x99", "'"},  // single quotes
    {"\xE2\x80\x9C", "\""}, {"\xE2\x80\x9D", "\""},  // double quotes
    {"\xE2\x80\xA6", "..."},                         // ellipsis
    {"\xC2\xA0", " "},                               // no-break space
};

std::string fmt(const char* f, double v) {
  char buf[48];
  std::snprintf(buf, sizeof buf, f, v);
  return buf;
}
}  // namespace

std::string ascii_display(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    bool folded = false;
    if (static_cast<unsigned char>(text[i]) >= 0x80) {
      for (const auto& f : kFolds) {
        if (text.substr(i, f.from.size()) == f.from) {
          out += f.to;
          i += f.from.size();
          folded = true;
          break;
        }
      }
    }
    if (!folded) out.push_back(text[i++]);
  }
  // " - " produced next to an existing space would double up.
  for (std::size_t p; (p = out.find("  - ")) != std::string::npos;) out.erase(p, 1);
  for (std::size_t p; (p = out.find(" -  ")) != std::string::npos;) out.erase(p + 3, 1);
  return out;
}

std::string format_bytes(std::uint64_t bytes) {
  constexpr const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  double v = static_cast<double>(bytes);
  int u = 0;
  while (v >= 1000.0 && u < 4) {
    v /= 1000.0;
    ++u;
  }
  if (u == 0) return std::to_string(bytes) + " B";
  return fmt(v < 100 ? "%.1f " : "%.0f ", v) + units[u];
}

std::string format_rate(double tok_s) {
  if (!(tok_s > 0)) return "--";
  return fmt(tok_s < 100 ? "%.1f" : "%.0f", tok_s) + " tok/s";
}

std::string format_millis(double ms) {
  if (!(ms > 0)) return "--";
  if (ms < 1000) return fmt("%.0f", ms) + " ms";
  return fmt("%.1f", ms / 1000.0) + " s";
}

std::string format_eta(std::optional<double> seconds) {
  if (!seconds || !std::isfinite(*seconds) || *seconds < 0) return {};
  return catalog::format_duration(*seconds) + " (estimate)";
}

std::string describe_error(ErrorCode code, std::string_view message) {
  std::string m = ascii_display(message);
  if (m.empty()) m = "Something went wrong";
  return m + " (" + std::string(to_string(code)) + ")";
}

// ---- Node state ----------------------------------------------------------------------------------------------

std::string_view to_string(NodeUiState s) noexcept {
  switch (s) {
    case NodeUiState::kUnreachable: return "Not running";
    case NodeUiState::kStarting: return "Starting";
    case NodeUiState::kAvailable: return "Available";
    case NodeUiState::kPreparing: return "Preparing";
    case NodeUiState::kReady: return "Ready";
    case NodeUiState::kInUse: return "In use";
    case NodeUiState::kBusy: return "Busy";
    case NodeUiState::kPaused: return "Paused";
    case NodeUiState::kCleanupNeeded: return "Cleanup needed";
    case NodeUiState::kStopping: return "Stopping";
  }
  return "Unknown";
}

std::string_view describe(NodeUiState s) noexcept {
  switch (s) {
    case NodeUiState::kUnreachable: return "The ClusterLM Node service is not answering. Start it from Windows Services.";
    case NodeUiState::kStarting: return "The Node is starting up.";
    case NodeUiState::kAvailable: return "This PC can help the Father when you are not using it.";
    case NodeUiState::kPreparing: return "Getting ready to help the Father. Temporary files are being received.";
    case NodeUiState::kReady: return "Ready to help the Father.";
    case NodeUiState::kInUse: return "Helping the Father right now.";
    case NodeUiState::kBusy: return "You are using this PC, so the Node is staying out of the way.";
    case NodeUiState::kPaused: return "Paused. This PC will not help until you resume.";
    case NodeUiState::kCleanupNeeded: return "Temporary files from an earlier session are being removed.";
    case NodeUiState::kStopping: return "The Node is shutting down.";
  }
  return "";
}

NodeUiState from_ipc(ipc::NodeState s) noexcept {
  switch (s) {
    case ipc::NodeState::kStarting: return NodeUiState::kStarting;
    case ipc::NodeState::kBusy: return NodeUiState::kBusy;
    case ipc::NodeState::kOffering: return NodeUiState::kAvailable;
    case ipc::NodeState::kPaused: return NodeUiState::kPaused;
    case ipc::NodeState::kSuspended: return NodeUiState::kBusy;  // locked/suspended session: out of the way
    case ipc::NodeState::kStopping: return NodeUiState::kStopping;
  }
  return NodeUiState::kUnreachable;
}

NodeUiState from_machine_state(catalog::MachineState s) noexcept {
  using M = catalog::MachineState;
  switch (s) {
    case M::kBusy: return NodeUiState::kBusy;
    case M::kAvailable: return NodeUiState::kAvailable;
    case M::kPreparing: return NodeUiState::kPreparing;
    case M::kReady: return NodeUiState::kReady;
    case M::kInferencing: return NodeUiState::kInUse;
    case M::kReleasing: return NodeUiState::kCleanupNeeded;
    case M::kCleanupPending: return NodeUiState::kCleanupNeeded;
    case M::kOffline: return NodeUiState::kUnreachable;
  }
  return NodeUiState::kUnreachable;
}

Status validate(const NodeSettings& s) {
  if (s.temp_storage_limit_gb < 1 || s.temp_storage_limit_gb > 4096)
    return make_error(ErrorCode::kInvalidArgument, "Temporary storage limit must be between 1 and 4096 GB.");
  if (s.cpu_cap_percent < 10 || s.cpu_cap_percent > 100)
    return make_error(ErrorCode::kInvalidArgument, "CPU limit must be between 10% and 100%.");
  if (s.gpu_memory_cap_percent < 10 || s.gpu_memory_cap_percent > 100)
    return make_error(ErrorCode::kInvalidArgument, "Graphics memory limit must be between 10% and 100%.");
  if (s.ram_cap_gb > 1024)
    return make_error(ErrorCode::kInvalidArgument, "Memory limit must be 0 (no limit) or between 1 and 1024 GB.");
  return Status::ok();
}

Status validate(const FatherSettings& s) {
  if (s.context_tokens < 512 || s.context_tokens > (1u << 20))
    return make_error(ErrorCode::kInvalidArgument, "Context size must be between 512 and 1048576 tokens.");
  if (s.max_new_tokens < 1 || s.max_new_tokens > s.context_tokens)
    return make_error(ErrorCode::kInvalidArgument, "Answer length must be at least 1 token and fit in the context.");
  if (s.system_prompt.size() > 16 * 1024)
    return make_error(ErrorCode::kInvalidArgument, "The system prompt is too long (limit 16 KB).");
  return Status::ok();
}

}  // namespace clusterlm::ui
