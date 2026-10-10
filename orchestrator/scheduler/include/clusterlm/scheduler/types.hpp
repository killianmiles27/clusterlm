#pragma once
// Scheduler-facing value types. These deliberately do not include the profile/library headers of workstream A: the
// scheduler consumes immutable *snapshots* (scheduler-admission-v1 §1) that an adapter fills from the profile store, so
// both sides can change independently. Nothing here carries prompt text or token ids — only counts (G: privacy).
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace clusterlm::scheduler {

// ---- stable error vocabulary (scheduler-admission-v1 §7) ----------------------------------------------------------

enum class Code : std::uint8_t {
  kInvalidApiKey, kInsufficientScope, kModelNotFound, kRequestTooLarge, kUnsupportedParameter,
  kContextLengthExceeded, kInvalidRequest, kConflict, kRateLimitExceeded, kQueueFull, kModelNotReady,
  kWorkersUnavailable, kInsufficientMemory, kWorkerLost, kShuttingDown, kRequestCancelled, kQueueTimeout,
  kInternalError,
};
std::string_view to_string(Code c) noexcept;   // the wire `code` string, e.g. "model_not_ready"
int http_status(Code c) noexcept;              // 401, 403, 404, 413, 400, 409, 429, 503, 504, 500
std::string_view error_type(Code c) noexcept;  // OpenAI `type`

// ---- readiness (readiness-state-machine-v1) -----------------------------------------------------------------------

enum class ReadyState : std::uint8_t { kUnavailable, kInstalled, kCompatible, kLoadable, kPreparing, kReady, kBusy };
std::string_view to_string(ReadyState s) noexcept;
inline bool is_prepared(ReadyState s) noexcept { return s == ReadyState::kReady || s == ReadyState::kBusy; }
inline bool is_loadable(ReadyState s) noexcept {
  return s == ReadyState::kLoadable || s == ReadyState::kPreparing || is_prepared(s);
}

struct ReadinessView {
  ReadyState state = ReadyState::kUnavailable;
  std::string reached = "none";                  // none | installed | compatible (meaningful for unavailable)
  std::vector<std::string> reasons;              // user-readable, first is the headline
  std::vector<std::string> blockers;             // machine codes
  std::optional<std::string> job_id;
  std::optional<double> percent;                 // preparing only
  std::optional<double> eta_seconds;             // estimate; absent without a measured rate
};

// ---- snapshots of profile / alias ----------------------------------------------------------------------------------

enum class Preparation : std::uint8_t { kManual, kOnDemand, kKeepReady };
enum class OnWorkerLoss : std::uint8_t { kStop, kFallbackProfile };

struct ProfileSnapshot {
  std::string id;                  // prof_*
  std::uint32_t revision = 0;
  std::string api_model_id;
  bool exposed_api = false;
  bool exposed_lan = false;
  // Everything that makes a prepared plan reusable (model identity, backend, topology, options): equal fingerprint ⇒ the
  // plan built for it serves a ticket pinned to this revision. The adapter hashes these; the scheduler only compares.
  std::string plan_fingerprint;
  std::string model_identity;      // family/quant/hash text for reporting (x_clusterlm), not for matching
  std::string backend_id;
  std::uint32_t context_max_tokens = 0;
  std::uint32_t context_default_tokens = 0;
  std::vector<std::uint32_t> offered_contexts;   // ascending; largest == context_max_tokens
  std::uint32_t max_completion_tokens_cap = 0;   // 0 = context_max_tokens
  Preparation preparation = Preparation::kOnDemand;
  bool prepare_when_available = false;
  std::uint32_t release_after_idle_seconds = 60;
  std::uint32_t max_sessions = 1;                // min(descriptor.max_sessions_per_domain, profile limit)
  bool per_client_isolation = false;
  OnWorkerLoss on_worker_loss = OnWorkerLoss::kStop;
  std::string fallback_profile_id;
};

enum class AliasStrategy : std::uint8_t { kFirstViable, kBestReady };
struct AliasSnapshot {
  std::string id;                  // alias_*
  std::uint32_t revision = 0;
  std::string api_model_id;
  bool exposed_lan = false;
  AliasStrategy strategy = AliasStrategy::kFirstViable;
  bool allow_prepare = false;
  std::vector<std::string> candidates;           // profile ids in list order
};

// ---- clients ---------------------------------------------------------------------------------------------------------

enum class Priority : std::uint8_t { kInteractive = 0, kApi = 1 };
inline constexpr std::size_t kPriorityCount = 2;

struct ClientLimits {
  std::uint32_t max_concurrent = 4;              // tickets in flight (queued + starting + running) for this client
  std::uint32_t max_queued = 8;                  // queue share
  std::uint32_t requests_per_minute = 0;         // 0 = unlimited
  bool lan = false;
  bool can_see_machine_names = false;            // holds `status:read`
  std::uint32_t prepare_wait_ms = 0;             // how long a request may wait for a prepare job; 0 = fail fast
};

struct ClientInfo {
  std::string id;                                // key id | "local-ui" | "mcp:<key id>"
  Priority priority = Priority::kApi;
  ClientLimits limits;
  std::vector<std::string> allowed_models;       // api_model_ids, or {"*"}
  bool scope_inference = true;
  bool revoked = false;
};

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

}  // namespace clusterlm::scheduler
