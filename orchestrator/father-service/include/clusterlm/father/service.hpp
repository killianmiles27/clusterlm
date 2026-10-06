#pragma once
// Father service API: the façade the Father UI and agent use.
//
//   list tiers (with honest readiness) -> select -> prepare (async, progress events) -> chat (async, streaming
//   token events + final stats) -> cancel -> release; plus a redacted diagnostics snapshot.
//
// Privacy boundary: chat roles, message text and the tokenizer exist ONLY in this Father-local layer. The
// Coordinator receives token arrays and nothing else; Nodes never see either. Logs and diagnostics carry
// counts, ids and timings — never prompts, responses or token arrays (the conversation is included in a
// snapshot only when the caller explicitly asks for it).
//
// Fallback policy: if a distributed session is invalidated (node lost, local activity, fault) the conversation
// is kept and the catalog's explicit per-tier policy decides: retry the same tier up to max_retries (only
// while the tier is still observed viable), then downgrade to the next viable lower tier or stop. A model is
// never substituted silently: every retry/downgrade emits a FallbackEvent naming both models, and every token
// event and the final stats name the model that produced them.
//
// Threading: all methods are thread-safe. One job (prepare or chat) runs at a time on a service-owned thread;
// events are delivered on that thread (or the caller's for list/select), serialized, in order. Sinks must not
// block and must not call release() or the destructor (cancel() is fine).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/catalog/readiness.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/father/tokenizer.hpp"

namespace clusterlm::father {

// ---- environment adapters (supplied by the app; mocks/fixtures in tests) --------------------------------

// Everything needed to run one tier: where the nodes are and the executable plan for the requested context.
// Binding a tier to concrete machines and a plan (placement search -> ClusterPlan) is the provider's job.
struct Deployment {
  coordinator::CoordinatorConfig config;
  coordinator::ClusterPlan plan;
  std::function<std::shared_ptr<domain::Drafter>()> make_drafter;  // required only for q > 1
};

class DeploymentProvider {
 public:
  virtual ~DeploymentProvider() = default;
  virtual Result<Deployment> resolve(const catalog::TierEntry& tier, std::uint32_t context_tokens) = 0;
};

// Observes the world for readiness: local model state, backend, paired machines + node state, power,
// provisioning progress, placement feasibility. The service overlays what only it knows (its own prepared plan
// and Father's domain state).
class ReadinessSource {
 public:
  virtual ~ReadinessSource() = default;
  virtual catalog::ReadinessInputs observe(const catalog::TierEntry& tier, const catalog::TierAssignment& assignment,
                                           std::uint32_t context_tokens) = 0;
};

// ---- events ----------------------------------------------------------------------------------------------

using RequestId = std::uint64_t;

struct TierSelectedEvent { std::string tier_id, model_name; };
struct PrepareProgressEvent {
  std::string tier_id, model_name;
  std::optional<double> percent;
  std::optional<double> eta_seconds;  // estimate
  std::string message;
};
struct TierReadyEvent { std::string tier_id, model_name; };
// Father-local: carries the generated token IDs and their text for the UI/agent only.
struct TokensEvent {
  RequestId request = 0;
  std::string tier_id, model_name;  // the model that produced exactly these tokens
  std::vector<std::int32_t> tokens;
  std::string text;
};
struct FallbackEvent {
  enum class Kind : std::uint8_t { kRetry, kDowngrade };
  RequestId request = 0;
  Kind kind = Kind::kDowngrade;
  std::string from_tier, to_tier, from_model, to_model;
  std::string message;  // readable, names both models
};
enum class FinishReason : std::uint8_t { kCompleted, kCancelled, kFailed };
std::string_view to_string(FinishReason r) noexcept;

struct AnswerSegment {
  std::string tier_id, model_name;
  std::uint32_t tokens = 0;
};
struct GenerationStats {
  double ttft_ms = 0;               // request start -> first token event (includes any tier preparation)
  double decode_tok_s = 0;          // observed on this run: tokens / generation wall time
  double accepted_per_round = 0;    // mean tokens accepted per decode round (1.0 without speculation)
  std::uint32_t tokens = 0;
  std::uint32_t rounds = 0;
  std::uint32_t fallbacks = 0;
  std::string final_tier, final_model;
  std::vector<AnswerSegment> answered_by;  // which model produced how many tokens, in order
};
struct FinishedEvent { RequestId request = 0; FinishReason reason = FinishReason::kCompleted; GenerationStats stats; };
struct ErrorEvent { RequestId request = 0; ErrorCode code = ErrorCode::kInternal; std::string message; };
struct ReleasedEvent { std::string message; };

using Event = std::variant<TierSelectedEvent, PrepareProgressEvent, TierReadyEvent, TokensEvent, FallbackEvent,
                           FinishedEvent, ErrorEvent, ReleasedEvent>;
using EventSink = std::function<void(const Event&)>;
using SubscriptionId = std::uint64_t;

// ---- requests / snapshots --------------------------------------------------------------------------------

struct ChatRequest {
  std::string user_message;
  std::optional<std::string> system_prompt;  // applied only when the conversation is empty
  std::uint32_t max_new_tokens = 32;
  std::uint32_t context_tokens = 4096;       // a catalog context profile
  std::uint32_t q = 1;                       // verification width
};

struct DiagnosticsSnapshot {
  struct TierLine { std::string tier_id, model_name, state, headline; };
  std::string selected_tier, active_tier, active_model;
  bool active_ready = false, job_running = false;
  std::uint64_t requests_started = 0, requests_completed = 0, requests_cancelled = 0, requests_failed = 0;
  std::uint64_t fallbacks = 0, tokens_emitted = 0;
  std::uint32_t conversation_messages = 0;
  std::vector<TierLine> tiers;
  std::string last_error;  // status text only
  std::vector<ChatMessage> conversation;  // empty unless include_text was requested
  std::string to_string() const;
};

struct ServiceOptions {
  std::chrono::milliseconds progress_poll{100};
  std::uint32_t default_context_tokens = 4096;
};

class FatherService {
 public:
  virtual ~FatherService() = default;

  virtual std::vector<catalog::TierReadiness> list_tiers(std::uint32_t context_tokens = 0) = 0;
  virtual Status select_tier(std::string_view tier_id) = 0;
  virtual std::string selected_tier() const = 0;
  // Async: emits PrepareProgressEvent... then TierReadyEvent, or ErrorEvent.
  virtual Status prepare_tier(std::string_view tier_id, std::uint32_t context_tokens) = 0;
  // Async: appends the user message to the Father-held conversation and streams the answer.
  virtual Result<RequestId> chat(ChatRequest request) = 0;
  virtual Status cancel(RequestId request) = 0;
  // Cancels any running job, waits for it, releases every lease and local domain.
  virtual Status release() = 0;
  // Fails (kFailedPrecondition) while a job is running.
  virtual Status reset_conversation() = 0;
  virtual std::vector<ChatMessage> conversation() const = 0;  // Father-local only
  virtual DiagnosticsSnapshot diagnostics(bool include_text = false) const = 0;

  virtual SubscriptionId subscribe(EventSink sink) = 0;
  virtual void unsubscribe(SubscriptionId id) = 0;
  // True when no job is running (waits up to `timeout`).
  virtual bool wait_idle(std::chrono::milliseconds timeout) = 0;
};

struct ServiceDeps {
  catalog::Catalog catalog;
  catalog::TierAssignment assignment;
  std::shared_ptr<Tokenizer> tokenizer;
  std::shared_ptr<ReadinessSource> readiness;
  std::shared_ptr<DeploymentProvider> deployments;
  ServiceOptions options;
};

Result<std::unique_ptr<FatherService>> make_father_service(ServiceDeps deps);

}  // namespace clusterlm::father
