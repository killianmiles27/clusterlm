#pragma once
// Admission and alias selection: pure functions (scheduler-admission-v1 §3). No clock, no I/O, no locks — the scheduler
// gathers the inputs under its lock and calls these, so every rejection is reproducible in a unit test from its inputs.
#include <string>
#include <vector>

#include "clusterlm/scheduler/types.hpp"

namespace clusterlm::scheduler {

// How a model id resolved for this client. The adapter fills it (profile store + alias store + key scopes).
struct ResolvedTarget {
  const ProfileSnapshot* profile = nullptr;      // the exact profile addressed, or nullptr when `alias` is set
  const AliasSnapshot* alias = nullptr;
  std::string api_model_id;                      // what the client asked for
};

struct CandidateView {
  const ProfileSnapshot* profile = nullptr;
  ReadinessView readiness;                       // for the needed context class
  bool visible = true;                           // visible to this key (exposure, LAN, allowed_models)
};

struct ResourceView {
  bool workers_allowed = true;                   // resource policy for the profile's required Workers, now
  std::string reason;                            // generic sentence when not allowed
  bool memory_blocked = false;                   // admission found the plan cannot fit (insufficient_memory)
};

struct RequestShape {
  std::uint32_t prompt_tokens = 0;
  std::optional<std::uint32_t> max_tokens;       // max_tokens / max_completion_tokens as given
  std::uint64_t body_bytes = 0;
  std::vector<std::string> unsupported;          // parameter names the backend/template cannot honour (n>1, logprobs…)
};

struct QueueView {
  std::size_t depth = 0;
  std::size_t client_queued = 0;
  std::size_t client_in_flight = 0;
  std::size_t client_requests_last_minute = 0;
  std::uint32_t max_queue_depth = 32;
  std::uint32_t max_queue_per_client = 8;
  bool slot_free = false;                        // an immediate start would be possible for the target plan
  bool shutting_down = false;
};

struct AdmissionPolicy {
  std::uint64_t max_body_bytes = 8ULL << 20;
  std::uint32_t max_completion_tokens_cap = 0;   // global cap; 0 = none beyond the profile
};

struct Decision {
  enum class Kind : std::uint8_t { kAdmit, kQueue, kReject } kind = Kind::kReject;
  Code code = Code::kInternalError;              // Reject only
  std::string message;
  std::vector<std::string> reasons;
  std::string profile_id;                        // resolved profile (alias: the recorded candidate)
  std::uint32_t profile_revision = 0;
  std::string routed_via;                        // alias api_model_id when routed
  std::uint32_t completion_budget = 0;
  std::uint32_t context_class = 0;               // smallest offered context >= prompt + budget
  ReadyState readiness = ReadyState::kUnavailable;
  bool start_prepare = false;                    // the scheduler must (idempotently) start/join a prepare job
  bool wait_for_prepare = false;                 // Queue only: hold the ticket until the job finishes
  std::optional<std::string> job_id;
  std::optional<std::uint32_t> retry_after_s;
};

// Smallest offered context >= need; nullopt when none (need > max).
std::optional<std::uint32_t> context_class_for(const ProfileSnapshot& p, std::uint32_t need);

// Completion budget per §3 step 3: requested, else the rest of the context, capped by policy.
std::uint32_t completion_budget(const ProfileSnapshot& p, const RequestShape& r, const AdmissionPolicy& pol);

// Alias selection. Candidates arrive in alias list order. Returns the index of the chosen candidate or nullopt (with
// `why` listing per-candidate reasons, generic when the client may not see machine names).
std::optional<std::size_t> select_candidate(const AliasSnapshot& alias, const std::vector<CandidateView>& cands,
                                            std::vector<std::string>* why);

// Maps a non-servable readiness to the rejection code of §7 (transient -> workers_unavailable, blocked ->
// model_not_ready, memory -> insufficient_memory).
Code readiness_reject_code(const ReadinessView& r, const ResourceView& res);

// Steps 1-8. `readiness` is the chosen profile's (for an alias, the chosen candidate's).
Decision admit(const ClientInfo& client, const ResolvedTarget& target, const ProfileSnapshot& chosen,
               const RequestShape& req, const ReadinessView& readiness, const ResourceView& resources,
               const QueueView& queue, const AdmissionPolicy& policy, bool alias_allows_prepare);

}  // namespace clusterlm::scheduler
