#include "clusterlm/scheduler/admission.hpp"

#include <algorithm>

namespace clusterlm::scheduler {

namespace {

bool model_allowed(const ClientInfo& c, const std::string& api_model_id) {
  return std::any_of(c.allowed_models.begin(), c.allowed_models.end(),
                     [&](const std::string& m) { return m == "*" || m == api_model_id; });
}

Decision reject(Code code, std::string message, const ProfileSnapshot& p, ReadyState st) {
  Decision d;
  d.kind = Decision::Kind::kReject;
  d.code = code;
  d.message = std::move(message);
  d.profile_id = p.id;
  d.profile_revision = p.revision;
  d.readiness = st;
  return d;
}

std::string generic_sentence(Code c) {
  switch (c) {
    case Code::kWorkersUnavailable: return "a required Worker is not available";
    case Code::kInsufficientMemory: return "not enough memory is available for this model";
    case Code::kModelNotReady: return "the model is not ready";
    default: return "the request cannot be served right now";
  }
}

void fill_reasons(Decision& d, const ClientInfo& c, const ReadinessView& r) {
  d.reasons = c.limits.can_see_machine_names ? r.reasons : std::vector<std::string>{generic_sentence(d.code)};
  if (d.reasons.empty()) d.reasons.push_back(generic_sentence(d.code));
}

}  // namespace

std::optional<std::uint32_t> context_class_for(const ProfileSnapshot& p, std::uint32_t need) {
  for (std::uint32_t c : p.offered_contexts)
    if (c >= need) return c;
  if (p.offered_contexts.empty() && need <= p.context_max_tokens) return p.context_max_tokens;
  return std::nullopt;
}

std::uint32_t completion_budget(const ProfileSnapshot& p, const RequestShape& r, const AdmissionPolicy& pol) {
  std::uint32_t cap = p.max_completion_tokens_cap ? p.max_completion_tokens_cap : p.context_max_tokens;
  if (pol.max_completion_tokens_cap) cap = std::min(cap, pol.max_completion_tokens_cap);
  const std::uint32_t rest = p.context_max_tokens > r.prompt_tokens ? p.context_max_tokens - r.prompt_tokens : 0;
  const std::uint32_t want = r.max_tokens ? *r.max_tokens : rest;
  return std::min(want, cap);
}

Code readiness_reject_code(const ReadinessView& r, const ResourceView& res) {
  if (res.memory_blocked) return Code::kInsufficientMemory;
  for (const auto& b : r.blockers)
    if (b == "insufficient_memory") return Code::kInsufficientMemory;
  // Transient (a Worker busy, offline, on battery, paused, no feasible placement right now) is `compatible`.
  if (r.state == ReadyState::kCompatible) return Code::kWorkersUnavailable;
  return Code::kModelNotReady;
}

std::optional<std::size_t> select_candidate(const AliasSnapshot& alias, const std::vector<CandidateView>& cands,
                                            std::vector<std::string>* why) {
  // Viable = could serve this request without a substitution: prepared; or loadable/preparing when the alias allows
  // starting a prepare ("the alias never starts a prepare job unless allow_prepare").
  const auto viable = [&](const CandidateView& c) {
    if (!c.profile || !c.visible) return false;
    const ReadyState s = c.readiness.state;
    if (is_prepared(s)) return true;
    return alias.allow_prepare && (s == ReadyState::kLoadable || s == ReadyState::kPreparing);
  };
  std::optional<std::size_t> first_viable;
  for (std::size_t i = 0; i < cands.size(); ++i) {
    if (!viable(cands[i])) continue;
    if (!first_viable) first_viable = i;
    if (alias.strategy == AliasStrategy::kBestReady && is_prepared(cands[i].readiness.state)) return i;
    if (alias.strategy == AliasStrategy::kFirstViable) return i;
  }
  if (first_viable) return first_viable;
  if (why) {
    for (const auto& c : cands) {
      if (!c.profile || !c.visible) continue;
      std::string line = c.profile->api_model_id + ": " + std::string(to_string(c.readiness.state));
      if (!c.readiness.reasons.empty()) line += " (" + c.readiness.reasons.front() + ")";
      why->push_back(std::move(line));
    }
  }
  return std::nullopt;
}

Decision admit(const ClientInfo& client, const ResolvedTarget& target, const ProfileSnapshot& p,
               const RequestShape& req, const ReadinessView& readiness, const ResourceView& res,
               const QueueView& q, const AdmissionPolicy& policy, bool alias_allows_prepare) {
  const ReadyState st = readiness.state;

  // 0. Server state first. (The v1.1 table lists it last; a draining server has already released its plans, so
  //    evaluating it last would answer `model_not_ready` for what is really `shutting_down`. See docs/scheduler-design.md.)
  if (q.shutting_down) return reject(Code::kShuttingDown, "the server is shutting down", p, st);

  // 1. Auth, scope, visibility.
  if (client.revoked) return reject(Code::kInvalidApiKey, "invalid API key", p, st);
  if (!client.scope_inference) return reject(Code::kInsufficientScope, "this key may not run inference", p, st);
  const std::string& asked = target.api_model_id.empty() ? p.api_model_id : target.api_model_id;
  const bool lan_blocked = client.limits.lan && !p.exposed_lan;
  if (!model_allowed(client, asked) || !p.exposed_api || lan_blocked)
    return reject(Code::kModelNotFound, "the model '" + asked + "' does not exist or is not available to this key", p, st);

  // 2. Shape.
  if (req.body_bytes > policy.max_body_bytes)
    return reject(Code::kRequestTooLarge, "request body is too large", p, st);
  if (!req.unsupported.empty())
    return reject(Code::kUnsupportedParameter, "parameter '" + req.unsupported.front() + "' is not supported by this model",
                  p, st);

  // 3. Context, against the profile (never the prepared plan).
  const std::uint32_t budget = completion_budget(p, req, policy);
  const std::uint64_t need64 = std::uint64_t{req.prompt_tokens} + budget;
  const auto cls = (budget > 0 && need64 <= p.context_max_tokens)
                       ? context_class_for(p, static_cast<std::uint32_t>(need64))
                       : std::nullopt;
  if (!cls)
    return reject(Code::kContextLengthExceeded,
                  "this model's maximum context is " + std::to_string(p.context_max_tokens) +
                      " tokens; the request needs " + std::to_string(std::max<std::uint64_t>(need64, req.prompt_tokens + 1ULL)),
                  p, st);

  // 4. Client limits.
  if (client.limits.requests_per_minute && q.client_requests_last_minute >= client.limits.requests_per_minute) {
    Decision d = reject(Code::kRateLimitExceeded, "rate limit exceeded", p, st);
    d.retry_after_s = 60;
    return d;
  }
  if (q.client_in_flight >= client.limits.max_concurrent ||
      q.client_queued >= std::min<std::size_t>(client.limits.max_queued, q.max_queue_per_client)) {
    Decision d = reject(Code::kRateLimitExceeded, "too many concurrent requests for this key", p, st);
    d.retry_after_s = 5;
    return d;
  }

  Decision d;
  d.profile_id = p.id;
  d.profile_revision = p.revision;
  d.readiness = st;
  d.completion_budget = budget;
  d.context_class = *cls;
  d.job_id = readiness.job_id;
  if (target.alias) d.routed_via = target.alias->api_model_id;

  // 5. Readiness for the context class.
  const auto not_ready = [&](Code code, std::string msg) {
    d.kind = Decision::Kind::kReject;
    d.code = code;
    d.message = std::move(msg);
    fill_reasons(d, client, readiness);
    if (d.job_id) d.retry_after_s = 5;
    return d;
  };
  const bool may_prepare = p.preparation != Preparation::kManual || alias_allows_prepare;
  switch (st) {
    case ReadyState::kReady:
    case ReadyState::kBusy: break;
    case ReadyState::kPreparing:
      if (client.limits.prepare_wait_ms == 0) return not_ready(Code::kModelNotReady, "the model is being prepared");
      d.wait_for_prepare = true;
      break;
    case ReadyState::kLoadable:
      if (!may_prepare) return not_ready(Code::kModelNotReady, "the model is not prepared and preparation is manual");
      d.start_prepare = true;
      if (client.limits.prepare_wait_ms == 0) return not_ready(Code::kModelNotReady, "the model is being prepared");
      d.wait_for_prepare = true;
      break;
    default:
      return not_ready(readiness_reject_code(readiness, res), "the model cannot be used right now");
  }

  // 6. Resource policy.
  if (!res.workers_allowed) {
    d.start_prepare = false;
    d.wait_for_prepare = false;
    return not_ready(res.memory_blocked ? Code::kInsufficientMemory : Code::kWorkersUnavailable,
                     res.reason.empty() ? "a required Worker is not available" : res.reason);
  }

  // 7. Queue capacity. An immediate start needs a free slot, an empty queue (FIFO) and a prepared plan.
  const bool immediate = q.slot_free && q.depth == 0 && !d.wait_for_prepare && is_prepared(st);
  if (immediate) {
    d.kind = Decision::Kind::kAdmit;
    return d;
  }
  if (q.depth >= q.max_queue_depth) {
    Decision r = reject(Code::kQueueFull, "the request queue is full", p, st);
    r.retry_after_s = 5;
    return r;
  }
  d.kind = Decision::Kind::kQueue;
  return d;
}

}  // namespace clusterlm::scheduler
