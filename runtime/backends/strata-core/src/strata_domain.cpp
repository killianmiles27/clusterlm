#include "clusterlm/backends/strata/strata_domain.hpp"

#include <algorithm>

#include "clusterlm/backends/strata_handoff.hpp"
#include "clusterlm/common/clock.hpp"

namespace clusterlm::backends::strata {
namespace {

using domain::StageRole;

Status precondition(std::string m) { return make_error(ErrorCode::kFailedPrecondition, std::move(m)); }
Status invalid(std::string m) { return make_error(ErrorCode::kInvalidArgument, std::move(m)); }

void add_timing(domain::StageTiming& total, const domain::StageTiming& t) {
  total.compute_ns += t.compute_ns;
  total.cpu_expert_ns += t.cpu_expert_ns;
  total.gpu_ns += t.gpu_ns;
  total.experts_selected += t.experts_selected;
  total.experts_cpu += t.experts_cpu;
  total.experts_gpu += t.experts_gpu;
}

}  // namespace

std::uint32_t token_free_first_layer(const objects::ModelGeometry& g) { return std::max<std::uint32_t>(2, g.ple_layer + 1); }

const std::int32_t* verifier_tokens(const EngineCaps& caps, std::span<const std::int32_t> tokens) {
  return caps.needs_tokens && !tokens.empty() ? tokens.data() : nullptr;
}

StrataDomain::StrataDomain(objects::ModelManifest manifest, domain::DomainSpec spec, std::unique_ptr<StrataEngine> engine,
                           StrataDomainOptions options)
    : manifest_(std::move(manifest)),
      spec_(spec),
      layout_(domain::BoundaryLayout::for_geometry(manifest_.geometry)),
      engine_(std::move(engine)),
      options_(options) {}

StrataDomain::~StrataDomain() {
  if (prepared_) (void)release();
}

Result<std::unique_ptr<StrataDomain>> StrataDomain::create(const objects::ModelManifest& manifest,
                                                           const domain::DomainSpec& spec,
                                                           std::unique_ptr<StrataEngine> engine,
                                                           StrataDomainOptions options) {
  if (!engine) return invalid("strata domain: no engine");
  std::unique_ptr<StrataDomain> d(new StrataDomain(manifest, spec, std::move(engine), options));
  CLM_RETURN_IF_ERROR(d->init());
  return d;
}

Status StrataDomain::init() {
  CLM_RETURN_IF_ERROR(manifest_.validate());
  const objects::ModelGeometry& g = manifest_.geometry;
  if (spec_.max_context == 0 || spec_.max_window == 0 || spec_.max_sessions == 0 || spec_.max_window > spec_.max_context)
    return invalid("strata domain: spec needs max_context >= max_window >= 1 and max_sessions >= 1");
  const objects::LayerRange r = spec_.layers;
  if (r.end > g.n_layers || r.end < r.begin) return invalid("strata domain: layer range outside the model");
  const std::uint32_t first_free = token_free_first_layer(g);
  switch (spec_.role) {
    case StageRole::kPrefix:
      if (r.begin != 0 || r.end < first_free)
        return invalid("strata domain: the prefix must start at layer 0 and hold layers 0.." + std::to_string(first_free - 1) +
                       " (embedding and the token-dependent PLE block)");
      if (r.end >= g.n_layers) return invalid("strata domain: the prefix cannot hold the last layer (the head is the tail's)");
      break;
    case StageRole::kMiddle:
      if (r.empty()) return invalid("strata domain: a middle domain needs at least one layer");
      [[fallthrough]];
    case StageRole::kTail:
      if (r.begin < first_free)
        return invalid("strata domain: a token-free domain cannot own layer " + std::to_string(r.begin) +
                       " (the PLE block needs token IDs; it stays on the Father prefix)");
      if (spec_.role == StageRole::kTail && r.end != g.n_layers)
        return invalid("strata domain: the tail must end at the last layer");
      if (spec_.role == StageRole::kMiddle && r.end >= g.n_layers)
        return invalid("strata domain: a middle domain cannot hold the last layer (the head follows it on the tail)");
      break;
  }
  caps_ = engine_->caps();
  if (caps_.max_window == 0) return invalid("strata domain: engine reports a zero window");
  if (caps_.needs_tokens != (spec_.role == StageRole::kPrefix))
    return make_error(ErrorCode::kInternal, "strata domain: engine token requirement disagrees with the domain role");
  if (caps_.has_head != (spec_.role == StageRole::kTail))
    return make_error(ErrorCode::kInternal, "strata domain: engine head disagrees with the domain role");
  if (caps_.has_head && caps_.vocab != g.vocab_size)
    return make_error(ErrorCode::kVersionMismatch, "strata domain: engine vocabulary differs from the manifest");
  if (spec_.max_local_batch > spec_.max_window) return invalid("strata domain: max_local_batch exceeds max_window");
  return Status::ok();
}

std::uint32_t StrataDomain::local_batch() const {
  const std::uint32_t want = spec_.max_local_batch != 0 ? spec_.max_local_batch : spec_.max_window;
  return std::max<std::uint32_t>(1, std::min(want, caps_.max_window));
}

Result<domain::DomainRequirements> StrataDomain::describe_requirements() const {
  const std::vector<std::string> names = required_objects(manifest_.geometry, spec_, options_.objects);
  for (const std::string& n : names)
    if (manifest_.find(n) == nullptr) return make_error(ErrorCode::kNotFound, "manifest lacks required object '" + n + "'");
  CLM_ASSIGN_OR_RETURN(domain::DomainRequirements req, engine_->requirements(names));
  req.required_objects = names;
  return req;
}

Status StrataDomain::prepare(const objects::ObjectResolver& resolver) {
  if (prepared_) return precondition("strata domain: already prepared; release() first");
  Status st = engine_->prepare(resolver);
  if (!st.is_ok()) {
    (void)engine_->release();  // never leave a half-bound domain behind
    return st;
  }
  prepared_ = true;
  return Status::ok();
}

Status StrataDomain::open_session(Epoch epoch, SessionId session) {
  if (!prepared_) return precondition("strata domain: not prepared");
  if (!sessions_.contains(session) && sessions_.size() >= spec_.max_sessions)
    return make_error(ErrorCode::kResourceExhausted, "strata domain: max_sessions reached");
  CLM_RETURN_IF_ERROR(ledger_.open_session(epoch, session));
  const Status st = engine_->open_session(session);  // a re-opened session starts from zeroed state
  if (!st.is_ok()) {
    (void)ledger_.abort_session(epoch, session);
    sessions_.erase(session);
    return st;
  }
  sessions_[session] = Session{epoch};
  windows_.erase(session);
  return Status::ok();
}

Result<domain::StageActivations> StrataDomain::run_prefix(const domain::WindowRequest& request,
                                                          std::span<const std::int32_t> tokens) {
  if (spec_.role != StageRole::kPrefix) return precondition("run_prefix is only valid on a prefix domain");
  return execute(request, nullptr, tokens, nullptr);
}

Result<domain::StageActivations> StrataDomain::run_window(const domain::WindowRequest& request,
                                                          const domain::StageActivations& input) {
  if (spec_.role == StageRole::kPrefix) return precondition("a prefix domain consumes tokens, not activations");
  if (spec_.role == StageRole::kTail)
    return precondition("a Strata tail runs its layers and the head as one Verifier window: use run_tail");
  return execute(request, &input, {}, nullptr);
}

Result<domain::Logits> StrataDomain::run_tail(const domain::WindowRequest& request, const domain::StageActivations& input) {
  if (spec_.role != StageRole::kTail) return precondition("run_tail is only valid on a tail domain");
  domain::Logits logits;
  auto r = execute(request, &input, {}, &logits);
  if (!r.is_ok()) return r.status();
  return logits;
}

void StrataDomain::invalidate(SessionId session) {
  auto it = sessions_.find(session);
  if (it != sessions_.end()) (void)ledger_.abort_session(it->second.epoch, session);
  (void)engine_->close_session(session);
  sessions_.erase(session);
  if (windows_.erase(session) != 0) ++metrics_.windows_aborted;
}

Result<domain::StageActivations> StrataDomain::execute(const domain::WindowRequest& req, const domain::StageActivations* input,
                                                       std::span<const std::int32_t> tokens, domain::Logits* logits) {
  if (!prepared_) return precondition("strata domain: not prepared");
  CLM_RETURN_IF_ERROR(ledger_.begin_window(req));
  auto reject = [&](Status st) -> Result<domain::StageActivations> {
    ledger_.fail_window(req.session);
    return st;
  };
  const objects::ModelGeometry& g = manifest_.geometry;
  if (req.positions > spec_.max_window) return reject(make_error(ErrorCode::kResourceExhausted, "window exceeds max_window"));
  if (req.base_position + req.positions > spec_.max_context)
    return reject(make_error(ErrorCode::kResourceExhausted, "window exceeds max_context"));
  if (input != nullptr) {
    const Status v = input->validate();
    if (!v.is_ok()) return reject(v);
    if (input->layout != layout_) return reject(make_error(ErrorCode::kVersionMismatch, "activation layout mismatch"));
    if (input->positions != req.positions || input->first_position != req.base_position)
      return reject(invalid("activations do not match the window request"));
  } else {
    if (tokens.size() != req.positions) return reject(invalid("tokens.size() != request.positions"));
    for (std::int32_t t : tokens)
      if (t < 0 || static_cast<std::uint32_t>(t) >= g.vocab_size) return reject(invalid("token outside the vocabulary"));
  }

  Stopwatch sw;
  timing_ = domain::StageTiming{};
  const std::uint32_t q = req.positions, B = local_batch();
  const std::size_t fpp = layout_.floats_per_position();
  domain::StageActivations out;
  out.layout = layout_;
  out.first_position = req.base_position;
  out.positions = q;
  const bool produce_activations = !caps_.has_head;
  if (produce_activations) out.data.assign(std::size_t{q} * fpp, 0.0f);
  if (logits != nullptr) {
    logits->positions = q;
    logits->vocab = caps_.vocab;
    logits->data.assign(std::size_t{q} * caps_.vocab, 0.0f);
  }
  // Field-major staging for one engine window: Strata's hand-off layout depends on the window length.
  std::vector<float> hin(input != nullptr ? std::size_t{B} * fpp : 0), hout(produce_activations ? std::size_t{B} * fpp : 0);

  Window w{req.window, q, 0};
  for (std::uint32_t c0 = 0; c0 < q; c0 += B) {
    const std::uint32_t n = std::min(B, q - c0);
    const std::size_t floats = std::size_t{n} * fpp;
    EngineWindow ew;
    ew.session = req.session;
    ew.pos0 = req.base_position + c0;
    ew.positions = n;
    if (caps_.needs_tokens) ew.tokens = tokens.subspan(c0, n);
    if (input != nullptr) {
      const Status cv = wire_to_strata(layout_, n, std::span<const float>(input->data).subspan(std::size_t{c0} * fpp, floats),
                                       std::span<float>(hin).first(floats));
      if (!cv.is_ok()) return reject(cv);
      ew.handoff_in = std::span<const float>(hin).first(floats);
    }
    if (produce_activations) ew.handoff_out = std::span<float>(hout).first(floats);
    if (logits != nullptr)
      ew.logits_out = std::span<float>(logits->data).subspan(std::size_t{c0} * caps_.vocab, std::size_t{n} * caps_.vocab);

    domain::StageTiming t;
    Status st = engine_->run(ew, t);
    add_timing(timing_, t);
    if (!st.is_ok()) {
      if (w.provisional > 0) {  // earlier sub-batches are committed in the engine: the ledger's state is gone
        invalidate(req.session);
        return make_error(ErrorCode::kAborted, "strata domain: sub-batch at position " + std::to_string(ew.pos0) +
                                                   " failed after earlier sub-batches were committed; the session was "
                                                   "invalidated (" + st.message() + ")");
      }
      if (!engine_->abort(req.session).is_ok()) {
        invalidate(req.session);
        return make_error(ErrorCode::kAborted, "strata domain: window failed and could not be rolled back; session "
                                                   "invalidated (" + st.message() + ")");
      }
      return reject(st);
    }
    if (produce_activations) {
      const Status cv = strata_to_wire(layout_, n, std::span<const float>(hout).first(floats),
                                       std::span<float>(out.data).subspan(std::size_t{c0} * fpp, floats));
      if (!cv.is_ok()) {
        (void)engine_->abort(req.session);
        invalidate(req.session);
        return cv;
      }
    }
    if (c0 + n < q) {
      st = engine_->commit(req.session, n);  // provisional: the next sub-batch must run on top of this one
      if (!st.is_ok()) {
        invalidate(req.session);
        return make_error(ErrorCode::kAborted, "strata domain: provisional sub-batch commit failed; session invalidated (" +
                                                   st.message() + ")");
      }
      w.provisional += n;
    }
  }
  windows_[req.session] = w;
  ++metrics_.windows_run;
  timing_.compute_ns = sw.elapsed_ns();
  metrics_.compute_ns_total += timing_.compute_ns;
  return out;
}

Result<domain::CommitAck> StrataDomain::commit_window(const domain::CommitRequest& request) {
  if (!prepared_) return precondition("strata domain: not prepared");
  CLM_ASSIGN_OR_RETURN(windows::WindowLedger::CommitDecision decision, ledger_.begin_commit(request));
  if (decision.kind == windows::WindowLedger::CommitDecision::Kind::kReplay) return decision.ack;  // never re-commit
  auto it = windows_.find(request.session);
  if (it == windows_.end() || it->second.id != request.window)
    return make_error(ErrorCode::kInternal, "ledger and domain window state disagree");
  const Window w = it->second;
  if (w.provisional > 0 && request.accepted != w.positions) {
    invalidate(request.session);
    return make_error(ErrorCode::kAborted,
                      "strata domain: window " + request.window.str() + " ran in local sub-batches of " +
                          std::to_string(local_batch()) + " and can only be committed whole (" +
                          std::to_string(w.positions) + " positions, " + std::to_string(request.accepted) +
                          " accepted); the session was invalidated and must be re-opened");
  }
  const Status st = engine_->commit(request.session, request.accepted - w.provisional);
  if (!st.is_ok()) {
    invalidate(request.session);
    return make_error(ErrorCode::kAborted, "strata domain: engine commit failed; session invalidated (" + st.message() + ")");
  }
  windows_.erase(it);
  ++metrics_.windows_committed;
  metrics_.positions_committed += request.accepted;
  return ledger_.finish_commit(request);
}

Result<domain::WindowAbortAck> StrataDomain::abort_window(Epoch epoch, SessionId session, WindowId window) {
  CLM_ASSIGN_OR_RETURN(domain::WindowAbortAck ack, ledger_.abort_window(epoch, session, window));
  auto it = windows_.find(session);
  if (it == windows_.end() || it->second.id != window) return ack;  // already committed/aborted/failed: no-op
  if (it->second.provisional > 0) {
    invalidate(session);
    return make_error(ErrorCode::kAborted, "strata domain: window " + window.str() +
                                               " ran in local sub-batches that are already committed; it cannot be "
                                               "dropped, so the session was invalidated and must be re-opened");
  }
  const Status st = engine_->abort(session);
  if (!st.is_ok()) {
    invalidate(session);
    return make_error(ErrorCode::kAborted, "strata domain: engine abort failed; session invalidated (" + st.message() + ")");
  }
  windows_.erase(it);
  ++metrics_.windows_aborted;
  return ack;
}

Status StrataDomain::abort_session(Epoch epoch, SessionId session) {
  CLM_RETURN_IF_ERROR(ledger_.abort_session(epoch, session));
  if (windows_.erase(session) != 0) ++metrics_.windows_aborted;
  if (sessions_.erase(session) != 0) CLM_RETURN_IF_ERROR(engine_->close_session(session));
  return Status::ok();
}

Status StrataDomain::release() {
  ledger_.clear();
  sessions_.clear();
  windows_.clear();
  prepared_ = false;
  return engine_->release();
}

domain::DomainMetrics StrataDomain::read_metrics() const {
  domain::DomainMetrics m = metrics_;
  const EngineCounters c = engine_->counters();
  m.resident_weight_bytes = c.resident_weight_bytes;
  m.state_bytes = c.session_state_bytes * sessions_.size();
  m.stale_rejections = ledger_.stale_rejections();
  return m;
}

}  // namespace clusterlm::backends::strata
