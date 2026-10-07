#include "clusterlm/domain/reference_domain.hpp"

#include <algorithm>
#include <optional>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/objects/tensor_codec.hpp"
#include "clusterlm/windows/window_ledger.hpp"
#include "reference_math.hpp"

namespace clusterlm::domain {

namespace {

using namespace objects;
namespace rm = refmath;

constexpr std::uint64_t kFloat = sizeof(float);

Status precondition(std::string m) { return make_error(ErrorCode::kFailedPrecondition, std::move(m)); }
Status invalid(std::string m) { return make_error(ErrorCode::kInvalidArgument, std::move(m)); }

std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

struct Expert {
  std::vector<float> gate, up, down;  // [ff][H], [ff][H], [H][ff] after exact dequantization
};

struct LayerWeights {
  LayerKind kind = LayerKind::kRecurrent;
  DenseLayout lay;
  std::vector<float> dense;
  std::vector<float> sh_gate, sh_up, sh_down;
  std::vector<Expert> experts;
  std::vector<AllocationTarget> expert_target;  // per expert, for experts_cpu / experts_gpu accounting
};

// Sequence state of one session in this domain (committed state only; window state lives in Window).
struct Session {
  Epoch epoch;
  std::vector<std::vector<float>> rec;      // per local layer: H floats (recurrent layers)
  std::vector<std::vector<float>> k, v;     // per local layer: max_context * nkv*hd (attention layers)
  std::vector<std::int32_t> history;        // prefix only: committed token history (PLE n-grams)
};

// Temporary state of the single outstanding window.
struct Window {
  WindowId id;
  std::uint64_t base = 0;
  std::uint32_t q = 0;
  std::vector<std::vector<float>> snapshots;  // per local layer: q*H floats, r after each position
  std::vector<std::int32_t> tokens;           // prefix only
};

struct Scratch {
  std::vector<float> u, x, t, m, h, q, k, v, o, scores, router, a, b, z, d, y, sh;
};

class ReferenceDomainImpl final : public ReferenceDomain {
 public:
  ReferenceDomainImpl(objects::ModelManifest manifest, DomainSpec spec)
      : manifest_(std::move(manifest)), g_(manifest_.geometry), spec_(spec), layout_(BoundaryLayout::for_geometry(g_)) {}

  Status init() {
    CLM_RETURN_IF_ERROR(manifest_.validate());
    if (spec_.max_context == 0 || spec_.max_window == 0 || spec_.max_sessions == 0 || spec_.max_window > spec_.max_context)
      return invalid("domain spec needs max_context >= max_window >= 1 and max_sessions >= 1");
    const LayerRange r = spec_.layers;
    if (r.end > g_.n_layers || r.end < r.begin) return invalid("layer range outside the model");
    const bool has_ple = r.contains(g_.ple_layer);
    switch (spec_.role) {
      case StageRole::kPrefix:
        if (r.begin != 0 || !has_ple) return invalid("prefix domain must start at layer 0 and contain the PLE layer");
        break;
      case StageRole::kMiddle:
        if (r.empty()) return invalid("middle domain needs at least one layer");
        if (has_ple) return invalid("only the prefix domain may contain the PLE layer");
        break;
      case StageRole::kTail:
        if (has_ple) return invalid("only the prefix domain may contain the PLE layer");
        if (r.end != g_.n_layers) return invalid("tail domain must end at the last layer");
        break;
    }
    return Status::ok();
  }

  const DomainSpec& spec() const override { return spec_; }
  BoundaryLayout boundary() const override { return layout_; }

  void set_allocation_targets(std::unordered_map<std::string, AllocationTarget> targets) override {
    targets_ = std::move(targets);
  }
  StageTiming last_timing() const override { return timing_; }

  // ---- sizing -------------------------------------------------------------------------------------

  std::vector<std::string> required_object_names() const {
    std::vector<std::string> names;
    if (spec_.role == StageRole::kPrefix) {
      names.emplace_back(kEmbeddingObjectName);
      names.emplace_back(kPleObjectName);
    }
    for (std::uint32_t L = spec_.layers.begin; L < spec_.layers.end; ++L) {
      names.push_back(dense_object_name(L));
      if (g_.shared_expert_ff > 0) names.push_back(shared_expert_object_name(L));
      for (std::uint32_t e = 0; e < g_.n_experts; ++e) names.push_back(expert_object_name(L, e));
    }
    if (spec_.role == StageRole::kTail) names.emplace_back(kHeadObjectName);
    return names;
  }

  AllocationTarget target_of(const std::string& name) const {
    auto it = targets_.find(name);
    return it == targets_.end() ? AllocationTarget::kCpuResident : it->second;
  }

  std::uint64_t per_session_state_bytes() const {
    std::uint64_t n = 0;
    const std::uint64_t kvd = std::uint64_t{g_.n_kv_heads} * g_.head_dim;
    for (std::uint32_t L = spec_.layers.begin; L < spec_.layers.end; ++L)
      n += g_.layer_kinds[L] == LayerKind::kRecurrent ? std::uint64_t{g_.hidden_size} * kFloat
                                                      : 2 * kvd * spec_.max_context * kFloat;
    if (spec_.role == StageRole::kPrefix) n += std::uint64_t{spec_.max_context} * sizeof(std::int32_t);
    return n;
  }

  // Recurrent-state snapshots of one window and (prefix) its token buffer, per session.
  std::uint64_t window_bytes_per_session() const {
    std::uint64_t rec_layers = 0;
    for (std::uint32_t L = spec_.layers.begin; L < spec_.layers.end; ++L)
      rec_layers += g_.layer_kinds[L] == LayerKind::kRecurrent ? 1 : 0;
    // KV written by a window lives in the context reservation; only snapshots and tokens are extra.
    return rec_layers * spec_.max_window * g_.hidden_size * kFloat +
           (spec_.role == StageRole::kPrefix ? std::uint64_t{spec_.max_window} * sizeof(std::int32_t) : 0);
  }

  std::uint64_t scratch_float_count() const {
    const std::uint64_t H = g_.hidden_size, qd = std::uint64_t{g_.n_heads} * g_.head_dim,
                        kvd = std::uint64_t{g_.n_kv_heads} * g_.head_dim;
    const std::uint64_t ffmax = std::max(g_.expert_ff, g_.shared_expert_ff);
    return 8 * H /*u x t m h d y sh*/ + 2 * qd /*q o*/ + 2 * kvd /*k v*/ + spec_.max_context /*scores*/ +
           g_.n_experts /*router*/ + 3 * ffmax /*a b z*/;
  }

  Result<DomainRequirements> describe_requirements() const override {
    DomainRequirements req;
    req.required_objects = required_object_names();
    for (const std::string& n : req.required_objects) {
      const ManifestObject* o = manifest_.find(n);
      if (o == nullptr) return make_error(ErrorCode::kNotFound, "manifest lacks required object '" + n + "'");
      (target_of(n) == AllocationTarget::kGpuResident ? req.gpu_weight_bytes : req.cpu_weight_bytes) += o->byte_size;
    }
    req.state_bytes = per_session_state_bytes() * spec_.max_sessions;
    req.window_bytes = window_bytes_per_session();
    req.scratch_bytes = scratch_float_count() * kFloat;
    req.staging_bytes = 0;  // reference weights are decoded straight from the resolver; no staging ring
    return req;
  }

  // ---- lifecycle ----------------------------------------------------------------------------------

  Status prepare(const ObjectResolver& resolver) override {
    if (prepared_) return precondition("domain already prepared; release() first");
    Status st = prepare_impl(resolver);
    if (!st.is_ok()) (void)release();  // never leave a half-bound domain behind
    return st;
  }

  Status prepare_impl(const ObjectResolver& resolver) {
    resident_bytes_ = 0;
    layers_.clear();
    const std::size_t H = g_.hidden_size, ff = g_.expert_ff, sff = g_.shared_expert_ff;

    auto fetch = [&](const std::string& name) -> Result<ProvisionedObject> {
      CLM_ASSIGN_OR_RETURN(ProvisionedObject p, resolver.resolve(name));
      if (p.entry == nullptr || p.entry->name != name) return make_error(ErrorCode::kInternal, "resolver returned a mismatched entry");
      if (p.bytes.size() != p.entry->byte_size)
        return make_error(ErrorCode::kDataLoss, "object '" + name + "' has an unexpected size");
      resident_bytes_ += p.entry->byte_size;
      return p;
    };
    auto decode_f32 = [&](const ProvisionedObject& p, std::size_t offset_floats, std::size_t n, std::vector<float>& out) {
      // The object's byte_size comes from the manifest and is only checked for self-consistency; the tensor span
      // must be re-checked against the geometry-derived size before it is sliced or sized from.
      if (n > p.bytes.size() / 4 || offset_floats > p.bytes.size() / 4 - n)
        return make_error(ErrorCode::kDataLoss, "object '" + p.entry->name + "' is smaller than its tensor layout");
      out.resize(n);
      return decode_tensor(p.entry->representation.quant_type, p.bytes.subspan(offset_floats * 4, n * 4), out);
    };

    if (spec_.role == StageRole::kPrefix) {
      CLM_ASSIGN_OR_RETURN(ProvisionedObject emb, fetch(std::string(kEmbeddingObjectName)));
      CLM_RETURN_IF_ERROR(decode_f32(emb, 0, std::size_t{g_.vocab_size} * H, embd_));
      CLM_ASSIGN_OR_RETURN(ProvisionedObject ple, fetch(std::string(kPleObjectName)));
      CLM_RETURN_IF_ERROR(decode_f32(ple, 0, std::size_t{g_.ple_rows} * H, ple_));
    }
    for (std::uint32_t L = spec_.layers.begin; L < spec_.layers.end; ++L) {
      LayerWeights lw;
      lw.kind = g_.layer_kinds[L];
      lw.lay = dense_layout(g_, lw.kind);
      CLM_ASSIGN_OR_RETURN(ProvisionedObject dense, fetch(dense_object_name(L)));
      if (dense.bytes.size() != lw.lay.total * 4) return make_error(ErrorCode::kDataLoss, "dense object has the wrong size");
      CLM_RETURN_IF_ERROR(decode_f32(dense, 0, lw.lay.total, lw.dense));
      if (sff > 0) {
        CLM_ASSIGN_OR_RETURN(ProvisionedObject sh, fetch(shared_expert_object_name(L)));
        if (sh.bytes.size() != (3 * sff * H) * 4) return make_error(ErrorCode::kDataLoss, "shared expert has the wrong size");
        CLM_RETURN_IF_ERROR(decode_f32(sh, 0, sff * H, lw.sh_gate));
        CLM_RETURN_IF_ERROR(decode_f32(sh, sff * H, sff * H, lw.sh_up));
        CLM_RETURN_IF_ERROR(decode_f32(sh, 2 * sff * H, H * sff, lw.sh_down));
      }
      lw.experts.resize(g_.n_experts);
      for (std::uint32_t e = 0; e < g_.n_experts; ++e) {
        const std::string name = expert_object_name(L, e);
        CLM_ASSIGN_OR_RETURN(ProvisionedObject ex, fetch(name));
        const std::string& qt = ex.entry->representation.quant_type;
        const std::uint64_t gb = tensor_bytes(qt, ff * H), db = tensor_bytes(qt, H * ff);
        if (gb == 0 || db == 0 || ex.bytes.size() != 2 * gb + db)
          return make_error(ErrorCode::kDataLoss, "expert object '" + name + "' has the wrong size for its representation");
        Expert& x = lw.experts[e];
        x.gate.resize(ff * H);
        x.up.resize(ff * H);
        x.down.resize(H * ff);
        CLM_RETURN_IF_ERROR(decode_tensor(qt, ex.bytes.subspan(0, gb), x.gate));
        CLM_RETURN_IF_ERROR(decode_tensor(qt, ex.bytes.subspan(gb, gb), x.up));
        CLM_RETURN_IF_ERROR(decode_tensor(qt, ex.bytes.subspan(2 * gb, db), x.down));
        lw.expert_target.push_back(target_of(name));
      }
      layers_.push_back(std::move(lw));
    }
    if (spec_.role == StageRole::kTail) {
      CLM_ASSIGN_OR_RETURN(ProvisionedObject head, fetch(std::string(kHeadObjectName)));
      if (head.bytes.size() != (H + std::size_t{g_.vocab_size} * H) * 4)
        return make_error(ErrorCode::kDataLoss, "head object has the wrong size");
      CLM_RETURN_IF_ERROR(decode_f32(head, 0, H, head_norm_));
      CLM_RETURN_IF_ERROR(decode_f32(head, H, std::size_t{g_.vocab_size} * H, head_w_));
    }

    const std::size_t qd = std::size_t{g_.n_heads} * g_.head_dim, kvd = std::size_t{g_.n_kv_heads} * g_.head_dim;
    const std::size_t ffmax = std::max(ff, sff);
    for (auto* v : {&sc_.u, &sc_.x, &sc_.t, &sc_.m, &sc_.h, &sc_.d, &sc_.y, &sc_.sh}) v->assign(H, 0.0f);
    sc_.q.assign(qd, 0.0f);
    sc_.o.assign(qd, 0.0f);
    sc_.k.assign(kvd, 0.0f);
    sc_.v.assign(kvd, 0.0f);
    sc_.scores.assign(spec_.max_context, 0.0f);
    sc_.router.assign(g_.n_experts, 0.0f);
    for (auto* v : {&sc_.a, &sc_.b, &sc_.z}) v->assign(ffmax, 0.0f);
    prepared_ = true;
    return Status::ok();
  }

  Status open_session(Epoch epoch, SessionId session) override {
    if (!prepared_) return precondition("domain not prepared");
    if (!sessions_.contains(session) && sessions_.size() >= spec_.max_sessions)
      return make_error(ErrorCode::kResourceExhausted, "max_sessions reached");
    CLM_RETURN_IF_ERROR(ledger_.open_session(epoch, session));
    Session s;
    s.epoch = epoch;
    const std::size_t kvd = std::size_t{g_.n_kv_heads} * g_.head_dim;
    for (std::uint32_t L = spec_.layers.begin; L < spec_.layers.end; ++L) {
      const bool rec = g_.layer_kinds[L] == LayerKind::kRecurrent;
      s.rec.emplace_back(rec ? g_.hidden_size : 0, 0.0f);
      s.k.emplace_back(rec ? 0 : spec_.max_context * kvd, 0.0f);
      s.v.emplace_back(rec ? 0 : spec_.max_context * kvd, 0.0f);
    }
    if (spec_.role == StageRole::kPrefix) s.history.reserve(spec_.max_context);
    sessions_[session] = std::move(s);
    return Status::ok();
  }

  // ---- execution ----------------------------------------------------------------------------------

  Result<StageActivations> run_prefix(const WindowRequest& request, std::span<const std::int32_t> tokens) override {
    if (spec_.role != StageRole::kPrefix) return precondition("run_prefix is only valid on a prefix domain");
    return execute(request, nullptr, tokens, nullptr);
  }

  Result<StageActivations> run_window(const WindowRequest& request, const StageActivations& input) override {
    if (spec_.role == StageRole::kPrefix) return precondition("a prefix domain consumes tokens, not activations");
    return execute(request, &input, {}, nullptr);
  }

  Result<Logits> run_tail(const WindowRequest& request, const StageActivations& input) override {
    if (spec_.role != StageRole::kTail) return precondition("run_tail is only valid on a tail domain");
    Logits logits;
    auto r = execute(request, &input, {}, &logits);
    if (!r.is_ok()) return r.status();
    return logits;
  }

  Result<CommitAck> commit_window(const CommitRequest& request) override {
    if (!prepared_) return precondition("domain not prepared");
    CLM_ASSIGN_OR_RETURN(windows::WindowLedger::CommitDecision decision, ledger_.begin_commit(request));
    if (decision.kind == windows::WindowLedger::CommitDecision::Kind::kReplay) return decision.ack;
    auto it = sessions_.find(request.session);
    if (it == sessions_.end() || !windows_.contains(request.session) || windows_[request.session].id != request.window)
      return make_error(ErrorCode::kInternal, "ledger and domain window state disagree");
    Session& s = it->second;
    Window& w = windows_[request.session];
    const std::uint32_t n = request.accepted;
    const std::size_t H = g_.hidden_size;
    // Keep exactly the first n positions: r = snapshot[n-1]; KV beyond base+n becomes unreachable because the
    // committed position (ledger) bounds every later attention read and the slots are rewritten before use.
    for (std::size_t li = 0; li < s.rec.size(); ++li)
      if (!s.rec[li].empty()) std::copy_n(w.snapshots[li].begin() + static_cast<std::ptrdiff_t>((n - 1) * H), H, s.rec[li].begin());
    if (spec_.role == StageRole::kPrefix) s.history.insert(s.history.end(), w.tokens.begin(), w.tokens.begin() + n);
    windows_.erase(request.session);
    ++metrics_.windows_committed;
    metrics_.positions_committed += n;
    return ledger_.finish_commit(request);
  }

  Result<WindowAbortAck> abort_window(Epoch epoch, SessionId session, WindowId window) override {
    CLM_ASSIGN_OR_RETURN(WindowAbortAck ack, ledger_.abort_window(epoch, session, window));
    // Window state is temporary by construction (snapshots, tentative KV slots past the committed position,
    // tentative PLE history), so discarding it restores the committed session exactly.
    auto it = windows_.find(session);
    if (it != windows_.end() && it->second.id == window) {
      windows_.erase(it);
      ++metrics_.windows_aborted;
    }
    return ack;
  }

  Status abort_session(Epoch epoch, SessionId session) override {
    CLM_RETURN_IF_ERROR(ledger_.abort_session(epoch, session));
    if (windows_.erase(session) != 0) ++metrics_.windows_aborted;
    sessions_.erase(session);
    return Status::ok();
  }

  Status release() override {
    ledger_.clear();
    sessions_.clear();
    windows_.clear();
    layers_.clear();
    layers_.shrink_to_fit();
    for (auto* v : {&embd_, &ple_, &head_norm_, &head_w_}) {
      v->clear();
      v->shrink_to_fit();
    }
    sc_ = Scratch{};
    resident_bytes_ = 0;
    prepared_ = false;
    return Status::ok();
  }

  Status enable_routing_aggregation(bool on) override {
    aggregate_routing_ = on;
    routing_ = RoutingAggregate{};
    routing_.first_layer = spec_.layers.begin;
    routing_.counts.assign(spec_.layers.size(), std::vector<std::uint64_t>(g_.n_experts, 0));
    return Status::ok();
  }
  Result<RoutingAggregate> routing_aggregate() const override {
    if (!aggregate_routing_) return make_error(ErrorCode::kFailedPrecondition, "routing aggregation is not enabled");
    return routing_;
  }

  DomainMetrics read_metrics() const override {
    DomainMetrics m = metrics_;
    m.resident_weight_bytes = resident_bytes_;
    m.state_bytes = per_session_state_bytes() * sessions_.size();
    m.window_bytes = window_bytes_per_session() * sessions_.size();
    m.stale_rejections = ledger_.stale_rejections();
    return m;
  }

 private:
  Result<StageActivations> execute(const WindowRequest& req, const StageActivations* input,
                                   std::span<const std::int32_t> tokens, Logits* logits_out) {
    if (!prepared_) return precondition("domain not prepared");
    const Status admitted = ledger_.begin_window(req);
    if (!admitted.is_ok()) return admitted;
    auto reject = [&](Status st) -> Result<StageActivations> {
      ledger_.fail_window(req.session);
      return st;
    };
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
        if (t < 0 || static_cast<std::uint32_t>(t) >= g_.vocab_size) return reject(invalid("token outside the vocabulary"));
    }
    Session& s = sessions_.at(req.session);

    Stopwatch sw;
    timing_ = StageTiming{};
    Window w;
    w.id = req.window;
    w.base = req.base_position;
    w.q = req.positions;
    const std::size_t H = g_.hidden_size, fpp = layout_.floats_per_position();

    StageActivations out;
    out.layout = layout_;
    out.first_position = req.base_position;
    out.positions = req.positions;
    if (input != nullptr) {
      out.data = input->data;
    } else {
      w.tokens.assign(tokens.begin(), tokens.end());
      out.data.assign(std::size_t{req.positions} * fpp, 0.0f);
      for (std::uint32_t i = 0; i < req.positions; ++i) {
        float* rec = out.data.data() + i * fpp;
        const float* e = embd_.data() + static_cast<std::size_t>(tokens[i]) * H;
        for (std::uint32_t j = 0; j < g_.residual_streams; ++j) std::copy_n(e, H, rec + j * H);  // P and g stay 0
      }
    }

    w.snapshots.resize(layers_.size());
    for (std::size_t li = 0; li < layers_.size(); ++li) {
      if (layers_[li].kind == LayerKind::kRecurrent) w.snapshots[li].assign(std::size_t{req.positions} * H, 0.0f);
      for (std::uint32_t i = 0; i < req.positions; ++i)
        layer_position(li, spec_.layers.begin + static_cast<std::uint32_t>(li), s, w, i, out.data.data() + i * fpp);
    }

    if (logits_out != nullptr) {
      logits_out->positions = req.positions;
      logits_out->vocab = g_.vocab_size;
      logits_out->data.assign(std::size_t{req.positions} * g_.vocab_size, 0.0f);
      for (std::uint32_t i = 0; i < req.positions; ++i)
        head_position(out.data.data() + i * fpp, logits_out->data.data() + std::size_t{i} * g_.vocab_size);
    }

    windows_[req.session] = std::move(w);
    ++metrics_.windows_run;
    timing_.compute_ns = sw.elapsed_ns();
    metrics_.compute_ns_total += timing_.compute_ns;
    return out;
  }

  std::int32_t token_at(const Session& s, const Window& w, std::uint64_t pos) const {
    return pos < w.base ? s.history[pos] : w.tokens[pos - w.base];
  }

  // The reference math for one layer at one position (see the module spec; steps 1-6).
  void layer_position(std::size_t li, std::uint32_t L, Session& s, Window& w, std::uint32_t i, float* record) {
    const LayerWeights& lw = layers_[li];
    const std::size_t H = g_.hidden_size, hc = g_.residual_streams;
    float* S = record;
    float* P = record + layout_.pending_offset();
    float* g = record + layout_.injection_offset();
    const std::uint64_t p = w.base + i;
    const float* dn = lw.dense.data();
    Scratch& sc = sc_;

    rm::fold(S, P, g, hc, H);
    rm::mean_streams(S, hc, H, sc.u.data());
    rm::rmsnorm(sc.u.data(), dn + lw.lay.norm, H, sc.x.data());

    if (L == g_.ple_layer) {  // prefix only: n-gram over the committed history + this window's tokens
      std::uint64_t hsh = 0x243F6A8885A308D3ull;
      for (std::uint32_t k = g_.ple_ngram; k-- > 0;) {
        const std::int32_t tok = p >= k ? token_at(s, w, p - k) : -1;
        hsh = mix64(hsh ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(tok)));
      }
      const float* row = ple_.data() + static_cast<std::size_t>(hsh % g_.ple_rows) * H;
      for (std::size_t c = 0; c < H; ++c) sc.x[c] += 0.1f * row[c];
    }

    if (lw.kind == LayerKind::kRecurrent) {
      std::vector<float>& r = w.snapshots[li];
      float* rcur = r.data() + std::size_t{i} * H;
      const float* rprev = i == 0 ? s.rec[li].data() : r.data() + std::size_t{i - 1} * H;
      rm::matvec(dn + lw.lay.w_in, H, H, sc.x.data(), sc.t.data());  // W_in x (t reused as the pre-activation)
      const float* decay = dn + lw.lay.decay;
      for (std::size_t c = 0; c < H; ++c) rcur[c] = decay[c] * rprev[c] + sc.t[c];
      for (std::size_t c = 0; c < H; ++c) sc.t[c] = std::tanh(rcur[c]);
      rm::matvec(dn + lw.lay.w_out, H, H, sc.t.data(), sc.m.data());
    } else {
      const std::size_t nh = g_.n_heads, nkv = g_.n_kv_heads, hd = g_.head_dim, kvd = nkv * hd, qd = nh * hd;
      rm::matvec(dn + lw.lay.wq, qd, H, sc.x.data(), sc.q.data());
      rm::matvec(dn + lw.lay.wk, kvd, H, sc.x.data(), sc.k.data());
      rm::matvec(dn + lw.lay.wv, kvd, H, sc.x.data(), sc.v.data());
      std::copy_n(sc.k.begin(), kvd, s.k[li].begin() + static_cast<std::ptrdiff_t>(p * kvd));
      std::copy_n(sc.v.begin(), kvd, s.v[li].begin() + static_cast<std::ptrdiff_t>(p * kvd));
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      const std::size_t group = nh / nkv;
      for (std::size_t head = 0; head < nh; ++head) {
        const std::size_t kvh = head / group;
        const float* qh = sc.q.data() + head * hd;
        float mx = 0.0f;
        for (std::uint64_t t = 0; t <= p; ++t) {
          const float sco = rm::dot(qh, s.k[li].data() + t * kvd + kvh * hd, hd) * scale;
          sc.scores[t] = sco;
          if (t == 0 || sco > mx) mx = sco;
        }
        float sum = 0.0f;
        for (std::uint64_t t = 0; t <= p; ++t) {
          sc.scores[t] = std::exp(sc.scores[t] - mx);
          sum += sc.scores[t];
        }
        float* oh = sc.o.data() + head * hd;
        for (std::size_t d = 0; d < hd; ++d) oh[d] = 0.0f;
        for (std::uint64_t t = 0; t <= p; ++t) {
          const float wt = sc.scores[t] / sum;
          const float* vt = s.v[li].data() + t * kvd + kvh * hd;
          for (std::size_t d = 0; d < hd; ++d) oh[d] += wt * vt[d];
        }
      }
      rm::matvec(dn + lw.lay.wo, H, qd, sc.o.data(), sc.m.data());
    }

    for (std::size_t c = 0; c < H; ++c) sc.h[c] = sc.x[c] + sc.m[c];

    // Router: top-k with ties to the lower expert index, softmax over the selected logits only.
    const std::size_t E = g_.n_experts, K = g_.n_active_experts;
    rm::matvec(dn + lw.lay.router, E, H, sc.h.data(), sc.router.data());
    std::vector<std::uint32_t>& sel = sel_;
    sel.clear();
    std::vector<char>& used = used_;
    used.assign(E, 0);
    for (std::size_t k = 0; k < K; ++k) {
      std::size_t best = E;
      for (std::size_t e = 0; e < E; ++e)
        if (!used[e] && (best == E || sc.router[e] > sc.router[best])) best = e;
      used[best] = 1;
      sel.push_back(static_cast<std::uint32_t>(best));
    }
    std::sort(sel.begin(), sel.end());
    float mx = sc.router[sel[0]];
    for (std::uint32_t e : sel) mx = std::max(mx, sc.router[e]);
    weights_.assign(K, 0.0f);
    float wsum = 0.0f;
    for (std::size_t k = 0; k < K; ++k) {
      weights_[k] = std::exp(sc.router[sel[k]] - mx);
      wsum += weights_[k];
    }
    for (std::size_t k = 0; k < K; ++k) weights_[k] /= wsum;

    const std::uint64_t t0 = monotonic_ns();
    for (std::size_t c = 0; c < H; ++c) sc.y[c] = 0.0f;
    const std::size_t ff = g_.expert_ff;
    for (std::size_t k = 0; k < K; ++k) {
      const Expert& ex = lw.experts[sel[k]];
      swiglu(ex.gate.data(), ex.up.data(), ex.down.data(), ff, sc.h.data());
      for (std::size_t c = 0; c < H; ++c) sc.y[c] += weights_[k] * sc.d[c];
      ++timing_.experts_selected;
      if (aggregate_routing_) ++routing_.counts[li][sel[k]];
      (lw.expert_target[sel[k]] == AllocationTarget::kGpuResident ? timing_.experts_gpu : timing_.experts_cpu)++;
    }
    if (g_.shared_expert_ff > 0) {
      swiglu(lw.sh_gate.data(), lw.sh_up.data(), lw.sh_down.data(), g_.shared_expert_ff, sc.h.data());
      for (std::size_t c = 0; c < H; ++c) sc.y[c] += sc.d[c];
    }
    timing_.cpu_expert_ns += monotonic_ns() - t0;

    if (aggregate_routing_ && li == 0) ++routing_.positions;
    for (std::size_t c = 0; c < H; ++c) P[c] = sc.m[c] + sc.y[c];
    for (std::size_t j = 0; j < hc; ++j) g[j] = rm::sigmoid(rm::dot(dn + lw.lay.inj + j * H, sc.x.data(), H));
  }

  // d = down * (silu(gate*h) ⊙ (up*h)); result in scratch.d
  void swiglu(const float* gate, const float* up, const float* down, std::size_t ff, const float* h) {
    const std::size_t H = g_.hidden_size;
    rm::matvec(gate, ff, H, h, sc_.a.data());
    rm::matvec(up, ff, H, h, sc_.b.data());
    for (std::size_t c = 0; c < ff; ++c) sc_.z[c] = rm::silu(sc_.a[c]) * sc_.b[c];
    rm::matvec(down, H, ff, sc_.z.data(), sc_.d.data());
  }

  void head_position(const float* record, float* logits) {
    const std::size_t H = g_.hidden_size, hc = g_.residual_streams;
    std::vector<float>& S = head_s_;
    S.assign(record, record + hc * H);
    rm::fold(S.data(), record + layout_.pending_offset(), record + layout_.injection_offset(), hc, H);
    rm::mean_streams(S.data(), hc, H, sc_.u.data());
    rm::rmsnorm(sc_.u.data(), head_norm_.data(), H, sc_.x.data());
    rm::matvec(head_w_.data(), g_.vocab_size, H, sc_.x.data(), logits);
  }

  objects::ModelManifest manifest_;
  const objects::ModelGeometry& g_;
  DomainSpec spec_;
  BoundaryLayout layout_;
  std::unordered_map<std::string, AllocationTarget> targets_;

  bool prepared_ = false;
  std::uint64_t resident_bytes_ = 0;
  std::vector<LayerWeights> layers_;
  std::vector<float> embd_, ple_, head_norm_, head_w_;
  Scratch sc_;
  std::vector<std::uint32_t> sel_;
  std::vector<char> used_;
  std::vector<float> weights_, head_s_;

  windows::WindowLedger ledger_;
  std::unordered_map<SessionId, Session> sessions_;
  std::unordered_map<SessionId, Window> windows_;
  DomainMetrics metrics_;
  StageTiming timing_;
  bool aggregate_routing_ = false;
  RoutingAggregate routing_;
};

}  // namespace

Result<std::unique_ptr<ReferenceDomain>> ReferenceDomain::create(const objects::ModelManifest& manifest,
                                                                 const DomainSpec& spec) {
  auto d = std::make_unique<ReferenceDomainImpl>(manifest, spec);
  CLM_RETURN_IF_ERROR(d->init());
  return std::unique_ptr<ReferenceDomain>(std::move(d));
}

}  // namespace clusterlm::domain
