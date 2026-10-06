#include "clusterlm/expert_domains/father_executor.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/domain/boundary.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/objects/manifest.hpp"
#include "expert_kernel.hpp"

namespace clusterlm::expert_domains {

namespace rm = domain::refmath;
using objects::LayerKind;

namespace {

Status precondition(std::string m) { return make_error(ErrorCode::kFailedPrecondition, std::move(m)); }
Status invalid(std::string m) { return make_error(ErrorCode::kInvalidArgument, std::move(m)); }

std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

struct LayerWeights {
  LayerKind kind = LayerKind::kRecurrent;
  objects::DenseLayout lay;
  std::vector<float> dense;
  ExpertWeights shared;               // gate/up/down of the shared expert (ff = shared_expert_ff)
  std::vector<ExpertWeights> own;     // Father-owned routed experts, by local index
};

struct Session {
  std::vector<std::vector<float>> rec;   // per layer: H floats (recurrent layers)
  std::vector<std::vector<float>> k, v;  // per layer: max_context * nkv*hd (attention layers)
  std::vector<std::int32_t> history;     // committed tokens (PLE n-grams)
};

struct Window {
  WindowId id;
  std::uint64_t base = 0;
  std::uint32_t q = 0;
  std::vector<std::vector<float>> snapshots;  // per layer: q*H, recurrent state after each position
  std::vector<std::int32_t> tokens;
};

using Clock = std::chrono::steady_clock;

}  // namespace

struct FatherExecutor::Impl {
  objects::ModelGeometry g;
  FatherExecutorConfig cfg;
  domain::BoundaryLayout layout;
  std::vector<RemoteLink> remotes;
  std::vector<LayerWeights> layers;
  std::vector<float> embd, ple, head_norm, head_w;

  // Scratch for the mixer path (per position, reused).
  std::vector<float> u, x, t, m, h, q, k, v, attn_o, scores, router, head_s;
  ExpertScratch esc;
  std::vector<std::uint32_t> sel;
  std::vector<char> used;
  std::vector<float> wts;

  Session sess;
  std::optional<Window> window;
  std::uint64_t position = 0;  // committed
  std::uint64_t window_counter = 0;
  std::uint64_t correlation = 0;
  bool broken = false;
  WindowMetrics metrics;

  // ---- loading ------------------------------------------------------------------------------------

  Status load(const objects::ObjectResolver& resolver) {
    const std::size_t H = g.hidden_size, ff = g.expert_ff, sff = g.shared_expert_ff;
    auto fetch = [&](const std::string& name) -> Result<objects::ProvisionedObject> {
      CLM_ASSIGN_OR_RETURN(objects::ProvisionedObject p, resolver.resolve(name));
      if (p.entry == nullptr || p.entry->name != name || p.bytes.size() != p.entry->byte_size)
        return make_error(ErrorCode::kDataLoss, "object '" + name + "' is inconsistent");
      return p;
    };
    auto decode_f32 = [&](const objects::ProvisionedObject& p, std::size_t off, std::size_t n, std::vector<float>& out) {
      out.resize(n);
      return objects::decode_tensor(p.entry->representation.quant_type, p.bytes.subspan(off * 4, n * 4), out);
    };
    CLM_ASSIGN_OR_RETURN(auto emb, fetch(std::string(objects::kEmbeddingObjectName)));
    CLM_RETURN_IF_ERROR(decode_f32(emb, 0, std::size_t{g.vocab_size} * H, embd));
    CLM_ASSIGN_OR_RETURN(auto plep, fetch(std::string(objects::kPleObjectName)));
    CLM_RETURN_IF_ERROR(decode_f32(plep, 0, std::size_t{g.ple_rows} * H, ple));
    for (std::uint32_t L = 0; L < g.n_layers; ++L) {
      LayerWeights lw;
      lw.kind = g.layer_kinds[L];
      lw.lay = objects::dense_layout(g, lw.kind);
      CLM_ASSIGN_OR_RETURN(auto dense, fetch(objects::dense_object_name(L)));
      if (dense.bytes.size() != lw.lay.total * 4) return make_error(ErrorCode::kDataLoss, "dense object has the wrong size");
      CLM_RETURN_IF_ERROR(decode_f32(dense, 0, lw.lay.total, lw.dense));
      if (sff > 0) {
        CLM_ASSIGN_OR_RETURN(auto sh, fetch(objects::shared_expert_object_name(L)));
        if (sh.bytes.size() != 3 * sff * H * 4) return make_error(ErrorCode::kDataLoss, "shared expert has the wrong size");
        CLM_RETURN_IF_ERROR(decode_f32(sh, 0, sff * H, lw.shared.gate));
        CLM_RETURN_IF_ERROR(decode_f32(sh, sff * H, sff * H, lw.shared.up));
        CLM_RETURN_IF_ERROR(decode_f32(sh, 2 * sff * H, H * sff, lw.shared.down));
      }
      for (std::uint32_t e : cfg.assignment.owned[kFatherOwner]) {
        CLM_ASSIGN_OR_RETURN(auto ex, fetch(objects::expert_object_name(L, e)));
        ExpertWeights w;
        CLM_RETURN_IF_ERROR(load_expert_weights(ex, H, ff, w));
        lw.own.push_back(std::move(w));
      }
      layers.push_back(std::move(lw));
    }
    CLM_ASSIGN_OR_RETURN(auto head, fetch(std::string(objects::kHeadObjectName)));
    if (head.bytes.size() != (H + std::size_t{g.vocab_size} * H) * 4)
      return make_error(ErrorCode::kDataLoss, "head object has the wrong size");
    CLM_RETURN_IF_ERROR(decode_f32(head, 0, H, head_norm));
    CLM_RETURN_IF_ERROR(decode_f32(head, H, std::size_t{g.vocab_size} * H, head_w));

    const std::size_t qd = std::size_t{g.n_heads} * g.head_dim, kvd = std::size_t{g.n_kv_heads} * g.head_dim;
    for (auto* vec : {&u, &x, &t, &m, &h}) vec->assign(H, 0.0f);
    q.assign(qd, 0.0f);
    attn_o.assign(qd, 0.0f);
    k.assign(kvd, 0.0f);
    v.assign(kvd, 0.0f);
    scores.assign(cfg.max_context, 0.0f);
    router.assign(g.n_experts, 0.0f);
    esc.init(H, std::max<std::size_t>(ff, sff));
    reset_session();
    return Status::ok();
  }

  void reset_session() {
    const std::size_t kvd = std::size_t{g.n_kv_heads} * g.head_dim;
    sess = Session{};
    for (std::uint32_t L = 0; L < g.n_layers; ++L) {
      const bool rec = g.layer_kinds[L] == LayerKind::kRecurrent;
      sess.rec.emplace_back(rec ? g.hidden_size : 0, 0.0f);
      sess.k.emplace_back(rec ? 0 : cfg.max_context * kvd, 0.0f);
      sess.v.emplace_back(rec ? 0 : cfg.max_context * kvd, 0.0f);
    }
    sess.history.reserve(cfg.max_context);
    window.reset();
    position = 0;
  }

  // ---- window execution ---------------------------------------------------------------------------

  std::int32_t token_at(const Window& w, std::uint64_t pos) const {
    return pos < w.base ? sess.history[pos] : w.tokens[pos - w.base];
  }

  // Per-layer, per-window scratch for the split MoE: filled by mixer_position, consumed by the exchange.
  struct LayerState {
    std::vector<float> m_out;                  // q*H mixer outputs
    std::vector<float> h_in;                   // q*H router/expert input (x + m)
    std::vector<float> gate;                   // q*hc injection values
    std::vector<std::vector<std::uint32_t>> sel;  // per position, ascending expert ids
    std::vector<std::vector<float>> wts;          // per position, normalized weights (same order as sel)
    // routes[owner][position], and the lowest global expert id each owner got at each position
    std::vector<std::vector<std::vector<ExpertRoute>>> routes;
    std::vector<std::vector<std::uint32_t>> first_id;
    std::vector<std::vector<float>> partial;   // [owner] q*H partial sums (owner 0 = Father)
    std::vector<float> shared_out;             // q*H shared-expert outputs
  };

  // Steps 1-6 of the reference layer up to and including the router; identical arithmetic and order.
  void mixer_position(std::uint32_t L, Window& w, std::uint32_t i, float* record, LayerState& ls) {
    const LayerWeights& lw = layers[L];
    const std::size_t H = g.hidden_size, hc = g.residual_streams;
    float* S = record;
    float* P = record + layout.pending_offset();
    float* gg = record + layout.injection_offset();
    const std::uint64_t p = w.base + i;
    const float* dn = lw.dense.data();

    rm::fold(S, P, gg, hc, H);
    rm::mean_streams(S, hc, H, u.data());
    rm::rmsnorm(u.data(), dn + lw.lay.norm, H, x.data());

    if (L == g.ple_layer) {
      std::uint64_t hsh = 0x243F6A8885A308D3ull;
      for (std::uint32_t kk = g.ple_ngram; kk-- > 0;) {
        const std::int32_t tok = p >= kk ? token_at(w, p - kk) : -1;
        hsh = mix64(hsh ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(tok)));
      }
      const float* row = ple.data() + static_cast<std::size_t>(hsh % g.ple_rows) * H;
      for (std::size_t c = 0; c < H; ++c) x[c] += 0.1f * row[c];
    }

    if (lw.kind == LayerKind::kRecurrent) {
      std::vector<float>& r = w.snapshots[L];
      float* rcur = r.data() + std::size_t{i} * H;
      const float* rprev = i == 0 ? sess.rec[L].data() : r.data() + std::size_t{i - 1} * H;
      rm::matvec(dn + lw.lay.w_in, H, H, x.data(), t.data());
      const float* decay = dn + lw.lay.decay;
      for (std::size_t c = 0; c < H; ++c) rcur[c] = decay[c] * rprev[c] + t[c];
      for (std::size_t c = 0; c < H; ++c) t[c] = std::tanh(rcur[c]);
      rm::matvec(dn + lw.lay.w_out, H, H, t.data(), m.data());
    } else {
      const std::size_t nh = g.n_heads, nkv = g.n_kv_heads, hd = g.head_dim, kvd = nkv * hd, qd = nh * hd;
      rm::matvec(dn + lw.lay.wq, qd, H, x.data(), q.data());
      rm::matvec(dn + lw.lay.wk, kvd, H, x.data(), k.data());
      rm::matvec(dn + lw.lay.wv, kvd, H, x.data(), v.data());
      std::copy_n(k.begin(), kvd, sess.k[L].begin() + static_cast<std::ptrdiff_t>(p * kvd));
      std::copy_n(v.begin(), kvd, sess.v[L].begin() + static_cast<std::ptrdiff_t>(p * kvd));
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      const std::size_t group = nh / nkv;
      for (std::size_t head = 0; head < nh; ++head) {
        const std::size_t kvh = head / group;
        const float* qh = q.data() + head * hd;
        float mx = 0.0f;
        for (std::uint64_t tt = 0; tt <= p; ++tt) {
          const float sco = rm::dot(qh, sess.k[L].data() + tt * kvd + kvh * hd, hd) * scale;
          scores[tt] = sco;
          if (tt == 0 || sco > mx) mx = sco;
        }
        float sum = 0.0f;
        for (std::uint64_t tt = 0; tt <= p; ++tt) {
          scores[tt] = std::exp(scores[tt] - mx);
          sum += scores[tt];
        }
        float* oh = attn_o.data() + head * hd;
        for (std::size_t d = 0; d < hd; ++d) oh[d] = 0.0f;
        for (std::uint64_t tt = 0; tt <= p; ++tt) {
          const float wt = scores[tt] / sum;
          const float* vt = sess.v[L].data() + tt * kvd + kvh * hd;
          for (std::size_t d = 0; d < hd; ++d) oh[d] += wt * vt[d];
        }
      }
      rm::matvec(dn + lw.lay.wo, H, qd, attn_o.data(), m.data());
    }

    for (std::size_t c = 0; c < H; ++c) h[c] = x[c] + m[c];

    const std::size_t E = g.n_experts, K = g.n_active_experts;
    rm::matvec(dn + lw.lay.router, E, H, h.data(), router.data());
    sel.clear();
    used.assign(E, 0);
    for (std::size_t kk = 0; kk < K; ++kk) {
      std::size_t best = E;
      for (std::size_t e = 0; e < E; ++e)
        if (!used[e] && (best == E || router[e] > router[best])) best = e;
      used[best] = 1;
      sel.push_back(static_cast<std::uint32_t>(best));
    }
    std::sort(sel.begin(), sel.end());
    float mx = router[sel[0]];
    for (std::uint32_t e : sel) mx = std::max(mx, router[e]);
    wts.assign(K, 0.0f);
    float wsum = 0.0f;
    for (std::size_t kk = 0; kk < K; ++kk) {
      wts[kk] = std::exp(router[sel[kk]] - mx);
      wsum += wts[kk];
    }
    for (std::size_t kk = 0; kk < K; ++kk) wts[kk] /= wsum;

    // Hand the position's mixer results to the exchange.
    std::copy_n(m.begin(), H, ls.m_out.begin() + static_cast<std::ptrdiff_t>(std::size_t{i} * H));
    std::copy_n(h.begin(), H, ls.h_in.begin() + static_cast<std::ptrdiff_t>(std::size_t{i} * H));
    for (std::size_t j = 0; j < hc; ++j) ls.gate[std::size_t{i} * hc + j] = rm::sigmoid(rm::dot(dn + lw.lay.inj + j * H, x.data(), H));
    ls.sel[i] = sel;
    ls.wts[i] = wts;
  }

  Status fail(Status st) {
    broken = true;
    window.reset();
    return st;
  }

  // One layer of a window: mixers, split MoE exchange, combine, finalize the records.
  Status run_layer(std::uint32_t L, Window& w, std::vector<float>& rec) {
    const std::size_t H = g.hidden_size, hc = g.residual_streams, fpp = layout.floats_per_position();
    const std::uint32_t qn = w.q;
    const std::size_t n_remote = remotes.size();
    const std::size_t sff = g.shared_expert_ff;
    LayerExchange lx;
    lx.layer = L;
    lx.positions = qn;
    const std::uint64_t t_layer0 = monotonic_ns();

    LayerState ls;
    ls.m_out.assign(std::size_t{qn} * H, 0.0f);
    ls.h_in.assign(std::size_t{qn} * H, 0.0f);
    ls.gate.assign(std::size_t{qn} * hc, 0.0f);
    ls.sel.resize(qn);
    ls.wts.resize(qn);

    if (layers[L].kind == LayerKind::kRecurrent) w.snapshots[L].assign(std::size_t{qn} * H, 0.0f);
    for (std::uint32_t i = 0; i < qn; ++i) mixer_position(L, w, i, rec.data() + std::size_t{i} * fpp, ls);
    const std::uint64_t t_moe0 = monotonic_ns();
    lx.dense_ns = t_moe0 - t_layer0;

    // ---- split selected experts by owner -------------------------------------------------------
    const ExpertAssignment& as = cfg.assignment;
    ls.routes.assign(as.n_owners, std::vector<std::vector<ExpertRoute>>(qn));
    ls.first_id.assign(as.n_owners, std::vector<std::uint32_t>(qn, 0));
    std::vector<char> seen(g.n_experts, 0);
    std::vector<std::uint64_t> routed_to(as.n_owners, 0);
    for (std::uint32_t i = 0; i < qn; ++i)
      for (std::size_t kk = 0; kk < ls.sel[i].size(); ++kk) {
        const std::uint32_t e = ls.sel[i][kk];
        const std::uint32_t o = as.owner_of[e];
        auto& per = ls.routes[o][i];
        if (per.empty()) ls.first_id[o][i] = e;
        per.push_back({as.local_index[e], ls.wts[i][kk]});
        ++routed_to[o];
        ++lx.selections;
        if (o != kFatherOwner) ++lx.remote_selections;
        seen[e] = 1;
      }
    for (char c : seen) lx.distinct_experts += c != 0;

    // ---- one batch per participating remote domain ----------------------------------------------
    std::vector<std::uint64_t> sent_corr(n_remote, 0);
    std::vector<char> participating(n_remote, 0);
    const std::uint64_t t_send0 = monotonic_ns();
    for (std::size_t r = 0; r < n_remote; ++r) {
      if (routed_to[r + 1] == 0) continue;
      ExpertBatch b;
      b.epoch = cfg.epoch;
      b.window = w.id;
      b.layer = L;
      b.positions = qn;
      b.hidden = static_cast<std::uint32_t>(H);
      b.activations = ls.h_in;
      b.routes = ls.routes[r + 1];
      transport::Frame f = to_frame(b, sent_corr[r] = ++correlation);
      lx.bytes_sent += f.payload.size();
      const Status st = remotes[r].connection->send(f);
      if (!st.is_ok()) return fail(make_error(st.code(), "send to expert domain '" + remotes[r].name + "' failed"));
      participating[r] = 1;
      ++lx.domains_participating;
    }

    // ---- Father's own experts + the shared expert (overlapped with the remote work) -------------
    ls.partial.assign(as.n_owners, {});
    ls.shared_out.assign(std::size_t{qn} * H, 0.0f);
    auto local_work = [&] {
      const std::uint64_t t0 = monotonic_ns();
      const LayerWeights& lw = layers[L];
      ls.partial[kFatherOwner].assign(std::size_t{qn} * H, 0.0f);
      for (std::uint32_t i = 0; i < qn; ++i) {
        float* y = ls.partial[kFatherOwner].data() + std::size_t{i} * H;
        const float* hin = ls.h_in.data() + std::size_t{i} * H;
        for (const ExpertRoute& r : ls.routes[kFatherOwner][i])
          accumulate_expert(lw.own[r.local_expert], H, g.expert_ff, hin, r.weight, esc, y);
        if (sff > 0) {
          swiglu(lw.shared.gate.data(), lw.shared.up.data(), lw.shared.down.data(), H, sff, hin, esc);
          std::copy_n(esc.d.begin(), H, ls.shared_out.begin() + static_cast<std::ptrdiff_t>(std::size_t{i} * H));
        }
      }
      lx.local_expert_ns = monotonic_ns() - t0;
    };
    if (cfg.overlap_local) local_work();

    // ---- the barrier: one result per participating domain ----------------------------------------
    for (std::size_t r = 0; r < n_remote; ++r) {
      if (!participating[r]) continue;
      const std::uint64_t t_wait0 = monotonic_ns();
      auto frame = remotes[r].connection->receive(cfg.layer_timeout);
      lx.barrier_wait_ns += monotonic_ns() - t_wait0;
      if (!frame.is_ok()) {
        const ErrorCode code = frame.status().code() == ErrorCode::kDeadlineExceeded ? ErrorCode::kDeadlineExceeded
                                                                                       : ErrorCode::kUnavailable;
        return fail(make_error(code, "expert domain '" + remotes[r].name + "' lost at layer " + std::to_string(L) + ": " +
                                         frame.status().message()));
      }
      lx.bytes_received += frame->payload.size();
      if (frame->type == kMsgExpertError) {
        auto e = decode_error(frame->payload, cfg.limits);
        if (!e.is_ok()) return fail(e.status());
        return fail(make_error(e->code, "expert domain '" + remotes[r].name + "': " + e->message));
      }
      if (frame->type != kMsgExpertResult || frame->correlation != sent_corr[r])
        return fail(make_error(ErrorCode::kProtocolError, "unexpected reply from expert domain '" + remotes[r].name + "'"));
      auto res = decode_result(frame->payload, cfg.limits);
      if (!res.is_ok()) return fail(res.status());
      if (res->epoch != cfg.epoch || res->window != w.id || res->layer != L || res->positions != qn || res->hidden != H)
        return fail(make_error(ErrorCode::kStaleEpoch, "expert domain '" + remotes[r].name + "' answered a different request"));
      lx.remote_compute_ns_sum += res->compute_ns;
      lx.remote_compute_ns_max = std::max(lx.remote_compute_ns_max, res->compute_ns);
      ls.partial[r + 1] = std::move(res->partial);
    }
    const std::uint64_t t_recv_done = monotonic_ns();
    lx.exchange_ns = lx.domains_participating > 0 ? t_recv_done - t_send0 : 0;
    if (!cfg.overlap_local) local_work();

    // ---- deterministic combine + finalize ---------------------------------------------------------
    std::vector<std::uint32_t> order;
    for (std::uint32_t i = 0; i < qn; ++i) {
      float* record = rec.data() + std::size_t{i} * fpp;
      float* P = record + layout.pending_offset();
      float* gg = record + layout.injection_offset();
      order.clear();
      for (std::uint32_t o = 0; o < as.n_owners; ++o)
        if (!ls.routes[o][i].empty()) order.push_back(o);
      // Owner order is a function of the route alone (lowest selected expert id), never of arrival order.
      std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return ls.first_id[a][i] < ls.first_id[b][i];
      });
      std::vector<float>& y = h;  // scratch: the mixer scratch is free again
      std::fill(y.begin(), y.end(), 0.0f);
      for (std::uint32_t o : order) {
        const float* pp = ls.partial[o].data() + std::size_t{i} * H;
        for (std::size_t c = 0; c < H; ++c) y[c] += pp[c];
      }
      if (sff > 0) {
        const float* sh = ls.shared_out.data() + std::size_t{i} * H;
        for (std::size_t c = 0; c < H; ++c) y[c] += sh[c];
      }
      const float* mo = ls.m_out.data() + std::size_t{i} * H;
      for (std::size_t c = 0; c < H; ++c) P[c] = mo[c] + y[c];
      for (std::size_t j = 0; j < hc; ++j) gg[j] = ls.gate[std::size_t{i} * hc + j];
    }
    const std::uint64_t t_end = monotonic_ns();
    lx.moe_ns = t_end - t_moe0;
    lx.layer_ns = t_end - t_layer0;
    lx.dense_ns += t_end - t_recv_done;  // finalize counts as Father dense work
    if (!cfg.overlap_local) lx.dense_ns -= lx.local_expert_ns;
    metrics.layers.push_back(lx);
    return Status::ok();
  }

  Result<domain::Logits> run_window(std::span<const std::int32_t> tokens) {
    if (broken) return make_error(ErrorCode::kAborted, "executor is broken (a domain was lost); rebuild it");
    if (window.has_value()) return precondition("a window is already outstanding");
    const auto qn = static_cast<std::uint32_t>(tokens.size());
    if (qn == 0 || qn > cfg.max_window || qn > cfg.limits.max_positions) return invalid("window size out of range");
    if (position + qn > cfg.max_context) return make_error(ErrorCode::kResourceExhausted, "window exceeds max_context");
    for (std::int32_t tk : tokens)
      if (tk < 0 || static_cast<std::uint32_t>(tk) >= g.vocab_size) return invalid("token outside the vocabulary");

    const std::uint64_t t0 = monotonic_ns();
    metrics = WindowMetrics{};
    metrics.positions = qn;
    Window w;
    w.id = WindowId{++window_counter};
    w.base = position;
    w.q = qn;
    w.tokens.assign(tokens.begin(), tokens.end());
    w.snapshots.resize(g.n_layers);

    const std::size_t H = g.hidden_size, fpp = layout.floats_per_position();
    std::vector<float> rec(std::size_t{qn} * fpp, 0.0f);
    for (std::uint32_t i = 0; i < qn; ++i) {
      float* r = rec.data() + std::size_t{i} * fpp;
      const float* e = embd.data() + static_cast<std::size_t>(tokens[i]) * H;
      for (std::uint32_t j = 0; j < g.residual_streams; ++j) std::copy_n(e, H, r + j * H);
    }
    for (std::uint32_t L = 0; L < g.n_layers; ++L) CLM_RETURN_IF_ERROR(run_layer(L, w, rec));

    domain::Logits out;
    out.positions = qn;
    out.vocab = g.vocab_size;
    out.data.assign(std::size_t{qn} * g.vocab_size, 0.0f);
    const std::size_t hc = g.residual_streams;
    for (std::uint32_t i = 0; i < qn; ++i) {
      const float* record = rec.data() + std::size_t{i} * fpp;
      head_s.assign(record, record + hc * H);
      rm::fold(head_s.data(), record + layout.pending_offset(), record + layout.injection_offset(), hc, H);
      rm::mean_streams(head_s.data(), hc, H, u.data());
      rm::rmsnorm(u.data(), head_norm.data(), H, x.data());
      rm::matvec(head_w.data(), g.vocab_size, H, x.data(), out.data.data() + std::size_t{i} * g.vocab_size);
    }
    window = std::move(w);
    metrics.total_ns = monotonic_ns() - t0;
    return out;
  }

  Status commit(std::uint32_t accepted) {
    if (broken) return make_error(ErrorCode::kAborted, "executor is broken");
    if (!window.has_value()) return precondition("no outstanding window");
    if (accepted == 0 || accepted > window->q) return invalid("accepted length outside [1, q]");
    const std::size_t H = g.hidden_size;
    for (std::size_t L = 0; L < sess.rec.size(); ++L)
      if (!sess.rec[L].empty())
        std::copy_n(window->snapshots[L].begin() + static_cast<std::ptrdiff_t>((accepted - 1) * H), H, sess.rec[L].begin());
    sess.history.insert(sess.history.end(), window->tokens.begin(), window->tokens.begin() + accepted);
    position += accepted;
    window.reset();
    return Status::ok();
  }
};

FatherExecutor::FatherExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FatherExecutor::~FatherExecutor() { close(); }

Result<std::unique_ptr<FatherExecutor>> FatherExecutor::create(const objects::ModelManifest& manifest,
                                                               const objects::ObjectResolver& resolver,
                                                               FatherExecutorConfig config,
                                                               std::vector<RemoteLink> remotes) {
  CLM_RETURN_IF_ERROR(manifest.validate());
  CLM_RETURN_IF_ERROR(config.assignment.validate());
  const objects::ModelGeometry& g = manifest.geometry;
  if (config.assignment.n_experts != g.n_experts) return invalid("assignment does not match the model's expert count");
  if (config.assignment.n_owners != remotes.size() + 1) return invalid("one remote link per non-Father owner is required");
  for (std::size_t r = 0; r < remotes.size(); ++r)
    if (config.assignment.owned[r + 1].empty() || !remotes[r].connection)
      return invalid("every remote owner needs experts and a connection");
  if (config.max_window == 0 || config.max_window > config.max_context) return invalid("bad window limits");
  if (config.limits.max_hidden < g.hidden_size || config.limits.max_layer < g.n_layers ||
      config.limits.max_positions < config.max_window)
    return invalid("decode limits smaller than the model geometry or window");
  auto impl = std::make_unique<Impl>();
  impl->g = g;
  impl->layout = domain::BoundaryLayout::for_geometry(g);
  impl->cfg = std::move(config);
  impl->remotes = std::move(remotes);
  for (RemoteLink& r : impl->remotes) r.connection->set_max_payload(impl->cfg.limits.max_payload_bytes());
  CLM_RETURN_IF_ERROR(impl->load(resolver));
  return std::unique_ptr<FatherExecutor>(new FatherExecutor(std::move(impl)));
}

Result<domain::Logits> FatherExecutor::run_window(std::span<const std::int32_t> tokens) { return impl_->run_window(tokens); }
Status FatherExecutor::commit(std::uint32_t accepted) { return impl_->commit(accepted); }
void FatherExecutor::reset_session() { impl_->reset_session(); }
std::uint64_t FatherExecutor::committed_position() const { return impl_->position; }
bool FatherExecutor::broken() const { return impl_->broken; }
const WindowMetrics& FatherExecutor::last_window_metrics() const { return impl_->metrics; }
void FatherExecutor::set_epoch(Epoch epoch) { impl_->cfg.epoch = epoch; }
void FatherExecutor::close() {
  if (!impl_) return;
  for (RemoteLink& r : impl_->remotes)
    if (r.connection) r.connection->close();
}

}  // namespace clusterlm::expert_domains
