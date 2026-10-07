#include "clusterlm/expert_domains/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/domain/reference_domain.hpp"
#include "clusterlm/expert_domains/analytic.hpp"
#include "clusterlm/expert_domains/rig.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "local_cluster.hpp"

namespace clusterlm::expert_domains {

namespace fs = std::filesystem;
using bench::Distribution;

namespace {

double ms(std::uint64_t ns) { return static_cast<double>(ns) / 1e6; }

std::int32_t synthetic_token(std::uint64_t i, std::uint32_t vocab) {
  return static_cast<std::int32_t>((i * 7919u + 13u) % vocab);
}

// ---- grouped design ------------------------------------------------------------------------------------------

struct GroupedStats {
  Distribution window_ms, layer_exchange_ms, layer_wait_ms, layer_moe_ms, layer_ms;
  double layers = 0, windows = 0;
  double messages = 0, bytes = 0, selections = 0, remote_selections = 0, distinct = 0, participating = 0;
  double local_ns = 0, remote_ns_sum = 0, moe_ns = 0, wait_ns = 0, exchange_ns = 0, dense_ns = 0;
  double max_rel_diff = 0;
  bool reference_ok = true, argmax_ok = true;
};

// Reference execution of the same windows on the unsplit reference domains (prefix [0,n) + head-only tail).
class ReferenceRunner {
 public:
  static Result<std::unique_ptr<ReferenceRunner>> create(const objects::CanonicalModelStore& store, std::uint32_t max_context,
                                                         std::uint32_t max_window) {
    auto r = std::unique_ptr<ReferenceRunner>(new ReferenceRunner());
    const auto& m = store.manifest();
    const std::uint32_t n = m.geometry.n_layers;
    domain::DomainSpec ps;
    ps.stage = StageId{0};
    ps.role = domain::StageRole::kPrefix;
    ps.layers = {0, n};
    ps.max_context = max_context;
    ps.max_window = max_window;
    domain::DomainSpec ts = ps;
    ts.stage = StageId{1};
    ts.role = domain::StageRole::kTail;
    ts.layers = {n, n};
    CLM_ASSIGN_OR_RETURN(r->prefix_, domain::ReferenceDomain::create(m, ps));
    CLM_ASSIGN_OR_RETURN(r->tail_, domain::ReferenceDomain::create(m, ts));
    CLM_RETURN_IF_ERROR(r->prefix_->prepare(store));
    CLM_RETURN_IF_ERROR(r->tail_->prepare(store));
    CLM_RETURN_IF_ERROR(r->prefix_->open_session(Epoch{1}, SessionId{1}));
    CLM_RETURN_IF_ERROR(r->tail_->open_session(Epoch{1}, SessionId{1}));
    return r;
  }
  Result<domain::Logits> run(std::span<const std::int32_t> tokens) {
    domain::WindowRequest req;
    req.epoch = Epoch{1};
    req.session = SessionId{1};
    req.window = WindowId{++window_};
    req.base_position = position_;
    req.expected_state = StateVersion{state_};
    req.positions = static_cast<std::uint32_t>(tokens.size());
    last_ = req;
    CLM_ASSIGN_OR_RETURN(auto act, prefix_->run_prefix(req, tokens));
    return tail_->run_tail(req, act);
  }
  Status commit(std::uint32_t accepted) {
    domain::CommitRequest c;
    c.epoch = last_.epoch;
    c.session = last_.session;
    c.window = last_.window;
    c.accepted = accepted;
    c.expected_state = last_.expected_state;
    CLM_RETURN_IF_ERROR(prefix_->commit_window(c).status());
    CLM_RETURN_IF_ERROR(tail_->commit_window(c).status());
    position_ += accepted;
    ++state_;
    return Status::ok();
  }

 private:
  ReferenceRunner() = default;
  std::unique_ptr<domain::ReferenceDomain> prefix_, tail_;
  domain::WindowRequest last_;
  std::uint64_t window_ = 0, position_ = 0, state_ = 0;
};

double max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double mx = 0, ref = 1e-30;
  for (std::size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    ref = std::max(ref, std::fabs(static_cast<double>(b[i])));
  }
  return mx / ref;
}

std::int32_t argmax_row(const float* row, std::uint32_t n) {
  std::uint32_t best = 0;
  for (std::uint32_t i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return static_cast<std::int32_t>(best);
}

Result<GroupedStats> run_grouped(const SimulationOptions& opt, const objects::CanonicalModelStore& store,
                                 const std::optional<transport::NetworkConditions>& net, std::uint32_t q) {
  const std::uint32_t vocab = store.manifest().geometry.vocab_size;
  const std::uint32_t max_q = *std::max_element(opt.q_values.begin(), opt.q_values.end());
  RigOptions ro;
  ro.store = &store;
  ro.remote_domains = opt.peers.empty() ? opt.remote_domains : static_cast<std::uint32_t>(opt.peers.size());
  ro.strided = opt.strided;
  ro.kernel = opt.kernel;
  if (!opt.peers.empty()) ro.connect_peers = [&opt] { return connect_peers(opt.peers, opt.peer_security); };
  ro.network = net;
  ro.max_window = std::max<std::uint32_t>(max_q, 8);
  ro.max_context = opt.prompt_len + opt.windows * max_q + 16;
  ro.overlap_local = opt.overlap_local;
  ro.layer_timeout = std::chrono::milliseconds(30'000);
  CLM_ASSIGN_OR_RETURN(auto rig, GroupedRig::create(ro));
  std::unique_ptr<ReferenceRunner> ref;
  if (opt.check_reference && !opt.kernel.quantized()) {
    CLM_ASSIGN_OR_RETURN(ref, ReferenceRunner::create(store, ro.max_context, ro.max_window));
  }
  FatherExecutor& ex = rig->executor();
  std::uint64_t tok = 0;
  GroupedStats st;

  auto step = [&](std::uint32_t n, bool measured) -> Status {
    std::vector<std::int32_t> tokens;
    for (std::uint32_t i = 0; i < n; ++i) tokens.push_back(synthetic_token(tok++, vocab));
    CLM_ASSIGN_OR_RETURN(domain::Logits lg, ex.run_window(tokens));
    if (ref) {
      CLM_ASSIGN_OR_RETURN(domain::Logits rl, ref->run(tokens));
      st.max_rel_diff = std::max(st.max_rel_diff, max_rel_diff(lg.data, rl.data));
      for (std::uint32_t p = 0; p < n; ++p)
        if (argmax_row(lg.data.data() + std::size_t{p} * vocab, vocab) != argmax_row(rl.data.data() + std::size_t{p} * vocab, vocab))
          st.argmax_ok = false;
      CLM_RETURN_IF_ERROR(ref->commit(n));
    }
    CLM_RETURN_IF_ERROR(ex.commit(n));
    if (!measured) return Status::ok();
    const WindowMetrics& wm = ex.last_window_metrics();
    st.window_ms.add(ms(wm.total_ns));
    ++st.windows;
    for (const LayerExchange& l : wm.layers) {
      ++st.layers;
      if (l.domains_participating > 0) st.layer_exchange_ms.add(ms(l.exchange_ns));  // a barrier only exists if someone was asked
      st.layer_wait_ms.add(ms(l.barrier_wait_ns));
      st.layer_moe_ms.add(ms(l.moe_ns));
      st.layer_ms.add(ms(l.layer_ns));
      st.messages += 2.0 * l.domains_participating;
      st.bytes += static_cast<double>(l.bytes_sent + l.bytes_received);
      st.selections += l.selections;
      st.remote_selections += l.remote_selections;
      st.distinct += l.distinct_experts;
      st.participating += l.domains_participating;
      st.local_ns += static_cast<double>(l.local_expert_ns);
      st.remote_ns_sum += static_cast<double>(l.remote_compute_ns_sum);
      st.moe_ns += static_cast<double>(l.moe_ns);
      st.wait_ns += static_cast<double>(l.barrier_wait_ns);
      st.exchange_ns += static_cast<double>(l.exchange_ns);
      st.dense_ns += static_cast<double>(l.dense_ns);
    }
    return Status::ok();
  };

  for (std::uint32_t done = 0; done < opt.prompt_len;) {  // prefill (not measured)
    const std::uint32_t n = std::min<std::uint32_t>(ro.max_window, opt.prompt_len - done);
    CLM_RETURN_IF_ERROR(step(n, false));
    done += n;
  }
  for (std::uint32_t w = 0; w < opt.windows; ++w) CLM_RETURN_IF_ERROR(step(q, true));
  st.reference_ok = !ref || st.max_rel_diff <= opt.reference_tolerance;
  return st;
}

// ---- layer-domain design (production Coordinator, node processes) -----------------------------------------------

struct LayerDomainStats {
  Distribution round_ms, remote_ms, prefix_ms, tail_ms, commit_ms;
  double rounds = 0, boundary_messages = 0, control_messages = 0, boundary_bytes = 0;
};

Result<std::map<std::uint32_t, LayerDomainStats>> run_layer_domain(const SimulationOptions& opt, const fs::path& model_dir,
                                                                   const std::string& preset) {
  bench::LocalClusterOptions lo;
  lo.work_dir = opt.work_dir / ("layer-domain-" + preset);
  lo.tls = false;
  for (int i = 0; i < 2; ++i) {
    bench::LocalNodeOptions n;
    n.name = "node" + std::to_string(i);
    if (preset != "unlimited") n.impair = preset;
    lo.nodes.push_back(n);
  }
  CLM_ASSIGN_OR_RETURN(auto cluster, bench::LocalCluster::start(lo));
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = model_dir;
  cfg.security = cluster->father_security();
  cfg.nodes = cluster->endpoints();
  cfg.direct_peer = true;
  if (preset != "unlimited") {
    CLM_ASSIGN_OR_RETURN(auto cond, transport::network_preset(preset));
    cfg.impairment = cond;
  }
  cfg.window_timeout = std::chrono::milliseconds(30'000);
  CLM_ASSIGN_OR_RETURN(auto coord, coordinator::Coordinator::create(cfg));
  const auto& g = coord->manifest().geometry;
  const std::uint32_t n = g.n_layers;
  const std::uint32_t a = std::max(n / 4, g.ple_layer + 1), b = n * 5 / 8, c = n * 13 / 16;
  if (!(a < b && b < c && c < n)) return make_error(ErrorCode::kInvalidArgument, "too few layers for a 4-stage plan");
  CLM_ASSIGN_OR_RETURN(auto plan, coordinator::ClusterPlan::parse(std::to_string(0) + "-" + std::to_string(a) + "@father," +
                                                                      std::to_string(a) + "-" + std::to_string(b) + "@0," +
                                                                      std::to_string(b) + "-" + std::to_string(c) + "@1," +
                                                                      std::to_string(c) + "-" + std::to_string(n) + "@father",
                                                                  n));
  CLM_RETURN_IF_ERROR(coord->connect());
  CLM_RETURN_IF_ERROR(coord->prepare(plan).status());
  std::map<std::uint32_t, LayerDomainStats> out;
  for (std::uint32_t q : opt.q_values) {
    coordinator::GenerationRequest req;
    for (std::uint32_t i = 0; i < opt.windows * q; ++i) req.prompt.push_back(synthetic_token(i, g.vocab_size));
    req.max_new_tokens = 1;
    req.q = 1;
    req.prefill_chunk = q;  // prefill rounds of exactly q positions stand in for q-wide verification rounds
    CLM_ASSIGN_OR_RETURN(auto gen, coord->generate(req));
    LayerDomainStats& s = out[q];
    for (const auto& r : gen.rounds) {
      if (r.positions != q) continue;
      ++s.rounds;
      s.round_ms.add(r.total_ms);
      s.remote_ms.add(r.remote_ms);
      s.prefix_ms.add(r.prefix_ms);
      s.tail_ms.add(r.tail_ms);
      s.commit_ms.add(r.commit_ms);
      s.boundary_messages += r.boundary_messages;
      s.control_messages += r.control_messages;
      s.boundary_bytes += static_cast<double>(r.boundary_payload_bytes);
    }
  }
  CLM_RETURN_IF_ERROR(coord->release().status());
  return out;
}

double mean_of(const Distribution& d) {
  if (d.samples.empty()) return 0;
  double s = 0;
  for (double v : d.samples) s += v;
  return s / static_cast<double>(d.samples.size());
}

AnalyticInputs analytic_for(const std::string& preset, std::uint32_t q) {
  AnalyticInputs in;
  in.q = q;
  if (preset == "unlimited") {
    in.bandwidth_bytes_per_s = 1e12;  // effectively infinite
    in.one_way_latency_ms = 0;
  } else if (auto n = transport::network_preset(preset); n.is_ok()) {
    in.bandwidth_bytes_per_s = n->bandwidth_bytes_per_s;
    in.one_way_latency_ms = n->latency_ms + n->jitter_ms / 2.0;  // mean of U(0, jitter)
  }
  return in;
}

}  // namespace

Status run_simulation(const SimulationOptions& opt, bench::BenchmarkResult& r) {
  if (opt.q_values.empty() || opt.presets.empty()) return make_error(ErrorCode::kInvalidArgument, "need q values and presets");
  if (!opt.peers.empty())
    for (const auto& p : opt.presets)
      if (p != "unlimited")
        return make_error(ErrorCode::kInvalidArgument,
                          "peer mode runs over the real link: use --presets unlimited (impairment presets are simulations)");
  for (std::uint32_t q : opt.q_values)
    if (q == 0 || q > 8) return make_error(ErrorCode::kInvalidArgument, "q must be in 1..8");
  std::error_code ec;
  fs::create_directories(opt.work_dir, ec);
  const fs::path model_dir = opt.work_dir / "fixture-model";
  CLM_ASSIGN_OR_RETURN(auto manifest, objects::write_fixture_model(opt.fixture, model_dir));
  CLM_ASSIGN_OR_RETURN(auto store, objects::CanonicalModelStore::open(model_dir));
  const objects::ModelGeometry& g = manifest.geometry;

  r.mark_simulated("fixture_model", true);
  {
    bool all_loopback = true;
    for (const auto& p : opt.peers) all_loopback = all_loopback && p.endpoint.is_loopback();
    r.mark_simulated("localhost_cluster", all_loopback);  // false: the domains ran on other machines (still a fixture model)
  }
  r.config("topology", "grouped expert-domain (experimental) vs layer-domain pipeline");
  r.config("fixture", {{"layers", g.n_layers}, {"hidden", g.hidden_size}, {"experts", g.n_experts}, {"active", g.n_active_experts},
                       {"expert_ff", g.expert_ff}, {"shared_expert_ff", g.shared_expert_ff}});
  r.config("remote_domains", opt.peers.empty() ? opt.remote_domains : static_cast<std::uint32_t>(opt.peers.size()));
  r.config("peer_mode", !opt.peers.empty());
  if (!opt.peers.empty()) {
    nlohmann::json names = nlohmann::json::array();
    for (const auto& p : opt.peers) names.push_back(p.name + "@" + p.endpoint.str());
    r.config("peers", names);
    r.config("peer_transport", opt.peer_security.mode == transport::SecurityConfig::Mode::kMutualTls ? "mutual-tls" : "insecure-loopback");
  }
  r.config("expert_kernel", opt.kernel.quantized() ? "strata-" + opt.kernel.representation + " (synthetic blobs)" : std::string("fixture-fp32"));
  r.config("ownership", opt.strided ? "strided" : "ranges");
  r.config("overlap_local", opt.overlap_local);
  r.config("windows_per_cell", opt.windows);
  r.config("presets", opt.presets);
  r.config("q_values", opt.q_values);
  r.model({{"artifact_id", manifest.artifact_id}, {"root_hash", manifest.root_hash().hex()}, {"fixture", true}});
  r.backend("reference", "reference");

#define SAY(...)                                                \
  do {                                                          \
    if (opt.report != nullptr) std::fprintf(opt.report, __VA_ARGS__); \
  } while (0)
  const std::string primary = std::find(opt.presets.begin(), opt.presets.end(), "gige-simulated") != opt.presets.end()
                                  ? "gige-simulated"
                                  : opt.presets.front();

  {
    std::string joined;
    for (const std::string& p : opt.presets) joined += (joined.empty() ? "" : ",") + p;
    r.mark_simulated("network_preset", joined);
  }
  bool all_reference_ok = true, all_argmax_ok = true;
  double worst_diff = 0;
  for (const std::string& preset : opt.presets) {
    std::optional<transport::NetworkConditions> net;
    if (preset != "unlimited") {
      CLM_ASSIGN_OR_RETURN(auto cond, transport::network_preset(preset));
      net = cond;
    }
    std::map<std::uint32_t, LayerDomainStats> ld;
    if (opt.layer_domain) {
      auto res = run_layer_domain(opt, model_dir, preset);
      if (res.is_ok()) {
        ld = std::move(res).value();
      } else {
        r.check("layer_domain_" + preset + "_ran", false, res.status().to_string());
      }
    }
    SAY("\n== preset %s (Synthetic: fixture model, localhost, simulated link) ==\n", preset.c_str());
    SAY("%-3s | %-34s | %-32s\n", "q", "grouped expert-domain (per window)", "layer-domain pipeline (per round)");
    for (std::uint32_t q : opt.q_values) {
      CLM_ASSIGN_OR_RETURN(GroupedStats gs, run_grouped(opt, *store, net, q));
      const std::string k = "grouped." + preset + ".q" + std::to_string(q);
      const double layers = std::max(1.0, gs.layers);
      const double wins = std::max(1.0, gs.windows);
      r.metric(k + ".window_ms", gs.window_ms, "ms");
      r.metric(k + ".layer_exchange_ms", gs.layer_exchange_ms, "ms");
      r.metric(k + ".layer_barrier_wait_ms", gs.layer_wait_ms, "ms");
      r.metric(k + ".layer_moe_ms", gs.layer_moe_ms, "ms");
      r.metric(k + ".layer_ms", gs.layer_ms, "ms");
      r.metric(k + ".messages_per_layer", gs.messages / layers);
      r.metric(k + ".messages_per_layer_per_position", gs.messages / layers / q);
      r.metric(k + ".messages_per_window", gs.messages / wins);
      r.metric(k + ".bytes_per_layer", gs.bytes / layers);
      r.metric(k + ".bytes_per_window", gs.bytes / wins);
      r.metric(k + ".barriers_per_window", layers / wins);
      r.metric(k + ".selections_per_layer", gs.selections / layers);
      r.metric(k + ".distinct_experts_per_layer", gs.distinct / layers);
      r.metric(k + ".expert_union_growth", gs.selections > 0 ? gs.distinct / gs.selections : 0.0);
      r.metric(k + ".domains_participating_per_layer", gs.participating / layers);
      // CPU concurrency over the MoE section: (Father's local expert work + the domains' compute) / wall time.
      r.metric(k + ".moe_cpu_concurrency", gs.moe_ns > 0 ? (gs.local_ns + gs.remote_ns_sum) / gs.moe_ns : 0.0);
      r.metric(k + ".moe_overlap_fraction", gs.exchange_ns > 0 ? std::min(1.0, gs.local_ns / gs.exchange_ns) : 0.0);
      r.metric(k + ".critical_path_ms_per_window", mean_of(gs.window_ms));
      r.metric(k + ".max_rel_logit_diff_vs_reference", gs.max_rel_diff);
      all_reference_ok = all_reference_ok && gs.reference_ok;
      all_argmax_ok = all_argmax_ok && gs.argmax_ok;
      worst_diff = std::max(worst_diff, gs.max_rel_diff);

      double ld_round = 0;
      auto it = ld.find(q);
      if (it != ld.end() && it->second.rounds > 0) {
        const LayerDomainStats& s = it->second;
        const std::string lk = "layer_domain." + preset + ".q" + std::to_string(q);
        r.metric(lk + ".round_ms", s.round_ms, "ms");
        r.metric(lk + ".remote_ms", s.remote_ms, "ms");
        r.metric(lk + ".commit_ms", s.commit_ms, "ms");
        r.metric(lk + ".boundary_messages_per_round", s.boundary_messages / s.rounds);
        r.metric(lk + ".control_messages_per_round", s.control_messages / s.rounds);
        r.metric(lk + ".boundary_bytes_per_round", s.boundary_bytes / s.rounds);
        r.metric(lk + ".critical_path_ms_per_round", mean_of(s.round_ms));
        ld_round = mean_of(s.round_ms);
      }
      const double gw = mean_of(gs.window_ms);
      if (ld_round > 0) r.metric("compare." + preset + ".q" + std::to_string(q) + ".net_gain_ms", ld_round - gw);
      if (preset == primary && q == opt.q_values.front()) {
        r.metric("grouped.layer_barrier_ms", gs.layer_exchange_ms, "ms");
        if (ld_round > 0) r.metric("grouped.net_gain_ms", ld_round - gw);
      }
      SAY("%-3u | %6.2f ms/win, %5.1f msg/layer, %6.0f B/layer | ", q, gw, gs.messages / layers, gs.bytes / layers);
      if (ld_round > 0) {
        const LayerDomainStats& s = ld[q];
        SAY("%6.2f ms/round, %4.1f boundary msg + %4.1f ctl msg, %7.0f B\n", ld_round, s.boundary_messages / s.rounds,
            s.control_messages / s.rounds, s.boundary_bytes / s.rounds);
      } else {
        SAY("(layer-domain not run)\n");
      }
      SAY("      barrier wait/layer p50 %.3f ms, exchange/layer p50 %.3f ms, distinct experts/layer %.1f of %.1f selections, "
          "MoE cpu concurrency %.2f, max rel logit diff %.2e\n",
          bench::Distribution{gs.layer_wait_ms}.to_json("ms").value("p50", 0.0),
          bench::Distribution{gs.layer_exchange_ms}.to_json("ms").value("p50", 0.0), gs.distinct / layers,
          gs.selections / layers, gs.moe_ns > 0 ? (gs.local_ns + gs.remote_ns_sum) / gs.moe_ns : 0.0, gs.max_rel_diff);

      // Analytic model for the REAL geometry beside the simulation (calculation, Synthetic inputs).
      const AnalyticInputs ain = analytic_for(preset, q);
      const AnalyticOutputs aout = analytic_model(ain);
      const std::string ak = "analytic_calc." + preset + ".q" + std::to_string(q);
      r.metric(ak + ".barrier_ms", aout.barrier_ms);
      r.metric(ak + ".network_floor_ms_per_pass", aout.network_floor_ms_per_pass);
      r.metric(ak + ".exposed_wait_ms_per_pass", aout.exposed_wait_ms_per_pass);
      r.metric(ak + ".bytes_per_pass", aout.bytes_per_pass);
      r.metric(ak + ".messages_per_pass", aout.messages_per_pass);
      r.metric(ak + ".layer_domain_network_ms_per_pass", aout.layer_domain_network_ms_per_pass);
      r.trace({{"kind", "analytic_calculation"}, {"preset", preset}, {"model", analytic_to_json(ain, aout)}});
      if (q == opt.q_values.back() || q == 1) SAY("%s", analytic_report(ain, aout).c_str());
    }
  }
  char detail[256];
  std::snprintf(detail, sizeof detail,
                "max relative logit difference %.3e (tolerance %.1e); float sums associate differently across owners, see "
                "docs/experimental/expert-domains.md",
                worst_diff, opt.reference_tolerance);
  r.check("grouped_matches_unsplit_reference_within_tolerance", !opt.check_reference || opt.kernel.quantized() || all_reference_ok, detail);
  r.check("grouped_argmax_identical_to_reference", !opt.check_reference || opt.kernel.quantized() || all_argmax_ok);
  if (opt.kernel.quantized())
    r.check("quantized_kernel_reference_comparison_not_applicable", true,
            "synthetic IQ expert blobs: logits are not comparable with the FP32 reference; this mode measures cost, not accuracy");
  for (const char* id : {"HQ-P0C-01", "HQ-NET-02", "HQ-PERF-01"}) r.pending(id);
#undef SAY
  return Status::ok();
}

}  // namespace clusterlm::expert_domains
