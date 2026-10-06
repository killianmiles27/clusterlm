// WP12: candidate space, local batches, workload sweeps, pruning, measured inputs and determinism.
// Everything runs on SYNTHETIC (or explicitly hand-written "measured") fixtures and asserts rational direction
// and invariants, never optimality on real hardware.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>
#include <set>

#include "clusterlm/placement/placement.hpp"
#include "clusterlm/placement/workload.hpp"

using namespace clusterlm;
using namespace clusterlm::placement;

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
const char* kFather = "Father-4060Ti-7600";
const char* kG14 = "Node-G14-4070-8945HS";
const char* kN3060 = "Node-3060-5600";

std::string fixture(const std::string& name) { return std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/" + name; }
std::string measured_fixture(const std::string& name) { return std::string(CLUSTERLM_SOURCE_DIR) + "/tests/placement/fixtures/" + name; }

HardwareProfile load_hw(const std::string& path) {
  auto r = load_hardware_profile(path);
  REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
  return *r;
}
NetworkProfile load_net(const std::string& path) {
  auto r = load_network_profile(path);
  REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
  return *r;
}

PlacementRequest base_request() {
  PlacementRequest r;
  r.father = load_hw(fixture("Father-4060Ti-7600.json"));
  r.nodes = {load_hw(fixture("Node-G14-4070-8945HS.json")), load_hw(fixture("Node-3060-5600.json"))};
  r.network = load_net(fixture("network-gige-simulated.json"));
  r.model = flash_next_planning_estimate();
  r.context_tokens = 4096;
  r.default_lease = {28800, 288000};
  return r;
}

// A third node: a copy of the 3060 profile with its own id and links to everyone.
void add_third_node(PlacementRequest& r, const std::string& id) {
  HardwareProfile n = r.nodes.back();
  n.id = id;
  r.nodes.push_back(n);
  const LinkProfile base = *r.network.find_link(kFather, kN3060);
  for (const char* other : {kFather, kG14, kN3060}) {
    LinkProfile l = base;
    l.from = other;
    l.to = id;
    r.network.links.push_back(l);
  }
}

PlacementResult search(const PlacementRequest& r) {
  auto res = search_placements(r);
  REQUIRE_MESSAGE(res.is_ok(), res.status().to_string());
  return std::move(*res);
}

const PlacementPlan& recommended(const PlacementResult& r) {
  REQUIRE(r.recommended.has_value());
  return r.candidates[*r.recommended];
}

std::size_t remote_nodes(const PlacementPlan& p) { return p.node_ids().size(); }

const MemoryLedger* ledger_of(const PlacementPlan& p, const std::string& id) {
  for (const auto& l : p.ledgers)
    if (l.domain_id == id) return &l;
  return nullptr;
}

// Renames every domain id of a request (profiles, links, per-node maps).
PlacementRequest relabel(const PlacementRequest& in, const std::map<std::string, std::string>& names) {
  PlacementRequest r = in;
  auto nm = [&](const std::string& s) { return names.count(s) ? names.at(s) : s; };
  r.father.id = nm(r.father.id);
  for (auto& n : r.nodes) n.id = nm(n.id);
  for (auto& l : r.network.links) {
    l.from = nm(l.from);
    l.to = nm(l.to);
  }
  r.node_lease.clear();
  for (const auto& [k, v] : in.node_lease) r.node_lease[nm(k)] = v;
  r.already_provisioned.clear();
  for (const auto& [k, v] : in.already_provisioned) r.already_provisioned[nm(k)] = v;
  return r;
}

}  // namespace

TEST_CASE("candidate space: prefix sweep, tail sweep, node orders and up to three remote nodes") {
  PlacementRequest r = base_request();
  add_third_node(r, "Node-X-3060");
  r.sweep_prefix = true;
  r.prefix_sweep_span = 8;
  PlacementResult res = search(r);

  std::set<std::uint32_t> prefixes, tails;
  std::set<std::size_t> node_counts;
  std::set<std::string> orders;
  bool father_only = false;
  auto consider = [&](const std::vector<PlanStage>& stages) {
    std::uint32_t tail = 0;
    std::string order;
    std::size_t nodes = 0;
    for (const auto& s : stages) {
      if (s.role == StageRole::kPrefix) prefixes.insert(s.layers.size());
      if (s.role == StageRole::kTail) tail = s.layers.size();
      if (s.role == StageRole::kFull) father_only = true;
      if (s.role == StageRole::kMiddle) {
        ++nodes;
        order += s.domain_id + ">";
      }
    }
    if (nodes > 0) {
      tails.insert(tail);
      node_counts.insert(nodes);
      orders.insert(order);
    }
  };
  for (const auto& p : res.candidates) consider(p.stages);
  for (const auto& x : res.rejected) consider(x.stages);

  CHECK(father_only);
  // Smallest legal prefix is ple_layer + 1 = 3, then the multiples of 4 up to 3 + 8 (the default option 4 is one).
  CHECK(prefixes == std::set<std::uint32_t>{3, 4, 8});
  CHECK(tails.count(0) == 1);   // empty Father tail
  CHECK(tails.size() > 4);      // and many tail sizes
  CHECK(node_counts == std::set<std::size_t>{1, 2, 3});
  // Every ordered subset of the three nodes appears: 3 + 6 + 6.
  CHECK(orders.size() == 15);
  CHECK(res.enumerated == res.candidates.size() + res.rejected.size());

  // max_remote_nodes caps the chain; 0 leaves only the Father-only plan.
  PlacementRequest two = r;
  two.max_remote_nodes = 2;
  PlacementResult r2 = search(two);
  for (const auto& p : r2.candidates) CHECK(remote_nodes(p) <= 2);
  PlacementRequest none = r;
  none.max_remote_nodes = 0;
  PlacementResult r0 = search(none);
  CHECK(r0.candidates.size() + r0.rejected.size() == 1);

  // allow_node_orders=false keeps only request order.
  PlacementRequest fixed = r;
  fixed.allow_node_orders = false;
  PlacementResult rf = search(fixed);
  for (const auto& p : rf.candidates) {
    const auto ids = p.node_ids();
    for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
      const auto pos = [&](const std::string& id) {
        for (std::size_t k = 0; k < fixed.nodes.size(); ++k)
          if (fixed.nodes[k].id == id) return k;
        return std::size_t{999};
      };
      CHECK(pos(ids[i]) < pos(ids[i + 1]));
    }
  }

  // An explicit prefix that cannot hold the PLE layer is rejected with a reason, never silently legal.
  PlacementRequest small_prefix = base_request();
  small_prefix.prefix_layers_options = {2};
  PlacementResult sp = search(small_prefix);
  const RejectedPlan* rej = nullptr;
  for (const auto& x : sp.rejected)
    for (const auto& why : x.reasons)
      if (why.find("per-layer-embedding") != std::string::npos) rej = &x;
  CHECK(rej != nullptr);
}

TEST_CASE("search at the default space stays fast for 48 layers and three remote nodes") {
  PlacementRequest r = base_request();
  add_third_node(r, "Node-X-3060");
  r.sweep_prefix = true;
  r.prune_dominated = true;
  const auto t0 = std::chrono::steady_clock::now();
  PlacementResult res = search(r);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  INFO("enumerated " << res.enumerated << " pruned " << res.pruned_dominated << " in " << s << " s");
  CHECK(res.enumerated > 1000);
  CHECK(res.pruned_dominated > 0);
  // Release target is < 1 s; the bound is generous so a loaded Debug CI machine does not flake.
  CHECK(s < 30.0);
}

TEST_CASE("dominance pruning never changes the recommendation or the Pareto frontier") {
  PlacementRequest r = base_request();
  r.sweep_prefix = true;
  PlacementResult full = search(r);
  PlacementRequest p = r;
  p.prune_dominated = true;
  PlacementResult pruned = search(p);
  CHECK(pruned.pruned_dominated > 0);
  CHECK(pruned.candidates.size() + pruned.pruned_dominated == full.candidates.size());
  CHECK(recommended(pruned).plan_hash == recommended(full).plan_hash);
  REQUIRE(pruned.pareto.size() == full.pareto.size());
  for (std::size_t i = 0; i < full.pareto.size(); ++i)
    CHECK(pruned.candidates[pruned.pareto[i]].plan_hash == full.candidates[full.pareto[i]].plan_hash);
  CHECK(pruned.candidates[*pruned.best_throughput].metrics.decode_tok_s == full.candidates[*full.best_throughput].metrics.decode_tok_s);
  CHECK(pruned.candidates[*pruned.best_preparation].metrics.prepare_s == full.candidates[*full.best_preparation].metrics.prepare_s);
}

TEST_CASE("per-domain local batch comes from that domain's VRAM headroom, never from the smallest GPU") {
  PlacementRequest r = base_request();
  r.prefix_layers_options = {4};
  PlacementResult res = search(r);
  const PlacementPlan* two = nullptr;
  for (const auto& p : res.candidates)
    if (remote_nodes(p) == 2) two = &p;
  REQUIRE(two != nullptr);
  for (const auto& l : two->ledgers) {
    CHECK(l.local_batch == 512);  // roomy GPUs run the full chunk
    CHECK(l.batch_workspace == std::uint64_t{512} * r.model.batch_scratch_bytes_per_token);
    CHECK(l.vram_used() <= l.vram_budget);
  }

  // Squeeze ONLY the G14: its headroom after dense + state + scratch is 200 MiB.
  const MemoryLedger* g = ledger_of(*two, kG14);
  REQUIRE(g != nullptr);
  PlacementRequest tight = r;
  for (auto& n : tight.nodes)
    if (n.id == kG14) {
      const double fixed = static_cast<double>(g->dense + g->state + g->scratch_vram);
      n.gpu.vram_budget = Quantity::synthetic(fixed + 200.0 * 1024 * 1024);
      n.gpu.vram_total = Quantity::synthetic(std::max(n.gpu.vram_total.value, n.gpu.vram_budget.value));
      // Fewer GPU experts means more CPU-resident ones: give the host room so only the batch is under test.
      n.memory.ram_total = Quantity::synthetic(96.0 * kGiB);
      n.memory.ram_safe_allowance = Quantity::synthetic(64.0 * kGiB);
    }
  PlanEvaluation ev = evaluate_plan(tight, two->stages);
  REQUIRE_MESSAGE(ev.plan.has_value(), (ev.reasons.empty() ? "" : ev.reasons.front()));
  const PlacementPlan& t = *ev.plan;
  const MemoryLedger* gt = ledger_of(t, kG14);
  REQUIRE(gt != nullptr);
  // 25% of 200 MiB at 512 KiB/token = 100 tokens -> largest power of two = 64.
  CHECK(gt->local_batch == 64);
  CHECK(gt->vram_used() <= gt->vram_budget);
  for (const auto& l : t.ledgers)
    if (l.domain_id != kG14) CHECK(l.local_batch == 512);  // the pipeline chunk is not shrunk globally
  // The squeezed stage streams its CPU-resident experts once per micro-batch: prefill is not cheaper.
  double chunk_before = 0, chunk_after = 0;
  for (const auto& s : two->metrics.stages)
    if (s.domain_id == kG14) chunk_before = s.prefill_chunk_ms;
  for (const auto& s : t.metrics.stages) {
    if (s.domain_id == kG14) {
      chunk_after = s.prefill_chunk_ms;
      CHECK(s.local_batch == 64);
    }
  }
  CHECK(chunk_after >= chunk_before);
  CHECK(t.metrics.prefill_s >= two->metrics.prefill_s);
  // Less VRAM left for experts: decode cannot get faster.
  CHECK(t.metrics.decode_tok_s <= two->metrics.decode_tok_s + 1e-9);

  // Headroom that cannot hold even min_local_batch tokens makes the domain inadmissible, with a reason.
  PlacementRequest starved = tight;
  for (auto& n : starved.nodes)
    if (n.id == kG14) n.gpu.vram_budget = Quantity::synthetic(static_cast<double>(g->dense + g->state + g->scratch_vram) + 4.0 * 1024 * 1024);
  PlanEvaluation bad = evaluate_plan(starved, two->stages);
  CHECK_FALSE(bad.plan.has_value());
  bool saw = false;
  for (const auto& why : bad.reasons) saw |= why.find("local batch") != std::string::npos;
  CHECK(saw);

  // The local batch always covers the verify window.
  PlacementRequest wide = tight;
  wide.q = 4;
  wide.min_local_batch = 1;
  PlanEvaluation w = evaluate_plan(wide, two->stages);
  REQUIRE(w.plan.has_value());
  CHECK(ledger_of(*w.plan, kG14)->local_batch >= 4);
}

TEST_CASE("residency tie-breaking is deterministic and has no memory of a previous plan") {
  PlacementRequest r = base_request();
  FixtureEstimateSpec spec;
  spec.n_layers = 8;
  spec.n_experts = 32;
  spec.n_active = 4;
  spec.zipf_exponent = 0.0;  // uniform routing: every expert ties
  spec.expert_bytes = 4ull << 20;
  spec.dense_bytes = 16ull << 20;
  r.model = fixture_estimate(spec);
  r.prefix_layers_options = {2};
  // Room for only part of the experts on each GPU so the fill order matters.
  // (scratch is ~0.8 GiB in the fixtures, so budget = scratch + ~0.5 GiB of experts).
  r.father.gpu.vram_budget = Quantity::synthetic(r.father.overheads.scratch_vram.value + 0.5 * kGiB);
  r.father.gpu.vram_total = Quantity::synthetic(4.0 * kGiB);
  for (auto& n : r.nodes) {
    n.gpu.vram_budget = Quantity::synthetic(n.overheads.scratch_vram.value + 0.5 * kGiB);
    n.gpu.vram_total = Quantity::synthetic(4.0 * kGiB);
  }
  PlacementResult a = search(r), b = search(r);
  REQUIRE_FALSE(a.candidates.empty());
  REQUIRE(a.candidates.size() == b.candidates.size());
  for (std::size_t i = 0; i < a.candidates.size(); ++i) {
    CHECK(a.candidates[i].plan_hash == b.candidates[i].plan_hash);
    CHECK(a.candidates[i].key == b.candidates[i].key);
  }
  // Ties resolve to the lowest layer first and to the lowest expert ids within a layer.
  int checked = 0;
  for (const auto& p : a.candidates)
    for (const auto& res : p.residency) {
      std::uint64_t prev = ~0ull;
      for (const auto& per_layer : res.gpu_experts_by_layer) {
        if (per_layer.empty()) continue;
        for (std::size_t i = 0; i < per_layer.size(); ++i) CHECK(per_layer[i] == i);
        CHECK(per_layer.size() <= prev);
        prev = per_layer.size();
        ++checked;
      }
    }
  CHECK(checked > 0);
  // Evaluating a plan after unrelated searches gives the same hash (no hidden state).
  PlanEvaluation again = evaluate_plan(r, a.candidates.front().stages);
  REQUIRE(again.plan.has_value());
  CHECK(again.plan->plan_hash == a.candidates.front().plan_hash);
}

TEST_CASE("workload sweep: per-context frontiers, state growth displaces residency, feasibility can flip") {
  PlacementRequest r = base_request();
  r.sweep_prefix = true;
  WorkloadSpec spec;
  spec.workload = {1024, 256, 100};
  auto res = search_workload(r, spec);
  REQUIRE_MESSAGE(res.is_ok(), res.status().to_string());
  REQUIRE(res->contexts.size() == 6);
  CHECK(res->contexts.front().context_tokens == 4096);
  CHECK(res->contexts.back().context_tokens == 131072);
  REQUIRE(res->selected.has_value());
  CHECK(res->contexts[*res->selected].context_tokens == 4096);  // 1024 + 256 fits the first profile
  REQUIRE(res->recommended() != nullptr);
  CHECK(res->recommended()->context_tokens == 4096);
  CHECK(res->provenance == Provenance::kSynthetic);

  // Frontiers are mutually non-dominated per context, and state growth shows up in the ledgers.
  std::uint64_t last_state = 0;
  for (const auto& c : res->contexts) {
    if (!c.feasible) continue;
    REQUIRE_FALSE(c.frontier.empty());
    for (std::size_t i = 0; i < c.frontier.size(); ++i)
      for (std::size_t j = 0; j < c.frontier.size(); ++j) {
        if (i == j) continue;
        const auto& a = c.frontier[i].metrics;
        const auto& b = c.frontier[j].metrics;
        const bool dom = a.prepare_s <= b.prepare_s && a.decode_tok_s >= b.decode_tok_s && a.prefill_s <= b.prefill_s &&
                         (a.prepare_s < b.prepare_s || a.decode_tok_s > b.decode_tok_s || a.prefill_s < b.prefill_s);
        CHECK_FALSE(dom);
      }
    CHECK(c.recommended->context_tokens == c.context_tokens);
    std::uint64_t state = 0;
    for (const auto& l : c.recommended->ledgers) state += l.state;
    CHECK(state >= last_state);
    last_state = state;
  }

  // The same stage assignment at a longer context holds more state and therefore fewer GPU experts.
  PlacementRequest short_ctx = r;
  short_ctx.prefix_layers_options = {4};
  PlacementResult sr = search(short_ctx);
  const PlacementPlan& p4k = recommended(sr);
  PlacementRequest long_ctx = short_ctx;
  long_ctx.context_tokens = 65536;
  long_ctx.prompt_tokens = 4096;
  PlanEvaluation e = evaluate_plan(long_ctx, p4k.stages);
  REQUIRE(e.plan.has_value());
  for (std::size_t i = 0; i < p4k.ledgers.size(); ++i) {
    CHECK(e.plan->ledgers[i].state >= p4k.ledgers[i].state);
    CHECK(e.plan->ledgers[i].gpu_experts <= p4k.ledgers[i].gpu_experts);
  }
  CHECK(e.plan->residency.size() == p4k.residency.size());

  // Feasibility flips: with a small VRAM budget everywhere, the longest context becomes infeasible while 4K is not.
  PlacementRequest small = base_request();
  small.father.gpu.vram_budget = Quantity::synthetic(6.5 * kGiB);
  small.father.gpu.vram_total = Quantity::synthetic(16.0 * kGiB);
  for (auto& n : small.nodes) {
    n.gpu.vram_budget = Quantity::synthetic(6.5 * kGiB);
    n.gpu.vram_total = Quantity::synthetic(12.0 * kGiB);
  }
  small.sweep_prefix = true;
  auto sw = search_workload(small, spec);
  REQUIRE(sw.is_ok());
  CHECK(sw->contexts.front().feasible);
  bool flipped = false;
  for (const auto& c : sw->contexts) {
    if (!c.feasible) {
      flipped = true;
      CHECK_FALSE(c.infeasible_reasons.empty());
      CHECK(c.frontier.empty());
      CHECK_FALSE(c.recommended.has_value());
    }
  }
  CHECK(flipped);
  CHECK_FALSE(sw->notes.empty());
  // Once a context is infeasible, every longer one is too (state only grows).
  bool seen_infeasible = false;
  for (const auto& c : sw->contexts) {
    if (!c.feasible) seen_infeasible = true;
    if (seen_infeasible) CHECK_FALSE(c.feasible);
  }
}

TEST_CASE("workload sweep: q values use supplied acceptance and are never invented") {
  PlacementRequest r = base_request();
  WorkloadSpec spec;
  spec.context_profiles = {4096};
  spec.q_values = {1, 2, 3, 4};

  // Missing acceptance for q > 1 is an error.
  auto missing = search_workload(r, spec);
  CHECK_FALSE(missing.is_ok());

  // No benefit from wider verification: q = 1 wins.
  for (std::uint32_t q : {2u, 3u, 4u}) spec.acceptance_by_q[q] = Quantity::synthetic(1.0);
  auto flat = search_workload(r, spec);
  REQUIRE_MESSAGE(flat.is_ok(), flat.status().to_string());
  REQUIRE(flat->recommended() != nullptr);
  CHECK(flat->recommended()->q == 1);
  REQUIRE(flat->contexts[0].per_q.size() == 4);

  // Strong acceptance at q = 3: more tokens per (barely longer) round, so a wider q is recommended.
  spec.acceptance_by_q[3] = Quantity::synthetic(2.6);
  auto good = search_workload(r, spec);
  REQUIRE(good.is_ok());
  REQUIRE(good->recommended() != nullptr);
  CHECK(good->recommended()->q == 3);
  CHECK(good->recommended()->metrics.decode_tok_s > flat->recommended()->metrics.decode_tok_s);
  // The frontier mixes q values and each plan states its own.
  std::set<std::uint32_t> seen;
  for (const auto& p : good->contexts[0].frontier) seen.insert(p.q);
  CHECK(seen.count(3) == 1);

  // Provenance of the workload is the weakest of every acceptance used.
  CHECK(good->provenance == Provenance::kSynthetic);
}

TEST_CASE("workload description sets the context profile, output length and lease share") {
  PlacementRequest r = base_request();
  WorkloadSpec spec;
  spec.context_profiles = {4096, 8192, 32768};
  spec.workload = {6000, 500, 50};
  auto res = search_workload(r, spec);
  REQUIRE(res.is_ok());
  REQUIRE(res->selected.has_value());
  CHECK(res->contexts[*res->selected].context_tokens == 8192);
  const PlacementPlan* rec = res->recommended();
  REQUIRE(rec != nullptr);
  CHECK(rec->output_tokens == 500);
  CHECK(rec->prompt_tokens == 6000);
  CHECK(rec->context_tokens == 8192);

  // More requests per lease amortise preparation, which can only lower the best objective.
  WorkloadSpec many = spec;
  many.workload.requests_per_lease = 5000;
  PlacementRequest cold = r;
  cold.default_lease = {600, 1};  // overwritten by the workload's derivation
  auto few = search_workload(cold, spec);
  auto lots = search_workload(cold, many);
  REQUIRE(few.is_ok());
  REQUIRE(lots.is_ok());
  CHECK(lots->recommended()->metrics.objective <= few->recommended()->metrics.objective + 1e-9);

  // A workload that no profile covers is reported, not silently clamped.
  WorkloadSpec huge = spec;
  huge.workload.expected_prompt_tokens = 100000;
  auto h = search_workload(r, huge);
  REQUIRE(h.is_ok());
  CHECK_FALSE(h->selected.has_value());
  CHECK(h->recommended() == nullptr);
  CHECK_FALSE(h->notes.empty());
}

TEST_CASE("measured inputs: provenance is Measured only when every used input is Measured") {
  PlacementRequest r;
  r.father = load_hw(measured_fixture("measured-Father-4060Ti-7600.json"));
  r.nodes = {load_hw(measured_fixture("measured-Node-G14-4070-8945HS.json")), load_hw(measured_fixture("measured-Node-3060-5600.json"))};
  r.network = load_net(measured_fixture("measured-network-gige.json"));
  r.model = flash_next_planning_estimate();
  r.default_lease = {28800, 288000};
  CHECK(weakest_provenance(r.father) == Provenance::kMeasured);
  CHECK(weakest_provenance(r.network) == Provenance::kMeasured);

  // Model structure, draft time and acceptance are still Synthetic: the plan is Synthetic and says why.
  PlacementResult syn = search(r);
  REQUIRE_FALSE(syn.candidates.empty());
  CHECK(recommended(syn).provenance == Provenance::kSynthetic);
  bool names_model = false;
  for (const auto& s : recommended(syn).non_qualified_inputs) names_model |= s.find("model:structure=synthetic") != std::string::npos;
  CHECK(names_model);

  r.model.provenance = Provenance::kMeasured;
  r.model.draft_ms.provenance = Provenance::kMeasured;
  r.acceptance.provenance = Provenance::kMeasured;
  PlacementResult meas = search(r);
  REQUIRE_FALSE(meas.candidates.empty());
  CHECK(meas.provenance == Provenance::kMeasured);
  for (const auto& p : meas.candidates) {
    CHECK(p.provenance == Provenance::kMeasured);
    CHECK_FALSE(require_qualified(p).is_ok());  // measured is never qualified
  }
  auto banner = nlohmann::json::parse(report_to_json(meas));
  CHECK(banner["provenance"] == "measured");
  CHECK(banner["provenance_banner"].get<std::string>().find("NOT QUALIFIED") != std::string::npos);

  // The measured values are what the planner uses: a measured G14 CPU 20% slower than the synthetic one
  // yields a slower G14 stage for the same assignment than the synthetic profile does.
  PlacementRequest syn_req = base_request();
  const PlacementPlan* with_g14 = nullptr;
  for (const auto& c : meas.candidates)
    for (const auto& s : c.stages)
      if (s.domain_id == kG14) with_g14 = &c;
  REQUIRE(with_g14 != nullptr);
  PlanEvaluation es = evaluate_plan(syn_req, with_g14->stages);
  REQUIRE(es.plan.has_value());
  double cpu_meas = 0, cpu_syn = 0;
  for (const auto& s : with_g14->metrics.stages)
    if (s.domain_id == kG14) cpu_meas += s.cpu_miss_bytes_expected;
  for (const auto& s : es.plan->metrics.stages)
    if (s.domain_id == kG14) cpu_syn += s.cpu_miss_bytes_expected;
  CHECK(cpu_meas == doctest::Approx(cpu_syn));  // same bytes (same residency inputs)...
  double ms_meas = 0, ms_syn = 0;
  for (const auto& s : with_g14->metrics.stages)
    if (s.domain_id == kG14) ms_meas += s.cpu_ms;
  for (const auto& s : es.plan->metrics.stages)
    if (s.domain_id == kG14) ms_syn += s.cpu_ms;
  CHECK(ms_meas > ms_syn);  // ...served slower, because the measured throughput is 20% lower

  // A Synthetic field the plan USES drags it down and is named; a Synthetic field it does not use does not.
  PlacementRequest mixed = r;
  mixed.nodes[0].cpu.q_scaling = Quantity::synthetic(0.05);  // G14 q_scaling stays synthetic
  PlacementResult mx = search(mixed);
  for (const auto& c : mx.candidates) {
    bool uses_g14 = false;
    for (const auto& s : c.stages) uses_g14 |= s.domain_id == kG14;
    if (uses_g14) {
      CHECK(c.provenance == Provenance::kSynthetic);
      bool named = false;
      for (const auto& s : c.non_qualified_inputs) named |= s.find("cpu.q_scaling") != std::string::npos && s.find(kG14) != std::string::npos;
      CHECK(named);
    } else {
      CHECK(c.provenance == Provenance::kMeasured);  // plans that never touch G14 stay Measured
    }
  }
  CHECK(mx.provenance == Provenance::kSynthetic);  // the request as a whole is only as strong as its weakest input

  PlacementRequest unused_quant = r;
  unused_quant.nodes[0].cpu.expert_bytes_per_s["q8_0"] = Quantity::synthetic(9e9);  // the model has no q8_0 layer
  PlacementResult uq = search(unused_quant);
  CHECK(recommended(uq).provenance == Provenance::kMeasured);
}

TEST_CASE("the plan is a function of its inputs only: relabelling nodes relabels the plan, nothing else") {
  PlacementRequest base = base_request();
  base.sweep_prefix = true;
  PlacementResult rb = search(base);
  const PlacementPlan& pb = recommended(rb);

  // Rename every domain and reverse the node order in the request.
  const std::map<std::string, std::string> names = {{kFather, "alpha"}, {kG14, "zulu"}, {kN3060, "bravo"}};
  PlacementRequest re = relabel(base, names);
  std::reverse(re.nodes.begin(), re.nodes.end());
  PlacementResult rr = search(re);
  const PlacementPlan& pr = recommended(rr);

  REQUIRE(pr.stages.size() == pb.stages.size());
  for (std::size_t i = 0; i < pb.stages.size(); ++i) {
    CHECK(pr.stages[i].domain_id == names.at(pb.stages[i].domain_id));
    CHECK(pr.stages[i].layers == pb.stages[i].layers);
    CHECK(pr.stages[i].role == pb.stages[i].role);
  }
  CHECK(pr.metrics.objective == doctest::Approx(pb.metrics.objective));
  CHECK(pr.metrics.decode_tok_s == doctest::Approx(pb.metrics.decode_tok_s));
  CHECK(pr.metrics.prepare_s == doctest::Approx(pb.metrics.prepare_s));
  REQUIRE(pr.residency.size() == pb.residency.size());
  for (std::size_t i = 0; i < pb.residency.size(); ++i) CHECK(pr.residency[i].gpu_experts_by_layer == pb.residency[i].gpu_experts_by_layer);
  // The whole ranking maps one to one (same objective sequence), so there is no name-dependent preference.
  REQUIRE(rr.candidates.size() == rb.candidates.size());
  for (std::size_t i = 0; i < rb.candidates.size(); ++i) CHECK(rr.candidates[i].metrics.objective == doctest::Approx(rb.candidates[i].metrics.objective));

  // Changing only GPU names does nothing (hash included).
  PlacementRequest gpu_names = base;
  gpu_names.father.gpu.name = "x";
  for (auto& n : gpu_names.nodes) n.gpu.name = "y";
  PlacementResult rg = search(gpu_names);
  REQUIRE(rg.candidates.size() == rb.candidates.size());
  for (std::size_t i = 0; i < rb.candidates.size(); ++i) CHECK(rg.candidates[i].plan_hash == rb.candidates[i].plan_hash);

  // Swapping the NUMBERS of the two nodes (keeping ids) swaps who gets which range: capability, not identity,
  // decides. Nothing in the planner knows the illustrative Father/G14/3060 split.
  PlacementRequest swapped = base;
  std::swap(swapped.nodes[0].cpu, swapped.nodes[1].cpu);
  std::swap(swapped.nodes[0].gpu, swapped.nodes[1].gpu);
  std::swap(swapped.nodes[0].memory, swapped.nodes[1].memory);
  std::swap(swapped.nodes[0].overheads, swapped.nodes[1].overheads);
  PlacementResult rs = search(swapped);
  const std::map<std::string, std::string> swap_names = {{kG14, kN3060}, {kN3060, kG14}};
  PlacementRequest relabelled = relabel(base, swap_names);  // same numbers, ids swapped, request order kept
  // relabel() renamed ids but kept profile positions: node[0] now carries the 3060 id with G14 numbers.
  PlacementResult rl = search(relabelled);
  REQUIRE(rs.candidates.size() == rl.candidates.size());
  CHECK(recommended(rs).metrics.objective == doctest::Approx(recommended(rl).metrics.objective));
  CHECK(recommended(rs).metrics.decode_tok_s == doctest::Approx(recommended(rl).metrics.decode_tok_s));
}
