// Placement tests run on SYNTHETIC fixtures: they check rational direction of change and invariants, never
// that a plan is optimal for real hardware.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "clusterlm/placement/placement.hpp"

using namespace clusterlm;
using namespace clusterlm::placement;

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
const char* kFatherId = "Father-4060Ti-7600";
const char* kG14 = "Node-G14-4070-8945HS";
const char* kN3060 = "Node-3060-5600";

std::string fixture(const char* name) { return std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/" + name; }

HardwareProfile load_hw(const char* name) {
  auto r = load_hardware_profile(fixture(name));
  REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
  return *r;
}

PlacementRequest base_request() {
  PlacementRequest r;
  r.father = load_hw("Father-4060Ti-7600.json");
  r.nodes = {load_hw("Node-G14-4070-8945HS.json"), load_hw("Node-3060-5600.json")};
  auto n = load_network_profile(fixture("network-gige-simulated.json"));
  REQUIRE_MESSAGE(n.is_ok(), n.status().to_string());
  r.network = *n;
  r.model = flash_next_planning_estimate();
  r.context_tokens = 4096;
  r.q = 1;
  r.default_lease = {28800, 288000};  // 8 h lease: preparation is heavily amortised
  return r;
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

const MemoryLedger* ledger_of(const PlacementPlan& p, const std::string& id) {
  for (const auto& l : p.ledgers)
    if (l.domain_id == id) return &l;
  return nullptr;
}

bool uses(const PlacementPlan& p, const std::string& id) {
  for (const auto& s : p.stages)
    if (s.domain_id == id) return true;
  return false;
}

double cpu_served_bytes(const PlacementPlan& p, const std::string& id) {
  double b = 0;
  for (const auto& s : p.metrics.stages)
    if (s.domain_id == id) b += s.cpu_miss_bytes_expected;
  return b;
}

double cpu_share(const PlacementPlan& p, const std::string& id) {
  double total = 0;
  for (const auto& s : p.metrics.stages) total += s.cpu_miss_bytes_expected;
  return total > 0 ? cpu_served_bytes(p, id) / total : 0.0;
}

void scale_cpu(HardwareProfile& p, double f) {
  for (auto& [k, q] : p.cpu.expert_bytes_per_s) q.value *= f;
}

HardwareProfile& node(PlacementRequest& r, const char* id) {
  for (auto& n : r.nodes)
    if (n.id == id) return n;
  FAIL("no such node");
  return r.nodes.front();
}

LinkProfile& link_to(PlacementRequest& r, const char* id) {
  for (auto& l : r.network.links)
    if (l.from == kFatherId && l.to == id) return l;
  FAIL("no such link");
  return r.network.links.front();
}

}  // namespace

TEST_CASE("fixtures load, are synthetic everywhere, and round-trip through JSON preserving provenance") {
  for (const char* f : {"Father-4060Ti-7600.json", "Node-G14-4070-8945HS.json", "Node-3060-5600.json"}) {
    HardwareProfile p = load_hw(f);
    CHECK(p.synthetic_fixture);
    CHECK(weakest_provenance(p) == Provenance::kSynthetic);
    int n = 0;
    for_each_quantity(p, [&](const std::string&, Quantity& q) {
      ++n;
      CHECK(q.provenance == Provenance::kSynthetic);
      CHECK(q.source.find("synthetic") != std::string::npos);
    });
    CHECK(n > 15);

    // Round trip, with a measured field mixed in to prove provenance is per field.
    p.cpu.q_scaling = Quantity::measured(0.07, "bench:run-42");
    const std::string text = to_json(p);
    auto back = hardware_profile_from_json(text);
    REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
    CHECK(to_json(*back) == text);
    CHECK(back->cpu.q_scaling == p.cpu.q_scaling);
    CHECK(back->cpu.q_scaling.provenance == Provenance::kMeasured);
    CHECK(back->cpu.q_scaling.source == "bench:run-42");
    CHECK(back->id == p.id);
    CHECK(back->cpu.features == p.cpu.features);
    CHECK(back->gpu.dense_layer_ms.at("attention") == p.gpu.dense_layer_ms.at("attention"));
  }
  auto n = load_network_profile(fixture("network-gige-simulated.json"));
  REQUIRE(n.is_ok());
  CHECK(n->synthetic_fixture);
  auto back = network_profile_from_json(to_json(*n));
  REQUIRE(back.is_ok());
  CHECK(to_json(*back) == to_json(*n));
  CHECK(back->find_link(kG14, kFatherId) != nullptr);  // reverse lookup
  CHECK(back->find_link(kG14, "nope") == nullptr);

  auto bad = hardware_profile_from_json(R"({"id":"x"})");
  CHECK_FALSE(bad.is_ok());
  CHECK_FALSE(hardware_profile_from_json("not json").is_ok());

  const auto tmp = std::filesystem::temp_directory_path() / "clusterlm_placement_profile_test.json";
  HardwareProfile p = load_hw("Node-3060-5600.json");
  REQUIRE(save_hardware_profile(p, tmp.string()).is_ok());
  auto loaded = load_hardware_profile(tmp.string());
  REQUIRE(loaded.is_ok());
  CHECK(to_json(*loaded) == to_json(p));
  std::filesystem::remove(tmp);
}

TEST_CASE("a synthetic plan cannot pass require_qualified; qualification demands non-synthetic inputs") {
  PlacementRequest r = base_request();
  PlacementResult res = search(r);
  REQUIRE_FALSE(res.candidates.empty());
  CHECK(res.provenance == Provenance::kSynthetic);
  for (const auto& p : res.candidates) {
    CHECK(p.provenance == Provenance::kSynthetic);
    Status st = require_qualified(p);
    CHECK_FALSE(st.is_ok());
    CHECK(st.code() == ErrorCode::kFailedPrecondition);
  }

  // Marking Qualified with synthetic fields fails and leaves the profile untouched.
  HardwareProfile p = load_hw("Node-3060-5600.json");
  const std::string before = to_json(p);
  Status st = mark_qualified(p, "acceptance-run-1");
  CHECK_FALSE(st.is_ok());
  CHECK(to_json(p) == before);
  NetworkProfile net = r.network;
  CHECK_FALSE(mark_qualified(net, "acceptance-run-1").is_ok());

  // Promote everything through the explicit path: synthetic -> measured (a benchmark), then qualified.
  auto to_measured = [](const std::string&, Quantity& q) {
    q.provenance = Provenance::kMeasured;
    q.source = "bench:run-1";
  };
  for_each_quantity(r.father, to_measured);
  for (auto& n : r.nodes) for_each_quantity(n, to_measured);
  for_each_quantity(r.network, to_measured);
  SUBCASE("measured but unqualified stays below Qualified") {
    r.model.provenance = Provenance::kMeasured;
    r.model.draft_ms.provenance = Provenance::kMeasured;
    r.acceptance.provenance = Provenance::kMeasured;
    PlacementResult m = search(r);
    REQUIRE_FALSE(m.candidates.empty());
    CHECK(recommended(m).provenance == Provenance::kMeasured);
    CHECK_FALSE(require_qualified(recommended(m)).is_ok());
  }
  SUBCASE("fully qualified inputs pass") {
    CHECK(mark_qualified(r.father, "q1").is_ok());
    for (auto& n : r.nodes) CHECK(mark_qualified(n, "q1").is_ok());
    CHECK(mark_qualified(r.network, "q1").is_ok());
    r.model.provenance = Provenance::kQualified;
    r.model.draft_ms.provenance = Provenance::kQualified;
    r.acceptance.provenance = Provenance::kQualified;
    PlacementResult m = search(r);
    REQUIRE_FALSE(m.candidates.empty());
    CHECK(m.provenance == Provenance::kQualified);
    CHECK(require_qualified(recommended(m)).is_ok());
    CHECK(recommended(m).non_qualified_inputs.empty());
    // One synthetic input anywhere in the plan drags it down.
    r.acceptance.provenance = Provenance::kSynthetic;
    PlacementResult m2 = search(r);
    CHECK_FALSE(require_qualified(recommended(m2)).is_ok());
  }
}

TEST_CASE("admission rejects plans over RAM and VRAM with reasons; Father-only is decided explicitly") {
  PlacementRequest r = base_request();
  PlacementResult res = search(r);

  // The flash-next estimate (~50 GB of experts) cannot fit on Father alone.
  const std::string key = plan_key({{kFatherId, StageRole::kFull, {0, 48}}});
  const PlacementPlan* feasible = res.find(key);
  const RejectedPlan* rejected = res.find_rejected(key);
  CHECK((feasible != nullptr) != (rejected != nullptr));  // exactly one outcome, never silently dropped
  REQUIRE(rejected != nullptr);
  REQUIRE_FALSE(rejected->reasons.empty());
  CHECK(rejected->reasons.front().find("RAM") != std::string::npos);

  // Every feasible plan respects every ledger limit; every rejected plan says why.
  for (const auto& p : res.candidates)
    for (const auto& l : p.ledgers) {
      CHECK(l.ram_used() <= l.ram_safe_allowance);
      CHECK(l.vram_used() <= l.vram_budget);
    }
  for (const auto& x : res.rejected) CHECK_FALSE(x.reasons.empty());
  CHECK(res.rejected.size() > 0);

  // VRAM: a Father whose GPU budget cannot hold its dense layers rejects every plan (a tiny budget means the
  // prefix's dense weights alone overflow).
  PlacementRequest v = base_request();
  v.father.gpu.vram_budget = Quantity::synthetic(0.05 * kGiB);
  PlacementResult vr = search(v);
  CHECK(vr.candidates.empty());
  REQUIRE_FALSE(vr.rejected.empty());
  bool saw_vram = false;
  for (const auto& x : vr.rejected)
    for (const auto& why : x.reasons) saw_vram |= why.find("VRAM") != std::string::npos;
  CHECK(saw_vram);

  // Small model: Father-only becomes feasible and has no transport or provisioning.
  PlacementRequest s = base_request();
  FixtureEstimateSpec spec;
  spec.n_layers = 8;
  spec.expert_bytes = 1ull << 20;
  s.model = fixture_estimate(spec);
  s.prefix_layers_options = {2};
  PlacementResult sr = search(s);
  const PlacementPlan* fo = sr.find(plan_key({{kFatherId, StageRole::kFull, {0, 8}}}));
  REQUIRE(fo != nullptr);
  CHECK(fo->metrics.transports.empty());
  CHECK(fo->metrics.provisioning_bytes == 0);
  CHECK(fo->metrics.commit_ms == 0);
  CHECK(fo->node_ids().empty());
}

TEST_CASE("no placement decision depends on device names") {
  PlacementRequest a = base_request();
  PlacementRequest b = base_request();
  b.father.gpu.name = "Totally Different GPU";
  for (auto& n : b.nodes) n.gpu.name = "Mystery Accelerator";
  PlacementResult ra = search(a), rb = search(b);
  REQUIRE(ra.candidates.size() == rb.candidates.size());
  for (std::size_t i = 0; i < ra.candidates.size(); ++i) {
    CHECK(ra.candidates[i].plan_hash == rb.candidates[i].plan_hash);
    CHECK(ra.candidates[i].metrics.decode_tok_s == rb.candidates[i].metrics.decode_tok_s);
  }
}

TEST_CASE("search output is well formed: sorted, pareto-optimal, deterministic hash") {
  PlacementRequest r = base_request();
  PlacementResult res = search(r);
  REQUIRE(res.candidates.size() > 10);
  for (std::size_t i = 1; i < res.candidates.size(); ++i)
    CHECK(res.candidates[i - 1].metrics.objective <= res.candidates[i].metrics.objective);
  CHECK(*res.recommended == 0);
  CHECK(res.candidates[*res.best_throughput].metrics.decode_tok_s >= recommended(res).metrics.decode_tok_s);
  CHECK(res.candidates[*res.best_preparation].metrics.prepare_s <= recommended(res).metrics.prepare_s);

  // Pareto (3-D: prepare_s min, decode_tok_s max, prefill_s min): members are mutually non-dominated; every
  // non-member is dominated by or tied with some member.
  REQUIRE_FALSE(res.pareto.empty());
  auto dominates = [&](const PredictedMetrics& a, const PredictedMetrics& b) {
    return a.prepare_s <= b.prepare_s && a.decode_tok_s >= b.decode_tok_s && a.prefill_s <= b.prefill_s &&
           (a.prepare_s < b.prepare_s || a.decode_tok_s > b.decode_tok_s || a.prefill_s < b.prefill_s);
  };
  for (std::size_t i = 0; i < res.candidates.size(); ++i) {
    const bool member = std::find(res.pareto.begin(), res.pareto.end(), i) != res.pareto.end();
    bool dominated = false;
    for (std::size_t j = 0; j < res.candidates.size(); ++j)
      dominated |= dominates(res.candidates[j].metrics, res.candidates[i].metrics);
    if (member) CHECK_FALSE(dominated);
    else {
      bool covered = false;
      for (std::size_t j : res.pareto)
        covered |= res.candidates[j].metrics.prepare_s <= res.candidates[i].metrics.prepare_s &&
                   res.candidates[j].metrics.decode_tok_s >= res.candidates[i].metrics.decode_tok_s &&
                   res.candidates[j].metrics.prefill_s <= res.candidates[i].metrics.prefill_s;
      CHECK(covered);
    }
  }

  // Hash: deterministic across runs and across re-evaluation of the stored stage list; distinct plans differ.
  PlacementResult again = search(r);
  REQUIRE(again.candidates.size() == res.candidates.size());
  for (std::size_t i = 0; i < res.candidates.size(); ++i) CHECK(res.candidates[i].plan_hash == again.candidates[i].plan_hash);
  PlanEvaluation re = evaluate_plan(r, recommended(res).stages);
  REQUIRE(re.plan.has_value());
  CHECK(re.plan->plan_hash == recommended(res).plan_hash);
  CHECK(res.candidates[0].plan_hash != res.candidates[1].plan_hash);
  CHECK(recommended(res).plan_hash.hex().size() == 64);
  PlacementRequest r8 = base_request();
  r8.context_tokens = 8192;
  PlanEvaluation other_ctx = evaluate_plan(r8, recommended(res).stages);
  if (other_ctx.plan) CHECK(other_ctx.plan->plan_hash != recommended(res).plan_hash);

  // Report JSON states provenance prominently and parses.
  const std::string report = report_to_json(res);
  auto j = nlohmann::json::parse(report);
  CHECK(j["provenance"] == "synthetic");
  CHECK(j["provenance_banner"].get<std::string>().find("SYNTHETIC") == 0);
  CHECK(j["recommended_plan"]["provenance"] == "synthetic");
  CHECK(j["recommended_plan"]["plan_hash"] == recommended(res).plan_hash.hex());
  CHECK(j["rejected"].size() == res.rejected.size());
  auto full = nlohmann::json::parse(plan_to_json(recommended(res), true));
  CHECK(full["expert_residency"][0].contains("gpu_experts_by_layer"));
}

TEST_CASE("search over the flash-next estimate at granularity 4 is fast") {
  PlacementRequest r = base_request();
  const auto t0 = std::chrono::steady_clock::now();
  PlacementResult res = search(r);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  CHECK(s < 2.0);
  CHECK(res.candidates.size() + res.rejected.size() > 100);
}

TEST_CASE("residency fills VRAM by routing frequency and respects the ledger") {
  PlacementRequest r = base_request();
  PlacementResult res = search(r);
  const PlacementPlan& p = recommended(res);
  for (std::size_t i = 0; i < p.ledgers.size(); ++i) {
    const auto& led = p.ledgers[i];
    const auto& rs = p.residency[i];
    // Spare VRAM is smaller than the largest expert (greedy fill leaves no usable gap).
    CHECK(led.vram_budget - led.vram_used() < 2'030'000);
    std::uint64_t counted = 0;
    for (std::size_t l = 0; l < rs.gpu_experts_by_layer.size(); ++l) {
      counted += rs.gpu_experts_by_layer[l].size();
      // Hot experts first: every GPU-resident expert is at least as frequent as the hottest non-resident one
      // unless that layer's stream was cut off by a full budget.
      if (rs.gpu_experts_by_layer[l].empty()) continue;
      double min_in = 1.0;
      for (auto e : rs.gpu_experts_by_layer[l]) min_in = std::min(min_in, r.model.routing_freq[l][e]);
      std::vector<bool> in(512, false);
      for (auto e : rs.gpu_experts_by_layer[l]) in[e] = true;
      double max_out = 0;
      for (std::size_t e = 0; e < 512; ++e)
        if (!in[e]) max_out = std::max(max_out, r.model.routing_freq[l][e]);
      CHECK(min_in >= max_out);
    }
    CHECK(counted == rs.gpu_expert_count);
    CHECK(rs.gpu_expert_count * 2'030'000 == led.gpu_experts);
  }
}

TEST_CASE("scenario: slower G14 CPU does not raise predicted decode nor G14's CPU-served share") {
  PlacementRequest base = base_request();
  PlacementRequest slow = base_request();
  scale_cpu(node(slow, kG14), 0.7);
  PlacementResult rb = search(base), rs = search(slow);
  REQUIRE_FALSE(rb.candidates.empty());
  REQUIRE_FALSE(rs.candidates.empty());
  const PlacementPlan& pb = recommended(rb);
  const PlacementPlan& ps = recommended(rs);
  INFO("baseline " << pb.key << " slow " << ps.key);
  CHECK(ps.metrics.decode_tok_s <= pb.metrics.decode_tok_s + 1e-9);
  CHECK(rs.candidates[*rs.best_throughput].metrics.decode_tok_s <= rb.candidates[*rb.best_throughput].metrics.decode_tok_s + 1e-9);
  CHECK(cpu_share(ps, kG14) <= cpu_share(pb, kG14) + 1e-9);
  // The same plan is never faster when a participant gets slower.
  for (const auto& p : rb.candidates) {
    if (!uses(p, kG14)) continue;
    const PlacementPlan* q = rs.find(p.key);
    REQUIRE(q != nullptr);
    CHECK(q->metrics.decode_tok_s <= p.metrics.decode_tok_s + 1e-9);
  }
}

TEST_CASE("scenario: 3060 limited to 8.8 GiB safe RAM") {
  PlacementRequest base = base_request();
  PlacementRequest tight = base_request();
  node(tight, kN3060).memory.ram_safe_allowance = Quantity::synthetic(8.8 * kGiB);
  PlacementResult rb = search(base), rt = search(tight);
  const double limit = 8.8 * kGiB;
  for (const auto& p : rt.candidates) {
    const MemoryLedger* l = ledger_of(p, kN3060);
    if (l) CHECK(static_cast<double>(l->ram_used()) <= limit);
  }
  int newly_rejected = 0;
  for (const auto& p : rb.candidates) {
    const MemoryLedger* l = ledger_of(p, kN3060);
    if (l && static_cast<double>(l->ram_used()) > limit) {
      ++newly_rejected;
      CHECK(rt.find(p.key) == nullptr);
      const RejectedPlan* x = rt.find_rejected(p.key);
      REQUIRE(x != nullptr);
      bool mentions = false;
      for (const auto& why : x->reasons) mentions |= why.find(kN3060) != std::string::npos && why.find("RAM") != std::string::npos;
      CHECK(mentions);
    }
  }
  CHECK(newly_rejected > 0);
  CHECK(rt.candidates.size() < rb.candidates.size());
}

TEST_CASE("scenario: slower Father->G14 link raises provisioning time of G14 plans") {
  PlacementRequest base = base_request();
  PlacementRequest slow = base_request();
  link_to(slow, kG14).bandwidth_bytes_per_s = Quantity::synthetic(72e6);
  PlacementResult rb = search(base), rs = search(slow);
  int checked = 0, strict = 0;
  for (const auto& p : rb.candidates) {
    if (!uses(p, kG14)) continue;
    const PlacementPlan* q = rs.find(p.key);
    REQUIRE(q != nullptr);
    CHECK(q->metrics.provisioning_s >= p.metrics.provisioning_s);
    CHECK(q->metrics.prepare_s >= p.metrics.prepare_s);
    CHECK(q->metrics.provisioning_bytes == p.metrics.provisioning_bytes);
    if (q->metrics.provisioning_s > p.metrics.provisioning_s) ++strict;
    ++checked;
  }
  CHECK(checked > 0);
  CHECK(strict > 0);
  // Plans without G14 are unaffected.
  for (const auto& p : rb.candidates) {
    if (uses(p, kG14)) continue;
    const PlacementPlan* q = rs.find(p.key);
    REQUIRE(q != nullptr);
    CHECK(q->metrics.provisioning_s == p.metrics.provisioning_s);
  }
}

TEST_CASE("provisioning shares the Father NIC: two nodes are not parallel-additive") {
  PlacementRequest r = base_request();
  PlacementResult res = search(r);
  int two_node = 0;
  for (const auto& p : res.candidates) {
    if (p.node_ids().size() != 2) continue;
    ++two_node;
    const double egress = r.network.father_egress_bytes_per_s.value;
    CHECK(p.metrics.provisioning_s >= p.metrics.provisioning_bytes / egress - 1e-9);
  }
  CHECK(two_node > 0);
}

TEST_CASE("scenario: G14 expected lease of 25 minutes") {
  PlacementRequest base = base_request();
  PlacementRequest short_lease = base_request();
  short_lease.node_lease[kG14] = {25 * 60.0, 15000};
  PlacementResult rb = search(base), rl = search(short_lease);
  const PlacementPlan& pb = recommended(rb);
  const PlacementPlan& pl = recommended(rl);
  INFO("baseline " << pb.key << " short " << pl.key);
  const bool flagged = !pl.metrics.flags.empty();
  CHECK((pl.metrics.prepare_s <= pb.metrics.prepare_s + 1e-9 || flagged));
  // A harsher expectation (a lease shorter than the preparation) is flagged on every plan using G14 that
  // needs more preparation than that, and is never preferred over an unflagged equivalent.
  PlacementRequest tiny = base_request();
  tiny.node_lease[kG14] = {5.0, 100};
  PlacementResult rt = search(tiny);
  for (const auto& p : rt.candidates) {
    const bool has_flag = std::find(p.metrics.flags.begin(), p.metrics.flags.end(), std::string("lease_overrun:") + kG14) != p.metrics.flags.end();
    if (uses(p, kG14) && p.metrics.provisioning_s > 5.0) CHECK(has_flag);
    if (!uses(p, kG14)) CHECK_FALSE(has_flag);
  }
  CHECK(recommended(rt).metrics.prepare_s <= pb.metrics.prepare_s + 1e-9);
}

TEST_CASE("scenario: already-provisioned layers avoid reprovisioning at a short lease") {
  PlacementRequest r = base_request();
  r.default_lease = {600, 256};  // short lease: preparation is charged in full
  PlacementResult res = search(r);
  const PlacementPlan* with_g14 = nullptr;
  for (const auto& p : res.candidates)
    if (uses(p, kG14) && p.metrics.provisioning_bytes > 10.0 * kGiB) {
      with_g14 = &p;
      break;
    }
  REQUIRE(with_g14 != nullptr);
  LayerRange mine;
  for (const auto& s : with_g14->stages)
    if (s.domain_id == kG14) mine = s.layers;

  PlanEvaluation cold = evaluate_plan(r, with_g14->stages);
  PlacementRequest warm_req = r;
  warm_req.already_provisioned[kG14] = {mine};
  PlanEvaluation warm = evaluate_plan(warm_req, with_g14->stages);
  REQUIRE(cold.plan.has_value());
  REQUIRE(warm.plan.has_value());
  CHECK(cold.plan->metrics.provisioning_bytes > 10.0 * kGiB);  // the "15 GB reprovisioning" case
  CHECK(warm.plan->metrics.provisioning_bytes < cold.plan->metrics.provisioning_bytes);
  CHECK(warm.plan->metrics.provisioning_s < cold.plan->metrics.provisioning_s);
  CHECK(warm.plan->metrics.t_request_s < cold.plan->metrics.t_request_s);
  CHECK(warm.plan->metrics.objective < cold.plan->metrics.objective);
  // Same assignment, same hash: provisioning state is not part of the placement identity.
  CHECK(warm.plan->plan_hash == cold.plan->plan_hash);
  // Partial reuse lies strictly between.
  PlacementRequest part_req = r;
  part_req.already_provisioned[kG14] = {{mine.begin, mine.begin + mine.size() / 2}};
  PlanEvaluation part = evaluate_plan(part_req, with_g14->stages);
  REQUIRE(part.plan.has_value());
  CHECK(part.plan->metrics.provisioning_bytes < cold.plan->metrics.provisioning_bytes);
  CHECK(part.plan->metrics.provisioning_bytes > warm.plan->metrics.provisioning_bytes);
  // Everything provisioned on every node: nothing to transfer.
  PlacementRequest all_req = r;
  for (const auto& s : with_g14->stages)
    if (s.role == StageRole::kMiddle) all_req.already_provisioned[s.domain_id] = {s.layers};
  PlanEvaluation all = evaluate_plan(all_req, with_g14->stages);
  REQUIRE(all.plan.has_value());
  CHECK(all.plan->metrics.provisioning_bytes == 0);
  CHECK(all.plan->metrics.provisioning_s == 0);
}

TEST_CASE("verify width q=4 scales transport bytes by 4 and never speeds up a round") {
  PlacementRequest r1 = base_request();
  PlacementResult res = search(r1);
  const PlacementPlan& p1 = [&]() -> const PlacementPlan& {
    for (const auto& p : res.candidates)
      if (p.node_ids().size() == 2) return p;
    return recommended(res);
  }();
  REQUIRE_FALSE(p1.metrics.transports.empty());
  PlacementRequest r4 = base_request();
  r4.q = 4;
  PlanEvaluation e4 = evaluate_plan(r4, p1.stages);
  REQUIRE(e4.plan.has_value());
  const auto& p4 = *e4.plan;
  REQUIRE(p4.metrics.transports.size() == p1.metrics.transports.size());
  CHECK(p4.metrics.transport_bytes_per_round == doctest::Approx(4.0 * p1.metrics.transport_bytes_per_round));
  for (std::size_t i = 0; i < p1.metrics.transports.size(); ++i)
    CHECK(p4.metrics.transports[i].bytes_per_round == doctest::Approx(4.0 * p1.metrics.transports[i].bytes_per_round));
  CHECK(p1.metrics.transport_bytes_per_round == doctest::Approx(static_cast<double>(p1.metrics.transports.size()) * 51'216.0));
  CHECK(p4.metrics.round_ms > p1.metrics.round_ms);
  // Expert union grows sublinearly in q: CPU-served bytes at q=4 are < 4x those at q=1.
  double c1 = 0, c4 = 0;
  for (const auto& s : p1.metrics.stages) c1 += s.cpu_miss_bytes_expected;
  for (const auto& s : p4.metrics.stages) c4 += s.cpu_miss_bytes_expected;
  CHECK(c4 > c1);
  CHECK(c4 < 4.0 * c1);
  // With acceptance A higher at q=4 the per-token cost can still win; the model just divides.
  r4.acceptance = Quantity::synthetic(3.0);
  PlanEvaluation e4b = evaluate_plan(r4, p1.stages);
  REQUIRE(e4b.plan.has_value());
  CHECK(e4b.plan->metrics.decode_tok_s == doctest::Approx(1000.0 * 3.0 / e4b.plan->metrics.round_ms));
}

TEST_CASE("cost model directions: network, overlap, GPU speed") {
  PlacementRequest r = base_request();
  PlacementResult res = search(r);
  const PlacementPlan& p = recommended(res);

  PlacementRequest slow_net = base_request();
  for (auto& l : slow_net.network.links) l.rtt_ms.value *= 10;
  PlanEvaluation a = evaluate_plan(slow_net, p.stages);
  REQUIRE(a.plan.has_value());
  if (!p.node_ids().empty()) {
    CHECK(a.plan->metrics.decode_tok_s < p.metrics.decode_tok_s);
    CHECK(a.plan->metrics.commit_ms > p.metrics.commit_ms);
  }

  PlacementRequest overlap = base_request();
  overlap.overlap_factor = 0.5;
  PlanEvaluation b = evaluate_plan(overlap, p.stages);
  REQUIRE(b.plan.has_value());
  CHECK(b.plan->metrics.decode_tok_s >= p.metrics.decode_tok_s);

  PlacementRequest fast_gpu = base_request();
  fast_gpu.father.gpu.gpu_expert_bytes_per_s.value *= 2;
  for (auto& n : fast_gpu.nodes) n.gpu.gpu_expert_bytes_per_s.value *= 2;
  PlanEvaluation c = evaluate_plan(fast_gpu, p.stages);
  REQUIRE(c.plan.has_value());
  CHECK(c.plan->metrics.decode_tok_s >= p.metrics.decode_tok_s);

  PlacementRequest bigger_ctx = base_request();
  bigger_ctx.context_tokens = 16384;
  PlanEvaluation d = evaluate_plan(bigger_ctx, p.stages);
  if (d.plan) {
    CHECK(d.plan->metrics.prefill_s > p.metrics.prefill_s);
    for (std::size_t i = 0; i < d.plan->ledgers.size(); ++i) CHECK(d.plan->ledgers[i].state >= p.ledgers[i].state);
  }
}

TEST_CASE("invalid requests and missing measurements are errors or rejections, never defaults") {
  PlacementRequest r = base_request();
  r.q = 0;
  CHECK_FALSE(search_placements(r).is_ok());

  PlacementRequest dup = base_request();
  dup.nodes.push_back(dup.nodes.front());
  CHECK_FALSE(search_placements(dup).is_ok());

  // A node with no throughput for the model's quant type is rejected for every plan using it.
  PlacementRequest nq = base_request();
  node(nq, kG14).cpu.expert_bytes_per_s.clear();
  PlacementResult res = search(nq);
  for (const auto& p : res.candidates) CHECK_FALSE(uses(p, kG14));
  bool saw = false;
  for (const auto& x : res.rejected)
    for (const auto& why : x.reasons) saw |= why.find("quant") != std::string::npos;
  CHECK(saw);

  // No link between two nodes -> plans chaining them are rejected.
  PlacementRequest nl = base_request();
  std::erase_if(nl.network.links, [](const LinkProfile& l) { return l.from == kG14 && l.to == kN3060; });
  PlacementResult rn = search(nl);
  for (const auto& p : rn.candidates) {
    const auto ids = p.node_ids();
    CHECK(ids.size() < 2);
  }

  // Profile validation.
  HardwareProfile bad = load_hw("Node-3060-5600.json");
  bad.cpu.sustained_factor.value = 1.5;
  CHECK_FALSE(validate(bad).is_ok());
}

TEST_CASE("model estimates are structurally valid") {
  ModelCostInputs m = flash_next_planning_estimate();
  CHECK(validate(m).is_ok());
  CHECK(m.n_layers() == 48);
  CHECK(m.n_experts == 512);
  CHECK(m.n_active == 10);
  CHECK(m.boundary_bytes_per_position == 51'216);
  CHECK(m.ple_layer == 2);
  CHECK(m.provenance == Provenance::kSynthetic);
  CHECK(m.draft_ms.provenance == Provenance::kSynthetic);
  int attn = 0;
  for (const auto& l : m.layers) attn += l.kind == LayerKind::kAttention;
  CHECK(attn == 12);
  for (const auto& row : m.routing_freq) {
    double mx = 0;
    for (double p : row) mx = std::max(mx, p);
    CHECK(mx <= 1.0);
    CHECK(mx > 5.0 / 512);  // skewed, not uniform
  }
  CHECK(validate(fixture_estimate()).is_ok());
}
