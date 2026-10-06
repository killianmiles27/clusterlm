// Replan triggers. Scenarios from the design ("laptop CPU 30% slower", "3060 only has 8.8 GiB", "Father->G14
// bandwidth falls to 72 MB/s", "G14 leased only 25 minutes", "node needs 15 GB reprovisioning") assert the
// RATIONAL DIRECTION of the decision under the cost model, not optimality on real hardware.
#include <doctest/doctest.h>

#include <algorithm>

#include "clusterlm/placement/placement.hpp"
#include "clusterlm/placement/profile.hpp"
#include "clusterlm/planning/replan.hpp"

using namespace clusterlm;
using namespace clusterlm::placement;
using namespace clusterlm::planning;

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
const char* kFather = "Father-4060Ti-7600";
const char* kG14 = "Node-G14-4070-8945HS";
const char* kN3060 = "Node-3060-5600";

std::string fixture(const char* name) { return std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/" + name; }

PlacementRequest base_request() {
  PlacementRequest r;
  auto f = load_hardware_profile(fixture("Father-4060Ti-7600.json"));
  auto a = load_hardware_profile(fixture("Node-G14-4070-8945HS.json"));
  auto b = load_hardware_profile(fixture("Node-3060-5600.json"));
  auto n = load_network_profile(fixture("network-gige-simulated.json"));
  REQUIRE(f.is_ok());
  REQUIRE(a.is_ok());
  REQUIRE(b.is_ok());
  REQUIRE(n.is_ok());
  r.father = *f;
  r.nodes = {*a, *b};
  r.network = *n;
  r.model = flash_next_planning_estimate();
  r.context_tokens = 4096;
  r.default_lease = {28800, 288000};
  return r;
}

bool uses(const PlacementPlan& p, const char* id) {
  return std::any_of(p.stages.begin(), p.stages.end(), [&](const PlanStage& s) { return s.domain_id == id; });
}

// A deployed plan that runs on the given node (best-ranked one), from the baseline world.
PlacementPlan deployed_using(const PlacementRequest& r, const char* node) {
  auto res = search_placements(r);
  REQUIRE(res.is_ok());
  for (const auto& p : res->candidates)
    if (uses(p, node)) return p;
  FAIL("no feasible plan uses " << node);
  return {};
}

bool has_reason(const ReplanDecision& d, ReplanReasonKind k) {
  return std::any_of(d.reasons.begin(), d.reasons.end(), [&](const ReplanReason& r) { return r.kind == k; });
}
const ReplanReason* find_reason(const ReplanDecision& d, ReplanReasonKind k) {
  for (const auto& r : d.reasons)
    if (r.kind == k) return &r;
  return nullptr;
}

double ledger_ram(const PlacementPlan& p, const char* id) {
  for (const auto& l : p.ledgers)
    if (l.domain_id == id) return static_cast<double>(l.ram_used());
  return 0;
}

double cpu_served(const PlacementPlan& p, const char* id) {
  double b = 0;
  for (const auto& s : p.metrics.stages)
    if (s.domain_id == id) b += s.cpu_miss_bytes_expected;
  return b;
}

ReplanInputs inputs(const PlacementRequest& r) {
  ReplanInputs in;
  in.request = r;
  in.policy.decode_regression = 0.001;  // report even small slowdowns so the direction is visible
  return in;
}

}  // namespace

TEST_CASE("unchanged inputs: keep the plan, nothing to reprovision, no regression reasons") {
  const PlacementRequest r = base_request();
  const PlacementPlan cur = deployed_using(r, kG14);
  auto d = should_replan(cur, inputs(r));
  REQUIRE_MESSAGE(d.is_ok(), d.status().to_string());
  CHECK_FALSE(d->replan);
  CHECK_FALSE(d->reprovision_needed);
  CHECK(d->reprovision_bytes == 0);
  CHECK(d->reasons.empty());
  REQUIRE(d->current_under_new_inputs.has_value());
  CHECK(d->current_under_new_inputs->plan_hash == cur.plan_hash);
  CHECK(d->current_under_new_inputs->metrics.decode_tok_s == doctest::Approx(cur.metrics.decode_tok_s));
}

TEST_CASE("scenario: laptop CPU expert execution becomes 30% slower") {
  const PlacementRequest r = base_request();
  const PlacementPlan cur = deployed_using(r, kG14);
  PlacementRequest slow = r;
  for (auto& n : slow.nodes)
    if (n.id == kG14)
      for (auto& [k, q] : n.cpu.expert_bytes_per_s) q.value *= 0.7;
  auto d = should_replan(cur, inputs(slow));
  REQUIRE_MESSAGE(d.is_ok(), d.status().to_string());
  REQUIRE(d->current_under_new_inputs.has_value());
  // The deployed plan gets slower, and the reason names the slowed stage.
  CHECK(d->current_under_new_inputs->metrics.decode_tok_s < cur.metrics.decode_tok_s);
  const ReplanReason* dec = find_reason(*d, ReplanReasonKind::kDecodeRegression);
  REQUIRE(dec != nullptr);
  CHECK(dec->detail.find(kG14) != std::string::npos);
  CHECK(dec->magnitude > 0);
  // If it replans, the candidate is strictly better under the new inputs and leans less on the laptop's CPU.
  if (d->replan) {
    REQUIRE(d->candidate.has_value());
    CHECK(d->candidate->metrics.objective < d->current_under_new_inputs->metrics.objective);
    CHECK(cpu_served(*d->candidate, kG14) <= cpu_served(cur, kG14) + 1.0);
  }
  // Hysteresis: demanding a 99.99% gain suppresses the better-plan trigger but not the regression report.
  ReplanInputs strict = inputs(slow);
  strict.policy.min_objective_gain = 0.9999;
  auto s = should_replan(cur, strict);
  REQUIRE(s.is_ok());
  CHECK_FALSE(s->replan);
  CHECK(has_reason(*s, ReplanReasonKind::kDecodeRegression));
  // A zero-gain policy replans exactly when some other plan is strictly better.
  ReplanInputs eager = inputs(slow);
  eager.policy.min_objective_gain = 0.0;
  auto e = should_replan(cur, eager);
  REQUIRE(e.is_ok());
  CHECK((e->replan == (e->candidate && e->candidate->stages != cur.stages &&
                       e->candidate->metrics.objective < e->current_under_new_inputs->metrics.objective)));
}

TEST_CASE("scenario: the 3060 only has 8.8 GiB safely available") {
  const PlacementRequest r = base_request();
  auto res = search_placements(r);
  REQUIRE(res.is_ok());
  const PlacementPlan* cur = nullptr;
  for (const auto& p : res->candidates)
    if (uses(p, kN3060) && ledger_ram(p, kN3060) > 8.8 * kGiB) {
      cur = &p;
      break;
    }
  REQUIRE(cur != nullptr);
  PlacementRequest tight = r;
  for (auto& n : tight.nodes)
    if (n.id == kN3060) n.memory.ram_safe_allowance = Quantity::synthetic(8.8 * kGiB);
  auto d = should_replan(*cur, inputs(tight));
  REQUIRE_MESSAGE(d.is_ok(), d.status().to_string());
  CHECK(d->replan);
  CHECK_FALSE(d->current_under_new_inputs.has_value());
  const ReplanReason* why = find_reason(*d, ReplanReasonKind::kPlanInadmissible);
  REQUIRE(why != nullptr);
  CHECK(why->detail.find(kN3060) != std::string::npos);
  CHECK(why->detail.find("RAM") != std::string::npos);
  REQUIRE_FALSE(d->inadmissible_reasons.empty());
  REQUIRE(d->candidate.has_value());
  CHECK(d->candidate->stages != cur->stages);
  CHECK(ledger_ram(*d->candidate, kN3060) <= 8.8 * kGiB);  // the new plan respects the new limit (or skips the node)
  // Reprovisioning is reported in bytes and matches the candidate (nodes keep what they already hold).
  CHECK(d->reprovision_bytes == doctest::Approx(d->candidate->metrics.provisioning_bytes));
  CHECK(d->reprovision_needed == (d->reprovision_bytes > 0));
}

TEST_CASE("scenario: Father->G14 bandwidth falls to 72 MB/s") {
  const PlacementRequest r = base_request();
  const PlacementPlan cur = deployed_using(r, kG14);
  PlacementRequest slow = r;
  for (auto& l : slow.network.links)
    if (l.from == kFather && l.to == kG14) l.bandwidth_bytes_per_s = Quantity::synthetic(72e6);

  // A running deployment already holds its bytes: a slower provisioning link costs nothing, so keep the plan.
  auto warm = should_replan(cur, inputs(slow));
  REQUIRE_MESSAGE(warm.is_ok(), warm.status().to_string());
  CHECK_FALSE(warm->replan);
  CHECK_FALSE(warm->reprovision_needed);

  // A plan that still has to be provisioned pays for the slower link: the same bytes take longer.
  ReplanInputs cold_before = inputs(r);
  cold_before.current_plan_provisioned = false;
  ReplanInputs cold_after = inputs(slow);
  cold_after.current_plan_provisioned = false;
  auto before = should_replan(cur, cold_before);
  auto after = should_replan(cur, cold_after);
  REQUIRE(before.is_ok());
  REQUIRE(after.is_ok());
  REQUIRE(before->current_under_new_inputs.has_value());
  REQUIRE(after->current_under_new_inputs.has_value());
  const auto& mb = before->current_under_new_inputs->metrics;
  const auto& ma = after->current_under_new_inputs->metrics;
  CHECK(ma.provisioning_bytes == doctest::Approx(mb.provisioning_bytes));
  CHECK(ma.provisioning_s >= mb.provisioning_s);
  CHECK(ma.prepare_s >= mb.prepare_s);
  CHECK(ma.round_ms >= mb.round_ms);  // and the activation hop / commit through that link is no faster
}

TEST_CASE("scenario: G14 is expected to remain leased only 25 minutes") {
  const PlacementRequest r = base_request();
  const PlacementPlan cur = deployed_using(r, kG14);
  PlacementRequest short_lease = r;
  short_lease.node_lease[kG14] = {25 * 60.0, 15000};
  ReplanInputs in = inputs(short_lease);
  in.current_plan_provisioned = false;  // still has to be prepared: the lease now matters
  auto d = should_replan(cur, in);
  REQUIRE_MESSAGE(d.is_ok(), d.status().to_string());
  REQUIRE(d->current_under_new_inputs.has_value());
  // One request now bears a bigger share of the node's preparation: amortised cost does not fall.
  CHECK(d->current_under_new_inputs->metrics.effective_prepare_s >= cur.metrics.effective_prepare_s - 1e-9);
  if (d->current_under_new_inputs->metrics.effective_prepare_s > cur.metrics.effective_prepare_s * 1.1)
    CHECK(has_reason(*d, ReplanReasonKind::kPreparationRegression));

  // A lease shorter than the preparation itself is flagged as an overrun.
  PlacementRequest tiny = r;
  tiny.node_lease[kG14] = {5.0, 100};
  ReplanInputs tin = inputs(tiny);
  tin.current_plan_provisioned = false;
  auto t = should_replan(cur, tin);
  REQUIRE(t.is_ok());
  const ReplanReason* flag = find_reason(*t, ReplanReasonKind::kPreparationRegression);
  REQUIRE(flag != nullptr);
  CHECK(flag->detail.find(std::string("lease_overrun:") + kG14) != std::string::npos);
  // Any replacement avoids the penalty it can: it is not worse than staying.
  if (t->replan) {
    REQUIRE(t->candidate.has_value());
    CHECK(t->candidate->metrics.objective < t->current_under_new_inputs->metrics.objective);
  }
}

TEST_CASE("scenario: a node requires 15 GB of reprovisioning") {
  PlacementRequest r = base_request();
  r.default_lease = {600, 256};  // short lease: preparation is charged in full
  auto res = search_placements(r);
  REQUIRE(res.is_ok());
  const PlacementPlan* cur = nullptr;
  for (const auto& p : res->candidates)
    if (uses(p, kG14) && p.metrics.provisioning_bytes > 10.0 * kGiB) {
      cur = &p;
      break;
    }
  REQUIRE(cur != nullptr);

  // Running deployment: nodes hold their layers, nothing to move.
  auto warm = should_replan(*cur, inputs(r));
  REQUIRE(warm.is_ok());
  CHECK_FALSE(warm->reprovision_needed);
  CHECK(warm->reprovision_bytes == 0);

  // Not yet provisioned (or the node lost its store): the same plan needs the full transfer, in bytes.
  ReplanInputs cold = inputs(r);
  cold.current_plan_provisioned = false;
  auto c = should_replan(*cur, cold);
  REQUIRE(c.is_ok());
  if (!c->replan) {
    CHECK(c->reprovision_needed);
    CHECK(c->reprovision_bytes == doctest::Approx(cur->metrics.provisioning_bytes));
    CHECK(c->reprovision_bytes > 10.0 * kGiB);
    CHECK(c->reprovision_s > 0);
  } else {
    // Replanning away from a 15 GB transfer only happens to something that is better overall.
    REQUIRE(c->candidate.has_value());
    CHECK(c->candidate->metrics.objective < c->current_under_new_inputs->metrics.objective);
  }
  // The cold current plan is strictly more expensive to keep than the warm one.
  REQUIRE(c->current_under_new_inputs.has_value());
  REQUIRE(warm->current_under_new_inputs.has_value());
  CHECK(c->current_under_new_inputs->metrics.objective > warm->current_under_new_inputs->metrics.objective);
}

TEST_CASE("scenario: a node disappears") {
  const PlacementRequest r = base_request();
  const PlacementPlan cur = deployed_using(r, kG14);
  ReplanInputs in = inputs(r);
  in.unavailable_nodes = {kG14};
  auto d = should_replan(cur, in);
  REQUIRE_MESSAGE(d.is_ok(), d.status().to_string());
  CHECK(d->replan);
  const ReplanReason* gone = find_reason(*d, ReplanReasonKind::kNodeUnavailable);
  REQUIRE(gone != nullptr);
  CHECK(gone->detail.find(kG14) != std::string::npos);
  if (d->candidate) {
    CHECK_FALSE(uses(*d->candidate, kG14));
    CHECK(d->reprovision_bytes == doctest::Approx(d->candidate->metrics.provisioning_bytes));
  }

  // The same node simply absent from the request counts as gone too.
  PlacementRequest absent = r;
  std::erase_if(absent.nodes, [](const HardwareProfile& n) { return n.id == kG14; });
  auto a = should_replan(cur, inputs(absent));
  REQUIRE(a.is_ok());
  CHECK(a->replan);
  CHECK(has_reason(*a, ReplanReasonKind::kNodeUnavailable));

  // Everyone gone and the Father cannot hold the model alone: nothing feasible, reported as such.
  ReplanInputs alone = inputs(r);
  alone.unavailable_nodes = {kG14, kN3060};
  auto n = should_replan(cur, alone);
  REQUIRE(n.is_ok());
  CHECK(n->replan);
  CHECK_FALSE(n->candidate.has_value());
  CHECK(has_reason(*n, ReplanReasonKind::kNoFeasiblePlan));
  CHECK_FALSE(n->reprovision_needed);
}

TEST_CASE("scenario: the context requirement grows") {
  const PlacementRequest r = base_request();
  PlacementPlan cur = deployed_using(r, kG14);
  REQUIRE(cur.context_tokens == 4096);
  PlacementRequest big = r;
  big.context_tokens = 65536;
  big.prompt_tokens = 4096;
  auto d = should_replan(cur, inputs(big));
  REQUIRE_MESSAGE(d.is_ok(), d.status().to_string());
  const ReplanReason* g = find_reason(*d, ReplanReasonKind::kContextGrowth);
  REQUIRE(g != nullptr);
  CHECK(g->magnitude > 1.0);
  // Longer sequences cost state on the same stages: if the plan is still admissible it holds more state and no
  // more GPU experts.
  if (d->current_under_new_inputs) {
    for (std::size_t i = 0; i < cur.ledgers.size(); ++i) {
      CHECK(d->current_under_new_inputs->ledgers[i].state >= cur.ledgers[i].state);
      CHECK(d->current_under_new_inputs->ledgers[i].gpu_experts <= cur.ledgers[i].gpu_experts);
    }
  }
  // Growth to a context nothing can hold must report infeasibility.
  PlacementRequest huge = r;
  huge.context_tokens = 4'000'000;
  huge.prompt_tokens = 4096;
  auto h = should_replan(cur, inputs(huge));
  REQUIRE(h.is_ok());
  CHECK(h->replan);
  CHECK(has_reason(*h, ReplanReasonKind::kPlanInadmissible));
  CHECK(has_reason(*h, ReplanReasonKind::kNoFeasiblePlan));
}

TEST_CASE("invalid inputs are errors, not decisions") {
  const PlacementRequest r = base_request();
  const PlacementPlan cur = deployed_using(r, kG14);
  PlacementRequest bad = r;
  bad.q = 0;
  CHECK_FALSE(should_replan(cur, inputs(bad)).is_ok());
  PlacementRequest other_father = r;
  other_father.father.id = "someone-else";
  for (auto& l : other_father.network.links)
    if (l.from == kFather) l.from = "someone-else";
  CHECK_FALSE(should_replan(cur, inputs(other_father)).is_ok());
}
