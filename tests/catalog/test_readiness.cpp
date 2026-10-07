// Tier readiness matrix: Ready only when every requirement holds; every blocker has a readable reason.
#include <doctest/doctest.h>

#include "clusterlm/catalog/catalog.hpp"
#include "clusterlm/catalog/readiness.hpp"

using namespace clusterlm;
using catalog::MachineState;
using catalog::TierState;

namespace {

const std::string kRoot(64, 'b');

catalog::Catalog load() {
  auto c = catalog::Catalog::load(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json");
  REQUIRE(c.is_ok());
  return std::move(c).value();
}

catalog::MachineInputs machine(const std::string& role, const std::string& id, MachineState st) {
  catalog::MachineInputs m;
  m.role = role;
  m.machine_id = id;
  m.state = st;
  return m;
}

// Inputs under which the tier is fully Ready, to be degraded one dimension at a time.
catalog::ReadinessInputs ready_inputs(const catalog::TierEntry& t, std::uint32_t ctx = 4096) {
  catalog::ReadinessInputs in;
  in.context_tokens = ctx;
  in.model = {true, true, kRoot, kRoot};
  in.backend = {"strata", true};
  for (const auto& r : t.roles) {
    const std::string id = r == "father" ? "Father" : r == "node:laptop-class" ? "G14" : "3060";
    in.machines.push_back(machine(r, id, MachineState::kReady));
  }
  in.plan = {true, true};
  return in;
}

catalog::MachineInputs& find(catalog::ReadinessInputs& in, const std::string& role) {
  for (auto& m : in.machines)
    if (m.role == role) return m;
  FAIL("role not in inputs: " << role);
  return in.machines.front();
}

bool mentions(const catalog::TierReadiness& r, const std::string& needle) {
  for (const auto& s : r.reasons)
    if (s.find(needle) != std::string::npos) return true;
  return false;
}

}  // namespace

TEST_CASE("every tier is Ready when everything holds, and says which model") {
  auto c = load();
  for (const auto& t : c.tiers()) {
    CAPTURE(t.id);
    auto r = catalog::evaluate(t, ready_inputs(t));
    CHECK(r.state == TierState::kReady);
    CHECK(r.model_name == t.model.display_name);
    CHECK(mentions(r, t.model.display_name));
  }
}

TEST_CASE("machine states map to honest tier states and readable reasons") {
  auto c = load();
  const auto& ultra = *c.find("ultra");
  struct Row { MachineState state; TierState expect; const char* reason; };
  const Row rows[] = {
      {MachineState::kBusy, TierState::kUnavailable, "G14 is in use"},
      {MachineState::kOffline, TierState::kUnavailable, "G14 is offline"},
      {MachineState::kCleanupPending, TierState::kUnavailable, "G14 needs cleanup"},
      {MachineState::kReleasing, TierState::kUnavailable, "G14 is releasing"},
      {MachineState::kInferencing, TierState::kReady, ""},  // our own current plan is serving
      {MachineState::kReady, TierState::kReady, ""},
      {MachineState::kPreparing, TierState::kPreparing, "Preparing Ultra"},
      {MachineState::kAvailable, TierState::kAvailable, "prepared"},
  };
  for (const auto& row : rows) {
    CAPTURE(to_string(row.state));
    auto in = ready_inputs(ultra);
    find(in, "node:laptop-class").state = row.state;
    if (row.state == MachineState::kAvailable) in.plan.plan_ready = false;
    if (row.state == MachineState::kPreparing) in.plan.plan_ready = false;
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == row.expect);
    if (row.reason[0]) CHECK(mentions(r, row.reason));
  }
}

TEST_CASE("3060 needs cleanup is reported by name") {
  auto c = load();
  auto in = ready_inputs(*c.find("ultra"));
  find(in, "node:designated-3060").state = MachineState::kCleanupPending;
  auto r = catalog::evaluate(*c.find("ultra"), in);
  CHECK(r.state == TierState::kUnavailable);
  CHECK(r.headline() == "3060 needs cleanup");
}

TEST_CASE("Ready is never reported unless the plan is current and every domain is Ready") {
  auto c = load();
  const auto& ultra = *c.find("ultra");
  {  // plan not ready although every machine says Ready
    auto in = ready_inputs(ultra);
    in.plan.plan_ready = false;
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kAvailable);
    CHECK(mentions(r, "prepared again"));
  }
  {  // plan ready but one domain merely Available
    auto in = ready_inputs(ultra);
    find(in, "node:designated-3060").state = MachineState::kAvailable;
    CHECK(catalog::evaluate(ultra, in).state != TierState::kReady);
  }
  {  // plan ready but Father's own domain not Ready
    auto in = ready_inputs(ultra);
    find(in, "father").state = MachineState::kAvailable;
    CHECK(catalog::evaluate(ultra, in).state != TierState::kReady);
  }
  {  // a machine inferencing for someone else's plan
    auto in = ready_inputs(ultra);
    in.plan.plan_ready = false;
    find(in, "node:laptop-class").state = MachineState::kInferencing;
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "serving another session"));
  }
  {  // a required role without a machine
    auto in = ready_inputs(ultra);
    in.machines.pop_back();
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "node:designated-3060"));
  }
}

TEST_CASE("model availability: missing, unverified, unpinned, mismatched") {
  auto c = load();
  const auto& ultra = *c.find("ultra");
  {
    auto in = ready_inputs(ultra);
    in.model.manifest_present = false;
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "not downloaded"));
  }
  {
    auto in = ready_inputs(ultra);
    in.model.hashes_verified = false;
    CHECK(mentions(catalog::evaluate(ultra, in), "not been verified"));
  }
  {  // unpinned catalog entry and the user never confirmed the manifest
    auto in = ready_inputs(ultra);
    in.model.user_confirmed_root_hex.clear();
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "unpinned"));
  }
  {  // user confirmed a different manifest than the one on disk
    auto in = ready_inputs(ultra);
    in.model.user_confirmed_root_hex = std::string(64, 'c');
    CHECK(catalog::evaluate(ultra, in).state == TierState::kUnavailable);
  }
  {  // a pinned catalog needs no user confirmation but the hash must match
    auto tier = ultra;
    tier.model.expected_root_hash = kRoot;
    auto in = ready_inputs(tier);
    in.model.user_confirmed_root_hex.clear();
    CHECK(catalog::evaluate(tier, in).state == TierState::kReady);
    in.model.manifest_root_hex = std::string(64, 'd');
    auto r = catalog::evaluate(tier, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "pinned catalog hash"));
  }
}

TEST_CASE("backend, power, pairing and placement feasibility each block readiness") {
  auto c = load();
  const auto& ultra = *c.find("ultra");
  {
    auto in = ready_inputs(ultra);
    in.backend.hardware_available = false;
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "strata-hybrid backend cannot run"));
  }
  {
    auto in = ready_inputs(ultra);
    in.tokenizer_problem = "unsupported pre-tokenizer \"foo\"";
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "tokenizer is unavailable: unsupported pre-tokenizer"));
  }
  {
    auto in = ready_inputs(ultra);
    find(in, "node:laptop-class").power.on_ac = false;
    CHECK(mentions(catalog::evaluate(ultra, in), "G14 is on battery power"));
  }
  {
    auto in = ready_inputs(ultra);
    find(in, "node:laptop-class").power.battery_saver = true;
    CHECK(mentions(catalog::evaluate(ultra, in), "battery saver"));
  }
  {
    auto in = ready_inputs(ultra);
    find(in, "node:designated-3060").paired = false;
    CHECK(mentions(catalog::evaluate(ultra, in), "3060 is not paired"));
  }
  {
    auto in = ready_inputs(ultra, 32 * 1024);
    in.plan.feasible_for_context = false;
    auto r = catalog::evaluate(ultra, in);
    CHECK(r.state == TierState::kUnavailable);
    CHECK(mentions(r, "No feasible placement for 32K"));
  }
}

TEST_CASE("context profiles: not offered blocks, unqualified is a note") {
  auto c = load();
  const auto& fast = *c.find("fast");
  auto in = ready_inputs(fast, 128 * 1024);
  auto r = catalog::evaluate(fast, in);
  CHECK(r.state == TierState::kUnavailable);
  CHECK(mentions(r, "128K context is not offered for Fast"));
  in = ready_inputs(fast, 8192);
  r = catalog::evaluate(fast, in);
  CHECK(r.state == TierState::kReady);
  REQUIRE(r.notes.size() == 1);
  CHECK(r.notes[0].find("not yet qualified") != std::string::npos);
  in = ready_inputs(fast, 5000);  // not a catalog profile at all
  CHECK(catalog::evaluate(fast, in).state == TierState::kUnavailable);
}

TEST_CASE("Preparing reports percent and a labelled ETA; no rate means no ETA") {
  auto c = load();
  const auto& ultra = *c.find("ultra");
  auto in = ready_inputs(ultra);
  in.plan.plan_ready = false;
  find(in, "node:designated-3060").state = MachineState::kPreparing;
  catalog::ProvisioningProgress p;
  p.bytes_total = 1000;
  p.bytes_done = 410;
  p.rate_bytes_per_s = 590.0 / 180.0;  // 180 s left
  p.rate_is_measured = false;
  in.provisioning = p;
  auto r = catalog::evaluate(ultra, in);
  REQUIRE(r.state == TierState::kPreparing);
  REQUIRE(r.progress.has_value());
  CHECK(r.progress->percent == doctest::Approx(41.0));
  REQUIRE(r.progress->eta_seconds.has_value());
  CHECK(*r.progress->eta_seconds == doctest::Approx(180.0));
  CHECK(r.progress->eta_is_estimate);
  CHECK(r.headline().find("Preparing Ultra") != std::string::npos);
  CHECK(r.headline().find("41%") != std::string::npos);
  CHECK(r.headline().find("about 3 min") != std::string::npos);
  CHECK(r.headline().find("estimate") != std::string::npos);

  in.provisioning->rate_bytes_per_s.reset();
  r = catalog::evaluate(ultra, in);
  REQUIRE(r.state == TierState::kPreparing);
  CHECK_FALSE(r.progress->eta_seconds.has_value());
  CHECK(r.headline().find("estimate") == std::string::npos);

  // provisioning in flight while nodes still say Available is also Preparing
  in = ready_inputs(ultra);
  in.plan.plan_ready = false;
  find(in, "node:laptop-class").state = MachineState::kAvailable;
  find(in, "node:designated-3060").state = MachineState::kAvailable;
  find(in, "father").state = MachineState::kAvailable;
  in.provisioning = p;
  CHECK(catalog::evaluate(ultra, in).state == TierState::kPreparing);
  // finished provisioning but plan not ready: Available again, not Ready
  in.provisioning->bytes_done = in.provisioning->bytes_total;
  CHECK(catalog::evaluate(ultra, in).state == TierState::kAvailable);
}

TEST_CASE("duration formatting") {
  CHECK(catalog::format_duration(10) == "under a minute");
  CHECK(catalog::format_duration(180) == "about 3 min");
  CHECK(catalog::format_duration(3900) == "about 1 h 5 min");
  CHECK(catalog::format_duration(7200) == "about 2 h");
  CHECK(catalog::format_duration(-1) == "unknown");
}

TEST_CASE("fallback suggestions list only viable lower tiers in catalog order") {
  auto c = load();
  std::vector<std::pair<std::string, catalog::ReadinessInputs>> all;
  for (const auto& t : c.tiers()) all.emplace_back(t.id, ready_inputs(t));
  auto ultra_r = catalog::evaluate_with_fallback(c, *c.find("ultra"), all);
  CHECK(ultra_r.state == TierState::kReady);
  CHECK(ultra_r.suggested_fallback == std::vector<std::string>{"strong", "fast"});

  // G14 becomes busy: Ultra and Strong are both blocked and only Fast remains suggested.
  for (auto& [id, in] : all)
    for (auto& m : in.machines)
      if (m.role == "node:laptop-class") m.state = MachineState::kBusy;
  ultra_r = catalog::evaluate_with_fallback(c, *c.find("ultra"), all);
  CHECK(ultra_r.state == TierState::kUnavailable);
  CHECK(ultra_r.headline() == "G14 is in use");
  CHECK(ultra_r.suggested_fallback == std::vector<std::string>{"fast"});
  auto fast_r = catalog::evaluate_with_fallback(c, *c.find("fast"), all);
  CHECK(fast_r.suggested_fallback.empty());

  // Fast is available-but-unprepared: still a viable fallback.
  for (auto& [id, in] : all)
    if (id == "fast") {
      in.plan.plan_ready = false;
      in.machines[0].state = MachineState::kAvailable;
    }
  CHECK(catalog::evaluate_with_fallback(c, *c.find("ultra"), all).suggested_fallback == std::vector<std::string>{"fast"});
  // Missing inputs for a tier is Unavailable, never silently Ready.
  auto none = catalog::evaluate_with_fallback(c, *c.find("strong"), {});
  CHECK(none.state == TierState::kUnavailable);
}
