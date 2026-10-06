// Routing aggregates (privacy-restricted input) and the placement report bench calls.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/placement/profile.hpp"
#include "clusterlm/planning/planning.hpp"
#include "clusterlm/planning/report.hpp"

using namespace clusterlm;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(const char* name) {
  auto p = fs::temp_directory_path() / (std::string("clm-routing-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

// Skewed aggregate counts for L layers x E experts summing to positions * K per row.
json counts_doc(std::uint32_t layers, std::uint32_t experts, std::uint32_t active, std::uint64_t positions) {
  json counts = json::array();
  for (std::uint32_t l = 0; l < layers; ++l) {
    std::vector<double> row(experts, static_cast<double>(positions) * active / experts);
    const double shift = row[0] * 0.5;  // expert 0 is hot, expert 1 cold
    row[0] += shift;
    row[1] -= shift;
    counts.push_back(row);
  }
  return {{"schema", "clusterlm.routing_aggregates.v1"}, {"provenance", "measured"}, {"source", "test:aggregate-counts"},
          {"n_layers", layers},                          {"n_experts", experts},     {"n_active", active},
          {"positions", positions},                      {"counts", counts}};
}

std::string text(const json& j) { return j.dump(); }

}  // namespace

TEST_CASE("routing aggregates: counts and frequencies load, normalise and round trip") {
  auto a = planning::routing_aggregates_from_json(text(counts_doc(3, 8, 2, 1000)));
  REQUIRE_MESSAGE(a.is_ok(), a.status().to_string());
  CHECK(a->provenance == placement::Provenance::kMeasured);
  CHECK(a->source == "test:aggregate-counts");
  CHECK(a->n_layers == 3);
  CHECK(a->positions == 1000);
  for (const auto& row : a->frequencies) {
    double sum = 0;
    for (double p : row) sum += p;
    CHECK(sum == doctest::Approx(2.0));
    CHECK(row[0] > row[1]);
  }
  // Round trip through the canonical writer.
  auto back = planning::routing_aggregates_from_json(planning::to_json(*a));
  REQUIRE(back.is_ok());
  REQUIRE(back->frequencies.size() == a->frequencies.size());
  for (std::size_t l = 0; l < a->frequencies.size(); ++l)
    for (std::size_t e = 0; e < a->frequencies[l].size(); ++e) CHECK(back->frequencies[l][e] == doctest::Approx(a->frequencies[l][e]).epsilon(1e-12));

  // Frequencies given with print rounding are renormalised exactly to n_active.
  json f = {{"schema", "clusterlm.routing_aggregates.v1"}, {"provenance", "synthetic"}, {"source", "test"},
            {"n_layers", 1}, {"n_experts", 4}, {"n_active", 2},
            {"frequencies", json::array({json::array({0.7501, 0.5, 0.4999, 0.25})})}};
  auto fr = planning::routing_aggregates_from_json(text(f));
  REQUIRE_MESSAGE(fr.is_ok(), fr.status().to_string());
  double sum = 0;
  for (double p : fr->frequencies[0]) sum += p;
  CHECK(sum == doctest::Approx(2.0).epsilon(1e-12));

  const auto dir = temp_dir("load");
  { std::ofstream(dir / "agg.json") << text(counts_doc(2, 4, 1, 100)); }
  auto loaded = planning::load_routing_aggregates((dir / "agg.json").string());
  CHECK(loaded.is_ok());
  CHECK_FALSE(planning::load_routing_aggregates((dir / "missing.json").string()).is_ok());
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("routing aggregates: only aggregates are accepted (privacy)") {
  const json good = counts_doc(2, 4, 1, 100);
  // Anything beyond the aggregate schema is refused, whatever it is called.
  for (const char* key : {"tokens", "token_ids", "sequences", "routes", "prompts", "text", "trace", "per_request"}) {
    json j = good;
    j[key] = json::array({1, 2, 3});
    auto r = planning::routing_aggregates_from_json(text(j));
    CHECK_FALSE(r.is_ok());
    if (!r.is_ok()) CHECK(r.status().message().find("only aggregate") != std::string::npos);
  }
  // Wrong shapes and claims.
  auto bad = [&](auto mutate) {
    json j = good;
    mutate(j);
    return planning::routing_aggregates_from_json(text(j)).is_ok();
  };
  CHECK_FALSE(bad([](json& j) { j["provenance"] = "qualified"; }));  // a file cannot qualify itself
  CHECK_FALSE(bad([](json& j) { j.erase("provenance"); }));
  CHECK_FALSE(bad([](json& j) { j["source"] = ""; }));
  CHECK_FALSE(bad([](json& j) { j["schema"] = "other"; }));
  CHECK_FALSE(bad([](json& j) { j["frequencies"] = j["counts"]; }));  // both given
  CHECK_FALSE(bad([](json& j) { j.erase("counts"); }));                // neither given
  CHECK_FALSE(bad([](json& j) { j["n_layers"] = 3; }));                // rows != n_layers
  CHECK_FALSE(bad([](json& j) { j["counts"][0][0] = 1e9; }));          // exceeds positions / breaks the row sum
  CHECK_FALSE(bad([](json& j) { j["counts"][0][2] = -1; }));
  CHECK_FALSE(bad([](json& j) { j["n_active"] = 9; }));                // more active than experts
  CHECK_FALSE(bad([](json& j) { j.erase("positions"); }));
  CHECK_FALSE(planning::routing_aggregates_from_json("not json").is_ok());
  CHECK_FALSE(planning::routing_aggregates_from_json("[1,2]").is_ok());
  CHECK(bad([](json&) {}));
}

TEST_CASE("routing aggregates feed cost inputs; provenance follows the file; dimension mismatches are errors") {
  const auto dir = temp_dir("inputs");
  objects::FixtureSpec spec;
  auto m = objects::write_fixture_model(spec, dir / "model");
  REQUIRE(m.is_ok());

  auto synthetic = planning::cost_inputs_from_manifest(*m);
  REQUIRE(synthetic.is_ok());
  CHECK(synthetic->provenance == placement::Provenance::kSynthetic);

  auto agg = planning::routing_aggregates_from_json(text(counts_doc(spec.n_layers, spec.n_experts, spec.n_active, 3200)));
  REQUIRE_MESSAGE(agg.is_ok(), agg.status().to_string());
  planning::CostInputOptions opt;
  planning::apply_routing_aggregates(opt, *agg);
  auto in = planning::cost_inputs_from_manifest(*m, opt);
  REQUIRE_MESSAGE(in.is_ok(), in.status().to_string());
  CHECK(in->provenance == placement::Provenance::kMeasured);
  CHECK(in->routing_freq == agg->frequencies);
  CHECK(in->source.find("test:aggregate-counts") != std::string::npos);
  CHECK(in->routing_freq[0][0] > in->routing_freq[0][1]);  // the skew reached the planner
  // The Father-side draft time is a separate input: still Synthetic unless measured.
  CHECK(in->draft_ms.provenance == placement::Provenance::kSynthetic);
  opt.draft_ms = placement::Quantity::measured(3.0, "bench:draft");
  auto in2 = planning::cost_inputs_from_manifest(*m, opt);
  REQUIRE(in2.is_ok());
  CHECK(in2->draft_ms.provenance == placement::Provenance::kMeasured);

  // Aggregates for another geometry are refused with a clear error.
  auto wrong_layers = planning::routing_aggregates_from_json(text(counts_doc(spec.n_layers + 1, spec.n_experts, spec.n_active, 3200)));
  REQUIRE(wrong_layers.is_ok());
  planning::CostInputOptions bad;
  planning::apply_routing_aggregates(bad, *wrong_layers);
  auto r1 = planning::cost_inputs_from_manifest(*m, bad);
  CHECK_FALSE(r1.is_ok());
  auto wrong_experts = planning::routing_aggregates_from_json(text(counts_doc(spec.n_layers, spec.n_experts * 2, spec.n_active, 3200)));
  REQUIRE(wrong_experts.is_ok());
  planning::CostInputOptions bad2;
  planning::apply_routing_aggregates(bad2, *wrong_experts);
  CHECK_FALSE(planning::cost_inputs_from_manifest(*m, bad2).is_ok());
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("placement reports state provenance, never claim Qualified and carry only aggregates") {
  placement::PlacementRequest r;
  auto dir = std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/";
  r.father = *placement::load_hardware_profile(dir + "Father-4060Ti-7600.json");
  r.nodes = {*placement::load_hardware_profile(dir + "Node-G14-4070-8945HS.json"),
             *placement::load_hardware_profile(dir + "Node-3060-5600.json")};
  r.network = *placement::load_network_profile(dir + "network-gige-simulated.json");
  r.model = placement::flash_next_planning_estimate();
  r.default_lease = {28800, 288000};

  auto single = placement::search_placements(r);
  REQUIRE(single.is_ok());
  const json sj = json::parse(planning::placement_report_json(r, *single));
  CHECK(sj["schema"] == "clusterlm.placement_report.v2");
  CHECK(sj["provenance"] == "synthetic");
  CHECK(sj["provenance_banner"].get<std::string>().find("SYNTHETIC") == 0);
  CHECK(sj["inputs"]["father"]["provenance"] == "synthetic");
  CHECK(sj["inputs"]["nodes"].size() == 2);
  CHECK(sj["inputs"]["acceptance"]["provenance"] == "synthetic");
  CHECK(sj["pareto"].size() == single->pareto.size());
  CHECK(sj["pareto"][0].contains("prefill_s"));
  CHECK(sj["enumerated"].get<std::uint64_t>() == single->enumerated);
  CHECK(sj["recommended_plan"]["stages"][0]["domain"] == r.father.id);

  placement::WorkloadSpec spec;
  spec.context_profiles = {4096, 16384};
  spec.q_values = {1, 2};
  spec.acceptance_by_q[2] = placement::Quantity::synthetic(1.4);
  auto wl = placement::search_workload(r, spec);
  REQUIRE_MESSAGE(wl.is_ok(), wl.status().to_string());
  const std::string text_report = planning::placement_report_json(r, *wl);
  const json wj = json::parse(text_report);
  CHECK(wj["schema"] == "clusterlm.placement_workload_report.v1");
  CHECK(wj["provenance"] == "synthetic");
  CHECK(wj["contexts"].size() == 2);
  CHECK(wj["contexts"][0]["context_tokens"] == 4096);
  CHECK(wj["contexts"][0]["per_q"].size() == 2);
  CHECK(wj["contexts"][0]["frontier"].size() == wl->contexts[0].frontier.size());
  CHECK(wj["selected_context"] == 4096);
  CHECK(wj["recommended_plan"]["provenance"] == "synthetic");
  CHECK(wj["pareto_axes"].size() == 3);
  // Tools label results Synthetic or Measured; "qualified" never appears as a provenance value.
  CHECK(text_report.find("\"qualified\"") == std::string::npos);
  CHECK(planning::placement_report_json(r, *single).find("\"provenance\": \"qualified\"") == std::string::npos);
  // Nothing token- or sequence-shaped is emitted.
  for (const char* forbidden : {"token_ids", "\"prompt\"", "logits", "routing_sequence", "activation"})
    CHECK(text_report.find(forbidden) == std::string::npos);

  // Infeasible contexts are reported with reasons, not dropped.
  placement::PlacementRequest tight = r;
  tight.father.gpu.vram_budget = placement::Quantity::synthetic(0.05 * 1024 * 1024 * 1024);
  auto none = placement::search_workload(tight, spec);
  REQUIRE(none.is_ok());
  const json nj = json::parse(planning::placement_report_json(tight, *none));
  CHECK(nj["contexts"][0]["feasible"] == false);
  CHECK(nj["contexts"][0]["infeasible_reasons"].size() > 0);
  CHECK(nj["recommended_plan"].is_null());
}
