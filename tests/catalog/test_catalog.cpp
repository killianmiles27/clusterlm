// Catalog loading and validation: the shipped catalog, malformed input and the structural rules.
#include <doctest/doctest.h>

#include <fstream>

#include <nlohmann/json.hpp>

#include "clusterlm/catalog/catalog.hpp"

using namespace clusterlm;
using nlohmann::json;

namespace {

std::string catalog_path() { return std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/catalog/clusterlm-catalog.json"; }

json shipped_json() {
  std::ifstream f(catalog_path());
  REQUIRE(f.good());
  return json::parse(f);
}

Result<catalog::Catalog> parse(const json& j) { return catalog::Catalog::parse(j.dump()); }

}  // namespace

TEST_CASE("the shipped catalog has exactly the three tiers with the specified identities") {
  auto c = catalog::Catalog::load(catalog_path());
  REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
  REQUIRE(c->tiers().size() == 3);
  const auto* fast = c->find("fast");
  const auto* strong = c->find("strong");
  const auto* ultra = c->find("ultra");
  REQUIRE(fast);
  REQUIRE(strong);
  REQUIRE(ultra);
  CHECK(fast->model.display_name == "Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S");
  CHECK(strong->model.display_name == "Swift-1.5-Qwen3.8-Flash-Next GSQ-RCO IQ2_XS");
  CHECK(ultra->model.display_name == "Qwen3.8-Flash-Next GSQ-RCO IQ3_S");
  CHECK(*ultra->model.artifact_id == "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF");
  CHECK(ultra->model.quant == "IQ3_S");
  REQUIRE(ultra->model.expected_files.size() == 2);
  CHECK(*ultra->model.expected_files[0].approx_bytes == 54'800'000'000ull);
  CHECK(*ultra->model.expected_files[1].approx_bytes == 28'800'000'000ull);
  CHECK(fast->backend == catalog::BackendKind::kLlamaLocal);
  CHECK(strong->backend == catalog::BackendKind::kStrataHybrid);
  CHECK(ultra->backend == catalog::BackendKind::kStrataHybrid);
  CHECK(fast->roles == std::vector<std::string>{"father"});
  CHECK(strong->roles == std::vector<std::string>{"father", "node:laptop-class"});
  CHECK(ultra->roles == std::vector<std::string>{"father", "node:laptop-class", "node:designated-3060"});
  CHECK(c->fallback_order() == std::vector<std::string>{"ultra", "strong", "fast"});
}

TEST_CASE("the shipped catalog promises nothing: unpinned, unqualified, targets pending") {
  auto c = catalog::Catalog::load(catalog_path());
  REQUIRE(c.is_ok());
  for (const auto& t : c->tiers()) {
    CHECK_FALSE(t.model.expected_root_hash.has_value());
    CHECK(t.model.pin_status() == catalog::PinStatus::kUnpinned);
    for (const auto& cx : t.contexts) {
      CHECK_FALSE(cx.qualified);
      CHECK(cx.has_requirement(catalog::kReqFeasiblePlacement));
    }
    for (const auto& tg : t.targets) CHECK(tg.status == catalog::TargetStatus::kPendingQualification);
    CHECK_FALSE(t.qualification_experiments.empty());
  }
  const auto* ultra = c->find("ultra");
  REQUIRE(ultra->targets.size() == 1);
  CHECK(ultra->targets[0].metric == "decode_tok_s_median");
  CHECK(ultra->targets[0].value == doctest::Approx(20.0));
  for (std::uint32_t k : {4u, 8u, 16u, 32u, 64u, 128u}) CHECK(ultra->find_context(k * 1024) != nullptr);
  CHECK_FALSE(c->find("fast")->find_context(128 * 1024)->offered);
}

TEST_CASE("fallback order is Ultra -> Strong -> Fast") {
  auto c = catalog::Catalog::load(catalog_path());
  REQUIRE(c.is_ok());
  auto after_ultra = c->fallbacks_after("ultra");
  REQUIRE(after_ultra.size() == 2);
  CHECK(after_ultra[0]->id == "strong");
  CHECK(after_ultra[1]->id == "fast");
  CHECK(c->fallbacks_after("fast").empty());
  CHECK(c->find("fast")->on_session_loss.then == catalog::LossAction::kStop);
  CHECK(c->find("ultra")->on_session_loss.then == catalog::LossAction::kDowngrade);
}

TEST_CASE("catalog validation rejects bad roles, missing fields and Node 3 in Ultra") {
  const auto good = shipped_json();
  REQUIRE(parse(good).is_ok());
  auto bad = [&](const char* what, auto&& mutate, const char* expect) {
    CAPTURE(what);
    json j = good;
    mutate(j);
    auto r = parse(j);
    REQUIRE_FALSE(r.is_ok());
    CHECK_MESSAGE(r.status().message().find(expect) != std::string::npos, r.status().message());
  };
  bad("unknown role", [](json& j) { j["tiers"][1]["roles"] = {"father", "node:g14"}; }, "unknown role");
  bad("machine name instead of role", [](json& j) { j["tiers"][2]["roles"][2] = "RTX-3060-box"; }, "unknown role");
  bad("third node in Ultra", [](json& j) { j["tiers"][2]["roles"].push_back("node:third"); }, "Node 3");
  bad("Node 3 role in Ultra", [](json& j) { j["tiers"][2]["roles"][2] = "node:3"; }, "unknown role");
  bad("ultra missing 3060", [](json& j) { j["tiers"][2]["roles"] = {"father", "node:laptop-class"}; }, "expects 2 node");
  bad("ultra wrong node order", [](json& j) { j["tiers"][2]["roles"] = {"father", "node:designated-3060", "node:laptop-class"}; }, "pipeline order");
  bad("father not first", [](json& j) { j["tiers"][1]["roles"] = {"node:laptop-class", "father"}; }, "first role");
  bad("duplicate role", [](json& j) { j["tiers"][2]["roles"][2] = "node:laptop-class"; }, "duplicate role");
  bad("fast with a node", [](json& j) { j["tiers"][0]["roles"].push_back("node:laptop-class"); }, "expects 0 node");
  bad("llama-local for strong", [](json& j) { j["tiers"][1]["backend"] = "llama-local"; }, "llama-local");
  bad("unknown backend", [](json& j) { j["tiers"][1]["backend"] = "vllm"; }, "backend");
  bad("missing model", [](json& j) { j["tiers"][0].erase("model"); }, "model is required");
  bad("missing quant", [](json& j) { j["tiers"][0]["model"].erase("quant"); }, "quant");
  bad("missing contexts", [](json& j) { j["tiers"][0].erase("contexts"); }, "contexts");
  bad("missing on_session_loss", [](json& j) { j["tiers"][0].erase("on_session_loss"); }, "on_session_loss");
  bad("two tiers", [](json& j) { j["tiers"].erase(j["tiers"].begin()); }, "exactly three");
  bad("duplicate tier", [](json& j) { j["tiers"][1]["id"] = "fast"; }, "fast");
  bad("unknown tier id", [](json& j) { j["tiers"][1]["id"] = "medium"; }, "fast, strong or ultra");
  bad("fallback order", [](json& j) { j["fallback_order"] = {"fast", "strong", "ultra"}; }, "fallback_order");
  bad("pin status lies", [](json& j) { j["tiers"][0]["model"]["pin_status"] = "pinned"; }, "pin_status");
  bad("bad hash", [](json& j) {
    j["tiers"][0]["model"]["expected_root_hash"] = "abc";
    j["tiers"][0]["model"]["pin_status"] = "pinned";
  }, "64 hex");
  bad("unknown requirement", [](json& j) { j["tiers"][0]["contexts"][0]["requirements"] = {"magic"}; }, "unknown requirement");
  bad("duplicate context", [](json& j) { j["tiers"][0]["contexts"][1]["tokens"] = 4096; }, "duplicate context");
  bad("non-positive target", [](json& j) { j["tiers"][2]["performance_targets"][0]["value"] = 0; }, "positive");
  bad("bad schema version", [](json& j) { j["schema_version"] = 2; }, "schema_version");
}

TEST_CASE("a pinned catalog entry parses and reports pinned") {
  json j = shipped_json();
  j["tiers"][2]["model"]["expected_root_hash"] = std::string(64, 'a');
  j["tiers"][2]["model"]["pin_status"] = "pinned";
  auto c = parse(j);
  REQUIRE(c.is_ok());
  CHECK(c->find("ultra")->model.pin_status() == catalog::PinStatus::kPinned);
}

TEST_CASE("JSON loading is bounded") {
  CHECK_FALSE(catalog::Catalog::parse("").is_ok());
  CHECK_FALSE(catalog::Catalog::parse("not json").is_ok());
  CHECK_FALSE(catalog::Catalog::parse("[]").is_ok());
  CHECK_FALSE(catalog::Catalog::parse(std::string(catalog::kMaxCatalogBytes + 1, ' ')).is_ok());
  std::string deep(200, '[');
  deep += std::string(200, ']');
  auto r = catalog::Catalog::parse(deep);
  REQUIRE_FALSE(r.is_ok());
  CHECK(r.status().message().find("nesting") != std::string::npos);
  // Brackets inside strings do not count towards depth.
  json j = shipped_json();
  j["note"] = std::string(100, '[');
  CHECK(parse(j).is_ok());
  CHECK(catalog::Catalog::load("/nonexistent/catalog.json").status().code() == ErrorCode::kNotFound);
}

TEST_CASE("TierAssignment binds roles to machines and rejects gaps and sharing") {
  auto c = catalog::Catalog::load(catalog_path());
  REQUIRE(c.is_ok());
  const auto* ultra = c->find("ultra");
  catalog::TierAssignment a;
  CHECK_FALSE(a.validate_for(*ultra).is_ok());  // nothing bound
  a.bind("father", "Father");
  a.bind("node:laptop-class", "G14");
  CHECK_FALSE(a.validate_for(*ultra).is_ok());  // 3060 missing
  CHECK(a.validate_for(*c->find("strong")).is_ok());
  a.bind("node:designated-3060", "G14");
  CHECK(a.validate_for(*ultra).code() == ErrorCode::kInvalidArgument);  // one machine, two roles
  a.bind("node:designated-3060", "3060");
  CHECK(a.validate_for(*ultra).is_ok());
  CHECK(*a.machine_for("node:laptop-class") == "G14");
  CHECK_FALSE(a.machine_for("node:other").has_value());
}
