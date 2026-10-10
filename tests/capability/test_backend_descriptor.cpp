// Backend capability descriptors: embedded set matches the documented examples, parser is strict, labels are computed
// from descriptor + model facts and never exceed the backend ceiling.
#include <doctest/doctest.h>

#include <fstream>
#include <functional>
#include <sstream>

#include <nlohmann/json.hpp>

#include "clusterlm/domain/backend_descriptor.hpp"

using namespace clusterlm;
using namespace clusterlm::domain;
using nlohmann::json;

namespace {
std::string read_example(const std::string& name) {
  std::ifstream in(std::string(CLUSTERLM_SOURCE_DIR) + "/docs/interfaces/examples/backend-" + name + ".example.json", std::ios::binary);
  REQUIRE(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
const BackendDescriptor& builtin(std::string_view id) {
  const auto* d = BackendRegistry::builtin().find(id);
  REQUIRE_MESSAGE(d != nullptr, std::string(id));
  return *d;
}
BackendRuntimeStatus ok_runtime() {
  BackendRuntimeStatus r;
  r.built = r.runtime_present = r.hardware_available = true;
  return r;
}
ModelFacts flash_next(std::vector<std::string> types = {"IQ3_S", "F32"}) {
  ModelFacts m;
  m.container = "gguf-split";
  m.family = "Qwen3.8-Flash-Next";
  m.architecture = "qwen4exp";
  m.tensor_types = std::move(types);
  return m;
}
}  // namespace

TEST_CASE("the embedded descriptors equal the documented examples and survive a JSON round trip") {
  for (const char* id : {"reference", "llama-local", "strata-hybrid"}) {
    auto doc = backend_descriptor_from_json(read_example(id));
    REQUIRE_MESSAGE(doc.is_ok(), doc.status().to_string());
    CHECK(*doc == builtin(id));  // embedded text has not drifted from docs/interfaces/examples
    auto again = backend_descriptor_from_json(to_json(*doc));
    REQUIRE(again.is_ok());
    CHECK(*again == *doc);
  }
  CHECK(BackendRegistry::builtin().list().size() == 3);
  CHECK(BackendRegistry::builtin().find("nope") == nullptr);
}

TEST_CASE("the parser requires every capability field and rejects unknown ones") {
  auto mut = [](const std::function<void(json&)>& f) {
    json j = json::parse(read_example("strata-hybrid"));
    f(j);
    return backend_descriptor_from_json(j.dump());
  };
  CHECK(mut([](json&) {}).is_ok());
  CHECK_FALSE(mut([](json& j) { j.erase("serving"); }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["state"].erase("per_client_isolation"); }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["speculation"].erase("max_q"); }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["devices"].erase("requires_runtime"); }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["bogus"] = 1; }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["execution"]["bogus"] = 1; }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["model_support"]["tensor_formats"] = {"pinned-llama.cpp-supported"}; }).is_ok());  // no placeholders
  CHECK_FALSE(mut([](json& j) { j["model_support"]["unlisted_family_status"] = "Supported and qualified"; }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["id"] = "llama-local"; }).is_ok());  // id/factory pair
  CHECK_FALSE(mut([](json& j) { j["speculation"]["window_commit_abort"] = false; }).is_ok());  // max_q must then be 1
  CHECK_FALSE(mut([](json& j) { j["execution"]["cross_machine"] = false; }).is_ok());  // max workers must then be 0
  CHECK_FALSE(mut([](json& j) { j["devices"].erase("min_compute_capability"); }).is_ok());
  CHECK_FALSE(mut([](json& j) { j["qualification"]["hardware_experiments"] = json::array(); }).is_ok());  // production needs experiments
  CHECK(mut([](json& j) { j["contract_version"] = 2; }).status().code() == ErrorCode::kVersionMismatch);
}

TEST_CASE("labels are computed from the descriptor, never typed") {
  const auto& strata = builtin("strata-hybrid");
  auto r = check_model(strata, flash_next(), ok_runtime());
  CHECK(r.label == CompatLabel::kSupportedAwaitingQualification);
  CHECK(r.distributable);
  CHECK(r.hosts_alone);
  CHECK(r.max_workers == strata.execution.validated_max_workers);
  CHECK_FALSE(r.has_blocker());
  CHECK(to_string(r.label) == "Supported, awaiting hardware qualification");

  // Family label alone is not enough when the architecture differs.
  auto other_arch = flash_next();
  other_arch.architecture = "llama";
  r = check_model(strata, other_arch, ok_runtime());
  CHECK(r.label == CompatLabel::kUnsupported);
  CHECK_FALSE(r.distributable);
  CHECK(r.max_workers == 0);
  CHECK(r.has_blocker());

  // An unlisted family is never above Experimental/Unsupported.
  auto stranger = flash_next();
  stranger.family = "Totally-Other-Model";
  r = check_model(strata, stranger, ok_runtime());
  CHECK(r.label == CompatLabel::kUnsupported);
}

TEST_CASE("every tensor type of the model must be executable, not only the dominant one") {
  const auto& strata = builtin("strata-hybrid");
  auto r = check_model(strata, flash_next({"IQ3_S", "SOME_FUTURE_TYPE"}), ok_runtime());
  CHECK(r.label == CompatLabel::kUnsupported);
  REQUIRE(r.has_blocker());
  bool named = false;
  for (const auto& f : r.findings) named |= f.code == "tensor_type_unsupported" && f.message.find("SOME_FUTURE_TYPE") != std::string::npos;
  CHECK(named);
  // Case-insensitive on the model side.
  CHECK(check_model(strata, flash_next({"iq3_s", "f32"}), ok_runtime()).label == CompatLabel::kSupportedAwaitingQualification);
  // Container is checked too.
  auto manifest_only = flash_next();
  manifest_only.container = "safetensors";
  CHECK(check_model(strata, manifest_only, ok_runtime()).label == CompatLabel::kUnsupported);
}

TEST_CASE("experimental models need an explicit opt-in, are never distributable and the backend label is a ceiling") {
  auto d = builtin("strata-hybrid");
  d.model_support.unlisted_family_status = CompatLabel::kExperimental;
  ModelFacts stranger = flash_next();
  stranger.family = "Other";
  auto r = check_model(d, stranger, ok_runtime());
  CHECK(r.label == CompatLabel::kExperimental);
  CHECK(r.has_blocker());  // opt-in missing
  CHECK_FALSE(r.distributable);
  CHECK(r.max_workers == 0);
  stranger.experimental_opt_in = true;
  r = check_model(d, stranger, ok_runtime());
  CHECK_FALSE(r.has_blocker());
  CHECK(r.hosts_alone);

  // Ceiling: a family entry claiming "Supported and qualified" cannot beat the backend's own label.
  auto high = builtin("strata-hybrid");
  high.model_support.families[0].status = CompatLabel::kSupportedQualified;
  CHECK(check_model(high, flash_next(), ok_runtime()).label == CompatLabel::kSupportedAwaitingQualification);
  // And a fixture backend never gets to claim a real model: development-only is flagged in the report.
  const auto& ref = builtin("reference");
  auto rr = check_model(ref, flash_next(), ok_runtime());
  bool flagged = false;
  for (const auto& f : rr.findings) flagged |= f.code == "development_only_backend";
  CHECK(flagged);
}

TEST_CASE("runtime availability is reported next to the label, never hidden") {
  const auto& strata = builtin("strata-hybrid");
  BackendRuntimeStatus none;  // not built
  auto r = check_model(strata, flash_next(), none);
  CHECK(r.has_blocker());
  CHECK(r.label == CompatLabel::kSupportedAwaitingQualification);  // the label is about the model, the blocker about this machine
  BackendRuntimeStatus nogpu = ok_runtime();
  nogpu.hardware_available = false;
  r = check_model(strata, flash_next(), nogpu);
  CHECK_FALSE(r.has_blocker());
  bool warned = false;
  for (const auto& f : r.findings) warned |= f.code == "hardware_unavailable" && f.severity == Finding::Severity::kWarning;
  CHECK(warned);
}

TEST_CASE("family globs") {
  CHECK(family_glob_match("*Flash-Next*", "Qwen3.8-Flash-Next-GSQ"));
  CHECK(family_glob_match("qwen*", "Qwen3"));
  CHECK_FALSE(family_glob_match("qwen", "Qwen3"));
  CHECK(family_glob_match("*", "anything"));
  CHECK_FALSE(family_glob_match("*Flash-Next", "Flash-Next-x"));
  CHECK_FALSE(family_glob_match(std::string(200, '*'), "x"));
}
