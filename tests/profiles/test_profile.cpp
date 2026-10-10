// Profile and routing-alias documents: the frozen interface examples must parse, round-trip and validate as a set;
// strictness (unknown fields, bad ids, cross-object rules) is enforced.
#include <doctest/doctest.h>

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "clusterlm/profiles/validate.hpp"

using namespace clusterlm;
using namespace clusterlm::profiles;
using nlohmann::json;

namespace {
std::string read_file(const std::string& rel) {
  std::ifstream in(std::string(CLUSTERLM_SOURCE_DIR) + "/" + rel, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), rel);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
Profile example(const char* name) {
  auto p = profile_from_json(read_file(std::string("docs/interfaces/examples/") + name));
  REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
  return std::move(p).value();
}
std::string mutate(const char* name, const std::function<void(json&)>& f) {
  json j = json::parse(read_file(std::string("docs/interfaces/examples/") + name));
  f(j);
  return j.dump();
}
}  // namespace

TEST_CASE("the frozen example profiles parse, round-trip and validate as a set") {
  std::vector<Profile> set;
  for (const char* n : {"profile-fast.example.json", "profile-strong.example.json", "profile-ultra.example.json",
                        "profile-generic-requirements.example.json"})
    set.push_back(example(n));
  auto alias = alias_from_json(read_file("docs/interfaces/examples/routing-alias.example.json"));
  REQUIRE_MESSAGE(alias.is_ok(), alias.status().to_string());

  for (const auto& p : set) {
    auto again = profile_from_json(to_json(p));
    REQUIRE_MESSAGE(again.is_ok(), again.status().to_string());
    CHECK(*again == p);
  }
  auto a2 = alias_from_json(to_json(*alias));
  REQUIRE(a2.is_ok());
  CHECK(*a2 == *alias);

  auto st = validate_set(set, {*alias});
  CHECK_MESSAGE(st.is_ok(), st.to_string());
  CHECK(fallback_chain(set, "prof_example_ultra") == std::vector<std::string>{"prof_example_ultra", "prof_example_strong", "prof_example_fast"});
}

TEST_CASE("example content is what the migration table promises") {
  const Profile ultra = example("profile-ultra.example.json");
  CHECK(ultra.model.identity.quant == "IQ3_S");
  CHECK_FALSE(ultra.model.identity.expected_root_hash);  // never auto-pinned
  CHECK(ultra.model.identity.expected_files.size() == 2);
  CHECK(ultra.topology.slots.size() == 3);
  CHECK(worker_slot_count(ultra.topology) == 2);
  CHECK(worker_count_bounds(ultra.topology).min == 2);
  CHECK(ultra.topology.slots[1].select->binding == "node:laptop-class");
  CHECK(ultra.qualification_experiments.size() == 4);
  CHECK(ultra.goals.size() == 1);
  const Profile generic = example("profile-generic-requirements.example.json");
  CHECK(worker_count_bounds(generic.topology).min == 0);
  CHECK(generic.topology.slots[1].select->mode == Selector::Mode::kRequirements);
}

TEST_CASE("writers omit defaults so an older reader still understands the document") {
  Profile p = example("profile-fast.example.json");
  p.lifecycle = {};
  p.speculation = {};
  p.placement = {};
  p.on_worker_loss = {};
  p.exposure = {};
  const json j = json::parse(to_json(p));
  for (const char* k : {"lifecycle", "speculation", "placement", "on_worker_loss", "exposure", "resources", "chat_template"})
    CHECK_FALSE(j.contains(k));
}

TEST_CASE("unknown fields are rejected at every level, typos can never silently drop a security field") {
  CHECK_FALSE(profile_from_json(mutate("profile-fast.example.json", [](json& j) { j["exposur"] = json::object(); })).is_ok());
  CHECK_FALSE(profile_from_json(mutate("profile-fast.example.json", [](json& j) { j["model"]["identity"]["famly"] = "x"; })).is_ok());
  CHECK_FALSE(profile_from_json(mutate("profile-fast.example.json", [](json& j) { j["topology"]["slots"][0]["selectt"] = 1; })).is_ok());
  CHECK_FALSE(profile_from_json(mutate("profile-fast.example.json", [](json& j) { j["exposure"]["allow_lna"] = true; })).is_ok());
}

TEST_CASE("per-document rules") {
  auto bad = [](const std::function<void(json&)>& f, const char* file = "profile-ultra.example.json") {
    return !profile_from_json(mutate(file, f)).is_ok();
  };
  CHECK(bad([](json& j) { j["id"] = "Prof_Upper"; }));
  CHECK(bad([](json& j) { j["schema_version"] = 2; }));
  CHECK(bad([](json& j) { j["revision"] = 0; }));
  CHECK(bad([](json& j) { j["context"]["default_tokens"] = 999999; }));
  CHECK(bad([](json& j) { j["context"]["max_tokens"] = 100000; }));  // not the largest offered size
  CHECK(bad([](json& j) { j["context"]["offered_profiles"] = {8192, 4096}; }));
  CHECK(bad([](json& j) { j["model"]["identity"]["expected_root_hash"] = "abc"; }));
  CHECK(bad([](json& j) { j["model"]["identity"]["expected_files"][0]["name"] = "../x.gguf"; }));
  CHECK(bad([](json& j) { j["topology"]["slots"][0]["kind"] = "worker"; }));                // slot 0 must be host
  CHECK(bad([](json& j) { j["topology"]["slots"][2]["slot"] = "w1"; }));                    // duplicate slot name
  CHECK(bad([](json& j) { j["topology"]["slots"][2]["select"]["binding"] = "node:laptop-class"; }));  // same binding twice
  CHECK(bad([](json& j) { j["topology"]["slots"][1]["select"]["machine"] = std::string(64, 'a'); }));  // binding + machine
  CHECK(bad([](json& j) { j["topology"]["max_workers"] = 3; }));
  CHECK(bad([](json& j) { j["topology"]["min_workers"] = 1; }));  // below the 2 required slots
  CHECK(bad([](json& j) { j["exposure"]["api"] = true; j["exposure"].erase("api_model_id"); }));
  CHECK(bad([](json& j) { j["exposure"]["api_model_id"] = "clusterlm-ultra"; }));
  CHECK(bad([](json& j) { j["exposure"]["api_model_id"] = "auto"; }));
  CHECK(bad([](json& j) { j["on_worker_loss"].erase("fallback_profile_id"); }));
  CHECK(bad([](json& j) { j["on_worker_loss"]["fallback_profile_id"] = j["id"]; }));
  CHECK(bad([](json& j) { j["lifecycle"]["preparation"] = "manual"; j["lifecycle"]["prepare_when_available"] = true; }));
  CHECK(bad([](json& j) { j["placement"]["mode"] = "manual"; }));  // manual without stages
  CHECK(bad([](json& j) { j["goals"][0]["status"] = "measured"; }));
  CHECK(bad([](json& j) { j["speculation"]["max_q"] = 99; }));
  CHECK(profile_from_json("[]").status().code() == ErrorCode::kInvalidArgument);
  CHECK(profile_from_json("{").status().code() == ErrorCode::kInvalidArgument);
  CHECK(profile_from_json(std::string(kMaxDocumentBytes + 1, ' ')).status().code() == ErrorCode::kInvalidArgument);
  CHECK(profile_from_json(std::string(40, '[') + std::string(40, ']')).status().code() == ErrorCode::kInvalidArgument);
  // A newer schema is refused as a version problem, not guessed at.
  CHECK(profile_from_json(mutate("profile-fast.example.json", [](json& j) { j["schema_version"] = 2; })).status().code() ==
        ErrorCode::kVersionMismatch);
}

TEST_CASE("set rules: unique ids and api ids, resolvable acyclic fallbacks, alias candidates") {
  const Profile fast = example("profile-fast.example.json");
  const Profile strong = example("profile-strong.example.json");
  const Profile ultra = example("profile-ultra.example.json");
  auto alias = alias_from_json(read_file("docs/interfaces/examples/routing-alias.example.json")).value();

  CHECK(validate_set({fast, strong, ultra}, {alias}).is_ok());
  CHECK_FALSE(validate_set({fast, fast}, {}).is_ok());  // duplicate id

  auto clash = strong;
  clash.exposure.api_model_id = fast.exposure.api_model_id;
  CHECK_FALSE(validate_set({fast, clash, ultra}, {}).is_ok());  // api_model_id reused (even with api:false)

  auto alias_clash = alias;
  alias_clash.api_model_id = *fast.exposure.api_model_id;
  CHECK_FALSE(validate_set({fast, strong, ultra}, {alias_clash}).is_ok());

  CHECK_FALSE(validate_set({strong, ultra}, {}).is_ok());  // strong's fallback (fast) is missing

  auto cyc = fast;
  cyc.on_worker_loss.then = OnWorkerLoss::Then::kFallbackProfile;
  cyc.on_worker_loss.fallback_profile_id = ultra.id;
  CHECK_FALSE(validate_set({cyc, strong, ultra}, {}).is_ok());  // fast -> ultra -> strong -> fast

  auto missing = alias;
  missing.candidates[0].profile_id = "prof_does_not_exist";
  CHECK_FALSE(validate_set({fast, strong, ultra}, {missing}).is_ok());
  auto wrong_slot = alias;
  wrong_slot.candidates[0].workers_available = {"w9"};
  CHECK_FALSE(validate_set({fast, strong, ultra}, {wrong_slot}).is_ok());
}
