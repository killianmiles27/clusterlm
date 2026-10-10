// Profile export/import: no machine identity leaves, nothing is auto-bound or auto-exposed, collisions are explicit.
#include <doctest/doctest.h>

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "clusterlm/profiles/exchange.hpp"

using namespace clusterlm;
using namespace clusterlm::profiles;
using nlohmann::json;

namespace {
Profile load(const char* name) {
  std::ifstream in(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles-example/" + name + ".profile.json", std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  auto p = profile_from_json(ss.str());
  REQUIRE_MESSAGE(p.is_ok(), p.status().to_string());
  return std::move(p).value();
}
std::string fp(char c) { return std::string(64, c); }
}  // namespace

TEST_CASE("export drops fingerprints and the local library id and names Workers by a stable slug") {
  Profile p = load("ultra");
  p.model.library_id = "mdl_" + std::string(24, 'a');
  p.topology.slots[1].select = Selector{Selector::Mode::kMachine, "", fp('a'), std::nullopt};
  p.topology.slots[1].select->binding.clear();
  p.topology.slots[2].select = Selector{Selector::Mode::kMachine, "", fp('b'), std::nullopt};
  p.topology.slots[2].select->binding.clear();
  p.topology.slots[1].select->machine = fp('a');
  p.topology.slots[2].select->machine = fp('b');
  const auto names = [](const std::string& f) { return f == std::string(64, 'a') ? "ROG Flow G14!" : "ROG Flow G14?"; };  // collide after slugging
  const Profile e = export_profile(p, names);
  CHECK_FALSE(e.model.library_id);
  CHECK(e.topology.slots[1].select->mode == Selector::Mode::kBinding);
  CHECK(e.topology.slots[1].select->binding == "worker:rog-flow-g14");
  CHECK(e.topology.slots[2].select->binding == "worker:rog-flow-g14-2");
  const std::string text = to_json(e);
  CHECK(text.find(fp('a')) == std::string::npos);
  CHECK(text.find(fp('b')) == std::string::npos);
  CHECK(profile_from_json(text).is_ok());
  CHECK(worker_binding_slug("") == "worker:worker");
  CHECK(worker_binding_slug("  !! ") == "worker:worker");
  CHECK(worker_binding_slug(std::string(80, 'x')).size() == 7 + 32);
}

TEST_CASE("import never exposes, never binds, and lists what is left to do") {
  Profile p = load("strong");
  p.exposure.api = true;
  p.exposure.allow_lan = true;
  library::ModelLibrary lib;
  BindingMap bindings;
  ImportContext ctx;
  ctx.library = &lib;
  ctx.bindings = &bindings;
  auto r = import_profile(to_json(p), ImportMode::kReject, ctx);
  REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
  CHECK(r->action == ImportResult::Action::kAdded);
  CHECK_FALSE(r->profile.exposure.api);
  CHECK_FALSE(r->profile.exposure.allow_lan);
  CHECK_FALSE(r->profile.model.library_id);
  REQUIRE(r->todo.size() >= 2);
  bool model = false, worker = false;
  for (const auto& t : r->todo) {
    model |= t.find("library") != std::string::npos;
    worker |= t.find("Assign a Worker") != std::string::npos;
  }
  CHECK(model);
  CHECK(worker);
}

TEST_CASE("a document that pins a device cannot be imported") {
  Profile p = load("strong");
  p.topology.slots[1].select = Selector{Selector::Mode::kMachine, "", fp('a'), std::nullopt};
  CHECK(import_profile(to_json(p), ImportMode::kReject, {}).status().code() == ErrorCode::kInvalidArgument);
}

TEST_CASE("same id: identical content is a no-op, different content follows the chosen mode") {
  std::vector<Profile> existing = {load("fast"), load("strong")};
  ImportContext ctx;
  ctx.existing = &existing;
  CHECK(import_profile(to_json(existing[0]), ImportMode::kReject, ctx)->action == ImportResult::Action::kUnchanged);

  Profile changed = existing[1];
  changed.description = "edited elsewhere";
  CHECK(import_profile(to_json(changed), ImportMode::kReject, ctx).status().code() == ErrorCode::kAlreadyExists);

  existing[1].exposure.api = true;  // the user's own exposure choice survives a replace
  auto rep = import_profile(to_json(changed), ImportMode::kReplace, ctx);
  REQUIRE_MESSAGE(rep.is_ok(), rep.status().to_string());
  CHECK(rep->action == ImportResult::Action::kReplaced);
  CHECK(rep->profile.revision == existing[1].revision + 1);
  CHECK(rep->profile.exposure.api);
  CHECK(rep->profile.description == "edited elsewhere");

  // Duplicate: fresh id, no example_of, and the api name must be changed because it collides.
  auto dup = import_profile(to_json(changed), ImportMode::kDuplicate, ctx);
  CHECK(dup.status().code() == ErrorCode::kAlreadyExists);
  ctx.new_api_model_id = "strong-copy";
  dup = import_profile(to_json(changed), ImportMode::kDuplicate, ctx);
  REQUIRE_MESSAGE(dup.is_ok(), dup.status().to_string());
  CHECK(dup->action == ImportResult::Action::kDuplicated);
  CHECK(dup->profile.id == "prof_strong");
  CHECK_FALSE(dup->profile.example_of);
  CHECK(dup->profile.revision == 1);
  CHECK(dup->profile.exposure.api_model_id == "strong-copy");
  ctx.new_api_model_id = "auto";
  CHECK_FALSE(import_profile(to_json(changed), ImportMode::kDuplicate, ctx).is_ok());
}

TEST_CASE("fresh ids are valid and unique") {
  std::vector<Profile> existing = {load("fast")};
  CHECK(fresh_profile_id("Coding (any NVIDIA Worker)", existing) == "prof_coding_any_nvidia_worker");
  existing[0].id = "prof_coding_any_nvidia_worker";
  CHECK(fresh_profile_id("Coding (any NVIDIA Worker)", existing) == "prof_coding_any_nvidia_worker_2");
  CHECK(is_profile_id(fresh_profile_id("", {})));
  CHECK(is_profile_id(fresh_profile_id("x", {})));
  CHECK(is_profile_id(fresh_profile_id(std::string(200, 'z'), {})));
}
