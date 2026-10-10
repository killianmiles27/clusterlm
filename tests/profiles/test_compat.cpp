// Profile level-2 validation: identity resolution + backend descriptor + topology limits, before any provisioning.
#include <doctest/doctest.h>

#include "clusterlm/objects/fixture_gguf.hpp"
#include "clusterlm/profiles/compat.hpp"
#include "test_util.hpp"

using namespace clusterlm;
using namespace clusterlm::profiles;
using clusterlm::testutil::TempDir;

namespace {

struct World {
  TempDir dir{"compat"};
  library::ModelLibrary lib;
  domain::BackendRegistry registry;
  Profile profile;
  CompatContext ctx;

  World() {
    auto m = objects::write_fixture_gguf(objects::FixtureSpec::tiny(), dir.path(), {});
    REQUIRE(m.is_ok());
    auto rec = library::identify_model({dir.path() / std::string(objects::kFixtureGgufShard0)}, {});
    REQUIRE_MESSAGE(rec.is_ok(), rec.status().to_string());
    REQUIRE(lib.upsert(*rec).is_ok());

    // A test-only descriptor derived from the shipped Strata one: it lists the fixture's architecture and tensor types.
    auto d = *domain::BackendRegistry::builtin().find("strata-hybrid");
    d.model_support.families = {{"*", rec->architecture, domain::CompatLabel::kSupportedAwaitingQualification, {"test"}}};
    for (const auto& t : rec->tensor_types)
      if (std::find(d.model_support.tensor_formats.begin(), d.model_support.tensor_formats.end(), t.type) == d.model_support.tensor_formats.end())
        d.model_support.tensor_formats.push_back(t.type);
    auto reg = domain::BackendRegistry::of({d, *domain::BackendRegistry::builtin().find("reference")});
    REQUIRE_MESSAGE(reg.is_ok(), reg.status().to_string());
    registry = std::move(reg).value();

    profile.id = "prof_test_one";
    profile.name = "Test";
    profile.model.identity.family = rec->family;
    profile.model.identity.quant = rec->quant;
    profile.backend.id = "strata-hybrid";
    profile.context = {4096, 4096, {}};
    profile.topology.slots = {Slot{Slot::Kind::kHost, "host", "Host", false, std::nullopt},
                              Slot{Slot::Kind::kWorker, "w1", "", false, Selector{Selector::Mode::kBinding, "g14", "", std::nullopt}}};
    ctx.library = &lib;
    ctx.backends = &registry;
    ctx.runtime = [](const domain::BackendDescriptor&) {
      domain::BackendRuntimeStatus r;
      r.built = r.runtime_present = r.hardware_available = true;
      return r;
    };
  }
  bool has(const ProfileCompat& c, const std::string& code) const {
    for (const auto& f : c.findings)
      if (f.code == code) return true;
    return false;
  }
};

}  // namespace

TEST_CASE("an unpinned identity that matches one library model is usable but flagged, never verified") {
  World w;
  auto c = check_profile(w.profile, w.ctx);
  CHECK(c.model.kind == library::Resolution::Kind::kUnpinnedMatch);
  CHECK(w.has(c, "model_unpinned"));
  CHECK_FALSE(c.has_blocker());
  REQUIRE(c.report);
  CHECK(c.label() == domain::CompatLabel::kSupportedAwaitingQualification);
  CHECK(c.report->max_workers == 2);
}

TEST_CASE("a missing model, unknown backend and a quantization that is not in the library are blockers") {
  World w;
  auto p = w.profile;
  p.model.identity.quant = "IQ3_S";  // not in the library: no nearest-quant substitution
  auto c = check_profile(p, w.ctx);
  CHECK(c.has_blocker());
  CHECK(w.has(c, "model_missing"));
  CHECK_FALSE(c.report);

  p = w.profile;
  p.backend.id = "vllm";
  c = check_profile(p, w.ctx);
  CHECK(w.has(c, "backend_unknown"));
  CHECK(c.has_blocker());
}

TEST_CASE("a pinned identity resolves only through the verified or user-confirmed root") {
  World w;
  auto p = w.profile;
  p.model.identity.expected_root_hash = std::string(64, 'e');
  CHECK(w.has(check_profile(p, w.ctx), "model_missing"));
  REQUIRE(w.lib.pin_root(w.lib.records().front().id, std::string(64, 'e')).is_ok());
  auto c = check_profile(p, w.ctx);
  CHECK(c.model.kind == library::Resolution::Kind::kVerified);
  CHECK_FALSE(c.has_blocker());
}

TEST_CASE("library_id never bypasses the identity check") {
  World w;
  auto p = w.profile;
  p.model.library_id = "mdl_" + std::string(24, 'a');
  CHECK(w.has(check_profile(p, w.ctx), "library_id_stale"));
  p.model.library_id = w.lib.records().front().id;
  CHECK_FALSE(check_profile(p, w.ctx).has_blocker());
  p.model.identity.quant = "Q2_K";  // identity no longer matches that record
  CHECK(check_profile(p, w.ctx).has_blocker());
}

TEST_CASE("topology, speculation, placement and options are bounded by the descriptor") {
  World w;
  auto p = w.profile;
  p.topology.slots.push_back(Slot{Slot::Kind::kWorker, "w2", "", false, Selector{Selector::Mode::kBinding, "a", "", std::nullopt}});
  p.topology.slots.push_back(Slot{Slot::Kind::kWorker, "w3", "", false, Selector{Selector::Mode::kBinding, "b", "", std::nullopt}});
  CHECK(w.has(check_profile(p, w.ctx), "too_many_workers"));  // validated_max_workers is 2

  p = w.profile;
  p.speculation.max_q = 8;  // descriptor validates 4
  CHECK(w.has(check_profile(p, w.ctx), "speculation_too_wide"));

  p = w.profile;
  p.backend.options = {{"local_micro_batch", std::int64_t{4}}};
  CHECK_FALSE(check_profile(p, w.ctx).has_blocker());
  p.backend.options = {{"local_micro_batch", std::int64_t{99}}};
  CHECK(w.has(check_profile(p, w.ctx), "backend_options_invalid"));
  p.backend.options = {{"gpu_layers", std::int64_t{10}}};  // placement-owned quantity: not offered
  CHECK(w.has(check_profile(p, w.ctx), "backend_options_invalid"));
}

TEST_CASE("the development-only reference backend cannot serve a product profile") {
  World w;
  auto p = w.profile;
  p.backend.id = "reference";
  p.topology.slots.resize(1);
  CHECK(w.has(check_profile(p, w.ctx), "backend_development_only"));
}

TEST_CASE("a backend that is not built blocks, whatever the model label") {
  World w;
  w.ctx.runtime = [](const domain::BackendDescriptor&) { return domain::BackendRuntimeStatus{}; };
  auto c = check_profile(w.profile, w.ctx);
  CHECK(c.has_blocker());
  REQUIRE(c.report);
  CHECK(c.report->has_blocker());
}
