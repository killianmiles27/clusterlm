// Placement -> executable plan -> distributed execution, end to end on the fixture model.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>

#include "clusterlm/node/node_worker.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/placement/profile.hpp"
#include "clusterlm/planning/planning.hpp"

using namespace clusterlm;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(const char* name) {
  auto p = fs::temp_directory_path() / (std::string("clm-planning-") + name + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

placement::HardwareProfile profile(const std::string& file) {
  auto p = placement::load_hardware_profile(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/" + file);
  REQUIRE(p.is_ok());
  // The fixture model uses its own tensor representations; give them a synthetic CPU throughput equal to the
  // profile's first entry so the search can cost them. This is test scaffolding, not a hardware claim.
  auto prof = p.value();
  REQUIRE_FALSE(prof.cpu.expert_bytes_per_s.empty());
  const auto any = prof.cpu.expert_bytes_per_s.begin()->second;
  for (const char* quant : {"f32", "q8_0-fixture"})
    prof.cpu.expert_bytes_per_s[quant] = placement::Quantity::synthetic(any.value, "synthetic: test scaffolding");
  return prof;
}

}  // namespace

TEST_CASE("cost inputs from a manifest reflect its exact object sizes and stay Synthetic without routing data") {
  const auto dir = temp_dir("inputs");
  auto m = objects::write_fixture_model(objects::FixtureSpec{}, dir / "model");
  REQUIRE(m.is_ok());
  auto in = planning::cost_inputs_from_manifest(m.value());
  REQUIRE(in.is_ok());
  CHECK(in->n_layers() == m->geometry.n_layers);
  CHECK(in->boundary_bytes_per_position == domain::BoundaryLayout::for_geometry(m->geometry).bytes_per_position());
  std::uint64_t dense = 0;
  for (const auto& l : in->layers) dense += l.dense_bytes;
  std::uint64_t manifest_dense = 0;
  for (const auto& o : m->objects)
    if (o.kind == objects::ObjectKind::kLayerDense || o.kind == objects::ObjectKind::kSharedExpert)
      manifest_dense += o.byte_size;
  CHECK(dense == manifest_dense);
  CHECK(in->provenance == placement::Provenance::kSynthetic);
  CHECK(in->draft_ms.provenance == placement::Provenance::kSynthetic);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("a placement plan converts to a ClusterPlan that executes correctly on two Node workers") {
  const auto dir = temp_dir("exec");
  auto manifest = objects::write_fixture_model(objects::FixtureSpec{}, dir / "model");
  REQUIRE(manifest.is_ok());
  auto inputs = planning::cost_inputs_from_manifest(manifest.value());
  REQUIRE(inputs.is_ok());

  placement::PlacementRequest req;
  req.father = profile("Father-4060Ti-7600.json");
  req.nodes = {profile("Node-G14-4070-8945HS.json"), profile("Node-3060-5600.json")};
  auto net = placement::load_network_profile(std::string(CLUSTERLM_SOURCE_DIR) +
                                             "/fixtures/profiles/network-gige-simulated.json");
  REQUIRE(net.is_ok());
  req.network = net.value();
  req.model = inputs.value();
  req.context_tokens = 256;
  req.q = 1;
  req.prefix_layers_options = {4};
  auto result = placement::search_placements(req);
  REQUIRE(result.is_ok());
  REQUIRE_FALSE(result->candidates.empty());
  CHECK(result->provenance == placement::Provenance::kSynthetic);
  CHECK_FALSE(placement::require_qualified(result->candidates.front()).is_ok());

  // Pick the best candidate that uses both Nodes, so the test exercises two remote stages.
  const placement::PlacementPlan* chosen = nullptr;
  for (const auto& c : result->candidates)
    if (c.node_ids().size() == 2) {
      chosen = &c;
      break;
    }
  REQUIRE(chosen != nullptr);

  // Two Node workers in this process — still reached only over TCP through the protocol.
  std::map<std::string, int> index;
  std::vector<std::unique_ptr<node::NodeWorker>> workers;
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = dir / "model";
  cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  for (std::size_t i = 0; i < req.nodes.size(); ++i) {
    node::NodeConfig nc;
    nc.name = req.nodes[i].id;
    nc.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    nc.staging_root = dir / ("staging-" + std::to_string(i));
    nc.ram_allowance = 1ull << 30;
    nc.vram_allowance = 1ull << 30;
    auto w = node::NodeWorker::start(nc);
    REQUIRE(w.is_ok());
    cfg.nodes.push_back({nc.name, w.value()->endpoint(), ""});
    index[nc.name] = static_cast<int>(i);
    workers.push_back(std::move(w).value());
  }
  auto plan = planning::to_cluster_plan(*chosen, manifest.value(), req.father.id, index, 512, 64);
  REQUIRE(plan.is_ok());
  CHECK(plan->stages.front().role == domain::StageRole::kPrefix);
  CHECK(plan->stages.back().role == domain::StageRole::kTail);

  auto run = [&](const coordinator::ClusterPlan& p, std::vector<coordinator::NodeEndpoint> nodes) {
    auto c = cfg;
    c.nodes = std::move(nodes);
    auto coord = coordinator::Coordinator::create(c);
    REQUIRE(coord.is_ok());
    REQUIRE(coord.value()->connect().is_ok());
    auto prep = coord.value()->prepare(p);
    REQUIRE_MESSAGE(prep.is_ok(), prep.status().to_string());
    coordinator::GenerationRequest g;
    g.prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    g.max_new_tokens = 12;
    auto gen = coord.value()->generate(g);
    REQUIRE_MESSAGE(gen.is_ok(), gen.status().to_string());
    auto rel = coord.value()->release();
    REQUIRE(rel.is_ok());
    for (const auto& n : rel->nodes) CHECK(n.storage_cleaned);
    return gen->tokens;
  };
  const auto distributed = run(plan.value(), cfg.nodes);
  auto father_only = coordinator::ClusterPlan::parse("0-16@father,16-16@father", 16);
  REQUIRE(father_only.is_ok());
  const auto reference = run(father_only.value(), {});
  CHECK(distributed == reference);
  // Destroy the workers before deleting their staging roots: on Windows open handles block deletion.
  workers.clear();
  std::error_code ec;
  fs::remove_all(dir, ec);
  CHECK_FALSE(ec);
}
