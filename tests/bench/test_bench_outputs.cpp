// Checks on files produced by the clusterlm-bench smoke commands registered in tests/bench/CMakeLists.txt.
// These tests run only when that directory is known (CMake defines CLUSTERLM_BENCH_RESULTS_DIR).
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "bench_profile.hpp"
#include "clusterlm/placement/placement.hpp"
#include "clusterlm/placement/profile.hpp"

using namespace clusterlm;
using namespace clusterlm::placement;
using nlohmann::json;

#ifdef CLUSTERLM_BENCH_RESULTS_DIR
namespace {
std::string results(const char* name) { return std::string(CLUSTERLM_BENCH_RESULTS_DIR) + "/" + name; }
json load_json(const std::string& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in, "missing " << path);
  return json::parse(in);
}
}  // namespace

TEST_CASE("calibrate writes a profile that placement loads directly, with Measured fields and nothing Qualified") {
  auto loaded = load_hardware_profile(results("profile-smoke-father.json"));
  REQUIRE_MESSAGE(loaded.is_ok(), loaded.status().to_string());
  HardwareProfile p = loaded.value();
  CHECK(p.id == "smoke-father");
  CHECK(p.role == DomainRole::kFather);

  const auto doc = load_json(results("result-calibrate.json"));
  const std::string run_id = doc["configuration"]["run_id"].get<std::string>();
  const std::string src = "bench:" + run_id + "@smoke-father";
  CHECK(doc["environment"]["host_role"] == "development-host");  // never a target unless --on-target
  CHECK(doc["provenance"] == "Measured");

  int measured = 0;
  for_each_quantity(p, [&](const std::string& path, Quantity& q) {
    CAPTURE(path);
    CHECK(q.provenance != Provenance::kQualified);
    if (q.provenance == Provenance::kMeasured) {
      ++measured;
      CHECK(q.source == src);
    }
  });
  CHECK(measured >= 5);
  CHECK(p.memory.ram_total.provenance == Provenance::kMeasured);
  CHECK(p.memory.ram_bandwidth.provenance == Provenance::kMeasured);
  CHECK(p.cpu.usable_threads.provenance == Provenance::kMeasured);
  CHECK(p.cpu.expert_bytes_per_s.at("f32").provenance == Provenance::kMeasured);
  // Fields this host cannot measure keep the base fixture's Synthetic values.
  CHECK(p.gpu.dense_layer_ms.at("attention").provenance == Provenance::kSynthetic);
  CHECK(weakest_provenance(p) == Provenance::kSynthetic);
  CHECK(mark_qualified(p, "x").is_ok() == false);  // the tool cannot be the path to Qualified
  CHECK(doc["metrics"]["profile.weakest_provenance"] == "synthetic");
}

TEST_CASE("a measured profile can drive the placement search") {
  auto father = load_hardware_profile(results("profile-smoke-father.json"));
  REQUIRE(father.is_ok());
  // Nodes and network from the Synthetic fixtures; Father's measured fields come from this host.
  PlacementRequest req;
  req.father = father.value();
  for (const char* n : {"Node-G14-4070-8945HS.json", "Node-3060-5600.json"}) {
    auto p = load_hardware_profile(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/" + n);
    REQUIRE(p.is_ok());
    req.nodes.push_back(p.value());
  }
  auto net = load_network_profile(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/network-gige-simulated.json");
  REQUIRE(net.is_ok());
  req.network = net.value();
  req.model = flash_next_planning_estimate();
  auto result = search_placements(req);
  // The search may legitimately find no feasible plan when the measured RAM of the development host is small;
  // what matters is that measured profiles are accepted as inputs and the outcome is a defined Status/Result.
  if (!result.is_ok()) CHECK(result.status().code() != ErrorCode::kInvalidArgument);
}

TEST_CASE("loopback network measurements stay Synthetic in the NetworkProfile") {
  auto net = load_network_profile(results("network-loopback.json"));
  REQUIRE_MESSAGE(net.is_ok(), net.status().to_string());
  CHECK(net->synthetic_fixture);
  CHECK(weakest_provenance(net.value()) == Provenance::kSynthetic);
  CHECK(net->links.size() == 2);
  for (const auto& l : net->links) {
    CHECK(l.bandwidth_bytes_per_s.value > 0);
    CHECK(l.rtt_ms.value > 0);
  }
  const auto doc = load_json(results("result-transport.json"));
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["metrics"].contains("concurrent.egress_bytes_per_s"));
  CHECK(doc["metrics"].contains("link.sim-node0.jitter_ms.64"));
  CHECK(doc["metrics"].contains("link.sim-node1.rx_bytes_per_s.1048576"));
}

TEST_CASE("node-to-node measurement is orchestrated through child processes and is Synthetic on localhost") {
  const auto doc = load_json(results("result-node-to-node.json"));
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["simulated"]["localhost_cluster"] == true);
  CHECK(doc["metrics"].contains("node_to_node.rtt_ms.64"));
  CHECK(doc["metrics"].contains("node_to_node.tx_bytes_per_s.1048576"));
  for (const auto& c : doc["checks"]) CHECK(c["passed"] == true);
}

TEST_CASE("cpu smoke result carries the sweep, q scaling, dequant split and pending HQ items") {
  const auto doc = load_json(results("result-cpu.json"));
  CHECK(doc["provenance"] == "Measured");
  const auto& m = doc["metrics"];
  for (const char* k : {"cpu.expert_bytes_per_s.f32", "cpu.expert_bytes_per_s.q8_0-fixture", "cpu.q_scaling", "cpu.best_threads",
                        "cpu.usable_threads", "cpu.isa_selected", "cpu.q8_0-fixture.dequant_fraction",
                        "cpu.f32.thread_scaling.t1", "cpu.f32.q4.disjoint.union_experts_mean"})
    CHECK_MESSAGE(m.contains(k), k);
  bool hq_cpu_01 = false;
  for (const auto& id : doc["pending_qualification"]) hq_cpu_01 |= id == "HQ-CPU-01";
  CHECK(hq_cpu_01);  // reference kernels are not the production CPU path
  const auto sus = load_json(results("result-cpu-sustained.json"));
  CHECK(sus["metrics"].contains("cpu.sustained_factor"));
  CHECK(sus["metrics"].contains("thermal.time_to_equilibrium_s"));
  CHECK(sus["trace"].size() >= 2);
}

TEST_CASE("gpu smoke without a CUDA device reports it and lists the pending items") {
  const auto doc = load_json(results("result-gpu.json"));
  const bool present = doc["metrics"]["gpu.present"].get<bool>();
  if (!present) {
    CHECK(doc["provenance"] == "Synthetic");  // nothing was measured
    bool gpu01 = false;
    for (const auto& id : doc["pending_qualification"]) gpu01 |= id == "HQ-GPU-01";
    CHECK(gpu01);
    CHECK(doc["metrics"].contains("gpu.unavailable_reason"));
  }
}

TEST_CASE("cluster smoke results: routing comparison and prepare-phase metrics") {
  const auto routing = load_json(results("result-cluster-routing.json"));
  CHECK(routing["provenance"] == "Synthetic");
  const auto& m = routing["metrics"];
  for (const char* k : {"direct.q2.round_ms", "relay.q2.round_ms", "direct.q2.prefill_tok_s", "direct.q2.decode_tok_s",
                        "direct.q2.mtp.acceptance_rate", "direct.q2.stage.node0.compute_ms", "direct.q2.stage.remote_wait_ms",
                        "relay.q2.draft_ms", "relay.q2.verify_ms", "relay.q2.commit_ms", "direct.prepare.total_ms"})
    CHECK_MESSAGE(m.contains(k), k);
  for (const auto& c : routing["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>());

  const auto prep = load_json(results("result-cluster-prepare.json"));
  CHECK(prep["metrics"]["prepare.total_ms"]["n"] == 2);  // one prepare per cycle
  CHECK(prep["metrics"].contains("prepare.node.node0.provision_bytes_per_s"));
  for (const auto& c : prep["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>());
}
#endif
