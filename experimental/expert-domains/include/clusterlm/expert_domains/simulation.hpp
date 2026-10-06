#pragma once
// Simulation bench: the grouped expert-domain topology (in-process domains over real loopback transport with
// simulated network conditions) next to the layer-domain pipeline (clusterlm-node processes driven by the
// production Coordinator), plus the analytic model for the real geometry.
//
// Everything this produces is Synthetic: a fixture model, localhost, simulated links. Nothing here measures the
// target hardware; HQ-P0C-01 does that.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "clusterlm/objects/fixture_model.hpp"
#include "result.hpp"  // bench::BenchmarkResult (bench/src, via clusterlm_bench_harness)

namespace clusterlm::expert_domains {

struct SimulationOptions {
  objects::FixtureSpec fixture;
  std::vector<std::uint32_t> q_values{1, 2, 3, 4};
  std::vector<std::string> presets{"unlimited", "gige-simulated", "gige-degraded"};
  std::uint32_t remote_domains = 2;
  std::uint32_t windows = 12;      // measured decode windows per (preset, q)
  std::uint32_t prompt_len = 8;    // prefilled before measuring
  bool strided = false;            // expert ownership: false = contiguous ranges, true = round-robin
  bool overlap_local = true;
  bool check_reference = true;     // compare every window's logits with the unsplit reference domain
  double reference_tolerance = 1e-5;  // max |grouped - reference| / max |reference| per window
  bool layer_domain = true;        // also run the layer-domain design through the Coordinator (spawns 2 nodes)
  std::filesystem::path work_dir;  // model + staging; created if absent
  std::FILE* report = nullptr;     // human-readable tables (nullptr = none)
};

// Appends configuration, metrics, checks and trace entries to `result`. Returns an error only for setup failures;
// failed comparisons are recorded as failed checks.
Status run_simulation(const SimulationOptions& options, bench::BenchmarkResult& result);

}  // namespace clusterlm::expert_domains
