#pragma once
// One measured pass over a freshly started LocalCluster: prepare the plan, run the generation configurations, release,
// and record everything in a BenchmarkResult. Shared by `cluster` and `placement-validate` (which runs one pass per
// candidate plan and compares the measurement with the placement model's prediction).
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "bench_backend.hpp"
#include "cli.hpp"
#include "local_cluster.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "result.hpp"

namespace clusterlm::bench {

// The prompts of one context length and the Father-only reference output for each (empty when --no-reference).
struct PromptSet {
  std::uint32_t context = 0;
  std::vector<std::vector<std::int32_t>> prompts;
  std::vector<std::vector<std::int32_t>> references;  // same size as prompts, or empty
};

// What one pass measured, for callers that compare it with a prediction (placement-validate). Means over the repetitions
// of the first configuration (first context, first q); 0 when the pass did not get that far.
struct PassSummary {
  bool completed = false;
  double prepare_s = 0;      // PreparePlan -> every domain Ready (provisioning + load), mean over cycles
  double prefill_s = 0;      // mean time to prefill the first context's prompt
  double decode_tok_s = 0;   // mean decode rate
  std::uint32_t remote_stages = 0;
};

struct PassContext {
  PassContext(const cli::Args& a, BenchmarkResult& result) : args(a), r(result) {}
  const cli::Args& args;
  BenchmarkResult& r;
  std::string prefix;  // metric key prefix: "" for a single pass, "direct." / "relay." / "cand<i>." otherwise
  std::filesystem::path work, model_dir;
  std::string plan_text;
  // When set, the plan to run (placement-validate converts a placement plan to a ClusterPlan); `plan_text` is then only
  // used to count the Nodes.
  std::optional<coordinator::ClusterPlan> plan;
  std::size_t node_count = 0;  // Nodes to start (derived from plan_text when 0 and no plan)
  // Already-running Nodes to use for this pass, in plan node-index order (instead of `--node` or spawned processes).
  std::optional<std::vector<coordinator::NodeEndpoint>> external;
  std::vector<LocalNodeOptions> node_options;  // per Node (RAM/VRAM/disk allowances); missing entries are defaults
  std::vector<std::uint32_t> qs, contexts;
  std::vector<PromptSet> prompt_sets;  // one per context, in order
  std::uint32_t max_new = 64, prefill_chunk = 16;
  int repeat = 3;
  bool direct_peer = true;
  bool fixture = true;
  bool keyed_contexts = false;  // --contexts: results are keyed ctx<N>. even for a single context
  bool resources = true;        // process RSS/commit (+ VRAM) series; --no-resources
  double sample_ms = 200;       // minimum spacing of per-round resource samples (--sample-ms; 0 = every round)
  double telemetry_s = 5;       // NVML interval during --minutes runs (--sample-s)
  bool census = false;          // --census: cache/temp before/after
  const BackendChoice* backend = nullptr;  // null: the reference backend
  PassSummary* summary = nullptr;
};

Status run_cluster_pass(PassContext& pc);

// Father-only reference run of the same model on `backend`: the correctness reference for every distributed run.
// `max_context` bounds the session (prompt + generation + slack).
Result<std::vector<std::int32_t>> reference_tokens(const std::filesystem::path& model_dir, const BackendChoice* backend,
                                                   const std::vector<std::int32_t>& prompt, std::uint32_t max_new,
                                                   std::uint32_t max_context = 2048);

// Deterministic synthetic prompt (no corpus): n tokens inside the vocabulary.
std::vector<std::int32_t> synthetic_prompt_tokens(std::uint32_t n, std::uint32_t vocab);

// Prompts (and, unless `with_reference` is false, their Father-only reference outputs) for every context length. Made
// BEFORE any cluster or main Coordinator exists so a real backend never holds two models in device memory at once.
Result<std::vector<PromptSet>> build_prompt_sets(const cli::Args& args, const std::filesystem::path& model_dir,
                                                 const BackendChoice* backend, const std::vector<std::uint32_t>& contexts,
                                                 std::uint32_t max_new, bool with_reference, BenchmarkResult& r);

// Number of plan stages that run on Nodes in "0-4@father,4-10@0,..." text.
std::size_t remote_stage_count(const std::string& plan_text);

// ---- shared by `cluster`, `faults` and `placement-validate` ----

// A started LocalCluster plus what Father needs to talk to it.
struct ClusterSetup {
  std::filesystem::path work;
  std::filesystem::path model_dir;
  std::unique_ptr<LocalCluster> cluster;
  std::optional<transport::NetworkConditions> impairment;
  std::shared_ptr<domain::BackendAdapter> backend;  // Father's prefix/tail backend (null: reference)
};

// `--node NAME=HOST:PORT[@FINGERPRINT]` (repeatable): Nodes that are already running on other machines (real links), in
// the order the plan's node indices refer to them. Empty when no --node was given. The same grammar as clusterlm-father.
Result<std::vector<coordinator::NodeEndpoint>> parse_external_nodes(const cli::Args& args);
// The same grammar for any flag's values (placement-validate: --endpoint PROFILE_ID=HOST:PORT@FINGERPRINT).
Result<std::vector<coordinator::NodeEndpoint>> parse_node_specs(const std::vector<std::string>& specs, const std::string& flag);
bool is_loopback_host(const std::string& host);

std::filesystem::path default_work_dir(const std::string& name);
// --model DIR, or the generated fixture model under `work`.
Result<std::filesystem::path> ensure_model(const cli::Args& args, const std::filesystem::path& work);
// Starts `nodes` Node processes (the options of Node i from `overrides[i]` when present) - or, with --node, attaches to the
// Nodes already running there (--identity DIR: Father's persistent identity, which the Nodes pin).
Result<ClusterSetup> setup(const cli::Args& args, const std::filesystem::path& work, const std::filesystem::path& model_dir,
                           std::size_t nodes, const BackendChoice* backend, const std::vector<LocalNodeOptions>& overrides = {},
                           bool supervised = false, const std::vector<coordinator::NodeEndpoint>* external = nullptr);
coordinator::CoordinatorConfig father_config(const ClusterSetup& s, const cli::Args& args,
                                             std::optional<bool> direct_peer = std::nullopt);
std::size_t remote_count(const coordinator::ClusterPlan& p);
// Check "<label>": every Node reports zero staged bytes.
void record_census(BenchmarkResult& r, LocalCluster& cluster, const std::string& label);

}  // namespace clusterlm::bench
