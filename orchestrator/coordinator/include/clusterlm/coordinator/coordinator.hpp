#pragma once
// Coordinator: ClusterLM Father's distributed inference orchestration.
//
// Father owns the canonical model, the token-dependent prefix (embedding + PLE) and the tail (head, sampling,
// MTP drafter), and drives Nodes through the protocol:
//   connect -> PreparePlan -> provision selected objects -> PlanReady -> (AuthorizePeer) -> sessions/windows
//   -> ReleaseLease.
// Every speculative window is a transaction: Father runs q positions through all stages, decides the accepted
// prefix locally, commits that length on every domain and waits for every CommitAck before the next dependent
// window. Any stage failure, stale epoch or unknown commit outcome invalidates the distributed session
// (AbortSession everywhere, epoch advanced); Father keeps the conversation and the caller decides on retry or
// downgrade. Activations are never persisted.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/digest.hpp"
#include "clusterlm/common/ids.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/execution_domain.hpp"
#include "clusterlm/objects/manifest.hpp"
#include "clusterlm/transport/impairment.hpp"
#include "clusterlm/transport/transport.hpp"

namespace clusterlm::domain {
class Drafter;
}

namespace clusterlm::coordinator {

struct NodeEndpoint {
  std::string name;
  transport::Endpoint endpoint;
  std::string device_id;  // pinned identity (TLS fingerprint); empty in insecure loopback mode
};

inline constexpr int kFatherDomain = -1;

struct StagePlan {
  StageId stage;
  domain::StageRole role = domain::StageRole::kMiddle;
  objects::LayerRange layers;
  int domain = kFatherDomain;  // index into CoordinatorConfig::nodes, or kFatherDomain
};

struct ClusterPlan {
  std::vector<StagePlan> stages;  // in pipeline order: prefix, middles..., tail
  std::uint32_t max_context = 2048;
  std::uint32_t max_window = 256;  // max(q, prefill chunk)
  // Per-object allocation target overrides (default kCpuResident). Names are manifest object names.
  std::vector<std::pair<std::string, objects::AllocationTarget>> targets;

  // "0-4@father,4-10@0,10-13@1,13-16@father": layer ranges [begin,end) and their owners (node index or father).
  static Result<ClusterPlan> parse(std::string_view text, std::uint32_t n_layers);
  Status validate(const objects::ModelGeometry& geometry, std::size_t node_count) const;
  Digest256 hash(const Digest256& model_root) const;
  std::string describe() const;
};

struct CoordinatorConfig {
  std::filesystem::path model_dir;      // canonical model directory (manifest.json + shards)
  transport::SecurityConfig security;
  std::vector<NodeEndpoint> nodes;
  bool direct_peer = true;              // Node->Node activation forwarding instead of Father relay
  std::optional<transport::NetworkConditions> impairment;  // simulation only
  std::shared_ptr<transport::FaultInjector> faults;
  std::uint32_t provision_chunk_bytes = 1u << 20;
  std::chrono::milliseconds request_timeout{10'000};
  std::chrono::milliseconds window_timeout{30'000};
  std::chrono::milliseconds prepare_timeout{600'000};
};

struct NodeProvisionReport {
  std::string node;
  std::uint64_t objects = 0;
  std::uint64_t bytes = 0;
  double prepare_ms = 0;  // PreparePlan sent -> PlanReady received
  std::uint64_t node_prepare_ns = 0;
};

struct PrepareReport {
  Digest256 plan_hash;
  std::vector<NodeProvisionReport> nodes;
  std::uint64_t father_resident_bytes = 0;
  double total_ms = 0;
};

struct GenerationRequest {
  std::vector<std::int32_t> prompt;
  std::uint32_t max_new_tokens = 32;
  std::uint32_t q = 1;               // verification width (1 = no speculation)
  std::uint32_t prefill_chunk = 128;
  std::shared_ptr<domain::Drafter> drafter;  // required when q > 1
};

// Per-round trace entry. Contains sizes, counts and timings only — never token IDs or activation values.
struct RoundTrace {
  bool prefill = false;
  std::uint32_t positions = 0;
  std::uint32_t accepted = 0;
  double draft_ms = 0;
  double prefix_ms = 0;
  double remote_ms = 0;   // send of first remote RunWindow -> final StageResult received
  double tail_ms = 0;
  double commit_ms = 0;
  double total_ms = 0;
  std::uint32_t boundary_messages = 0;   // activation-carrying messages sent or received by Father
  std::uint64_t boundary_payload_bytes = 0;
  std::uint32_t control_messages = 0;
  std::vector<domain::StageTiming> remote_timings;
};

struct GenerationResult {
  std::vector<std::int32_t> tokens;  // Father-local
  std::vector<RoundTrace> rounds;
  Epoch epoch;
  double prefill_ms = 0;
  double decode_ms = 0;
  double first_token_ms = 0;
  std::uint32_t decode_rounds = 0;
  std::uint32_t decode_tokens = 0;
};

struct ReleaseReport {
  struct NodeRelease {
    std::string node;
    bool resources_released = false;
    bool storage_cleaned = false;
    std::uint64_t residual_bytes = 0;
    double release_ms = 0;
    std::string errors;
  };
  std::vector<NodeRelease> nodes;
};

class Coordinator {
 public:
  static Result<std::unique_ptr<Coordinator>> create(CoordinatorConfig config);
  ~Coordinator();
  Coordinator(const Coordinator&) = delete;
  Coordinator& operator=(const Coordinator&) = delete;

  const objects::ModelManifest& manifest() const;
  // Open control channels to every configured Node and collect their resource offers.
  Status connect();
  // Provision and prepare a plan. Nodes become Ready; Father's local prefix/tail domains are prepared.
  Result<PrepareReport> prepare(const ClusterPlan& plan);
  // Run one generation as a fresh distributed session.
  Result<GenerationResult> generate(const GenerationRequest& request);
  // Release every Node lease and Father's local domains.
  Result<ReleaseReport> release();
  // True while every Node in the current plan is Ready under the lease Father prepared.
  bool ready() const;

  struct Impl;

 private:
  explicit Coordinator(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Greedy argmax over one position's logits (ties -> lowest token index). Father-local.
std::int32_t argmax(std::span<const float> logits);

}  // namespace clusterlm::coordinator
