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
#include <atomic>
#include <chrono>
#include <functional>
#include <span>
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
#include "clusterlm/domain/sampling.hpp"
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
  int provision_retries = 3;          // bulk-channel reconnects within one uninterrupted lease
  int commit_retries = 1;             // idempotent CommitWindow resends after an acknowledgement timeout
};

struct NodeProvisionReport {
  std::string node;
  std::uint64_t objects = 0;
  std::uint64_t bytes = 0;
  double prepare_ms = 0;  // PreparePlan sent -> PlanReady received
  std::uint32_t resumes = 0;  // provision-channel reconnects that resumed the transfer
  std::uint64_t node_prepare_ns = 0;
};

struct PrepareReport {
  Digest256 plan_hash;
  std::vector<NodeProvisionReport> nodes;
  std::uint64_t father_resident_bytes = 0;
  double total_ms = 0;
};

// A conversation's distributed session. Its sequence state (KV, recurrent, PLE history) stays allocated in
// every domain between turns, so a follow-up turn prefills only its new tokens. Token history is Father-local.
// A conversation becomes invalid when its distributed session is invalidated (stage loss, stale epoch); the
// caller then opens a new one and re-prefills from Father's own history.
struct ConversationState {
  bool valid = false;
  SessionId session;
  Epoch epoch;
  std::uint64_t position = 0;
  StateVersion state{0};
  WindowId window{0};
  std::vector<std::int32_t> committed;
  std::vector<std::int32_t> pending;
};

class Conversation {
 public:
  bool valid() const { return s_.valid; }
  SessionId session() const { return s_.session; }
  Epoch epoch() const { return s_.epoch; }
  std::uint64_t committed_positions() const { return s_.position; }
  // Tokens whose state is committed in every domain, then tokens emitted but not yet fed (at most one: the
  // model's last prediction, which the next turn feeds first).
  const std::vector<std::int32_t>& committed_tokens() const { return s_.committed; }
  const std::vector<std::int32_t>& pending_tokens() const { return s_.pending; }
  // Mutable state for the Coordinator that owns this conversation; not for other callers.
  ConversationState& coordinator_state() { return s_; }

 private:
  ConversationState s_;
};

struct GenerationRequest {
  std::vector<std::int32_t> prompt;  // new tokens for this turn (the whole prompt for a fresh session)
  std::uint32_t max_new_tokens = 32;
  std::uint32_t q = 1;               // verification width (1 = no speculation)
  std::uint32_t prefill_chunk = 128; // transport chunk; stages may execute it in smaller local batches
  std::uint32_t prefill_inflight = 2;  // chunks in flight across the pipeline (backpressure bound)
  std::shared_ptr<domain::Drafter> drafter;  // required when q > 1
  domain::SamplingParams sampling;   // temperature 0 = greedy
  std::vector<std::int32_t> stop_tokens;
  // Continue this conversation's session instead of a one-shot session (sequence state is reused).
  std::shared_ptr<Conversation> conversation;
  // Polled at every safe point (between windows, while waiting for stages). Prefill drains the chunks already in
  // flight so every domain stays consistent and the conversation remains usable; decode discards the
  // outstanding window with AbortWindow.
  std::shared_ptr<std::atomic<bool>> cancel;
  // Streaming: called with newly emitted tokens (Father-local) as soon as they are final.
  std::function<void(std::span<const std::int32_t>)> on_tokens;
};

// Per-round trace entry. Contains sizes, counts and timings only — never token IDs or activation values.
struct RoundTrace {
  bool prefill = false;
  std::uint32_t positions = 0;       // positions verified (q) or prefilled (chunk)
  std::uint32_t proposed = 0;        // drafted positions (q - 1)
  std::uint32_t accepted = 0;        // positions committed
  std::uint32_t accepted_drafts = 0; // drafts accepted
  std::uint32_t emitted = 0;         // tokens this round added to the output
  double draft_ms = 0;
  double prefix_ms = 0;
  double remote_ms = 0;   // first remote RunWindow sent -> final StageResult received
  double tail_ms = 0;
  double commit_ms = 0;
  double total_ms = 0;
  double wait_ms = 0;     // Father idle waiting for remote stages
  std::uint32_t in_flight = 0;       // windows in flight when this one was launched (prefill queue depth)
  std::uint32_t boundary_messages = 0;   // activation-carrying messages sent or received by Father
  std::uint64_t boundary_payload_bytes = 0;
  std::uint32_t control_messages = 0;
  std::uint32_t commit_retries = 0;
  std::vector<domain::StageTiming> remote_timings;
};

struct StagePrefillStats {
  StageId stage;
  double compute_ms = 0;  // summed stage compute over all prefill chunks
};

struct GenerationResult {
  std::vector<std::int32_t> tokens;  // Father-local; generated output only (never drafts that were rejected)
  std::vector<RoundTrace> rounds;
  Epoch epoch;
  SessionId session;
  bool cancelled = false;
  bool stopped_on_token = false;
  double prefill_ms = 0;
  double decode_ms = 0;
  double first_token_ms = 0;
  std::uint32_t prefill_tokens = 0;
  std::uint32_t prefill_chunks = 0;
  double prefill_tok_s = 0;
  std::uint32_t prefill_max_in_flight = 0;
  double prefill_wait_ms = 0;
  std::vector<StagePrefillStats> prefill_stages;
  std::uint32_t decode_rounds = 0;
  std::uint32_t decode_tokens = 0;
  std::uint64_t proposed_positions = 0;  // drafts proposed across decode rounds
  std::uint64_t accepted_drafts = 0;
  double draft_ms_total = 0, verify_ms_total = 0, commit_ms_total = 0;
  std::uint64_t peak_rss_bytes = 0;      // Father process peak resident set (0 if unavailable)
  double acceptance_rate() const {
    return proposed_positions ? static_cast<double>(accepted_drafts) / static_cast<double>(proposed_positions) : 0.0;
  }
  double tokens_per_round() const {
    return decode_rounds ? static_cast<double>(decode_tokens) / decode_rounds : 0.0;
  }
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
  // Open a conversation (a distributed session on every domain of the prepared plan).
  Result<std::shared_ptr<Conversation>> open_conversation();
  // Free a conversation's sequence state everywhere (the lease and weights stay Ready).
  Status close_conversation(Conversation& conversation);
  // Run one turn. Without `request.conversation` this is a one-shot session opened and closed around the call.
  Result<GenerationResult> generate(const GenerationRequest& request);
  // Cancel an in-progress prepare() (provisioning stops; the partial lease is released).
  void cancel_prepare();
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
