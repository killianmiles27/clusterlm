#pragma once
// ExecutionDomain: the semantic contract of one stage's executor, independent of where it runs.
//
// A domain owns a set of complete layers (its LayerRange): their weights (GPU-resident and CPU-resident
// complements), their sequence state (recurrent/KV/indexer), scratch, CPU pool and expert residency. It
// never shares pointers, CUDA handles or allocators with another domain. The same interface is served
// in-process (tests), from a Node process over localhost, or across the LAN — callers cannot tell which.
//
// Roles:
//   kPrefix — Father only. Consumes token IDs (Father-local) and runs embedding, PLE and its layer range.
//   kMiddle — token-free. Consumes and produces StageActivations only.
//   kTail   — Father only. Consumes activations, runs its layers, output head; produces logits locally.
// A single Father process hosts both prefix and tail as two domains sharing one physical machine but with
// disjoint layer state.
//
// Speculative-window contract (per session):
//   * At most one uncommitted window per session.
//   * run_window requires window_id > last window_id and base_position == committed position; the window's
//     state changes are temporary until commit_window.
//   * commit_window(accepted) keeps exactly the first `accepted` positions' state changes (1..q) and
//     discards the rest; repeating the same commit is idempotent and returns the same ack.
//   * abort_window discards the outstanding window (if it is the named one) and keeps the session at its
//     committed state — used when a window is cancelled or must be dropped without ending the conversation;
//     repeating it is harmless.
//   * abort_session discards all uncommitted work and all sequence state for the session.
//   * Any message with a stale epoch is rejected with kStaleEpoch.
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/common/ids.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/boundary.hpp"
#include "clusterlm/objects/manifest.hpp"
#include "clusterlm/objects/provisioned.hpp"

namespace clusterlm::domain {

enum class StageRole : std::uint8_t { kPrefix = 0, kMiddle = 1, kTail = 2 };
std::string_view to_string(StageRole role);

struct DomainSpec {
  StageId stage;
  StageRole role = StageRole::kMiddle;
  objects::LayerRange layers;
  std::uint32_t max_context = 0;      // positions of sequence state to reserve
  std::uint32_t max_window = 0;       // maximum q (verification width) or prefill chunk positions
  std::uint32_t max_sessions = 1;
  // Largest batch this domain processes in one local step (0 = max_window). A transport chunk larger than this
  // is executed in local sub-batches, so a small GPU never forces a smaller chunk size on every other stage.
  std::uint32_t max_local_batch = 0;
};

// Memory a domain needs, reported before allocation so admission can happen against live budgets.
struct DomainRequirements {
  std::uint64_t cpu_weight_bytes = 0;
  std::uint64_t gpu_weight_bytes = 0;
  std::uint64_t state_bytes = 0;     // sequence state at max_context for max_sessions
  std::uint64_t window_bytes = 0;    // temporary speculative-window state at max_window
  std::uint64_t scratch_bytes = 0;
  std::uint64_t staging_bytes = 0;   // bounded host staging/pinned ring
  std::vector<std::string> required_objects;
};

struct WindowRequest {
  Epoch epoch;
  SessionId session;
  WindowId window;
  std::uint64_t base_position = 0;     // committed position the window starts at
  StateVersion expected_state;         // committed state version the window is computed against
  std::uint32_t positions = 0;         // q (or prefill chunk length)
};

struct CommitRequest {
  Epoch epoch;
  SessionId session;
  WindowId window;
  std::uint32_t accepted = 0;          // accepted-prefix length; never candidate token IDs
  StateVersion expected_state;
};

struct WindowAbortAck {
  SessionId session;
  WindowId window;
  std::uint64_t committed_position = 0;
  StateVersion state;
};

struct CommitAck {
  SessionId session;
  WindowId window;
  std::uint64_t committed_position = 0;
  StateVersion state;
  friend bool operator==(const CommitAck&, const CommitAck&) = default;
};

struct StageTiming {
  std::uint64_t compute_ns = 0;
  std::uint64_t cpu_expert_ns = 0;
  std::uint64_t gpu_ns = 0;
  std::uint32_t experts_selected = 0;      // routed selections executed (count only; never which)
  std::uint32_t experts_cpu = 0;
  std::uint32_t experts_gpu = 0;
};

struct DomainMetrics {
  std::uint64_t windows_run = 0;
  std::uint64_t windows_committed = 0;
  std::uint64_t windows_aborted = 0;
  std::uint64_t positions_committed = 0;
  std::uint64_t resident_weight_bytes = 0;
  std::uint64_t state_bytes = 0;
  std::uint64_t compute_ns_total = 0;
  std::uint64_t stale_rejections = 0;
};

// Aggregate routed-expert selection counts for the layers a domain owns (calibration input for placement).
// Counts only — never which position selected what, never token-linked sequences (privacy contract). Counts every
// executed position, including verified-then-rejected speculative positions (they execute and cost the same).
struct RoutingAggregate {
  std::uint32_t first_layer = 0;
  std::vector<std::vector<std::uint64_t>> counts;  // [local layer][expert]
  std::uint64_t positions = 0;                     // positions executed per layer
};

// Tail output for one window: per-position logits (Father-local, never sent over the network).
struct Logits {
  std::uint32_t positions = 0;
  std::uint32_t vocab = 0;
  std::vector<float> data;
};

class ExecutionDomain {
 public:
  virtual ~ExecutionDomain() = default;

  virtual const DomainSpec& spec() const = 0;
  virtual BoundaryLayout boundary() const = 0;

  // Report memory needs and the object names this domain must have before prepare().
  virtual Result<DomainRequirements> describe_requirements() const = 0;
  // Bind provisioned objects and allocate state. All required objects must resolve locally.
  virtual Status prepare(const objects::ObjectResolver& resolver) = 0;
  virtual Status open_session(Epoch epoch, SessionId session) = 0;

  // Prefix stage: token IDs never leave Father. tokens.size() == request.positions.
  virtual Result<StageActivations> run_prefix(const WindowRequest& request, std::span<const std::int32_t> tokens) = 0;
  // Middle stage (and the layer part of a tail stage).
  virtual Result<StageActivations> run_window(const WindowRequest& request, const StageActivations& input) = 0;
  // Tail stage: layers + head.
  virtual Result<Logits> run_tail(const WindowRequest& request, const StageActivations& input) = 0;

  virtual Result<CommitAck> commit_window(const CommitRequest& request) = 0;
  virtual Result<WindowAbortAck> abort_window(Epoch epoch, SessionId session, WindowId window) = 0;
  virtual Status abort_session(Epoch epoch, SessionId session) = 0;
  // Release every allocation. After release the domain must be prepared again.
  virtual Status release() = 0;
  virtual DomainMetrics read_metrics() const = 0;
  // Opt-in aggregate routing statistics (off by default). Backends without the counter report kUnimplemented.
  virtual Status enable_routing_aggregation(bool) { return make_error(ErrorCode::kUnimplemented, "routing aggregation"); }
  virtual Result<RoutingAggregate> routing_aggregate() const {
    return make_error(ErrorCode::kUnimplemented, "routing aggregation");
  }
};

}  // namespace clusterlm::domain
