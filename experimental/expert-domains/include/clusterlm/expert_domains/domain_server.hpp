#pragma once
// ExpertDomainServer: one remote expert domain of the grouped topology.
//
// Owns a set of routed experts for EVERY layer (resident CPU FP32, decoded from objects resolved through the
// objects API) and nothing else: no dense weights, no embedding, no sequence state, no tokens. It is stateless
// between requests; each ExpertBatch is answered with exactly one ExpertResult (or one ExpertError).
//
// Runs in-process on its own thread over a transport::Connection (loopback TCP today, the LAN tomorrow), the
// same way LocalCluster stands in for remote Nodes. The pure compute path `execute()` is also callable
// directly, which the unit tests use.
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "clusterlm/expert_domains/quant_experts.hpp"
#include "clusterlm/expert_domains/wire.hpp"
#include "clusterlm/objects/canonical_store.hpp"

namespace clusterlm::expert_domains {

struct ExpertDomainConfig {
  std::string name = "expert-domain";
  std::vector<std::uint32_t> owned_experts;  // global expert ids, ascending; wire "local id" indexes this list
  std::uint32_t first_layer = 0;             // layers served: [first_layer, end_layer); end 0 = every layer
  std::uint32_t end_layer = 0;
  Epoch epoch;                               // batches naming any other epoch are rejected as stale
  ExpertDecodeLimits limits;
  // Default (empty representation): the fixture's FP32 experts resolved from the objects. "iq3_s" / "iq2_xs": the
  // Strata CPU kernels on synthetic blobs (quant_experts.hpp); no expert objects are resolved in that mode.
  ExpertKernelSpec kernel;
  // Test-only fault injection: after this many batches the server drops the connection WITHOUT replying
  // (0 = never). Emulates a domain that dies mid-window.
  std::uint32_t die_after_batches = 0;
};

struct ExpertDomainMetrics {
  std::uint64_t batches = 0;
  std::uint64_t rejected = 0;
  std::uint64_t executions = 0;      // (position, expert) pairs executed
  std::uint64_t compute_ns = 0;
  std::uint64_t bytes_received = 0;  // payload bytes
  std::uint64_t bytes_sent = 0;
  std::uint64_t resident_bytes = 0;  // decoded FP32 expert weights held
};

class ExpertDomainServer {
 public:
  // Resolves and decodes every owned expert of every served layer. Objects the resolver lacks are a plan error
  // (kNotFound); the server never fetches anything else.
  static Result<std::unique_ptr<ExpertDomainServer>> create(const objects::ModelManifest& manifest,
                                                            const objects::ObjectResolver& resolver,
                                                            ExpertDomainConfig config);
  ~ExpertDomainServer();
  ExpertDomainServer(const ExpertDomainServer&) = delete;
  ExpertDomainServer& operator=(const ExpertDomainServer&) = delete;

  // Validates and executes one batch. Thread-safe (serialized).
  Result<ExpertResult> execute(const ExpertBatch& batch);

  // Starts the serving thread on `connection`. Returns when the thread is running. Each new connection is a new
  // Father session: the window counter and the fault-injection counter restart (a LAN domain outlives its Fathers).
  Status serve(std::unique_ptr<transport::Connection> connection);
  // Closes the connection and joins the thread. Idempotent.
  void stop();
  // True while the serving thread is alive.
  bool serving() const { return running_.load(); }

  void set_epoch(Epoch epoch) { epoch_.store(epoch.value); }
  const ExpertDomainConfig& config() const { return config_; }
  std::string kernel_path() const;  // "" for the fixture FP32 experts
  ExpertDomainMetrics metrics() const;

  // Drops the decoded weights (ephemeral: a domain holds nothing after its lease).
  void release();

 private:
  ExpertDomainServer(const objects::ModelManifest& manifest, ExpertDomainConfig config);
  Status load(const objects::ObjectResolver& resolver);
  void run(std::shared_ptr<transport::Connection> connection);
  Result<ExpertResult> execute_locked(const ExpertBatch& batch);

  struct Impl;
  std::unique_ptr<Impl> impl_;
  ExpertDomainConfig config_;
  std::atomic<std::uint64_t> epoch_{0};
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::shared_ptr<transport::Connection> connection_;
};

// Builds a resolver that holds ONLY the routed-expert objects `owned_experts` for the given layers, copied from
// Father's canonical store: what a plan-scoped lease of this domain would contain.
Result<std::unique_ptr<objects::InMemoryResolver>> provision_expert_objects(
    const objects::CanonicalModelStore& father_store, const std::vector<std::uint32_t>& owned_experts,
    std::uint32_t first_layer, std::uint32_t end_layer);

}  // namespace clusterlm::expert_domains
