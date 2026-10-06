#pragma once
// StrataDomain: the ClusterLM ExecutionDomain of the Strata backend.
//
// It owns the domain contract - role rules, WindowLedger admission, idempotent commits, abort, the boundary ABI and
// its layout conversion, local sub-batching, metrics - and drives a StrataEngine (engine.hpp) for everything that
// touches the device. The CUDA engine is runtime/backends/strata; tests drive the same state machine with a fake
// engine, so every rule here is verified without a GPU.
//
// Role rules (Strata specifics, docs/backends/strata-port.md finding 2):
//   * the PREFIX starts at layer 0 and holds every token-dependent layer: Strata's PLE block is a layer-1 module
//     (Verifier::ple_stage) and the manifest's ple_layer must be in it too -> layers.end >= max(2, ple_layer + 1);
//   * MIDDLE and TAIL start after it and are token-free: they never receive, store or synthesize token IDs;
//   * the TAIL ends at the last layer and holds the head (logits stay Father-local). A Strata tail runs its layers
//     and the head as one Verifier window, so it serves run_tail only.
//
// Local sub-batching: a window of q positions runs in engine windows of at most
//   local_batch = min(max_local_batch (0 = max_window), engine max window (Strata: 8)).
// q <= local_batch is one engine window: any accepted prefix may be committed, and abort_window restores the
// committed state exactly. A larger window (a prefill transport chunk) runs as consecutive engine windows, each
// committed provisionally before the next one starts (a Strata window replays GDN state only at commit, so the
// next sub-batch must see the previous one committed). Such a window can only be committed whole - which is what
// prefill (auto_commit) does; a partial commit or an abort of it cannot be honoured, so the session is invalidated
// (kAborted) and Father re-opens and re-prefills it. ADR 0201.
//
// Not thread-safe: one caller drives a domain at a time (a Node's worker, a Father stage thread).
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>

#include "clusterlm/backends/strata/engine.hpp"
#include "clusterlm/backends/strata/object_map.hpp"
#include "clusterlm/domain/execution_domain.hpp"
#include "clusterlm/windows/window_ledger.hpp"

namespace clusterlm::backends::strata {

struct StrataDomainOptions {
  StrataObjectOptions objects;  // e.g. with_mtp on a tail that hosts the drafter
};

// The first layer a token-free domain may own: Strata's PLE layer (1) and the manifest's ple_layer stay on the prefix.
std::uint32_t token_free_first_layer(const objects::ModelGeometry& g);

// What the engine may be handed as the token pointer of a Strata Verifier call: the window's tokens on a domain that
// needs them, nothing (nullptr) on every other domain - Verifier::run takes tokens == nullptr there (strata patch
// 0002). The CUDA engine passes exactly this pointer to Strata; tests check it.
const std::int32_t* verifier_tokens(const EngineCaps& caps, std::span<const std::int32_t> tokens);

class StrataDomain final : public domain::ExecutionDomain {
 public:
  static Result<std::unique_ptr<StrataDomain>> create(const objects::ModelManifest& manifest,
                                                      const domain::DomainSpec& spec,
                                                      std::unique_ptr<StrataEngine> engine,
                                                      StrataDomainOptions options = {});
  ~StrataDomain() override;

  const domain::DomainSpec& spec() const override { return spec_; }
  domain::BoundaryLayout boundary() const override { return layout_; }

  Result<domain::DomainRequirements> describe_requirements() const override;
  Status prepare(const objects::ObjectResolver& resolver) override;
  Status open_session(Epoch epoch, SessionId session) override;

  Result<domain::StageActivations> run_prefix(const domain::WindowRequest& request,
                                              std::span<const std::int32_t> tokens) override;
  Result<domain::StageActivations> run_window(const domain::WindowRequest& request,
                                              const domain::StageActivations& input) override;
  Result<domain::Logits> run_tail(const domain::WindowRequest& request, const domain::StageActivations& input) override;

  Result<domain::CommitAck> commit_window(const domain::CommitRequest& request) override;
  Result<domain::WindowAbortAck> abort_window(Epoch epoch, SessionId session, WindowId window) override;
  Status abort_session(Epoch epoch, SessionId session) override;
  Status release() override;
  domain::DomainMetrics read_metrics() const override;

  // Positions per engine window (see the header comment).
  std::uint32_t local_batch() const;
  domain::StageTiming last_timing() const { return timing_; }
  StrataEngine& engine() { return *engine_; }
  const objects::ModelManifest& manifest() const { return manifest_; }

 private:
  StrataDomain(objects::ModelManifest manifest, domain::DomainSpec spec, std::unique_ptr<StrataEngine> engine,
               StrataDomainOptions options);
  Status init();

  struct Session {
    Epoch epoch;
  };
  struct Window {
    WindowId id;
    std::uint32_t positions = 0;
    std::uint32_t provisional = 0;  // positions already committed to the engine by local sub-batching
  };

  Result<domain::StageActivations> execute(const domain::WindowRequest& request, const domain::StageActivations* input,
                                           std::span<const std::int32_t> tokens, domain::Logits* logits);
  // Drops a session whose engine state no longer matches the ledger (a split window that cannot be undone, a failed
  // engine commit): the ledger forgets it, the engine frees it, and the caller must re-open it.
  void invalidate(SessionId session);

  objects::ModelManifest manifest_;
  domain::DomainSpec spec_;
  domain::BoundaryLayout layout_;
  std::unique_ptr<StrataEngine> engine_;
  StrataDomainOptions options_;
  EngineCaps caps_;

  bool prepared_ = false;
  windows::WindowLedger ledger_;
  std::unordered_map<SessionId, Session> sessions_;
  std::unordered_map<SessionId, Window> windows_;
  domain::DomainMetrics metrics_;
  domain::StageTiming timing_;
};

}  // namespace clusterlm::backends::strata
