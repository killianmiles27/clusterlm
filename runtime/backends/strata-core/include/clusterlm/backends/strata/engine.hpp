#pragma once
// StrataEngine: the device side of one Strata execution domain, behind ClusterLM types only.
//
// The ClusterLM StrataDomain (strata_domain.hpp) owns everything the ExecutionDomain contract says about windows -
// epochs, window IDs, state versions, idempotent commits, abort, the boundary ABI and its layout, local
// sub-batching - and drives an engine through this interface. The CUDA engine (runtime/backends/strata, built with
// CLUSTERLM_ENABLE_STRATA) implements it with Strata's own WeightTable, NativeDense/NativeHead, SessionState,
// ExpertCache, CPU ExpertPool and Verifier; tests implement it with a fake so the adapter's state machine is
// verified without a device.
//
// Contract (the CUDA engine maps each call onto one Strata call):
//   * run() executes ONE engine window of at most caps().max_window positions over the session's committed state.
//     The window's changes are tentative: exactly one of commit() or abort() follows before the next run().
//   * Token IDs reach the engine only on a domain that embeds them (caps().needs_tokens: the Father prefix). On every
//     other domain `tokens` is empty and the engine must never synthesize a token array.
//   * commit(keep) makes the first `keep` positions permanent and returns only once the device work is complete.
//   * abort() restores exactly the committed state (strata patch 0003, Verifier::abort_window).
//   * Hand-off buffers are Strata's FIELD-MAJOR layout ([R: T x hc*H][bo: T x H][inj: T x hc]); the domain converts
//     from and to the position-major wire (strata_handoff.hpp).
#include <cstdint>
#include <span>
#include <string>

#include "clusterlm/common/ids.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/execution_domain.hpp"
#include "clusterlm/objects/provisioned.hpp"

namespace clusterlm::backends::strata {

struct EngineCaps {
  std::uint32_t max_window = 8;  // positions per engine window (Strata kVerifyMaxT)
  bool needs_tokens = false;     // the domain embeds token IDs (layer 0) or runs the PLE block (layer 1)
  bool has_head = false;         // the domain ends at the last layer and produces logits
  std::uint32_t vocab = 0;       // logits per position when has_head
};

struct EngineWindow {
  SessionId session;
  std::uint64_t pos0 = 0;
  std::uint32_t positions = 0;
  std::span<const std::int32_t> tokens;  // needs_tokens only; otherwise empty
  std::span<const float> handoff_in;     // field-major; empty on a domain that starts at layer 0
  std::span<float> handoff_out;          // field-major; empty on a domain with the head
  std::span<float> logits_out;           // positions * vocab; has_head only
};

struct EngineCounters {
  std::uint64_t resident_weight_bytes = 0;  // weights actually bound (GPU arena + native + expert tiers)
  std::uint64_t session_state_bytes = 0;    // per open session
};

class StrataEngine {
 public:
  virtual ~StrataEngine() = default;
  virtual EngineCaps caps() const = 0;
  // Bytes this engine will allocate for its spec, before anything is allocated (Strata's own sizing:
  // session_bytes, Verifier::init_bytes, WeightTable::pool_bytes over the domain's objects).
  virtual Result<domain::DomainRequirements> requirements(
      const std::vector<std::string>& required_objects) const = 0;
  virtual Status prepare(const objects::ObjectResolver& resolver) = 0;
  // Fresh (zeroed) sequence state for `session`; an already open session is reset.
  virtual Status open_session(SessionId session) = 0;
  virtual Status close_session(SessionId session) = 0;
  virtual Status run(const EngineWindow& window, domain::StageTiming& timing) = 0;
  virtual Status commit(SessionId session, std::uint32_t keep) = 0;
  virtual Status abort(SessionId session) = 0;
  virtual Status release() = 0;
  virtual EngineCounters counters() const = 0;
};

}  // namespace clusterlm::backends::strata
