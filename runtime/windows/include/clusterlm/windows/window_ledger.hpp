#pragma once
// WindowLedger: per-domain bookkeeping for speculative windows, epochs and committed state versions.
//
// ("windows" here means speculative windows, not the operating system.)
//
// Every execution domain — reference, Strata, local or remote — delegates request admission to a ledger
// so that the transaction rules are identical everywhere:
//   * a session is opened under an epoch; requests naming another epoch are kStaleEpoch;
//   * at most one window is outstanding (run but not committed) per session;
//   * window IDs are strictly increasing; a RunWindow for an old or repeated window ID is rejected rather
//     than replayed (it may already have changed temporary state);
//   * base_position / expected_state must match the committed state exactly;
//   * commit accepted ∈ [1, q]; re-sending the commit for the last committed window is idempotent and
//     returns the identical ack; a commit for any other window is rejected;
//   * abort discards the outstanding window and invalidates the session (it must be re-opened).
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "clusterlm/common/ids.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/execution_domain.hpp"

namespace clusterlm::windows {

class WindowLedger {
 public:
  struct SessionState {
    Epoch epoch;
    std::uint64_t committed_position = 0;
    StateVersion state{0};
    std::optional<WindowId> last_window;          // highest window ID admitted
    std::optional<domain::WindowRequest> outstanding;  // window run but not yet committed
    std::optional<domain::CommitAck> last_ack;    // for idempotent commit replay
    std::uint32_t last_accepted = 0;              // accepted length of last_ack (replay must match it)
  };

  Status open_session(Epoch epoch, SessionId session);
  // Validate a RunWindow and mark it outstanding. On failure nothing changes.
  Status begin_window(const domain::WindowRequest& request);
  // Called by the domain if computing an admitted window failed: the window is dropped, the session stays
  // at its committed state (the window ID is still consumed).
  void fail_window(SessionId session);
  // Validate a commit. Returns:
  //   * kind == kApply: the domain must apply `accepted` positions of the outstanding window, then call
  //     finish_commit() to obtain the ack;
  //   * kind == kReplay: an idempotent repeat; return `ack` without touching state.
  struct CommitDecision {
    enum class Kind { kApply, kReplay } kind = Kind::kApply;
    domain::WindowRequest window;  // the outstanding window (kApply)
    domain::CommitAck ack;         // the previous ack (kReplay)
  };
  Result<CommitDecision> begin_commit(const domain::CommitRequest& request);
  domain::CommitAck finish_commit(const domain::CommitRequest& request);
  Status abort_session(Epoch epoch, SessionId session);
  void clear();

  const SessionState* session(SessionId id) const;
  std::uint64_t stale_rejections() const { return stale_rejections_; }

 private:
  Result<SessionState*> lookup(Epoch epoch, SessionId session);
  std::unordered_map<SessionId, SessionState> sessions_;
  std::uint64_t stale_rejections_ = 0;
};

}  // namespace clusterlm::windows
