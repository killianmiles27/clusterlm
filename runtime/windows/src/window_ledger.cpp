#include "clusterlm/windows/window_ledger.hpp"

namespace clusterlm::windows {

namespace {
Status failed(const std::string& m) { return make_error(ErrorCode::kFailedPrecondition, m); }
}  // namespace

Result<WindowLedger::SessionState*> WindowLedger::lookup(Epoch epoch, SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) return make_error(ErrorCode::kNotFound, "session not open");
  if (it->second.epoch != epoch) {
    ++stale_rejections_;
    return make_error(ErrorCode::kStaleEpoch, "request names epoch " + epoch.str() + " but session is at epoch " +
                                                  it->second.epoch.str());
  }
  return &it->second;
}

Status WindowLedger::open_session(Epoch epoch, SessionId session) {
  auto it = sessions_.find(session);
  if (it != sessions_.end()) {
    if (epoch < it->second.epoch) {
      ++stale_rejections_;
      return make_error(ErrorCode::kStaleEpoch, "cannot open session under an older epoch");
    }
    if (epoch == it->second.epoch) return make_error(ErrorCode::kAlreadyExists, "session already open");
    // A newer epoch re-establishes the session from scratch; everything under the old epoch is discarded.
    sessions_.erase(it);
  }
  SessionState s;
  s.epoch = epoch;
  sessions_.emplace(session, std::move(s));
  return Status::ok();
}

Status WindowLedger::begin_window(const domain::WindowRequest& request) {
  CLM_ASSIGN_OR_RETURN(SessionState * s, lookup(request.epoch, request.session));
  if (request.positions == 0) return make_error(ErrorCode::kInvalidArgument, "window must have at least one position");
  if (s->outstanding) return failed("a window is already outstanding for this session");
  if (s->last_window && !(*s->last_window < request.window))
    return failed("window id " + request.window.str() + " is not greater than the last admitted window id");
  if (request.base_position != s->committed_position)
    return failed("window base position " + std::to_string(request.base_position) + " != committed position " +
                  std::to_string(s->committed_position));
  if (request.expected_state != s->state)
    return failed("window expects state version " + request.expected_state.str() + " but committed state is " +
                  s->state.str());
  s->outstanding = request;
  s->last_window = request.window;
  return Status::ok();
}

void WindowLedger::fail_window(SessionId session) {
  auto it = sessions_.find(session);
  if (it != sessions_.end()) it->second.outstanding.reset();
}

Result<WindowLedger::CommitDecision> WindowLedger::begin_commit(const domain::CommitRequest& request) {
  CLM_ASSIGN_OR_RETURN(SessionState * s, lookup(request.epoch, request.session));
  CommitDecision d;
  if (s->outstanding && s->outstanding->window == request.window) {
    if (request.accepted == 0 || request.accepted > s->outstanding->positions)
      return make_error(ErrorCode::kInvalidArgument, "accepted must be in [1, q]");
    if (request.expected_state != s->state)
      return failed("commit expects state version " + request.expected_state.str() + " but committed state is " +
                    s->state.str());
    d.kind = CommitDecision::Kind::kApply;
    d.window = *s->outstanding;
    return d;
  }
  if (s->last_ack && s->last_ack->window == request.window) {
    // Replay of the commit we already applied: it must be the identical request.
    if (request.accepted != s->last_accepted || request.expected_state.value + 1 != s->last_ack->state.value)
      return failed("repeated commit differs from the one already applied");
    d.kind = CommitDecision::Kind::kReplay;
    d.ack = *s->last_ack;
    return d;
  }
  return failed("commit names window " + request.window.str() + " which is neither outstanding nor the last committed");
}

domain::CommitAck WindowLedger::finish_commit(const domain::CommitRequest& request) {
  auto it = sessions_.find(request.session);
  if (it == sessions_.end()) return {};
  SessionState& s = it->second;
  s.committed_position += request.accepted;
  s.state = s.state.next();
  s.outstanding.reset();
  s.last_accepted = request.accepted;
  s.last_ack = domain::CommitAck{request.session, request.window, s.committed_position, s.state};
  return *s.last_ack;
}

Result<domain::WindowAbortAck> WindowLedger::abort_window(Epoch epoch, SessionId session, WindowId window) {
  CLM_ASSIGN_OR_RETURN(SessionState * s, lookup(epoch, session));
  if (!s->last_window || *s->last_window < window)
    return failed("abort names window " + window.str() + " which was never admitted");
  if (s->outstanding && s->outstanding->window == window) s->outstanding.reset();
  return domain::WindowAbortAck{session, window, s->committed_position, s->state};
}

Status WindowLedger::abort_session(Epoch epoch, SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) return Status::ok();  // already gone: abort is idempotent cleanup
  if (it->second.epoch != epoch) {
    ++stale_rejections_;
    return make_error(ErrorCode::kStaleEpoch, "abort names a stale epoch");
  }
  sessions_.erase(it);
  return Status::ok();
}

void WindowLedger::clear() { sessions_.clear(); }

const WindowLedger::SessionState* WindowLedger::session(SessionId id) const {
  auto it = sessions_.find(id);
  return it == sessions_.end() ? nullptr : &it->second;
}

}  // namespace clusterlm::windows
