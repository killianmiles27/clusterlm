#include <doctest/doctest.h>

#include "clusterlm/windows/window_ledger.hpp"

using namespace clusterlm;
using namespace clusterlm::domain;
using clusterlm::windows::WindowLedger;

namespace {
const Epoch kE{3};
const SessionId kS{7};

WindowRequest win(std::uint64_t id, std::uint64_t base, std::uint64_t state, std::uint32_t q, Epoch e = kE) {
  WindowRequest r;
  r.epoch = e;
  r.session = kS;
  r.window = WindowId{id};
  r.base_position = base;
  r.expected_state = StateVersion{state};
  r.positions = q;
  return r;
}
CommitRequest com(std::uint64_t id, std::uint32_t accepted, std::uint64_t state, Epoch e = kE) {
  CommitRequest c;
  c.epoch = e;
  c.session = kS;
  c.window = WindowId{id};
  c.accepted = accepted;
  c.expected_state = StateVersion{state};
  return c;
}
// Run + commit window `id` fully so a test can reach a later state.
void advance(WindowLedger& l, std::uint64_t id, std::uint64_t base, std::uint64_t state, std::uint32_t q, std::uint32_t acc) {
  REQUIRE(l.begin_window(win(id, base, state, q)).is_ok());
  auto d = l.begin_commit(com(id, acc, state));
  REQUIRE(d.is_ok());
  l.finish_commit(com(id, acc, state));
}
}  // namespace

TEST_CASE("open_session: duplicate, stale and newer epochs") {
  WindowLedger l;
  CHECK(l.begin_window(win(1, 0, 0, 1)).code() == ErrorCode::kNotFound);
  REQUIRE(l.open_session(kE, kS).is_ok());
  CHECK(l.open_session(kE, kS).code() == ErrorCode::kAlreadyExists);
  CHECK(l.open_session(Epoch{2}, kS).code() == ErrorCode::kStaleEpoch);
  advance(l, 1, 0, 0, 2, 2);
  REQUIRE(l.open_session(Epoch{4}, kS).is_ok());  // re-establishment resets everything
  CHECK(l.session(kS)->committed_position == 0);
  CHECK(l.session(kS)->state == StateVersion{0});
  CHECK_FALSE(l.session(kS)->last_window.has_value());
}

TEST_CASE("stale epoch is rejected on every request kind and counted") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  CHECK(l.begin_window(win(1, 0, 0, 1, Epoch{2})).code() == ErrorCode::kStaleEpoch);
  CHECK(l.begin_window(win(1, 0, 0, 1, Epoch{9})).code() == ErrorCode::kStaleEpoch);
  REQUIRE(l.begin_window(win(1, 0, 0, 1)).is_ok());
  CHECK(l.begin_commit(com(1, 1, 0, Epoch{2})).status().code() == ErrorCode::kStaleEpoch);
  CHECK(l.abort_session(Epoch{2}, kS).code() == ErrorCode::kStaleEpoch);
  CHECK(l.stale_rejections() == 4);
  CHECK(l.session(kS) != nullptr);  // the stale abort did not touch the session
  CHECK(l.session(kS)->outstanding.has_value());
}

TEST_CASE("only one outstanding window per session") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  REQUIRE(l.begin_window(win(1, 0, 0, 4)).is_ok());
  CHECK(l.begin_window(win(2, 0, 0, 4)).code() == ErrorCode::kFailedPrecondition);
  CHECK(l.begin_window(win(2, 4, 0, 4)).code() == ErrorCode::kFailedPrecondition);
  CHECK(l.session(kS)->outstanding->window == WindowId{1});  // failed attempts changed nothing
}

TEST_CASE("window ids must strictly increase") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  advance(l, 5, 0, 0, 2, 2);
  CHECK(l.begin_window(win(5, 2, 1, 2)).code() == ErrorCode::kFailedPrecondition);  // repeat
  CHECK(l.begin_window(win(3, 2, 1, 2)).code() == ErrorCode::kFailedPrecondition);  // older
  CHECK(l.begin_window(win(6, 2, 1, 2)).is_ok());
}

TEST_CASE("base position and state version must match committed state") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  CHECK(l.begin_window(win(1, 1, 0, 2)).code() == ErrorCode::kFailedPrecondition);
  CHECK(l.begin_window(win(1, 0, 1, 2)).code() == ErrorCode::kFailedPrecondition);
  CHECK(l.begin_window(win(1, 0, 0, 0)).code() == ErrorCode::kInvalidArgument);
  CHECK_FALSE(l.session(kS)->last_window.has_value());  // rejected windows do not consume an id
  advance(l, 1, 0, 0, 3, 2);                             // committed position is now 2, state 1
  CHECK(l.begin_window(win(2, 3, 1, 2)).code() == ErrorCode::kFailedPrecondition);
  CHECK(l.begin_window(win(2, 2, 0, 2)).code() == ErrorCode::kFailedPrecondition);
  CHECK(l.begin_window(win(2, 2, 1, 2)).is_ok());
}

TEST_CASE("commit accepted must be within [1, q] and leaves the window outstanding when invalid") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  REQUIRE(l.begin_window(win(1, 0, 0, 3)).is_ok());
  CHECK(l.begin_commit(com(1, 0, 0)).status().code() == ErrorCode::kInvalidArgument);
  CHECK(l.begin_commit(com(1, 4, 0)).status().code() == ErrorCode::kInvalidArgument);
  CHECK(l.begin_commit(com(1, 2, 5)).status().code() == ErrorCode::kFailedPrecondition);  // wrong state version
  CHECK(l.session(kS)->outstanding.has_value());
  for (std::uint32_t accepted : {1u, 2u, 3u}) {
    WindowLedger m;
    REQUIRE(m.open_session(kE, kS).is_ok());
    REQUIRE(m.begin_window(win(1, 0, 0, 3)).is_ok());
    auto d = m.begin_commit(com(1, accepted, 0));
    REQUIRE(d.is_ok());
    CHECK(d->kind == WindowLedger::CommitDecision::Kind::kApply);
    CHECK(d->window.positions == 3);
    const CommitAck ack = m.finish_commit(com(1, accepted, 0));
    CHECK(ack.committed_position == accepted);
    CHECK(ack.state == StateVersion{1});
    CHECK(ack.window == WindowId{1});
    CHECK_FALSE(m.session(kS)->outstanding.has_value());
  }
}

TEST_CASE("commit replay is idempotent and returns the identical ack") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  REQUIRE(l.begin_window(win(1, 0, 0, 4)).is_ok());
  REQUIRE(l.begin_commit(com(1, 3, 0)).is_ok());
  const CommitAck first = l.finish_commit(com(1, 3, 0));

  auto replay = l.begin_commit(com(1, 3, 0));
  REQUIRE(replay.is_ok());
  CHECK(replay->kind == WindowLedger::CommitDecision::Kind::kReplay);
  CHECK(replay->ack == first);
  CHECK(l.session(kS)->committed_position == 3);  // unchanged
  CHECK(l.session(kS)->state == StateVersion{1});

  // A "replay" that differs from what was applied is an error, not a silent success.
  CHECK(l.begin_commit(com(1, 2, 0)).status().code() == ErrorCode::kFailedPrecondition);
  CHECK(l.begin_commit(com(1, 3, 1)).status().code() == ErrorCode::kFailedPrecondition);
}

TEST_CASE("commit for a window that is neither outstanding nor the last committed is rejected") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  CHECK(l.begin_commit(com(1, 1, 0)).status().code() == ErrorCode::kFailedPrecondition);  // nothing outstanding
  advance(l, 1, 0, 0, 2, 2);
  advance(l, 2, 2, 1, 2, 1);
  CHECK(l.begin_commit(com(1, 2, 0)).status().code() == ErrorCode::kFailedPrecondition);  // too old for replay
  REQUIRE(l.begin_window(win(3, 3, 2, 2)).is_ok());
  CHECK(l.begin_commit(com(9, 1, 2)).status().code() == ErrorCode::kFailedPrecondition);  // other window
  CHECK(l.begin_commit(com(2, 1, 1)).is_ok());  // replay of the last committed one still works
  CHECK(l.session(kS)->outstanding->window == WindowId{3});
}

TEST_CASE("fail_window drops the window but consumes its id") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  REQUIRE(l.begin_window(win(1, 0, 0, 2)).is_ok());
  l.fail_window(kS);
  CHECK_FALSE(l.session(kS)->outstanding.has_value());
  CHECK(l.session(kS)->committed_position == 0);
  CHECK(l.begin_window(win(1, 0, 0, 2)).code() == ErrorCode::kFailedPrecondition);
  CHECK(l.begin_window(win(2, 0, 0, 2)).is_ok());
}

TEST_CASE("abort discards the session; it must be re-opened") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, kS).is_ok());
  REQUIRE(l.begin_window(win(1, 0, 0, 2)).is_ok());
  REQUIRE(l.abort_session(kE, kS).is_ok());
  CHECK(l.session(kS) == nullptr);
  CHECK(l.begin_window(win(2, 0, 0, 2)).code() == ErrorCode::kNotFound);
  CHECK(l.begin_commit(com(1, 1, 0)).status().code() == ErrorCode::kNotFound);
  CHECK(l.abort_session(kE, kS).is_ok());  // idempotent
  REQUIRE(l.open_session(kE, kS).is_ok());
  CHECK(l.begin_window(win(1, 0, 0, 2)).is_ok());  // fresh session: window ids restart
}

TEST_CASE("sessions are independent and clear() forgets them") {
  WindowLedger l;
  REQUIRE(l.open_session(kE, SessionId{1}).is_ok());
  REQUIRE(l.open_session(kE, SessionId{2}).is_ok());
  WindowRequest a = win(1, 0, 0, 2);
  a.session = SessionId{1};
  REQUIRE(l.begin_window(a).is_ok());
  WindowRequest b = win(1, 0, 0, 2);
  b.session = SessionId{2};
  CHECK(l.begin_window(b).is_ok());
  l.clear();
  CHECK(l.session(SessionId{1}) == nullptr);
}
