// Local IPC: frame codec, round trips, bounds, garbage, authorization, per-user socket permissions.
#include <doctest/doctest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

#include "clusterlm/platform/ipc.hpp"
#include "clusterlm/platform/ipc_messages.hpp"
#include "clusterlm/platform/ipc_server_loop.hpp"

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

using namespace clusterlm;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

struct TempDir {
  fs::path path;
  explicit TempDir(const char* name) : path(fs::temp_directory_path() / (std::string("clm-ipc-") + name)) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
};

ipc::Endpoint endpoint(const TempDir& d, const char* name = "test.pipe") { return ipc::Endpoint{name, d.path}; }

std::unique_ptr<ipc::Server> make_server(const ipc::Endpoint& ep, ipc::Authorizer authz = {}, std::uint32_t max_frame = 4096) {
  ipc::ServerOptions so;
  so.endpoint = ep;
  so.max_frame_bytes = max_frame;
  so.authorizer = std::move(authz);
  auto s = ipc::Server::create(std::move(so));
  REQUIRE_MESSAGE(s.is_ok(), s.status().to_string());
  return std::move(s).value();
}

// Connects and accepts concurrently (Unix connect completes against the backlog, Windows needs a waiting accept).
struct Pair {
  std::unique_ptr<ipc::Connection> client, server;
};
Pair connect_pair(ipc::Server& srv, const ipc::Endpoint& ep, const ipc::ClientOptions& co = {}) {
  Pair p;
  Result<std::unique_ptr<ipc::Connection>> accepted = make_error(ErrorCode::kInternal, "unset");
  std::thread t([&] { accepted = srv.accept(3000ms); });
  auto c = ipc::connect(ep, co, 2000ms);
  t.join();
  REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
  REQUIRE_MESSAGE(accepted.is_ok(), accepted.status().to_string());
  p.client = std::move(c).value();
  p.server = std::move(accepted).value();
  return p;
}

}  // namespace

TEST_CASE("frame codec round trips and enforces bounds") {
  ipc::Envelope e{7, 3, Bytes{1, 2, 3}};
  auto frame = ipc::encode_frame(e);
  REQUIRE(frame.is_ok());
  REQUIRE(frame->size() == 4 + 4 + 3);
  CHECK((*frame)[0] == 7);  // little-endian body length (4 header bytes + 3 payload)
  auto body = ipc::decode_body(ByteSpan(frame->data() + 4, frame->size() - 4));
  REQUIRE(body.is_ok());
  CHECK(body->kind == 7);
  CHECK(body->version == 3);
  CHECK(body->payload == e.payload);

  CHECK_FALSE(ipc::encode_frame(ipc::Envelope{1, 1, Bytes(100)}, 64).is_ok());
  CHECK(ipc::check_frame_length(3, 64).code() == ErrorCode::kProtocolError);   // shorter than the envelope header
  CHECK(ipc::check_frame_length(65, 64).code() == ErrorCode::kProtocolError);  // longer than the bound
  CHECK(ipc::check_frame_length(4, 64).is_ok());
  CHECK(ipc::check_frame_length(64, 64).is_ok());
  CHECK_FALSE(ipc::decode_body(ByteSpan(frame->data() + 4, 3)).is_ok());
}

TEST_CASE("endpoint names are validated") {
  TempDir d("names");
  CHECK(ipc::Endpoint{"ClusterLM.Node.Helper", d.path}.native_path().is_ok());
  CHECK_FALSE(ipc::Endpoint{"", d.path}.native_path().is_ok());
  CHECK_FALSE(ipc::Endpoint{"a/b", d.path}.native_path().is_ok());
  CHECK_FALSE(ipc::Endpoint{"..\\x", d.path}.native_path().is_ok());
  CHECK_FALSE(ipc::Endpoint{std::string(65, 'a'), d.path}.native_path().is_ok());
}

TEST_CASE("round trip in both directions, ordering preserved") {
  TempDir d("roundtrip");
  const auto ep = endpoint(d);
  auto srv = make_server(ep);
  auto p = connect_pair(*srv, ep);
  for (std::uint16_t i = 0; i < 20; ++i) REQUIRE(p.client->send({i, 1, Bytes(i, static_cast<std::uint8_t>(i))}, 1000ms).is_ok());
  for (std::uint16_t i = 0; i < 20; ++i) {
    auto m = p.server->receive(1000ms);
    REQUIRE(m.is_ok());
    CHECK(m->kind == i);
    CHECK(m->payload.size() == i);
  }
  REQUIRE(p.server->send({99, 2, Bytes{9}}, 1000ms).is_ok());
  auto r = p.client->receive(1000ms);
  REQUIRE(r.is_ok());
  CHECK(r->kind == 99);
  CHECK(r->payload == Bytes{9});
}

TEST_CASE("receive times out cleanly and the connection stays usable; peer close is reported") {
  TempDir d("timeout");
  const auto ep = endpoint(d);
  auto srv = make_server(ep);
  auto p = connect_pair(*srv, ep);
  auto r = p.server->receive(50ms);
  REQUIRE_FALSE(r.is_ok());
  CHECK(r.status().code() == ErrorCode::kDeadlineExceeded);
  CHECK(p.server->is_open());
  REQUIRE(p.client->send({1, 1, {}}, 1000ms).is_ok());
  CHECK(p.server->receive(1000ms).is_ok());

  p.client->close();
  r = p.server->receive(1000ms);
  REQUIRE_FALSE(r.is_ok());
  CHECK(r.status().code() == ErrorCode::kUnavailable);
}

TEST_CASE("a blocked receive is woken by close() from another thread") {
  TempDir d("wake");
  const auto ep = endpoint(d);
  auto srv = make_server(ep);
  auto p = connect_pair(*srv, ep);
  std::atomic<bool> returned{false};
  std::thread t([&] {
    auto r = p.server->receive(10000ms);
    CHECK_FALSE(r.is_ok());
    returned.store(true);
  });
  std::this_thread::sleep_for(100ms);
  p.server->close();
  t.join();
  CHECK(returned.load());
}

TEST_CASE("an oversize message is refused on send and an oversize announced frame poisons the receiver") {
  TempDir d("oversize");
  const auto ep = endpoint(d);
  auto srv = make_server(ep, {}, 256);
  ipc::ClientOptions co;
  co.max_frame_bytes = 256;
  auto p = connect_pair(*srv, ep, co);
  CHECK(p.client->send({1, 1, Bytes(1000)}, 1000ms).code() == ErrorCode::kResourceExhausted);
  CHECK(p.client->is_open());  // refused locally, nothing was written

#ifndef _WIN32
  // A hostile peer announces a 1 GiB frame over a raw socket: the server must refuse without allocating it.
  const auto path = ep.native_path().value();
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof addr.sun_path, "%s", path.c_str());
  REQUIRE(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
  auto conn = srv->accept(2000ms);
  REQUIRE(conn.is_ok());
  const std::uint8_t huge[4] = {0, 0, 0, 0x40};  // 0x40000000
  REQUIRE(::write(fd, huge, 4) == 4);
  auto r = conn.value()->receive(1000ms);
  REQUIRE_FALSE(r.is_ok());
  CHECK(r.status().code() == ErrorCode::kProtocolError);
  CHECK_FALSE(conn.value()->is_open());
  ::close(fd);
#endif
}

#ifndef _WIN32
TEST_CASE("garbage frames: too-short length, truncated body and EOF mid-frame") {
  TempDir d("garbage");
  const auto ep = endpoint(d);
  auto srv = make_server(ep);
  const auto path = ep.native_path().value();
  auto raw = [&] {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof addr.sun_path, "%s", path.c_str());
    REQUIRE(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    return fd;
  };
  {  // length 2 < envelope header
    const int fd = raw();
    auto c = srv->accept(2000ms);
    REQUIRE(c.is_ok());
    const std::uint8_t bytes[6] = {2, 0, 0, 0, 0xAA, 0xBB};
    REQUIRE(::write(fd, bytes, 6) == 6);
    auto r = c.value()->receive(1000ms);
    REQUIRE_FALSE(r.is_ok());
    CHECK(r.status().code() == ErrorCode::kProtocolError);
    ::close(fd);
  }
  {  // header promises 100 bytes, peer sends 10 then hangs up
    const int fd = raw();
    auto c = srv->accept(2000ms);
    REQUIRE(c.is_ok());
    std::uint8_t bytes[14] = {100, 0, 0, 0};
    REQUIRE(::write(fd, bytes, 14) == 14);
    ::close(fd);
    auto r = c.value()->receive(1000ms);
    REQUIRE_FALSE(r.is_ok());
    CHECK(r.status().code() == ErrorCode::kUnavailable);
    CHECK_FALSE(c.value()->is_open());
  }
  {  // EOF with a partial length header
    const int fd = raw();
    auto c = srv->accept(2000ms);
    REQUIRE(c.is_ok());
    const std::uint8_t bytes[2] = {8, 0};
    REQUIRE(::write(fd, bytes, 2) == 2);
    ::close(fd);
    auto r = c.value()->receive(1000ms);
    REQUIRE_FALSE(r.is_ok());
    CHECK(r.status().code() == ErrorCode::kUnavailable);
  }
}

TEST_CASE("POSIX socket is owner-only: 0600 socket in a 0700 directory") {
  TempDir d("perms");
  const auto sub = d.path / "run";
  const ipc::Endpoint ep{"perm.pipe", sub};
  auto srv = make_server(ep);
  struct stat st {};
  REQUIRE(::stat((sub / "perm.pipe.sock").c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
  REQUIRE(::stat(sub.c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0700);
}

TEST_CASE("a socket directory that is open to other users is tightened to 0700") {
  TempDir d("open-dir");
  ::chmod(d.path.c_str(), 0777);
  auto srv = make_server(endpoint(d));
  struct stat st {};
  REQUIRE(::stat(d.path.c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0700);
}

TEST_CASE("a server refuses to take over a path that is not its own socket") {
  TempDir d("squat");
  const auto ep = endpoint(d);
  { std::ofstream(ep.native_path().value()) << "regular file"; }
  ipc::ServerOptions so;
  so.endpoint = ep;
  CHECK_FALSE(ipc::Server::create(std::move(so)).is_ok());
}
#endif

TEST_CASE("authorizer rejection disconnects the client and accept reports it; the next client still works") {
  TempDir d("authz");
  const auto ep = endpoint(d);
  std::atomic<int> calls{0};
  auto srv = make_server(ep, [&](const ipc::PeerCredentials&) -> Status {
    return ++calls == 1 ? make_error(ErrorCode::kPermissionDenied, "no") : Status::ok();
  });
  {
    Result<std::unique_ptr<ipc::Connection>> accepted = make_error(ErrorCode::kInternal, "unset");
    std::thread t([&] { accepted = srv->accept(3000ms); });
    auto c = ipc::connect(ep, {}, 2000ms);
    t.join();
    REQUIRE_FALSE(accepted.is_ok());
    CHECK(accepted.status().code() == ErrorCode::kPermissionDenied);
    REQUIRE(c.is_ok());  // connect itself succeeded: authorization is the server's decision
    auto r = c.value()->receive(1000ms);
    REQUIRE_FALSE(r.is_ok());  // but the server hung up
    CHECK(r.status().code() == ErrorCode::kUnavailable);
  }
  auto p = connect_pair(*srv, ep);
  CHECK(p.client->send({1, 1, {}}, 1000ms).is_ok());
}

TEST_CASE("AuthPolicy with injected credentials") {
  ipc::PeerCredentials interactive;
  interactive.pid = 100;
  interactive.session_known = true;
  interactive.session_id = 2;
  interactive.user_id = "S-1-5-21-1-2-3-1001";
  ipc::PeerCredentials services = interactive;
  services.session_id = 0;
  ipc::PeerCredentials unknown_session;
  unknown_session.user_id = interactive.user_id;
  ipc::PeerCredentials other_user = interactive;
  other_user.user_id = "S-1-5-21-1-2-3-1002";
  ipc::PeerCredentials no_user = interactive;
  no_user.user_id.clear();

  ipc::AuthPolicy open;  // no requirements
  CHECK(open.authorize(services).is_ok());

  ipc::AuthPolicy interactive_only;
  interactive_only.require_interactive_session = true;
  CHECK(interactive_only.authorize(interactive).is_ok());
  CHECK(interactive_only.authorize(services).code() == ErrorCode::kPermissionDenied);        // session 0
  CHECK(interactive_only.authorize(unknown_session).code() == ErrorCode::kPermissionDenied);  // not established by the OS

  ipc::AuthPolicy exact;
  exact.require_interactive_session = true;
  exact.allowed_user_ids = {"S-1-5-21-1-2-3-1001"};
  CHECK(exact.authorize(interactive).is_ok());
  CHECK(exact.authorize(other_user).code() == ErrorCode::kPermissionDenied);
  CHECK(exact.authorize(no_user).code() == ErrorCode::kUnauthenticated);
  CHECK(exact.authorize(services).code() == ErrorCode::kPermissionDenied);
}

TEST_CASE("client can require the server to run as an expected user") {
  TempDir d("server-id");
  const auto ep = endpoint(d);
  auto srv = make_server(ep);
  auto self = ipc::current_user_id();
  REQUIRE(self.is_ok());
  {
    ipc::ClientOptions co;
    co.expected_server_user_ids = {self.value()};
    auto p = connect_pair(*srv, ep, co);
    CHECK(p.client->send({1, 1, {}}, 1000ms).is_ok());
  }
  ipc::ClientOptions wrong;
  wrong.expected_server_user_ids = {"S-1-5-18-not-us"};
  auto c = ipc::connect(ep, wrong, 1000ms);
  REQUIRE_FALSE(c.is_ok());
  CHECK(c.status().code() == ErrorCode::kUnauthenticated);
}

TEST_CASE("connecting to a missing server fails fast with kUnavailable") {
  TempDir d("missing");
  auto c = ipc::connect(endpoint(d), {}, 200ms);
  REQUIRE_FALSE(c.is_ok());
  CHECK(c.status().code() == ErrorCode::kUnavailable);
}

TEST_CASE("ServerLoop serves several clients concurrently and stop() joins everything") {
  TempDir d("loop");
  const auto ep = endpoint(d);
  ipc::ServerOptions so;
  so.endpoint = ep;
  auto server = ipc::Server::create(std::move(so));
  REQUIRE(server.is_ok());
  ipc::ServerLoop loop(std::move(server).value(), [](const std::shared_ptr<ipc::Connection>& c, const std::atomic<bool>& stop) {
    while (!stop.load() && c->is_open()) {
      auto m = c->receive(100ms);
      if (!m.is_ok()) {
        if (m.status().code() == ErrorCode::kDeadlineExceeded) continue;
        return;
      }
      m->kind = static_cast<std::uint16_t>(m->kind + 1000);  // echo with a transformed kind
      if (!c->send(m.value(), 1000ms).is_ok()) return;
    }
  });
  loop.start();
  std::vector<std::unique_ptr<ipc::Connection>> clients;
  for (int i = 0; i < 3; ++i) {
    auto c = ipc::connect(ep, {}, 2000ms);
    REQUIRE(c.is_ok());
    clients.push_back(std::move(c).value());
  }
  for (std::size_t i = 0; i < clients.size(); ++i) {
    REQUIRE(clients[i]->send({static_cast<std::uint16_t>(i), 1, Bytes{static_cast<std::uint8_t>(i)}}, 1000ms).is_ok());
    auto r = clients[i]->receive(2000ms);
    REQUIRE(r.is_ok());
    CHECK(r->kind == i + 1000);
  }
  CHECK(loop.connections().size() == 3);
  loop.stop();
  auto r = clients[0]->receive(1000ms);
  CHECK_FALSE(r.is_ok());  // the server side closed
}

TEST_CASE("helper protocol messages round trip and reject version skew, wrong kinds and trailing bytes") {
  auto rep = ipc::decode_activity_report(ipc::encode(ipc::ActivityReport{42, true, 3}));
  REQUIRE(rep.is_ok());
  CHECK(rep->idle_seconds == 42);
  CHECK(rep->session_locked);
  CHECK(rep->session_id == 3);

  auto pause = ipc::decode_pause_request(ipc::encode(ipc::PauseRequest{600}));
  REQUIRE(pause.is_ok());
  CHECK(pause->duration_seconds == 600);
  CHECK(ipc::decode_resume_request(ipc::encode(ipc::ResumeRequest{})).is_ok());
  CHECK(ipc::decode_status_request(ipc::encode(ipc::StatusRequest{})).is_ok());

  ipc::StatusReply sr;
  sr.state = ipc::NodeState::kOffering;
  sr.paired_father = "abcd1234";
  sr.storage_bytes = 1ull << 33;
  sr.helper_reports_fresh = true;
  auto sr2 = ipc::decode_status_reply(ipc::encode(sr));
  REQUIRE(sr2.is_ok());
  CHECK(sr2->state == ipc::NodeState::kOffering);
  CHECK(sr2->paired_father == "abcd1234");
  CHECK(sr2->storage_bytes == (1ull << 33));
  CHECK(sr2->helper_reports_fresh);

  auto ack = ipc::decode_ack(ipc::encode(ipc::Ack{ErrorCode::kPermissionDenied, "nope"}));
  REQUIRE(ack.is_ok());
  CHECK(ack->code == ErrorCode::kPermissionDenied);
  CHECK(ack->message == "nope");

  auto skew = ipc::encode(ipc::ActivityReport{1, false, 0});
  skew.version = 99;
  CHECK(ipc::decode_activity_report(skew).status().code() == ErrorCode::kVersionMismatch);
  CHECK(ipc::decode_pause_request(ipc::encode(ipc::ActivityReport{1, false, 0})).status().code() == ErrorCode::kProtocolError);
  auto trailing = ipc::encode(ipc::ActivityReport{1, false, 0});
  trailing.payload.push_back(0);
  CHECK_FALSE(ipc::decode_activity_report(trailing).is_ok());
  auto truncated = ipc::encode(ipc::ActivityReport{1, false, 0});
  truncated.payload.pop_back();
  CHECK_FALSE(ipc::decode_activity_report(truncated).is_ok());
  auto bad_bool = ipc::encode(ipc::ActivityReport{1, false, 0});
  bad_bool.payload[4] = 7;
  CHECK_FALSE(ipc::decode_activity_report(bad_bool).is_ok());
  auto bad_state = ipc::encode(sr);
  bad_state.payload[0] = 200;
  CHECK_FALSE(ipc::decode_status_reply(bad_state).is_ok());
}

TEST_CASE("Father UI envelopes are version-stamped and pipe names are sanitised per user") {
  auto e = ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes{1, 2});
  CHECK(e.kind == 0x100);
  CHECK(e.version == ipc::kFatherUiProtocolVersion);
  CHECK(ipc::father_ui_pipe_name("S-1-5-21-1001") == "ClusterLM.Father.UI.S-1-5-21-1001");
  CHECK(ipc::father_ui_pipe_name("a b\\c/d") == "ClusterLM.Father.UI.a_b_c_d");
  CHECK(ipc::father_ui_pipe_name("") == "ClusterLM.Father.UI.default");
  CHECK(ipc::father_ui_pipe_name(std::string(200, 'x')).size() == std::string("ClusterLM.Father.UI.").size() + 40);
  CHECK(ipc::Endpoint{ipc::father_ui_pipe_name("S-1-5-21-1001"), "/tmp"}.native_path().is_ok());
}
