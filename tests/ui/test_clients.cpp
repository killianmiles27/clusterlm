// Client implementations: IpcNodeClient against a fake service on the real helper-pipe transport, the documented
// IpcFatherClient seam, ScriptedFatherClient behaviour, and the display helpers.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

#include "clusterlm/ui/clients.hpp"
#include "clusterlm/ui/format.hpp"

using namespace clusterlm;
using namespace clusterlm::ui;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

// A minimal Node service: answers StatusRequest / PauseRequest / ResumeRequest on the helper endpoint.
class FakeService {
 public:
  explicit FakeService(const char* name) : dir_(fs::temp_directory_path() / (std::string("clm-ui-") + name)) {
    fs::remove_all(dir_);
    fs::create_directories(dir_);
    endpoint_ = {ipc::kNodeHelperPipeName, dir_};
    start();
  }
  ~FakeService() {
    stop();
    fs::remove_all(dir_);
  }
  void start() {
    ipc::ServerOptions so;
    so.endpoint = endpoint_;
    so.access = ipc::PipeAccess::kOwnerSystemAndInteractiveUsers;
    auto s = ipc::Server::create(std::move(so));
    REQUIRE_MESSAGE(s.is_ok(), s.status().to_string());
    server_ = std::move(s).value();
    running_ = true;
    thread_ = std::thread([this] { loop(); });
  }
  void stop() {
    running_ = false;
    if (server_) server_->close();
    if (thread_.joinable()) thread_.join();
    server_.reset();
  }
  ipc::Endpoint endpoint() const { return endpoint_; }
  std::atomic<int> pauses{0}, resumes{0}, status_calls{0};
  std::atomic<bool> reject_pause{false};
  ipc::StatusReply reply{ipc::NodeState::kOffering, "feedbeef", 123456789, true};

 private:
  void loop() {
    while (running_) {
      auto c = server_->accept(100ms);
      if (!c.is_ok()) continue;
      auto conn = std::move(c).value();
      while (running_) {
        auto env = conn->receive(100ms);
        if (!env.is_ok()) {
          if (env.status().code() == ErrorCode::kDeadlineExceeded) continue;
          break;
        }
        switch (static_cast<ipc::MessageKind>(env->kind)) {
          case ipc::MessageKind::kStatusRequest:
            ++status_calls;
            (void)conn->send(ipc::encode(reply));
            break;
          case ipc::MessageKind::kPauseRequest:
            ++pauses;
            (void)conn->send(ipc::encode(reject_pause ? ipc::Ack{ErrorCode::kPermissionDenied, "pause refused"} : ipc::Ack{}));
            break;
          case ipc::MessageKind::kResumeRequest:
            ++resumes;
            (void)conn->send(ipc::encode(ipc::Ack{}));
            break;
          default: break;
        }
      }
    }
  }
  fs::path dir_;
  ipc::Endpoint endpoint_;
  std::unique_ptr<ipc::Server> server_;
  std::thread thread_;
  std::atomic<bool> running_{false};
};

IpcNodeClient make_client(const ipc::Endpoint& ep) {
  IpcNodeClient::Options o;
  o.endpoint = ep;
  return IpcNodeClient(std::move(o));
}

}  // namespace

TEST_CASE("IpcNodeClient reads status over the helper pipe") {
  FakeService svc("status");
  auto c = make_client(svc.endpoint());
  auto st = c.status();
  REQUIRE(st.is_ok());
  CHECK(st->state == NodeUiState::kAvailable);
  CHECK(st->paired_father == "feedbeef");
  CHECK(st->storage_bytes == 123456789);
  CHECK(st->fresh);
  svc.reply.state = ipc::NodeState::kPaused;
  CHECK(c.status()->state == NodeUiState::kPaused);
}

TEST_CASE("IpcNodeClient pause and resume reach the service; a refusal is reported") {
  FakeService svc("pause");
  auto c = make_client(svc.endpoint());
  REQUIRE(c.pause().is_ok());
  REQUIRE(c.resume().is_ok());
  CHECK(svc.pauses == 1);
  CHECK(svc.resumes == 1);
  svc.reject_pause = true;
  auto s = c.pause();
  CHECK(s.code() == ErrorCode::kPermissionDenied);
  CHECK(s.message() == "pause refused");
}

TEST_CASE("IpcNodeClient treats an absent service as a state, and reconnects when it returns") {
  fs::path dir = fs::temp_directory_path() / "clm-ui-absent";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto c = make_client(ipc::Endpoint{ipc::kNodeHelperPipeName, dir});
  auto st = c.status();
  REQUIRE(st.is_ok());  // not an error: the UI shows "Not running"
  CHECK(st->state == NodeUiState::kUnreachable);
  CHECK_FALSE(st->detail.empty());
  CHECK_FALSE(c.pause().is_ok());
  fs::remove_all(dir);

  FakeService svc("reconnect");
  auto c2 = make_client(svc.endpoint());
  REQUIRE(c2.status()->state == NodeUiState::kAvailable);
  svc.stop();
  CHECK(c2.status()->state == NodeUiState::kUnreachable);
  svc.start();
  CHECK(c2.status()->state == NodeUiState::kAvailable);  // fresh connection after the restart
}

TEST_CASE("IpcNodeClient settings: defaults come back, saving is Unimplemented and says why") {
  IpcNodeClient c{IpcNodeClient::Options{}};
  auto g = c.get_settings();
  REQUIRE(g.is_ok());
  CHECK(g->allow_when_idle);
  NodeSettings s;
  auto st = c.set_settings(s);
  CHECK(st.code() == ErrorCode::kUnimplemented);
  CHECK(st.message().find("cannot save Node settings") != std::string::npos);
  s.temp_storage_limit_gb = 0;
  CHECK(c.set_settings(s).code() == ErrorCode::kInvalidArgument);  // validation still comes first
}

TEST_CASE("ScriptedFatherClient delivers events to subscribers and stops after unsubscribe") {
  ScriptedFatherClient c;
  int n = 0;
  auto id = c.subscribe([&](const father::Event&) { ++n; });
  c.emit(father::ReleasedEvent{"x"});
  c.unsubscribe(id);
  c.emit(father::ReleasedEvent{"y"});
  CHECK(n == 1);
  c.fail_next("release", make_error(ErrorCode::kInternal, "boom"));
  CHECK(c.release().code() == ErrorCode::kInternal);
  CHECK(c.release().is_ok());  // failure is one-shot
}

TEST_CASE("FatherSettings validation") {
  FatherSettings s;
  CHECK(validate(s).is_ok());
  s.context_tokens = 100;
  CHECK_FALSE(validate(s).is_ok());
  s = FatherSettings{};
  s.max_new_tokens = 0;
  CHECK_FALSE(validate(s).is_ok());
  s.max_new_tokens = s.context_tokens + 1;
  CHECK_FALSE(validate(s).is_ok());
}

TEST_CASE("display helpers") {
  CHECK(format_bytes(0) == "0 B");
  CHECK(format_bytes(999) == "999 B");
  CHECK(format_bytes(1'500'000) == "1.5 MB");
  CHECK(format_bytes(640'000'000) == "640 MB");
  CHECK(format_rate(0) == "--");
  CHECK(format_rate(8.04) == "8.0 tok/s");
  CHECK(format_millis(820) == "820 ms");
  CHECK(format_millis(1450) == "1.4 s");
  CHECK(format_eta(std::nullopt).empty());
  CHECK(format_eta(30.0) == "under a minute (estimate)");
  CHECK(format_eta(-1.0).empty());
  CHECK(ascii_display("a\xE2\x80\x94" "b") == "a - b");
  CHECK(ascii_display("a \xE2\x80\x94 b") == "a - b");
  CHECK(ascii_display("plain") == "plain");
  CHECK(describe_error(ErrorCode::kCancelled, "") == "Something went wrong (CANCELLED)");
}
