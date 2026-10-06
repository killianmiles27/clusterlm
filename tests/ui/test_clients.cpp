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

// A minimal Node service: answers Status / Pause / Resume / Settings / PairingMode requests on the helper endpoint.
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
  ipc::NodeSettingsView view;
  std::atomic<int> settings_reads{0}, settings_updates{0}, pairing_requests{0};
  ipc::NodeSettingsView last_update;
  Status update_result;  // ok unless a test sets it
  Status pairing_result;
  bool no_settings = false;  // answer like a service without a settings store

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
          case ipc::MessageKind::kSettingsRequest:
            ++settings_reads;
            if (no_settings) (void)conn->send(ipc::encode(ipc::Ack{ErrorCode::kFailedPrecondition, "settings are not available"}));
            else (void)conn->send(ipc::encode(ipc::SettingsReply{view}));
            break;
          case ipc::MessageKind::kSettingsUpdate: {
            ++settings_updates;
            auto u = ipc::decode_settings_update(env.value());
            if (u.is_ok() && update_result.is_ok()) {
              last_update = u->settings;
              view = u->settings;
            }
            (void)conn->send(ipc::encode(ipc::Ack{update_result.code(), update_result.message()}));
            break;
          }
          case ipc::MessageKind::kPairingModeRequest:
            ++pairing_requests;
            if (pairing_result.is_ok())
              (void)conn->send(ipc::encode(ipc::PairingModeReply{"ABCD-EFGH", "192.168.1.20:47601", "ab12-cd34-ef56", 300}));
            else
              (void)conn->send(ipc::encode(ipc::Ack{pairing_result.code(), pairing_result.message()}));
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

TEST_CASE("IpcNodeClient status carries the worker's lease state and counts") {
  FakeService svc("lease");
  auto c = make_client(svc.endpoint());
  CHECK(c.status()->state == NodeUiState::kAvailable);
  CHECK(c.status()->lease == ipc::LeaseState::kNone);
  svc.reply.lease_state = ipc::LeaseState::kPreparing;
  svc.reply.lease_objects_sealed = 3;
  svc.reply.lease_objects_total = 12;
  auto st = c.status();
  REQUIRE(st.is_ok());
  CHECK(st->state == NodeUiState::kPreparing);
  CHECK(st->lease_parts_done == 3);
  CHECK(st->lease_parts_total == 12);
  svc.reply.lease_state = ipc::LeaseState::kInferencing;
  CHECK(c.status()->state == NodeUiState::kInUse);
  svc.reply.state = ipc::NodeState::kBusy;  // the user came back: out of the way whatever the worker holds
  CHECK(c.status()->state == NodeUiState::kBusy);
}

TEST_CASE("IpcNodeClient settings are read and saved through the service") {
  FakeService svc("settings");
  svc.view.idle_seconds = 900;
  svc.view.ram_gib = 8;
  svc.view.vram_gib = 6;
  svc.view.temp_storage_limit_gib = 100;
  svc.view.threads = 0;
  auto c = make_client(svc.endpoint());
  auto g = c.get_settings();
  REQUIRE(g.is_ok());
  CHECK(g->ram_gb == 8);
  CHECK(g->gpu_memory_gb == 6);
  CHECK(g->temp_storage_limit_gb == 100);
  CHECK(g->cpu_cap_percent == 100);
  NodeSettings s = g.value();
  s.ram_gb = 12;
  s.allow_when_idle = false;
  s.cpu_cap_percent = 50;
  REQUIRE(c.set_settings(s).is_ok());
  CHECK(svc.settings_updates == 1);
  CHECK(svc.last_update.ram_gib == 12);
  CHECK_FALSE(svc.last_update.allow_when_idle);
  CHECK(svc.last_update.idle_seconds == 900);  // a field this window does not edit survives the save
  CHECK(svc.last_update.threads >= 1);
  // The service's refusal is the user's answer, not a silent success.
  svc.update_result = make_error(ErrorCode::kResourceExhausted, "settings were just changed; try again in a moment");
  auto st = c.set_settings(s);
  CHECK(st.code() == ErrorCode::kResourceExhausted);
  CHECK(st.message().find("try again") != std::string::npos);
  // Validation still comes first: nothing invalid reaches the service.
  const int before = svc.settings_updates;
  s.ram_gb = 0;
  CHECK(c.set_settings(s).code() == ErrorCode::kInvalidArgument);
  CHECK(svc.settings_updates == before);
}

TEST_CASE("IpcNodeClient settings against a service without a settings store, or no service") {
  FakeService svc("nosettings");
  svc.no_settings = true;
  auto c = make_client(svc.endpoint());
  auto g = c.get_settings();
  CHECK(g.status().code() == ErrorCode::kFailedPrecondition);
  CHECK(c.set_settings(NodeSettings{}).code() == ErrorCode::kFailedPrecondition);
  CHECK(svc.settings_updates == 0);
  fs::path dir = fs::temp_directory_path() / "clm-ui-nosvc";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto absent = make_client(ipc::Endpoint{ipc::kNodeHelperPipeName, dir});
  CHECK_FALSE(absent.get_settings().is_ok());
  CHECK_FALSE(absent.enter_pairing_mode().is_ok());
  fs::remove_all(dir);
}

TEST_CASE("IpcNodeClient pairing mode returns the code line; refusals come back as errors") {
  FakeService svc("pairing");
  auto c = make_client(svc.endpoint());
  auto p = c.enter_pairing_mode();
  REQUIRE(p.is_ok());
  CHECK(p->code == "ABCD-EFGH");
  CHECK(p->endpoint == "192.168.1.20:47601");
  CHECK(p->fingerprint == "ab12-cd34-ef56");
  CHECK(p->window_seconds == 300);
  svc.pairing_result = make_error(ErrorCode::kResourceExhausted, "pairing mode was just started; wait a few seconds");
  auto again = c.enter_pairing_mode();
  CHECK(again.status().code() == ErrorCode::kResourceExhausted);
  CHECK(svc.pairing_requests == 2);
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
