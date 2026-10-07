// FatherAgent transport: envelope routing, error mapping, events, per-user access.
#include <doctest/doctest.h>

#include <filesystem>

#include "clusterlm/father/father_agent.hpp"

using namespace clusterlm;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

struct TempDir {
  fs::path path;
  explicit TempDir(const char* name) : path(fs::temp_directory_path() / (std::string("clm-agent-") + name)) {
    { std::error_code ec_rm; fs::remove_all(path, ec_rm); }
    fs::create_directories(path);
  }
  ~TempDir() { { std::error_code ec_rm; fs::remove_all(path, ec_rm); } }
};

// Echoes the payload reversed; "fail" payloads return an error.
class EchoApi final : public father::FatherApiHandler {
 public:
  Result<ipc::Envelope> handle(const ipc::Envelope& req, const ipc::PeerCredentials&) override {
    ++calls;
    if (req.payload == Bytes{'f', 'a', 'i', 'l'}) return make_error(ErrorCode::kNotFound, "secret detail that must not leak");
    ipc::Envelope r;
    r.payload.assign(req.payload.rbegin(), req.payload.rend());
    return r;
  }
  std::atomic<int> calls{0};
};

struct Rig {
  TempDir dir;
  EchoApi api;
  std::unique_ptr<father::FatherAgent> agent;
  std::unique_ptr<ipc::Connection> ui;
  explicit Rig(const char* name) : dir(name) {
    father::FatherAgentConfig cfg;
    cfg.user_tag = "tester";
    cfg.ipc_dir = dir.path;
    auto a = father::FatherAgent::create(std::move(cfg), api);
    REQUIRE(a.is_ok());
    agent = std::move(a).value();
    REQUIRE_MESSAGE(agent->start().is_ok(), "start");
    auto c = ipc::connect(agent->endpoint(), {}, 2000ms);
    REQUIRE_MESSAGE(c.is_ok(), c.status().to_string());
    ui = std::move(c).value();
  }
};

}  // namespace

TEST_CASE("a UI request is routed to the handler and answered with a versioned reply") {
  Rig r("route");
  REQUIRE(r.ui->send(ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes{1, 2, 3}), 1000ms).is_ok());
  auto reply = r.ui->receive(2000ms);
  REQUIRE(reply.is_ok());
  CHECK(reply->kind == static_cast<std::uint16_t>(ipc::MessageKind::kFatherReply));
  CHECK(reply->version == ipc::kFatherUiProtocolVersion);
  CHECK(reply->payload == Bytes{3, 2, 1});
  CHECK(r.api.calls == 1);
}

TEST_CASE("handler errors become an Ack with the code only; version skew and wrong kinds never reach the handler") {
  Rig r("errors");
  REQUIRE(r.ui->send(ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes{'f', 'a', 'i', 'l'}), 1000ms).is_ok());
  auto reply = r.ui->receive(2000ms);
  REQUIRE(reply.is_ok());
  auto ack = ipc::decode_ack(reply.value());
  REQUIRE(ack.is_ok());
  CHECK(ack->code == ErrorCode::kNotFound);
  CHECK(ack->message.find("secret") == std::string::npos);

  auto skew = ipc::father_envelope(ipc::MessageKind::kFatherRequest, {});
  skew.version = 42;
  REQUIRE(r.ui->send(skew, 1000ms).is_ok());
  ack = ipc::decode_ack(r.ui->receive(2000ms).value());
  REQUIRE(ack.is_ok());
  CHECK(ack->code == ErrorCode::kVersionMismatch);

  REQUIRE(r.ui->send(ipc::encode(ipc::StatusRequest{}), 1000ms).is_ok());  // a helper message on the Father pipe
  ack = ipc::decode_ack(r.ui->receive(2000ms).value());
  REQUIRE(ack.is_ok());
  CHECK(ack->code == ErrorCode::kProtocolError);
  CHECK(r.api.calls == 1);  // only the first request reached the handler
}

TEST_CASE("the default placeholder API answers kUnimplemented") {
  TempDir dir("placeholder");
  father::UnimplementedFatherApi api;
  father::FatherAgentConfig cfg;
  cfg.user_tag = "tester";
  cfg.ipc_dir = dir.path;
  auto agent = father::FatherAgent::create(std::move(cfg), api);
  REQUIRE(agent.is_ok());
  REQUIRE(agent.value()->start().is_ok());
  auto ui = ipc::connect(agent.value()->endpoint(), {}, 2000ms);
  REQUIRE(ui.is_ok());
  REQUIRE(ui.value()->send(ipc::father_envelope(ipc::MessageKind::kFatherRequest, {}), 1000ms).is_ok());
  auto ack = ipc::decode_ack(ui.value()->receive(2000ms).value());
  REQUIRE(ack.is_ok());
  CHECK(ack->code == ErrorCode::kUnimplemented);
}

TEST_CASE("broadcast delivers events to connected UIs") {
  Rig r("events");
  // Round trip first so the agent has certainly registered the connection.
  REQUIRE(r.ui->send(ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes{1}), 1000ms).is_ok());
  REQUIRE(r.ui->receive(2000ms).is_ok());
  r.agent->broadcast(Bytes{7, 7});
  auto ev = r.ui->receive(2000ms);
  REQUIRE(ev.is_ok());
  CHECK(ev->kind == static_cast<std::uint16_t>(ipc::MessageKind::kFatherEvent));
  CHECK(ev->payload == Bytes{7, 7});
}

TEST_CASE("the agent keeps its Coordinator configuration and stops cleanly") {
  Rig r("config");
  CHECK(r.agent->coordinator_config().nodes.empty());
  r.agent->stop();
  auto c = ipc::connect(r.agent->endpoint(), {}, 200ms);
  CHECK_FALSE(c.is_ok());
}
