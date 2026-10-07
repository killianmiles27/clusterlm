#include "clusterlm/father/father_agent.hpp"

namespace clusterlm::father {

using namespace std::chrono_literals;

std::filesystem::path default_catalog_path(const std::filesystem::path& exe_dir) {
  namespace fs = std::filesystem;
  fs::path installed = (exe_dir / ".." / "catalog" / "clusterlm-catalog.json").lexically_normal();
  fs::path dev = exe_dir / "clusterlm-catalog.json";
  std::error_code ec;
  if (fs::is_regular_file(installed, ec)) return installed;
  return dev;
}

Result<std::unique_ptr<FatherAgent>> FatherAgent::create(FatherAgentConfig config, FatherApiHandler& handler) {
  ipc::Endpoint ep;
  ep.name = ipc::father_ui_pipe_name(config.user_tag);
  ep.socket_dir = config.ipc_dir;
  return std::unique_ptr<FatherAgent>(new FatherAgent(std::move(config), handler, std::move(ep)));
}

FatherAgent::~FatherAgent() { stop(); }

Status FatherAgent::start() {
  if (loop_) return make_error(ErrorCode::kFailedPrecondition, "agent already started");
  CLM_ASSIGN_OR_RETURN(const std::string self, ipc::current_user_id());
  ipc::ServerOptions so;
  so.endpoint = endpoint_;
  so.access = ipc::PipeAccess::kOwnerAndSystem;
  so.max_frame_bytes = cfg_.max_frame_bytes;
  // Defence in depth on top of the pipe ACL / socket permissions: only the agent's own user may connect.
  ipc::AuthPolicy policy;
  policy.allowed_user_ids = {self};
  so.authorizer = [policy](const ipc::PeerCredentials& p) { return policy.authorize(p); };
  CLM_ASSIGN_OR_RETURN(auto server, ipc::Server::create(std::move(so)));
  loop_ = std::make_unique<ipc::ServerLoop>(
      std::move(server), [this](const std::shared_ptr<ipc::Connection>& c, const std::atomic<bool>& stop) { serve(c, stop); });
  loop_->start();
  return Status::ok();
}

void FatherAgent::stop() {
  if (loop_) loop_->stop();
  loop_.reset();
}

void FatherAgent::broadcast(Bytes payload) {
  if (!loop_) return;
  const auto env = ipc::father_envelope(ipc::MessageKind::kFatherEvent, std::move(payload));
  for (auto& c : loop_->connections()) (void)c->send(env, 250ms);  // a failed send closes that connection
}

void FatherAgent::serve(const std::shared_ptr<ipc::Connection>& conn, const std::atomic<bool>& stop) {
  while (!stop.load() && conn->is_open()) {
    auto req = conn->receive(200ms);
    if (!req.is_ok()) {
      if (req.status().code() == ErrorCode::kDeadlineExceeded) continue;
      break;
    }
    ipc::Envelope reply;
    if (req->kind != static_cast<std::uint16_t>(ipc::MessageKind::kFatherRequest)) {
      reply = ipc::encode(ipc::Ack{ErrorCode::kProtocolError, "unsupported message kind"});
    } else if (req->version != ipc::kFatherUiProtocolVersion) {
      reply = ipc::encode(ipc::Ack{ErrorCode::kVersionMismatch, "unsupported Father UI protocol version"});
    } else {
      auto r = handler_.handle(req.value(), conn->peer());
      if (r.is_ok()) {
        reply = std::move(r).value();
        reply.kind = static_cast<std::uint16_t>(ipc::MessageKind::kFatherReply);
        reply.version = ipc::kFatherUiProtocolVersion;
      } else {
        reply = ipc::encode(ipc::Ack{r.status().code(), "request failed"});  // detail stays in the agent's logs
      }
    }
    if (!conn->send(reply, 2000ms).is_ok()) break;
  }
}

}  // namespace clusterlm::father
