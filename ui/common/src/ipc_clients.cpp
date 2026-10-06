// IpcNodeClient (helper pipe messages). IpcFatherClient lives in ipc_father_client.cpp.
#include "clusterlm/ui/clients.hpp"

#include <thread>

namespace clusterlm::ui {

// ---- Node ----------------------------------------------------------------------------------------------------

IpcNodeClient::~IpcNodeClient() {
  std::lock_guard lk(mu_);
  if (conn_) conn_->close();
}

Result<ipc::Envelope> IpcNodeClient::call(const ipc::Envelope& request) {
  std::lock_guard lk(mu_);
  Status last = make_error(ErrorCode::kUnavailable, "not connected");
  // One reconnect: a stale connection (service restarted) is the common failure.
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (!conn_ || !conn_->is_open()) {
      auto c = ipc::connect(opts_.endpoint, opts_.client, opts_.connect_timeout);
      if (!c.is_ok()) return c.status();
      conn_ = std::move(c).value();
    }
    auto sent = conn_->send(request, opts_.io_timeout);
    if (sent.is_ok()) {
      auto r = conn_->receive(opts_.io_timeout);
      if (r.is_ok()) return r;
      last = r.status();
    } else {
      last = sent;
    }
    conn_->close();
    conn_.reset();
  }
  return last;
}

Status IpcNodeClient::expect_ack(const ipc::Envelope& reply) {
  auto a = ipc::decode_ack(reply);
  if (!a.is_ok()) return a.status();
  if (a->code != ErrorCode::kOk) return make_error(a->code, a->message);
  return Status::ok();
}

Result<NodeStatus> IpcNodeClient::status() {
  NodeStatus out;
  auto reply = call(ipc::encode(ipc::StatusRequest{}));
  if (!reply.is_ok()) {
    out.state = NodeUiState::kUnreachable;
    out.detail = reply.status().message();
    return out;
  }
  auto sr = ipc::decode_status_reply(reply.value());
  if (!sr.is_ok()) {
    out.state = NodeUiState::kUnreachable;
    out.detail = "The Node service sent an answer this window does not understand.";
    return out;
  }
  out.state = from_ipc(sr->state, sr->lease_state);
  out.lease = sr->lease_state;
  out.lease_parts_done = sr->lease_objects_sealed;
  out.lease_parts_total = sr->lease_objects_total;
  out.paired_father = sr->paired_father;
  out.storage_bytes = sr->storage_bytes;
  out.fresh = sr->helper_reports_fresh;
  return out;
}

Status IpcNodeClient::pause() {
  auto r = call(ipc::encode(ipc::PauseRequest{0}));
  if (!r.is_ok()) return r.status();
  return expect_ack(r.value());
}

Status IpcNodeClient::resume() {
  auto r = call(ipc::encode(ipc::ResumeRequest{}));
  if (!r.is_ok()) return r.status();
  return expect_ack(r.value());
}

Result<ipc::NodeSettingsView> IpcNodeClient::read_view() {
  auto reply = call(ipc::encode(ipc::SettingsRequest{}));
  if (!reply.is_ok()) return reply.status();
  if (auto sr = ipc::decode_settings_reply(reply.value()); sr.is_ok()) return sr->settings;
  // Not a SettingsReply: the service answered with an Ack carrying the reason.
  if (Status st = expect_ack(reply.value()); !st.is_ok()) return st;
  return make_error(ErrorCode::kProtocolError, "the Node service sent an answer this window does not understand");
}

Result<NodeSettings> IpcNodeClient::get_settings() {
  CLM_ASSIGN_OR_RETURN(auto view, read_view());
  return from_view(view, std::thread::hardware_concurrency());
}

Status IpcNodeClient::set_settings(const NodeSettings& s) {
  if (auto st = validate(s); !st.is_ok()) return st;
  CLM_ASSIGN_OR_RETURN(auto base, read_view());
  auto r = call(ipc::encode(ipc::SettingsUpdate{to_view(s, base, std::thread::hardware_concurrency())}));
  if (!r.is_ok()) return r.status();
  return expect_ack(r.value());
}

Result<NodePairingInfo> IpcNodeClient::enter_pairing_mode() {
  auto reply = call(ipc::encode(ipc::PairingModeRequest{}));
  if (!reply.is_ok()) return reply.status();
  if (auto pr = ipc::decode_pairing_mode_reply(reply.value()); pr.is_ok()) {
    NodePairingInfo info;
    info.code = pr->code;
    info.endpoint = pr->endpoint;
    info.fingerprint = pr->fingerprint;
    info.window_seconds = pr->window_seconds;
    return info;
  }
  if (Status st = expect_ack(reply.value()); !st.is_ok()) return st;
  return make_error(ErrorCode::kProtocolError, "the Node service sent an answer this window does not understand");
}

}  // namespace clusterlm::ui
