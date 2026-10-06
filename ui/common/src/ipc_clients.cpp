// IpcFatherClient (documented seam, not functional) and IpcNodeClient (helper pipe messages).
#include "clusterlm/ui/clients.hpp"

namespace clusterlm::ui {

// ---- Father seam ---------------------------------------------------------------------------------------------

std::string_view IpcFatherClient::seam_message() {
  return "The connection to the Father agent is not available in this build. The agent protocol is not wired yet.";
}

namespace {
Status unavailable() { return make_error(ErrorCode::kUnavailable, std::string(IpcFatherClient::seam_message())); }
Status unimplemented() { return make_error(ErrorCode::kUnimplemented, std::string(IpcFatherClient::seam_message())); }
}  // namespace

Result<std::vector<catalog::TierReadiness>> IpcFatherClient::list_tiers(std::uint32_t) { return unavailable(); }
Result<std::vector<TierParticipant>> IpcFatherClient::participants(std::string_view) { return unavailable(); }
Status IpcFatherClient::select_tier(std::string_view) { return unavailable(); }
Status IpcFatherClient::prepare_tier(std::string_view, std::uint32_t) { return unavailable(); }
Result<father::RequestId> IpcFatherClient::send_chat(father::ChatRequest) { return unavailable(); }
Status IpcFatherClient::cancel(father::RequestId) { return unavailable(); }
Status IpcFatherClient::release() { return unavailable(); }
Status IpcFatherClient::reset_conversation() { return unavailable(); }
Result<ContextUse> IpcFatherClient::context_use() { return unavailable(); }
Result<DiagnosticsExport> IpcFatherClient::export_diagnostics(bool) { return unavailable(); }
Result<FatherSettings> IpcFatherClient::get_settings() { return FatherSettings{}; }
Status IpcFatherClient::set_settings(const FatherSettings& s) { return validate(s); }
Result<std::vector<PairedMachine>> IpcFatherClient::paired_machines() { return unimplemented(); }
Status IpcFatherClient::start_pairing() { return unimplemented(); }
Status IpcFatherClient::unpair(std::string_view) { return unimplemented(); }

// ---- Node ----------------------------------------------------------------------------------------------------

IpcNodeClient::~IpcNodeClient() {
  std::lock_guard lk(mu_);
  if (conn_) conn_->close();
}

std::string_view IpcNodeClient::settings_unavailable_message() {
  return "This build cannot save Node settings yet (the Node configuration store is not available). Changes apply "
         "to this window only.";
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
  out.state = from_ipc(sr->state);
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

Result<NodeSettings> IpcNodeClient::get_settings() { return NodeSettings{}; }

Status IpcNodeClient::set_settings(const NodeSettings& s) {
  if (auto st = validate(s); !st.is_ok()) return st;
  return make_error(ErrorCode::kUnimplemented, std::string(settings_unavailable_message()));
}

}  // namespace clusterlm::ui
