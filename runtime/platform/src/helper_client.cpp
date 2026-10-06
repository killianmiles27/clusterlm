#include "clusterlm/platform/helper_client.hpp"

namespace clusterlm::platform {

HelperClient::HelperClient(Options options, ActivityMonitor& source) : opts_(std::move(options)), source_(source) {}

bool HelperClient::connected() const {
  std::lock_guard lock(mu_);
  return conn_ && conn_->is_open();
}

void HelperClient::disconnect() {
  std::lock_guard lock(mu_);
  conn_.reset();
}

Status HelperClient::ensure_connected() {
  if (conn_ && conn_->is_open()) return Status::ok();
  conn_.reset();
  CLM_ASSIGN_OR_RETURN(conn_, ipc::connect(opts_.endpoint, opts_.client, opts_.connect_timeout));
  return Status::ok();
}

Result<ipc::Envelope> HelperClient::call(const ipc::Envelope& request) {
  CLM_RETURN_IF_ERROR(ensure_connected());
  if (auto st = conn_->send(request, opts_.io_timeout); !st.is_ok()) {
    conn_.reset();
    return st;
  }
  auto reply = conn_->receive(opts_.io_timeout);
  if (!reply.is_ok()) conn_.reset();  // timeout or loss: start over with a fresh connection next time
  return reply;
}

Status HelperClient::expect_ack(const ipc::Envelope& reply) {
  CLM_ASSIGN_OR_RETURN(auto ack, ipc::decode_ack(reply));
  if (ack.code != ErrorCode::kOk) return make_error(ack.code, "service rejected request: " + ack.message);
  return Status::ok();
}

Status HelperClient::report_once() {
  ipc::ActivityReport report;
  report.session_id = opts_.session_id;
  auto sample = source_.sample();
  if (sample.is_ok()) {
    report.idle_seconds = sample->idle_seconds;
    report.session_locked = sample->session_locked;
  }  // else: idle 0 / unlocked = in use
  std::lock_guard lock(mu_);
  CLM_ASSIGN_OR_RETURN(auto reply, call(ipc::encode(report)));
  return expect_ack(reply);
}

Result<ipc::StatusReply> HelperClient::status() {
  std::lock_guard lock(mu_);
  CLM_ASSIGN_OR_RETURN(auto reply, call(ipc::encode(ipc::StatusRequest{})));
  return ipc::decode_status_reply(reply);
}

Status HelperClient::pause(std::chrono::seconds duration) {
  std::lock_guard lock(mu_);
  CLM_ASSIGN_OR_RETURN(auto reply, call(ipc::encode(ipc::PauseRequest{static_cast<std::uint32_t>(duration.count())})));
  return expect_ack(reply);
}

Status HelperClient::resume() {
  std::lock_guard lock(mu_);
  CLM_ASSIGN_OR_RETURN(auto reply, call(ipc::encode(ipc::ResumeRequest{})));
  return expect_ack(reply);
}

}  // namespace clusterlm::platform
