#include "clusterlm/node/service_core.hpp"

#include <algorithm>
#include <optional>

#include "clusterlm/common/log.hpp"
#include "clusterlm/platform/fs_safety.hpp"

namespace clusterlm::node {

using namespace std::chrono_literals;

namespace {
constexpr std::size_t kMaxHelperConnections = 8;

ipc::Envelope ack(ErrorCode code, std::string message = {}) { return ipc::encode(ipc::Ack{code, std::move(message)}); }
}  // namespace

ServiceCore::ServiceCore(ServiceCoreConfig config, platform::PowerMonitor& power, std::unique_ptr<platform::ProcessJob> job)
    : cfg_(std::move(config)),
      activity_(cfg_.helper_report_stale_after),
      supervisor_(cfg_.supervisor, activity_, power, std::move(job)) {}

ServiceCore::~ServiceCore() { stop(); }

void ServiceCore::set_event_sink(EventSink sink) {
  std::lock_guard lock(sink_mu_);
  sink_ = std::move(sink);
}

void ServiceCore::emit(const std::vector<SupervisorEvent>& events) {
  std::lock_guard lock(sink_mu_);
  if (!sink_) return;
  for (const auto& e : events) sink_(e);
}

Status ServiceCore::start() {
  ipc::ServerOptions so;
  so.endpoint = cfg_.helper_endpoint;
  so.access = cfg_.helper_pipe_access;
  const ipc::AuthPolicy policy = cfg_.helper_auth;
  so.authorizer = [policy](const ipc::PeerCredentials& p) { return policy.authorize(p); };
  CLM_ASSIGN_OR_RETURN(auto server, ipc::Server::create(std::move(so)));
  ipc_loop_ = std::make_unique<ipc::ServerLoop>(
      std::move(server), [this](const std::shared_ptr<ipc::Connection>& c, const std::atomic<bool>& stop) { serve(c, stop); },
      kMaxHelperConnections);
  {
    std::lock_guard lock(mu_);
    CLM_RETURN_IF_ERROR(supervisor_.start());
    refresh_sessions_locked();
  }
  started_.store(true);
  ipc_loop_->start();
  return Status::ok();
}

void ServiceCore::refresh_sessions_locked() {
  if (!cfg_.helper_host) return;
  auto sessions = cfg_.helper_host->sessions();
  if (sessions.is_ok()) {
    const bool any_user = std::any_of(sessions->begin(), sessions->end(),
                                      [](const platform::UserSession& s) { return s.id != 0 && s.active; });
    activity_.set_no_interactive_sessions(!any_user);
  }
  if (auto st = platform::reconcile_helpers(*cfg_.helper_host, cfg_.helper_startup, helper_state_); !st.is_ok())
    log::warn("helper_startup", {{"status", st.to_string()}});  // non-fatal: no helper means fail-closed Busy
}

Result<std::vector<SupervisorEvent>> ServiceCore::run_once() {
  std::lock_guard lock(mu_);
  auto r = supervisor_.tick();
  if (r.is_ok()) emit(r.value());
  return r;
}

void ServiceCore::run() {
  while (!stopping_.load()) {
    auto r = run_once();
    if (!r.is_ok()) log::warn("tick_failed", {{"status", r.status().to_string()}});
    std::unique_lock lock(wake_mu_);
    wake_.wait_for(lock, cfg_.tick_interval, [this] { return stopping_.load(); });
  }
  stop();
}

void ServiceCore::request_stop() {
  stopping_.store(true);
  std::lock_guard lock(wake_mu_);
  wake_.notify_all();
}

void ServiceCore::stop() {
  stopping_.store(true);
  {
    std::lock_guard lock(wake_mu_);
    wake_.notify_all();
  }
  if (ipc_loop_) ipc_loop_->stop();
  if (started_.exchange(false)) {
    std::lock_guard lock(mu_);
    supervisor_.stop();
  }
}

Status ServiceCore::set_trusted_fathers(std::vector<std::string> fingerprints, std::string paired_father_short) {
  std::lock_guard lock(mu_);
  cfg_.paired_father = std::move(paired_father_short);
  if (!started_.load()) {
    cfg_.supervisor.trusted_peers = fingerprints;
    return supervisor_.set_trusted_peers(std::move(fingerprints)).status();
  }
  auto r = supervisor_.set_trusted_peers(std::move(fingerprints));
  if (!r.is_ok()) return r.status();
  emit(r.value());
  return Status::ok();
}

void ServiceCore::handle_power_event(const platform::PowerEvent& event) {
  using K = platform::PowerEventKind;
  std::lock_guard lock(mu_);
  if (!started_.load()) return;
  switch (event.kind) {
    case K::kSuspend: {
      log::info("power_suspend");
      auto r = supervisor_.on_suspend();
      if (r.is_ok()) {
        emit(r.value());
      } else {
        log::warn("suspend_revoke_failed", {{"status", r.status().to_string()}});
      }
      break;
    }
    case K::kResume: {
      log::info("power_resume");
      activity_.invalidate();  // pre-sleep reports describe a world that no longer exists
      auto r = supervisor_.on_resume();
      if (!r.is_ok()) log::warn("resume_failed", {{"status", r.status().to_string()}});
      refresh_sessions_locked();
      break;
    }
    case K::kPowerSourceChanged:
    case K::kPowerSavingChanged: break;  // PowerMonitor is re-sampled on every tick
  }
  std::lock_guard w(wake_mu_);
  wake_.notify_all();
}

void ServiceCore::handle_session_event(const platform::SessionEvent& event) {
  using K = platform::SessionEventKind;
  std::lock_guard lock(mu_);
  if (!started_.load()) return;
  if (event.kind == K::kLogoff || event.kind == K::kDisconnect) activity_.session_ended(event.session_id);
  if (event.kind == K::kLogon || event.kind == K::kConnect || event.kind == K::kLogoff || event.kind == K::kDisconnect)
    refresh_sessions_locked();
}

ipc::NodeState ServiceCore::state() const {
  if (stopping_.load()) return ipc::NodeState::kStopping;
  if (!started_.load()) return ipc::NodeState::kStarting;
  std::lock_guard lock(mu_);
  if (supervisor_.suspended()) return ipc::NodeState::kSuspended;
  if (activity_.paused()) return ipc::NodeState::kPaused;
  return supervisor_.eligible() ? ipc::NodeState::kOffering : ipc::NodeState::kBusy;
}

ipc::StatusReply ServiceCore::status() const {
  ipc::StatusReply s;
  s.state = state();
  {
    std::lock_guard lock(mu_);
    s.paired_father = cfg_.paired_father;
  }
  s.helper_reports_fresh = activity_.reports_fresh();
  if (!cfg_.staging_root.empty()) {
    auto census = platform::census_under(cfg_.staging_root);
    if (census.is_ok()) {
      // Same accounting as the worker: the content-free lease journal is not staged model data.
      std::error_code ec;
      const auto journal = std::filesystem::file_size(cfg_.staging_root / "journal.log", ec);
      s.storage_bytes = census->bytes - (ec ? 0 : std::min<std::uint64_t>(journal, census->bytes));
    }
  }
  return s;
}

std::string ServiceCore::worker_endpoint() const {
  std::lock_guard lock(mu_);
  return supervisor_.worker_endpoint();
}
std::string ServiceCore::worker_device_id() const {
  std::lock_guard lock(mu_);
  return supervisor_.worker_device_id();
}

// ---- Helper IPC ----------------------------------------------------------------------------------------
void ServiceCore::serve(const std::shared_ptr<ipc::Connection>& conn, const std::atomic<bool>& stop) {
  while (!stop.load() && conn->is_open()) {
    auto msg = conn->receive(200ms);
    if (!msg.is_ok()) {
      if (msg.status().code() == ErrorCode::kDeadlineExceeded) continue;
      break;  // closed, or a protocol violation (the connection has already been closed)
    }
    if (!conn->send(handle_message(msg.value(), conn->peer()), 2000ms).is_ok()) break;
  }
}

ipc::Envelope ServiceCore::handle_message(const ipc::Envelope& request, const ipc::PeerCredentials& peer) {
  using ipc::MessageKind;
  switch (static_cast<MessageKind>(request.kind)) {
    case MessageKind::kActivityReport: {
      auto r = ipc::decode_activity_report(request);
      if (!r.is_ok()) return ack(r.status().code(), "bad ActivityReport");
      if (cfg_.enforce_report_session_match && peer.session_known && r->session_id != peer.session_id)
        return ack(ErrorCode::kPermissionDenied, "report names a different session than the sender's");
      activity_.submit(r.value());
      return ack(ErrorCode::kOk);
    }
    case MessageKind::kPauseRequest: {
      auto r = ipc::decode_pause_request(request);
      if (!r.is_ok()) return ack(r.status().code(), "bad PauseRequest");
      activity_.pause(r->duration_seconds == 0 ? std::nullopt : std::optional(std::chrono::seconds(r->duration_seconds)));
      return ack(ErrorCode::kOk);
    }
    case MessageKind::kResumeRequest: {
      if (auto st = ipc::decode_resume_request(request); !st.is_ok()) return ack(st.code(), "bad ResumeRequest");
      activity_.resume();
      return ack(ErrorCode::kOk);
    }
    case MessageKind::kStatusRequest: {
      if (auto st = ipc::decode_status_request(request); !st.is_ok()) return ack(st.code(), "bad StatusRequest");
      return ipc::encode(status());
    }
    default: return ack(ErrorCode::kProtocolError, "unsupported message kind");
  }
}

}  // namespace clusterlm::node
