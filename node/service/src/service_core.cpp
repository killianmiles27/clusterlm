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

// Settings and pairing are for the person at the machine: a peer the OS says is in the services session (0) is not.
// POSIX peers carry no session id; the socket's uid check already restricted them to the service's own user.
bool interactive_peer(const ipc::PeerCredentials& peer) { return !(peer.session_known && peer.session_id == 0); }
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
  std::optional<std::string> notice;
  auto r = [&] {
    std::lock_guard lock(mu_);
    auto tick = supervisor_.tick();
    if (tick.is_ok()) emit(tick.value());
    notice = supervisor_.take_unpair_notice();
    return tick;
  }();
  if (notice) on_unpair_notice(*notice);  // outside the lock: it restarts the worker through set_trusted_fathers
  return r;
}

namespace {
std::string lower(std::string v) {
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v;
}

ipc::NodeSettingsView to_view(const config::NodeSettings& s) {
  ipc::NodeSettingsView v;
  v.allow_when_idle = s.allow_when_idle;
  v.idle_seconds = s.idle_seconds;
  v.ac_only = s.ac_only;
  v.temp_storage_limit_gib = s.temp_storage_limit_gib;
  v.start_with_system = s.start_with_system;
  v.ram_gib = s.caps.ram_gib;
  v.vram_gib = s.caps.vram_gib;
  v.threads = s.caps.threads;
  return v;
}
}  // namespace

// The worker accepted an UnpairNotice from a pinned Father. Honour it only if that Father is the one this service
// has recorded as paired; the worker already stopped trusting it in memory, so on a mismatch the trust list is
// restored by restarting it unchanged.
void ServiceCore::on_unpair_notice(const std::string& peer) {
  std::optional<config::PairedDevice> paired;
  if (cfg_.settings) paired = cfg_.settings->get().paired_father;
  if (cfg_.settings && (!paired || (!peer.empty() && lower(peer) != lower(paired->fingerprint)))) {
    log::warn("unpair_notice_ignored", {{"reason", paired ? "not the paired father" : "nothing paired"}});
    std::vector<std::string> trusted;
    std::string label;
    {
      std::lock_guard lock(mu_);
      trusted = supervisor_.trusted_peers();
      label = cfg_.paired_father;
    }
    if (auto st = set_trusted_fathers(std::move(trusted), std::move(label)); !st.is_ok())
      log::warn("trust_restore_failed", {{"status", st.to_string()}});
    return;
  }
  if (auto st = unpair_father(); !st.is_ok()) log::warn("unpair_notice_apply_failed", {{"status", st.to_string()}});
  else log::info("unpaired_by_father");
}

Status ServiceCore::unpair_father() {
  if (cfg_.settings) {
    CLM_RETURN_IF_ERROR(cfg_.settings->update([](config::NodeSettings& s) {
      s.paired_father.reset();
      return Status::ok();
    }));
  }
  return set_trusted_fathers({}, "");
}

void ServiceCore::set_pairing_starter(PairingModeStarter starter) {
  std::lock_guard lock(pairing_mu_);
  pairing_starter_ = std::move(starter);
}

Result<ipc::NodeSettingsView> ServiceCore::settings_view() const {
  if (!cfg_.settings) return make_error(ErrorCode::kFailedPrecondition, "this Node has no settings store");
  return to_view(cfg_.settings->get());
}

Status ServiceCore::apply_settings(const ipc::NodeSettingsView& v) {
  if (!cfg_.settings) return make_error(ErrorCode::kFailedPrecondition, "this Node has no settings store");
  std::lock_guard lock(mu_);
  const auto now = std::chrono::steady_clock::now();
  if (last_settings_apply_ != std::chrono::steady_clock::time_point{} &&
      now - last_settings_apply_ < cfg_.settings_min_interval)
    return make_error(ErrorCode::kResourceExhausted, "settings were just changed; try again in a moment");
  // Only the user-safe fields are copied; the name, the paired Father and every other field keep their values.
  CLM_RETURN_IF_ERROR(cfg_.settings->update([&](config::NodeSettings& s) {
    s.allow_when_idle = v.allow_when_idle;
    s.idle_seconds = v.idle_seconds;
    s.ac_only = v.ac_only;
    s.temp_storage_limit_gib = v.temp_storage_limit_gib;
    s.start_with_system = v.start_with_system;
    s.caps.ram_gib = v.ram_gib;
    s.caps.vram_gib = v.vram_gib;
    s.caps.threads = v.threads;
    return Status::ok();
  }));
  last_settings_apply_ = now;
  // Applied to the running Node. Policy is immediate; changed caps restart the worker after a cooperative release.
  IdlePolicy policy = supervisor_.policy();
  policy.idle_seconds_required = v.idle_seconds;
  policy.require_ac_power = v.ac_only;
  WorkerCaps caps;
  caps.ram_gib = v.ram_gib;
  caps.vram_gib = v.vram_gib;
  caps.disk_gib = v.temp_storage_limit_gib;
  caps.threads = v.threads;
  // The policy flag is separate from the user's tray pause: allowing participation again never ends a pause.
  activity_.set_participation_allowed(v.allow_when_idle);
  auto r = supervisor_.reconfigure(with_worker_caps(supervisor_.worker_args(), caps), policy);
  if (!r.is_ok()) {
    log::warn("settings_apply_failed", {{"status", r.status().to_string()}});
    return make_error(ErrorCode::kInternal, "saved, but the Node could not apply the new limits yet");
  }
  emit(r.value());
  std::lock_guard w(wake_mu_);
  wake_.notify_all();
  return Status::ok();
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
  if (activity_.paused() || !activity_.participation_allowed()) return ipc::NodeState::kPaused;
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
  {
    std::lock_guard lock(mu_);
    const auto lv = supervisor_.worker_lease();
    if (lv.known) {
      s.lease_objects_sealed = lv.sealed_objects;
      s.lease_objects_total = lv.planned_objects;
      if (lv.state == "Preparing") s.lease_state = ipc::LeaseState::kPreparing;
      else if (lv.state == "Ready") s.lease_state = ipc::LeaseState::kReady;
      else if (lv.state == "Inferencing") s.lease_state = ipc::LeaseState::kInferencing;
      else if (lv.state == "Releasing") s.lease_state = ipc::LeaseState::kReleasing;
      else if (lv.state == "CleanupPending") s.lease_state = ipc::LeaseState::kCleanupPending;
    }
  }
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
std::vector<std::string> ServiceCore::worker_args() const {
  std::lock_guard lock(mu_);
  return supervisor_.worker_args();
}
IdlePolicy ServiceCore::policy() const {
  std::lock_guard lock(mu_);
  return supervisor_.policy();
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
    case MessageKind::kSettingsRequest: {
      if (auto st = ipc::decode_settings_request(request); !st.is_ok()) return ack(st.code(), "bad SettingsRequest");
      auto v = settings_view();
      if (!v.is_ok()) return ack(v.status().code(), "settings are not available");
      return ipc::encode(ipc::SettingsReply{v.value()});
    }
    case MessageKind::kSettingsUpdate: {
      auto r = ipc::decode_settings_update(request);
      if (!r.is_ok()) return ack(r.status().code(), "bad SettingsUpdate");
      if (!interactive_peer(peer)) return ack(ErrorCode::kPermissionDenied, "settings can only be changed by a signed-in user");
      auto st = apply_settings(r->settings);
      return ack(st.code(), st.is_ok() ? std::string{} : std::string(st.message()));
    }
    case MessageKind::kPairingModeRequest: {
      if (auto st = ipc::decode_pairing_mode_request(request); !st.is_ok()) return ack(st.code(), "bad PairingModeRequest");
      if (!interactive_peer(peer)) return ack(ErrorCode::kPermissionDenied, "pairing can only be started by a signed-in user");
      PairingModeStarter starter;
      {
        std::lock_guard lock(pairing_mu_);
        if (!pairing_starter_) return ack(ErrorCode::kUnimplemented, "pairing is not available");
        const auto now = std::chrono::steady_clock::now();
        if (pairing_requested_ && now - last_pairing_request_ < cfg_.pairing_min_interval)
          return ack(ErrorCode::kResourceExhausted, "pairing mode was just started; wait a few seconds");
        pairing_requested_ = true;
        last_pairing_request_ = now;
        starter = pairing_starter_;
      }
      auto offer = starter();  // not under any lock: it asks this core for the worker endpoint
      if (!offer.is_ok()) return ack(offer.status().code(), "could not start pairing mode");
      return ipc::encode(offer.value());
    }
    default: return ack(ErrorCode::kProtocolError, "unsupported message kind");
  }
}

}  // namespace clusterlm::node
