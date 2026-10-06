#include "clusterlm/ui/node_viewmodel.hpp"

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

NodeViewModel::NodeViewModel(NodeClient& client) : client_(client) {
  load_settings();
  rebuild();
}

// The service is the authority on the settings: they are read when it first answers, and again after every save and
// reconnect. While it cannot answer, the note says why and the controls keep their defaults.
void NodeViewModel::load_settings() {
  auto s = client_.get_settings();
  if (s.is_ok()) {
    st_.settings = s.value();
    ++st_.settings_revision;
    settings_loaded_ = true;
    st_.settings_note.clear();
  } else {
    settings_loaded_ = false;
    st_.settings_note = "Could not read the Node's settings: " + describe_error(s.status().code(), s.status().message());
  }
}

void NodeViewModel::rebuild() {
  auto& s = st_;
  s.state_label = std::string(to_string(s.status.state));
  s.description = std::string(describe(s.status.state));
  if (s.status.state == NodeUiState::kUnreachable && !s.status.detail.empty())
    s.description += " (" + ascii_display(s.status.detail) + ")";
  s.father_label = s.status.paired_father.empty() ? "Not paired with a Father yet"
                                                  : "Paired with Father " + ascii_display(s.status.paired_father);
  s.storage_label = "Temporary storage in use: " + format_bytes(s.status.storage_bytes);
  s.storage_label += s.settings.temp_storage_limit_gb > 0 ? " of " + std::to_string(s.settings.temp_storage_limit_gb) + " GB"
                                                          : " (no limit set)";
  s.lease_label.clear();
  const auto& stat = s.status;
  switch (stat.lease) {
    case ipc::LeaseState::kNone: break;
    case ipc::LeaseState::kPreparing:
      s.lease_label = "Receiving temporary files: " + std::to_string(stat.lease_parts_done) + " of " +
                      std::to_string(stat.lease_parts_total);
      break;
    case ipc::LeaseState::kReady:
    case ipc::LeaseState::kInferencing:
      s.lease_label = "Holding " + std::to_string(stat.lease_parts_total) + " temporary files";
      break;
    case ipc::LeaseState::kReleasing:
    case ipc::LeaseState::kCleanupPending: s.lease_label = "Removing temporary files"; break;
  }
  // The lease is only shown for a Node that is actually helping; a busy or paused PC stays out of the way.
  if (stat.state != NodeUiState::kPreparing && stat.state != NodeUiState::kReady && stat.state != NodeUiState::kInUse &&
      stat.state != NodeUiState::kCleanupNeeded)
    s.lease_label.clear();
  s.can_pair = stat.state != NodeUiState::kUnreachable && stat.state != NodeUiState::kStopping &&
               stat.state != NodeUiState::kStarting;
  s.tooltip = "ClusterLM Node - " + s.state_label;
  s.can_pause = s.status.state != NodeUiState::kPaused && s.status.state != NodeUiState::kUnreachable &&
                s.status.state != NodeUiState::kStopping;
  s.can_resume = s.status.state == NodeUiState::kPaused;
}

void NodeViewModel::tick(bool force) {
  const auto now = std::chrono::steady_clock::now();
  if (!force && polled_ && now - last_poll_ < std::chrono::seconds(2)) return;
  last_poll_ = now;
  polled_ = true;
  auto r = client_.status();
  if (r.is_ok()) {
    st_.status = r.value();
  } else {
    st_.status = NodeStatus{};
    st_.status.detail = r.status().message();
  }
  // Settings come from the service: (re)read them whenever it is reachable and they are not loaded yet.
  if (st_.status.state == NodeUiState::kUnreachable) {
    settings_loaded_ = false;
  } else if (!settings_loaded_) {
    load_settings();
  }
  rebuild();
}

Status NodeViewModel::pause() {
  auto s = client_.pause();
  if (!s.is_ok()) st_.banner = "Could not pause: " + describe_error(s.code(), s.message());
  else st_.banner.clear();
  tick(true);
  return s;
}

Status NodeViewModel::resume() {
  auto s = client_.resume();
  if (!s.is_ok()) st_.banner = "Could not resume: " + describe_error(s.code(), s.message());
  else st_.banner.clear();
  tick(true);
  return s;
}

Status NodeViewModel::apply_settings(const NodeSettings& s) {
  if (auto v = validate(s); !v.is_ok()) {
    st_.settings_error = v.message();
    return v;
  }
  st_.settings_error.clear();
  st_.settings = s;
  auto r = client_.set_settings(s);
  if (r.is_ok()) {
    st_.settings_persisted = true;
    st_.settings_note.clear();
    if (auto fresh = client_.get_settings(); fresh.is_ok()) {  // what the service actually stored
      st_.settings = fresh.value();
      ++st_.settings_revision;
    }
  } else {
    st_.settings_persisted = false;
    st_.settings_note = "Settings not saved: " + describe_error(r.code(), r.message());
  }
  rebuild();
  return r;
}

Status NodeViewModel::enter_pairing_mode() {
  st_.pairing_line.clear();  // never leave an old code on screen
  st_.pairing_error.clear();
  auto r = client_.enter_pairing_mode();
  if (!r.is_ok()) {
    st_.pairing_error = "Could not start pairing: " + describe_error(r.status().code(), r.status().message());
    return r.status();
  }
  const auto& p = r.value();
  st_.pairing_line = "Pairing code " + ascii_display(p.code) + ". On the Father, add this PC at " + ascii_display(p.endpoint) +
                     " and enter the code. This PC's fingerprint is " + ascii_display(p.fingerprint) +
                     " - check that the Father shows the same. Pairing mode closes after " +
                     std::to_string((p.window_seconds + 59) / 60) + " minutes.";
  return Status::ok();
}

}  // namespace clusterlm::ui
