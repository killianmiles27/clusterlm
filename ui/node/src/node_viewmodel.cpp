#include "clusterlm/ui/node_viewmodel.hpp"

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

NodeViewModel::NodeViewModel(NodeClient& client) : client_(client) {
  if (auto s = client_.get_settings(); s.is_ok()) st_.settings = s.value();
  rebuild();
}

void NodeViewModel::rebuild() {
  auto& s = st_;
  s.state_label = std::string(to_string(s.status.state));
  s.description = std::string(describe(s.status.state));
  if (s.status.state == NodeUiState::kUnreachable && !s.status.detail.empty())
    s.description += " (" + ascii_display(s.status.detail) + ")";
  s.father_label = s.status.paired_father.empty() ? "Not paired with a Father yet"
                                                  : "Paired with Father " + ascii_display(s.status.paired_father);
  s.storage_label = "Temporary storage in use: " + format_bytes(s.status.storage_bytes) + " of " +
                    std::to_string(s.settings.temp_storage_limit_gb) + " GB";
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
  } else if (r.code() == ErrorCode::kUnimplemented) {
    st_.settings_persisted = false;
    st_.settings_note = ascii_display(r.message());
  } else {
    st_.settings_persisted = false;
    st_.settings_note = "Settings not saved: " + describe_error(r.code(), r.message());
  }
  rebuild();
  return r;
}

}  // namespace clusterlm::ui
