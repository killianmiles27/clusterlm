#pragma once
// NodeViewModel: state and behaviour of the (deliberately boring) Node window and tray menu. No rendering, no
// platform code. There is no chat, no model selection and no GGUF vocabulary anywhere in this UI.
//
// Threading: single-threaded (the UI thread). tick() polls the Node service; NodeClient::status() may block for
// up to its I/O timeout when the service hangs, which is acceptable for a status window and documented in
// docs/ui.md.
#include <chrono>
#include <string>

#include "clusterlm/ui/node_client.hpp"

namespace clusterlm::ui {

struct NodeViewState {
  NodeStatus status;
  std::string state_label;  // "Available"
  std::string description;  // one plain sentence
  std::string father_label; // "Paired with Father abc123" / "Not paired with a Father yet"
  std::string storage_label;  // "Temporary storage in use: 1.2 GB of 64 GB"
  std::string tooltip;      // tray tooltip
  bool can_pause = false;
  bool can_resume = false;

  NodeSettings settings;        // what the controls show
  bool settings_persisted = false;  // true once the service accepted the current values
  std::string settings_error;   // validation message (the values were NOT sent)
  std::string settings_note;    // e.g. "This build cannot save Node settings yet..."
  std::string banner;           // last command failure
};

class NodeViewModel {
 public:
  explicit NodeViewModel(NodeClient& client);

  const NodeViewState& state() const { return st_; }

  // Polls status when `force` or the interval elapsed.
  void tick(bool force = false);
  Status pause();
  Status resume();
  // Validates, then sends. An invalid edit is rejected with a field-level message and nothing is sent. Unimplemented
  // from the service keeps the values in this window and says plainly that they were not saved.
  Status apply_settings(const NodeSettings& s);
  void dismiss_banner() { st_.banner.clear(); }

 private:
  void rebuild();

  NodeClient& client_;
  NodeViewState st_;
  std::chrono::steady_clock::time_point last_poll_{};
  bool polled_ = false;
};

}  // namespace clusterlm::ui
