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
  std::string lease_label;    // "Receiving temporary files: 3 of 12"; empty when the Node holds nothing
  std::string tooltip;      // tray tooltip
  bool can_pause = false;
  bool can_resume = false;

  NodeSettings settings;        // what the controls show
  std::uint32_t settings_revision = 0;  // bumped whenever `settings` was replaced from the service (draw code reloads)
  bool settings_persisted = false;  // true once the service accepted the current values
  std::string settings_error;   // validation message (the values were NOT sent)
  std::string settings_note;    // the service could not read or save them: the reason, in words
  std::string banner;           // last command failure

  // Pairing mode (the Pair button). `pairing_line` is what to show while it is on: the code, where to type it on the
  // Father and the fingerprint to compare afterwards.
  std::string pairing_line;
  std::string pairing_error;
  bool can_pair = false;        // the service is running and not stopping
};

class NodeViewModel {
 public:
  explicit NodeViewModel(NodeClient& client);

  const NodeViewState& state() const { return st_; }

  // Polls status when `force` or the interval elapsed.
  void tick(bool force = false);
  Status pause();
  Status resume();
  // Validates, then sends. An invalid edit is rejected with a field-level message and nothing is sent. A refusal or
  // failure from the service keeps the values in this window and says plainly that they were not saved; on success
  // the values are re-read from the service, which is the authority on what was stored.
  Status apply_settings(const NodeSettings& s);
  // Asks the service to enter pairing mode and shows the line it returns. A refusal (rate limit, not signed in) is
  // shown as `pairing_error`; the previous line is dropped so a stale code is never displayed.
  Status enter_pairing_mode();
  void dismiss_banner() { st_.banner.clear(); }

 private:
  void rebuild();

  void load_settings();

  NodeClient& client_;
  NodeViewState st_;
  bool settings_loaded_ = false;
  std::chrono::steady_clock::time_point last_poll_{};
  bool polled_ = false;
};

}  // namespace clusterlm::ui
