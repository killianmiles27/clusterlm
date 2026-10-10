#pragma once
// ConnectionsViewModel: the app-facing OpenAI-compatible server, its keys, and copy-ready client setup text.
//
// Rules (docs/interfaces/auth-scopes-v1.md): the server is off and this-PC-only by default; letting other PCs connect
// is an explicit confirmed action and needs encryption; a key's value is shown exactly once (`reveal`), is held only
// until dismissed and never appears in any snippet, notice or row; new keys default to chat-only, this-PC-only;
// management scopes are warned about and always cover every model. Snippets are starting points built from the live
// address and model names: they are not tested against the client programs here.
#include <string>
#include <vector>

#include "clusterlm/ui/host_admin.hpp"
#include "clusterlm/ui/host_common.hpp"

namespace clusterlm::ui {

struct Snippet {
  std::string client;     // "curl", "Python (openai)", "OpenCode", "Pi", "Any other app"
  std::string file_hint;  // where this text goes
  std::string text;
};

struct ScopePreset {
  std::string id;           // "chat" | "chat-status" | "manage"
  std::string label;
  std::string description;
  std::vector<std::string> scopes;
  std::string warning;      // non-empty for presets that can change things
  bool all_models_only = false;
};

struct KeyRow {
  std::string id, name;
  std::string what;         // plain language: "Chat only", "Chat and status"
  std::vector<std::string> scopes;
  std::string models;       // "All models" | "example-fast, example-strong"
  std::string network;      // "This PC only" | "Other PCs allowed"
  std::string created, last_used;
  bool revoked = false;
  std::string warning;
};

struct RevealedKey {
  bool shown = false;
  std::string name, secret;
};

struct ConnectionsState {
  bool supported = true;
  std::string unsupported_text;
  bool stale = false;
  ServerSettings settings;
  bool listening = false;
  std::string mode_text;     // "Off" | "On - this PC only (127.0.0.1:11434)" | "On - other PCs on your network can connect"
  std::string address;
  std::string base_url;
  std::string tls_fingerprint;
  std::string server_note;
  std::vector<std::string> exposed_models;
  std::vector<KeyRow> keys;
  std::vector<ScopePreset> presets;
  RevealedKey reveal;
  std::vector<Snippet> snippets;
  std::string snippet_model;
  std::string snippet_warning;
  Notice notice;
};

class ConnectionsViewModel {
 public:
  explicit ConnectionsViewModel(HostAdminClient& client);
  const ConnectionsState& state() const { return st_; }

  void refresh();
  Status set_enabled(bool enabled);
  // Other PCs may connect. `i_understand` comes from an explicit confirmation in the UI. Turns encryption on with it.
  Status set_allow_lan(bool allow, bool i_understand);
  Status set_port(std::uint16_t port);
  Status set_no_key_on_loopback(bool on, bool i_understand);
  Status create_key(const std::string& name, const std::string& preset_id, const std::vector<std::string>& models, bool lan);
  Status revoke_key(const std::string& id, bool i_confirm);
  void dismiss_secret();   // overwrites and drops the secret
  void set_snippet_model(const std::string& model);
  void dismiss_notice() { st_.notice = {}; }

  // Exposed for tests and the draw layer.
  static std::vector<Snippet> make_snippets(const std::string& base_url, const std::string& model);

 private:
  void fail(const Status& s);
  Status save(const ServerSettings& s);
  void rebuild_snippets();
  HostAdminClient& client_;
  ConnectionsState st_;
};

}  // namespace clusterlm::ui
