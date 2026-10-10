#include "clusterlm/ui/connections_viewmodel.hpp"

#include <algorithm>

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

namespace {
std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) out += (i ? sep : "") + v[i];
  return out;
}

std::vector<ScopePreset> presets() {
  return {
      {"chat", "Chat only", "The app can list models and ask them questions. It cannot change anything.", {"inference"}, "", false},
      {"chat-status", "Chat and status", "Also lets the app see which profiles are ready and how the machines are doing.", {"inference", "status:read"}, "", false},
      {"manage", "Manage profiles (advanced)", "Lets the app create, change and prepare profiles. Only give this to software you fully trust.",
       {"status:read", "profiles:prepare", "profiles:admin"}, "This key can change your profiles. It always covers every model.", true},
  };
}

std::string what_for(const std::vector<std::string>& scopes) {
  auto has = [&](const char* s) { return std::find(scopes.begin(), scopes.end(), s) != scopes.end(); };
  if (has("profiles:admin") || has("keys:admin")) return "Can manage profiles" + std::string(has("inference") ? " and chat" : "");
  if (has("inference") && has("status:read")) return "Chat and status";
  if (has("inference") && scopes.size() == 1) return "Chat only";
  return join(scopes, ", ");
}

constexpr const char* kKeyPlaceholder = "YOUR_CLUSTERLM_KEY";
}  // namespace

ConnectionsViewModel::ConnectionsViewModel(HostAdminClient& client) : client_(client) { st_.presets = presets(); }

void ConnectionsViewModel::fail(const Status& s) { st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())}; }

std::vector<Snippet> ConnectionsViewModel::make_snippets(const std::string& base_url, const std::string& model) {
  std::vector<Snippet> out;
  out.push_back({"curl", "Any terminal. Set CLUSTERLM_API_KEY first.",
                 "curl " + base_url + "/chat/completions \\\n  -H \"Authorization: Bearer $CLUSTERLM_API_KEY\" \\\n  -H \"Content-Type: application/json\" \\\n"
                 "  -d '{\"model\": \"" + model + "\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"});
  out.push_back({"Python (openai)", "A Python script. Set CLUSTERLM_API_KEY first.",
                 "import os\nfrom openai import OpenAI\n\nclient = OpenAI(base_url=\"" + base_url + "\", api_key=os.environ[\"CLUSTERLM_API_KEY\"])\n"
                 "reply = client.chat.completions.create(\n    model=\"" + model + "\",\n    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n)\n"
                 "print(reply.choices[0].message.content)"});
  out.push_back({"OpenCode", "opencode.json in your project or OpenCode's config folder. Set CLUSTERLM_API_KEY first.",
                 "{\n  \"$schema\": \"https://opencode.ai/config.json\",\n  \"provider\": {\n    \"clusterlm\": {\n      \"npm\": \"@ai-sdk/openai-compatible\",\n"
                 "      \"name\": \"ClusterLM\",\n      \"options\": {\n        \"baseURL\": \"" + base_url + "\",\n        \"apiKey\": \"{env:CLUSTERLM_API_KEY}\"\n      },\n"
                 "      \"models\": {\n        \"" + model + "\": { \"name\": \"" + model + "\" }\n      }\n    }\n  }\n}"});
  out.push_back({"Pi", "Pi's models.json. Set CLUSTERLM_API_KEY first.",
                 "{\n  \"providers\": {\n    \"clusterlm\": {\n      \"baseUrl\": \"" + base_url + "\",\n      \"api\": \"openai-completions\",\n"
                 "      \"apiKey\": \"CLUSTERLM_API_KEY\",\n      \"models\": [ { \"id\": \"" + model + "\" } ]\n    }\n  }\n}"});
  out.push_back({"Any other app", "Look for 'OpenAI-compatible' in the app's settings.",
                 "Base URL: " + base_url + "\nModel:    " + model + "\nAPI key:  " + std::string(kKeyPlaceholder) + " (the key you created here)"});
  return out;
}

void ConnectionsViewModel::rebuild_snippets() {
  st_.snippet_warning.clear();
  if (st_.exposed_models.empty()) {
    st_.snippet_model = "your-model-name";
    st_.snippet_warning = "No model is offered to apps yet. Turn on app access for a profile (Profiles, Edit), then use its name here.";
  } else if (std::find(st_.exposed_models.begin(), st_.exposed_models.end(), st_.snippet_model) == st_.exposed_models.end()) {
    st_.snippet_model = st_.exposed_models.front();
  }
  if (!st_.settings.enabled) {
    const std::string off = "The connection is switched off, so these will not work until you turn it on.";
    st_.snippet_warning += (st_.snippet_warning.empty() ? "" : " ") + off;
  }
  if (st_.settings.allow_lan) {
    st_.snippet_warning += (st_.snippet_warning.empty() ? "" : " ") +
                           std::string("Replace <this-PC-address> with this PC's name or IP address. The app must trust the certificate shown above.");
  }
  st_.snippets = make_snippets(st_.base_url, st_.snippet_model);
}

void ConnectionsViewModel::refresh() {
  auto status = client_.server_status();
  if (!status.is_ok()) {
    st_.stale = true;
    st_.listening = false;
    st_.mode_text = "Status unknown";
    fail(status.status());
    return;
  }
  st_.stale = false;
  const ServerStatus& s = status.value();
  st_.supported = s.available;
  st_.unsupported_text = s.available ? "" : (s.note.empty() ? "This Host cannot serve apps yet." : ascii_display(s.note));
  st_.settings = s.settings;
  st_.listening = s.listening;
  st_.address = s.address;
  st_.tls_fingerprint = s.tls_fingerprint;
  st_.server_note = ascii_display(s.note);
  st_.exposed_models = s.exposed_models;
  const std::string port = std::to_string(s.settings.port);
  if (!s.settings.enabled) st_.mode_text = "Off";
  else if (s.settings.allow_lan) st_.mode_text = "On - other PCs on your network can connect (encrypted, key required)";
  else st_.mode_text = "On - this PC only (" + (s.address.empty() ? "127.0.0.1:" + port : s.address) + ")";
  st_.base_url = s.settings.allow_lan ? "https://<this-PC-address>:" + port + "/v1" : "http://127.0.0.1:" + port + "/v1";

  st_.keys.clear();
  if (auto ks = client_.list_keys(); ks.is_ok()) {
    for (const auto& k : ks.value()) {
      KeyRow r;
      r.id = k.id;
      r.name = ascii_display(k.name);
      r.scopes = k.scopes;
      r.what = what_for(k.scopes);
      r.models = (k.allowed_models.size() == 1 && k.allowed_models.front() == "*") ? "All models" : join(k.allowed_models, ", ");
      r.network = k.lan_allowed ? "Other PCs allowed" : "This PC only";
      r.created = k.created;
      r.last_used = k.last_used.empty() ? "never" : k.last_used;
      r.revoked = k.revoked;
      if (std::find(k.scopes.begin(), k.scopes.end(), "profiles:admin") != k.scopes.end()) r.warning = "Can change your profiles.";
      if (k.lan_allowed) r.warning += std::string(r.warning.empty() ? "" : " ") + "Works from other PCs.";
      st_.keys.push_back(std::move(r));
    }
  }
  rebuild_snippets();
}

Status ConnectionsViewModel::save(const ServerSettings& s) {
  auto r = client_.set_server(s);
  if (!r.is_ok()) { fail(r); return r; }
  refresh();
  return r;
}

Status ConnectionsViewModel::set_enabled(bool enabled) {
  ServerSettings s = st_.settings;
  s.enabled = enabled;
  auto r = save(s);
  if (r.is_ok()) st_.notice = Notice{Notice::Kind::kSuccess, enabled ? "Apps on this PC can now connect with a key." : "Connections are off."};
  return r;
}

Status ConnectionsViewModel::set_allow_lan(bool allow, bool i_understand) {
  if (allow && !i_understand) {
    Status s = make_error(ErrorCode::kFailedPrecondition,
                          "Confirm that you want other PCs on your network to reach this PC's ClusterLM port. They will still need a key.");
    fail(s);
    return s;
  }
  ServerSettings s = st_.settings;
  s.allow_lan = allow;
  s.tls = allow ? true : s.tls;  // other PCs only ever connect over an encrypted connection
  auto r = save(s);
  if (r.is_ok()) st_.notice = Notice{Notice::Kind::kSuccess, allow ? "Other PCs can connect over an encrypted connection. Compare the certificate fingerprint on the app." : "Back to this PC only."};
  return r;
}

Status ConnectionsViewModel::set_port(std::uint16_t port) {
  ServerSettings s = st_.settings;
  s.port = port;
  return save(s);
}

Status ConnectionsViewModel::set_no_key_on_loopback(bool on, bool i_understand) {
  if (on && !i_understand) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "Confirm that any program on this PC may then chat without a key.");
    fail(s);
    return s;
  }
  ServerSettings s = st_.settings;
  s.no_key_on_loopback = on;
  return save(s);
}

Status ConnectionsViewModel::create_key(const std::string& name, const std::string& preset_id, const std::vector<std::string>& models, bool lan) {
  auto it = std::find_if(st_.presets.begin(), st_.presets.end(), [&](const ScopePreset& p) { return p.id == preset_id; });
  if (it == st_.presets.end()) {
    Status s = make_error(ErrorCode::kInvalidArgument, "Choose what the key may do.");
    fail(s);
    return s;
  }
  if (lan && !st_.settings.allow_lan) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "Allow other PCs to connect first, or leave this key for this PC only.");
    fail(s);
    return s;
  }
  std::vector<std::string> allowed = it->all_models_only ? std::vector<std::string>{"*"} : models;
  auto r = client_.create_key(name, it->scopes, allowed, lan);
  if (!r.is_ok()) { fail(r.status()); return r.status(); }
  st_.reveal = RevealedKey{true, ascii_display(r->key.name), r->secret};
  st_.notice = Notice{Notice::Kind::kInfo, "Copy the key now. It is shown only once and cannot be looked up later."};
  r->secret.assign(r->secret.size(), '\0');
  refresh();
  return Status::ok();
}

Status ConnectionsViewModel::revoke_key(const std::string& id, bool i_confirm) {
  if (!i_confirm) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "Confirm to revoke this key. Apps using it stop working at once.");
    fail(s);
    return s;
  }
  auto s = client_.revoke_key(id);
  if (!s.is_ok()) { fail(s); return s; }
  st_.notice = Notice{Notice::Kind::kSuccess, "Key revoked. Anything it was running has been stopped."};
  refresh();
  return s;
}

void ConnectionsViewModel::dismiss_secret() {
  st_.reveal.secret.assign(st_.reveal.secret.size(), '\0');
  st_.reveal = {};
}

void ConnectionsViewModel::set_snippet_model(const std::string& model) {
  if (std::find(st_.exposed_models.begin(), st_.exposed_models.end(), model) == st_.exposed_models.end()) return;
  st_.snippet_model = model;
  rebuild_snippets();
}

}  // namespace clusterlm::ui
