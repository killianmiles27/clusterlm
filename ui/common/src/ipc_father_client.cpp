// IpcFatherClient: the Father UI's client of the Father agent's JSON IPC API (docs/father-ipc.md).
#include <nlohmann/json.hpp>

#include <condition_variable>
#include <map>

#include "clusterlm/ui/clients.hpp"

namespace clusterlm::ui {

using nlohmann::json;

namespace {

ErrorCode code_from_name(const std::string& name) {
  for (std::uint16_t i = 0; i <= static_cast<std::uint16_t>(ErrorCode::kHardwareUnavailable); ++i)
    if (to_string(static_cast<ErrorCode>(i)) == name) return static_cast<ErrorCode>(i);
  return ErrorCode::kInternal;
}

catalog::TierState state_from_name(const std::string& n) {
  if (n == "Ready") return catalog::TierState::kReady;
  if (n == "Preparing") return catalog::TierState::kPreparing;
  if (n == "Available") return catalog::TierState::kAvailable;
  return catalog::TierState::kUnavailable;
}

template <typename T>
T get(const json& j, const char* key, T fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  try {
    return it->get<T>();
  } catch (...) {
    return fallback;
  }
}

std::optional<double> opt_num(const json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end() || !it->is_number()) return std::nullopt;
  return it->get<double>();
}

std::vector<std::string> strings(const json& j, const char* key) {
  std::vector<std::string> out;
  auto it = j.find(key);
  if (it == j.end() || !it->is_array()) return out;
  for (const auto& e : *it)
    if (e.is_string()) out.push_back(e.get<std::string>());
  return out;
}

// agent event JSON -> service event. Returns nullopt for anything unknown or malformed (ignored, never fatal).
std::optional<father::Event> parse_event(const json& j) {
  if (!j.is_object()) return std::nullopt;
  const std::string kind = get<std::string>(j, "event", "");
  if (kind == "tier_selected") return father::TierSelectedEvent{get<std::string>(j, "tier_id", ""), get<std::string>(j, "model_name", "")};
  if (kind == "prepare_progress")
    return father::PrepareProgressEvent{get<std::string>(j, "tier_id", ""), get<std::string>(j, "model_name", ""), opt_num(j, "percent"),
                                        opt_num(j, "eta_seconds"), get<std::string>(j, "message", "")};
  if (kind == "tier_ready") return father::TierReadyEvent{get<std::string>(j, "tier_id", ""), get<std::string>(j, "model_name", "")};
  if (kind == "tokens") {
    father::TokensEvent t;
    t.request = get<std::uint64_t>(j, "request_id", 0);
    t.tier_id = get<std::string>(j, "tier_id", "");
    t.model_name = get<std::string>(j, "model_name", "");
    t.text = get<std::string>(j, "text", "");
    const auto n = std::min<std::uint64_t>(get<std::uint64_t>(j, "token_count", 0), 1u << 16);
    t.tokens.assign(static_cast<std::size_t>(n), 0);  // IDs never leave the agent: placeholders so counts work
    return t;
  }
  if (kind == "fallback") {
    father::FallbackEvent f;
    f.request = get<std::uint64_t>(j, "request_id", 0);
    f.kind = get<std::string>(j, "kind", "") == "retry" ? father::FallbackEvent::Kind::kRetry : father::FallbackEvent::Kind::kDowngrade;
    f.from_tier = get<std::string>(j, "from_tier", "");
    f.to_tier = get<std::string>(j, "to_tier", "");
    f.from_model = get<std::string>(j, "from_model", "");
    f.to_model = get<std::string>(j, "to_model", "");
    f.message = get<std::string>(j, "message", "");
    return f;
  }
  if (kind == "finished") {
    father::FinishedEvent f;
    f.request = get<std::uint64_t>(j, "request_id", 0);
    const std::string r = get<std::string>(j, "reason", "");
    f.reason = r == "cancelled" ? father::FinishReason::kCancelled : r == "failed" ? father::FinishReason::kFailed : father::FinishReason::kCompleted;
    auto it = j.find("stats");
    if (it != j.end() && it->is_object()) {
      const json& s = *it;
      f.stats.ttft_ms = get<double>(s, "ttft_ms", 0);
      f.stats.decode_tok_s = get<double>(s, "decode_tok_s", 0);
      f.stats.accepted_per_round = get<double>(s, "accepted_per_round", 0);
      f.stats.tokens = get<std::uint32_t>(s, "tokens", 0);
      f.stats.rounds = get<std::uint32_t>(s, "rounds", 0);
      f.stats.fallbacks = get<std::uint32_t>(s, "fallbacks", 0);
      f.stats.final_tier = get<std::string>(s, "final_tier", "");
      f.stats.final_model = get<std::string>(s, "final_model", "");
      auto ab = s.find("answered_by");
      if (ab != s.end() && ab->is_array())
        for (const auto& seg : *ab)
          f.stats.answered_by.push_back({get<std::string>(seg, "tier_id", ""), get<std::string>(seg, "model_name", ""), get<std::uint32_t>(seg, "tokens", 0)});
    }
    return f;
  }
  if (kind == "error")
    return father::ErrorEvent{get<std::uint64_t>(j, "request_id", 0), code_from_name(get<std::string>(j, "code", "")), get<std::string>(j, "message", "")};
  if (kind == "released") return father::ReleasedEvent{get<std::string>(j, "message", "")};
  return std::nullopt;
}

}  // namespace

std::string_view IpcFatherClient::not_running_message() {
  return "The ClusterLM Father agent is not running. Start it and this window will connect.";
}

// ---- implementation ------------------------------------------------------------------------------------------

struct IpcFatherClient::Impl {
  struct Pending {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    Status status;
    json result;
  };

  Options opts;
  std::atomic<bool>* dev_fixture = nullptr;

  std::mutex mu;  // guards conn/reader/pending/next_id/sinks/caches
  std::shared_ptr<ipc::Connection> conn;
  std::thread reader;
  std::map<std::uint64_t, std::shared_ptr<Pending>> pending;
  std::uint64_t next_id = 0;
  std::map<father::SubscriptionId, father::EventSink> sinks;
  father::SubscriptionId next_sub = 1;
  bool stopping = false;

  // caches
  std::string selected;
  std::map<std::string, std::vector<std::string>> roles;  // tier -> roles from the last list
  json pairing_cache;
  std::chrono::steady_clock::time_point pairing_at{};
  FatherSettings local_settings;  // answer length and system prompt are UI-side; context size lives in the agent

  void fail_all(const Status& s) {  // under mu
    for (auto& [id, p] : pending) {
      std::lock_guard lk(p->m);
      p->done = true;
      p->status = s;
      p->cv.notify_all();
    }
    pending.clear();
  }

  void dispatch(const father::Event& e) {
    std::vector<father::EventSink> targets;
    {
      std::lock_guard lk(mu);
      for (auto& [id, s] : sinks) targets.push_back(s);
    }
    for (auto& s : targets) s(e);
  }

  void read_loop(std::shared_ptr<ipc::Connection> c) {
    for (;;) {
      auto env = c->receive(std::chrono::milliseconds(200));
      if (!env.is_ok()) {
        if (env.status().code() == ErrorCode::kDeadlineExceeded) {
          std::lock_guard lk(mu);
          if (stopping || conn != c) return;
          continue;
        }
        std::lock_guard lk(mu);
        if (conn == c) {
          conn.reset();
          fail_all(make_error(ErrorCode::kUnavailable, std::string(not_running_message())));
        }
        return;
      }
      const auto kind = static_cast<ipc::MessageKind>(env->kind);
      if (kind == ipc::MessageKind::kAck) {
        // The agent refused a frame outright (bad version/kind/payload): fail whatever is waiting.
        auto a = ipc::decode_ack(env.value());
        std::lock_guard lk(mu);
        fail_all(a.is_ok() ? make_error(a->code, a->message) : a.status());
        continue;
      }
      json j = json::parse(env->payload.begin(), env->payload.end(), nullptr, false);
      if (j.is_discarded()) continue;
      if (kind == ipc::MessageKind::kFatherEvent) {
        try {
          if (auto ev = parse_event(j)) dispatch(*ev);
        } catch (...) {
        }
      } else if (kind == ipc::MessageKind::kFatherReply) {
        std::shared_ptr<Pending> p;
        {
          std::lock_guard lk(mu);
          auto it = pending.find(get<std::uint64_t>(j, "id", 0));
          if (it == pending.end()) continue;
          p = it->second;
          pending.erase(it);
        }
        std::lock_guard lk(p->m);
        if (get<bool>(j, "ok", false)) {
          auto r = j.find("result");
          if (r != j.end()) p->result = *r;
        } else {
          auto e = j.find("error");
          p->status = make_error(e != j.end() && e->is_object() ? code_from_name(get<std::string>(*e, "code", "")) : ErrorCode::kInternal,
                                 e != j.end() && e->is_object() ? get<std::string>(*e, "message", "request failed") : "request failed");
        }
        p->done = true;
        p->cv.notify_all();
      }
    }
  }

  // One raw round trip on an established connection.
  Result<json> round_trip(const std::shared_ptr<ipc::Connection>& c, json req, std::chrono::milliseconds timeout) {
    auto p = std::make_shared<Pending>();
    {
      std::lock_guard lk(mu);
      req["id"] = ++next_id;
      pending[next_id] = p;
    }
    const std::string text = req.dump(-1, ' ', false, json::error_handler_t::replace);
    auto sent = c->send(ipc::father_envelope(ipc::MessageKind::kFatherRequest, Bytes(text.begin(), text.end())), std::chrono::milliseconds(5000));
    if (!sent.is_ok()) {
      std::lock_guard lk(mu);
      pending.erase(req["id"].get<std::uint64_t>());
      return make_error(ErrorCode::kUnavailable, std::string(not_running_message()));
    }
    bool answered;
    {
      std::unique_lock lk(p->m);
      answered = p->cv.wait_for(lk, timeout, [&] { return p->done; });
    }  // never hold p->m while taking mu: fail_all takes them in the other order
    if (!answered) {
      std::lock_guard g(mu);
      pending.erase(req["id"].get<std::uint64_t>());
      return make_error(ErrorCode::kDeadlineExceeded, "The Father agent did not answer in time.");
    }
    std::lock_guard lk(p->m);
    if (!p->status.is_ok()) return p->status;
    return std::move(p->result);
  }

  Result<std::shared_ptr<ipc::Connection>> ensure_connected() {
    std::thread old_reader;
    {
      std::lock_guard lk(mu);
      if (conn && conn->is_open()) return conn;
      if (conn) conn.reset();
      old_reader = std::move(reader);
    }
    if (old_reader.joinable()) old_reader.join();
    auto c = ipc::connect(opts.endpoint, opts.client, opts.connect_timeout);
    if (!c.is_ok()) return make_error(ErrorCode::kUnavailable, std::string(not_running_message()));
    std::shared_ptr<ipc::Connection> sc(std::move(c).value());
    {
      std::lock_guard lk(mu);
      conn = sc;
      reader = std::thread([this, sc] { read_loop(sc); });
    }
    // Learn whether the agent runs the development fixture model (labels every number Synthetic).
    auto hello = round_trip(sc, {{"op", "hello"}}, opts.request_timeout);
    if (hello.is_ok() && dev_fixture) dev_fixture->store(get<bool>(hello.value(), "dev_fixture_model", false));
    return sc;
  }

  Result<json> call(json req, std::chrono::milliseconds timeout) {
    auto c = ensure_connected();
    if (!c.is_ok()) return c.status();
    return round_trip(c.value(), std::move(req), timeout);
  }
  Result<json> call(json req) { return call(std::move(req), opts.request_timeout); }
  Status call_status(json req) {
    auto r = call(std::move(req));
    return r.is_ok() ? Status::ok() : r.status();
  }

  Result<json> pairing_list(bool force) {
    {
      std::lock_guard lk(mu);
      if (!force && !pairing_cache.is_null() && std::chrono::steady_clock::now() - pairing_at < std::chrono::milliseconds(1500)) return pairing_cache;
    }
    auto r = call({{"op", "pairing.list"}});
    if (!r.is_ok()) return r;
    std::lock_guard lk(mu);
    pairing_cache = r.value();
    pairing_at = std::chrono::steady_clock::now();
    return pairing_cache;
  }
};

IpcFatherClient::IpcFatherClient(Options options) : impl_(std::make_unique<Impl>()) {
  impl_->opts = std::move(options);
  impl_->dev_fixture = &dev_fixture_;
}

IpcFatherClient::~IpcFatherClient() {
  std::shared_ptr<ipc::Connection> c;
  std::thread reader;
  {
    std::lock_guard lk(impl_->mu);
    impl_->stopping = true;
    c = impl_->conn;
    reader = std::move(impl_->reader);
  }
  if (c) c->close();
  if (reader.joinable()) reader.join();
}

Result<std::vector<catalog::TierReadiness>> IpcFatherClient::list_tiers(std::uint32_t context_tokens) {
  json req = {{"op", "tiers.list"}};
  if (context_tokens) req["context_tokens"] = context_tokens;
  auto r = impl_->call(req);
  if (!r.is_ok()) return r.status();
  std::vector<catalog::TierReadiness> out;
  std::map<std::string, std::vector<std::string>> roles;
  auto it = r->find("tiers");
  if (it != r->end() && it->is_array()) {
    for (const auto& t : *it) {
      catalog::TierReadiness tr;
      tr.tier_id = get<std::string>(t, "tier_id", "");
      tr.model_name = get<std::string>(t, "model_name", "");
      tr.state = state_from_name(get<std::string>(t, "state", ""));
      tr.reasons = strings(t, "reasons");
      tr.notes = strings(t, "notes");
      for (auto& d : strings(t, "details")) tr.notes.push_back(std::move(d));  // provenance / dev-override notes
      tr.suggested_fallback = strings(t, "suggested_fallback");
      auto pg = t.find("progress");
      if (pg != t.end() && pg->is_object()) {
        catalog::PrepareProgress p;
        p.percent = get<double>(*pg, "percent", 0);
        p.eta_seconds = opt_num(*pg, "eta_seconds");
        tr.progress = p;
      }
      roles[tr.tier_id] = strings(t, "roles");
      out.push_back(std::move(tr));
    }
  }
  std::lock_guard lk(impl_->mu);
  impl_->selected = get<std::string>(r.value(), "selected", "");
  impl_->roles = std::move(roles);
  return out;
}

Result<std::vector<TierParticipant>> IpcFatherClient::participants(std::string_view tier_id) {
  std::vector<std::string> roles;
  {
    std::lock_guard lk(impl_->mu);
    auto it = impl_->roles.find(std::string(tier_id));
    if (it == impl_->roles.end()) return std::vector<TierParticipant>{};
    roles = it->second;
  }
  auto pl = impl_->pairing_list(false);
  std::vector<TierParticipant> out;
  for (const auto& role : roles) {
    TierParticipant p{role, role};
    if (role == "father") {
      p.machine_id = "This PC";
    } else if (pl.is_ok()) {
      // role -> fingerprint (assignments) -> device name
      std::string fp;
      auto a = pl->find("assignments");
      if (a != pl->end() && a->is_object()) fp = get<std::string>(*a, role.c_str(), "");
      auto d = pl->find("devices");
      if (!fp.empty() && d != pl->end() && d->is_array())
        for (const auto& dev : *d)
          if (get<std::string>(dev, "fingerprint", "") == fp) {
            const std::string name = get<std::string>(dev, "name", "");
            p.machine_id = name.empty() ? get<std::string>(dev, "short", fp) : name;
          }
    }
    out.push_back(std::move(p));
  }
  return out;
}

std::string IpcFatherClient::selected_tier() {
  std::lock_guard lk(impl_->mu);
  return impl_->selected;
}

Status IpcFatherClient::select_tier(std::string_view tier_id) {
  auto r = impl_->call({{"op", "tiers.select"}, {"tier_id", std::string(tier_id)}});
  if (!r.is_ok()) return r.status();
  std::lock_guard lk(impl_->mu);
  impl_->selected = std::string(tier_id);
  return Status::ok();
}

Status IpcFatherClient::prepare_tier(std::string_view tier_id, std::uint32_t context_tokens) {
  json req = {{"op", "tiers.prepare"}, {"tier_id", std::string(tier_id)}};
  if (context_tokens) req["context_tokens"] = context_tokens;
  return impl_->call_status(req);
}

Result<father::RequestId> IpcFatherClient::send_chat(father::ChatRequest request) {
  json req = {{"op", "chat.send"}, {"message", request.user_message}, {"max_new_tokens", request.max_new_tokens},
              {"context_tokens", request.context_tokens}, {"q", request.q}};
  if (request.system_prompt) req["system_prompt"] = *request.system_prompt;
  auto r = impl_->call(req);
  if (!r.is_ok()) return r.status();
  const auto id = get<std::uint64_t>(r.value(), "request_id", 0);
  if (id == 0) return make_error(ErrorCode::kProtocolError, "The Father agent did not return a request id.");
  return id;
}

Status IpcFatherClient::cancel(father::RequestId request) { return impl_->call_status({{"op", "chat.cancel"}, {"request_id", request}}); }
Status IpcFatherClient::release() { return impl_->call_status({{"op", "session.release"}}); }
Status IpcFatherClient::reset_conversation() { return impl_->call_status({{"op", "conversation.reset"}}); }

Result<ContextUse> IpcFatherClient::context_use() {
  // The tokenizer lives in the agent and the API does not expose conversation length.
  return make_error(ErrorCode::kUnavailable, "context use is not reported by the Father agent");
}

Result<DiagnosticsExport> IpcFatherClient::export_diagnostics(bool include_text) {
  auto r = impl_->call({{"op", "diagnostics.get"}, {"include_text", include_text}});
  if (!r.is_ok()) return r.status();
  DiagnosticsExport e;
  e.text = get<std::string>(r.value(), "text", "");
  e.includes_conversation = include_text && get<bool>(r.value(), "text_included", false);
  if (e.includes_conversation) {
    auto c = r->find("conversation");
    if (c != r->end() && c->is_array()) {
      e.text += "\nconversation:\n";
      for (const auto& m : *c) e.text += get<std::string>(m, "role", "?") + ": " + get<std::string>(m, "content", "") + "\n";
    }
  }
  // The API reports counters and tier lines but no plan or per-stage timings: leave them empty (the UI says so).
  return e;
}

Result<FatherSettings> IpcFatherClient::get_settings() {
  FatherSettings s;
  {
    std::lock_guard lk(impl_->mu);
    s = impl_->local_settings;
  }
  auto r = impl_->call({{"op", "settings.get"}});
  if (r.is_ok()) s.context_tokens = get<std::uint32_t>(r.value(), "context_tokens", s.context_tokens);
  // An unreachable agent is not a settings error: the defaults let the window open and explain itself.
  return s;
}

Status IpcFatherClient::set_settings(const FatherSettings& s) {
  if (auto v = validate(s); !v.is_ok()) return v;
  if (auto st = impl_->call_status({{"op", "settings.set"}, {"patch", {{"context_tokens", s.context_tokens}}}}); !st.is_ok()) return st;
  std::lock_guard lk(impl_->mu);
  impl_->local_settings = s;
  return Status::ok();
}

Result<std::vector<PairedMachine>> IpcFatherClient::paired_machines() {
  auto r = impl_->pairing_list(true);
  if (!r.is_ok()) return r.status();
  std::vector<PairedMachine> out;
  auto d = r->find("devices");
  if (d != r->end() && d->is_array())
    for (const auto& dev : *d) {
      PairedMachine m;
      m.fingerprint = get<std::string>(dev, "short", "");
      m.machine_id = get<std::string>(dev, "name", "");
      if (m.machine_id.empty()) m.machine_id = m.fingerprint;
      m.role_hint = get<std::string>(dev, "role", "Node");
      out.push_back(std::move(m));
    }
  return out;
}

Status IpcFatherClient::start_pairing(const PairingRequest& request) {
  if (request.address.empty() || request.code.empty())
    return make_error(ErrorCode::kInvalidArgument, "Enter the address and the pairing code shown on the other PC.");
  json req = {{"op", "pairing.start"}, {"address", request.address}, {"code", request.code}};
  if (!request.name.empty()) req["name"] = request.name;
  auto r = impl_->call(req, impl_->opts.pairing_timeout);
  if (!r.is_ok()) return r.status();
  std::lock_guard lk(impl_->mu);
  impl_->pairing_cache = json();
  return Status::ok();
}

Status IpcFatherClient::unpair(std::string_view machine_id) {
  auto pl = impl_->pairing_list(true);
  if (!pl.is_ok()) return pl.status();
  std::string fp;
  auto d = pl->find("devices");
  if (d != pl->end() && d->is_array())
    for (const auto& dev : *d) {
      const std::string name = get<std::string>(dev, "name", "");
      const std::string sh = get<std::string>(dev, "short", "");
      if ((!name.empty() && name == machine_id) || sh == machine_id) fp = get<std::string>(dev, "fingerprint", "");
    }
  if (fp.empty()) return make_error(ErrorCode::kNotFound, "That machine is not paired.");
  auto st = impl_->call_status({{"op", "pairing.unpair"}, {"fingerprint", fp}});
  std::lock_guard lk(impl_->mu);
  impl_->pairing_cache = json();
  return st;
}

father::SubscriptionId IpcFatherClient::subscribe(father::EventSink sink) {
  std::lock_guard lk(impl_->mu);
  const auto id = impl_->next_sub++;
  impl_->sinks[id] = std::move(sink);
  return id;
}

void IpcFatherClient::unsubscribe(father::SubscriptionId id) {
  std::lock_guard lk(impl_->mu);
  impl_->sinks.erase(id);
}

}  // namespace clusterlm::ui
