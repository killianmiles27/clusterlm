#include "clusterlm/father/father_service_api.hpp"

#include <nlohmann/json.hpp>

#include "clusterlm/common/log.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/pairing/pairing.hpp"

namespace clusterlm::father {

using nlohmann::json;
using namespace std::chrono_literals;

namespace {

// A tokenizer for builds without a real one: chat is refused with an honest message instead of garbage.
class NoTokenizer final : public Tokenizer {
 public:
  std::uint32_t vocab_size() const override { return 0; }
  std::vector<std::int32_t> encode(std::string_view) const override { return {}; }
  std::string decode(std::span<const std::int32_t>) const override { return {}; }
  std::vector<std::int32_t> encode_chat(std::span<const ChatMessage>) const override { return {}; }
};

json error_json(const Status& s) { return {{"code", std::string(to_string(s.code()))}, {"message", s.message()}}; }

// Model output is arbitrary bytes (a fixture or a mid-character token cut): never let dump() throw on it.
std::string dump_safe(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

Bytes to_bytes(const json& j) {
  const std::string s = dump_safe(j);
  return Bytes(s.begin(), s.end());
}

std::int64_t now_unix() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

json device_json(const config::PairedDevice& d) {
  return {{"fingerprint", d.fingerprint}, {"short", pairing::short_fingerprint(d.fingerprint)}, {"name", d.name},
          {"address", d.address},         {"role", d.role},                                     {"paired_at_unix", d.paired_at_unix}};
}

template <typename T>
bool get_opt(const json& obj, const char* key, T& out) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return false;
  try {
    out = it->get<T>();
    return true;
  } catch (...) {
    return false;
  }
}

Status need_string(const json& req, const char* key, std::string& out) {
  if (!get_opt(req, key, out) || out.empty()) return make_error(ErrorCode::kInvalidArgument, std::string("missing or invalid '") + key + "'");
  return Status::ok();
}

json event_json(const Event& event) {
  return std::visit(
      [](const auto& e) -> json {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, TierSelectedEvent>) return {{"event", "tier_selected"}, {"tier_id", e.tier_id}, {"model_name", e.model_name}};
        else if constexpr (std::is_same_v<T, PrepareProgressEvent>) {
          json j = {{"event", "prepare_progress"}, {"tier_id", e.tier_id}, {"model_name", e.model_name}, {"message", e.message}};
          j["percent"] = e.percent ? json(*e.percent) : json(nullptr);
          j["eta_seconds"] = e.eta_seconds ? json(*e.eta_seconds) : json(nullptr);  // an estimate
          return j;
        } else if constexpr (std::is_same_v<T, TierReadyEvent>) return {{"event", "tier_ready"}, {"tier_id", e.tier_id}, {"model_name", e.model_name}};
        else if constexpr (std::is_same_v<T, TokensEvent>)
          return {{"event", "tokens"}, {"request_id", e.request}, {"tier_id", e.tier_id}, {"model_name", e.model_name},
                  {"token_count", e.tokens.size()}, {"text", e.text}};
        else if constexpr (std::is_same_v<T, FallbackEvent>)
          return {{"event", "fallback"}, {"request_id", e.request}, {"kind", e.kind == FallbackEvent::Kind::kRetry ? "retry" : "downgrade"},
                  {"from_tier", e.from_tier}, {"to_tier", e.to_tier}, {"from_model", e.from_model}, {"to_model", e.to_model}, {"message", e.message}};
        else if constexpr (std::is_same_v<T, FinishedEvent>) {
          json segs = json::array();
          for (const auto& s : e.stats.answered_by) segs.push_back({{"tier_id", s.tier_id}, {"model_name", s.model_name}, {"tokens", s.tokens}});
          return {{"event", "finished"}, {"request_id", e.request}, {"reason", std::string(to_string(e.reason))},
                  {"stats", {{"ttft_ms", e.stats.ttft_ms}, {"decode_tok_s", e.stats.decode_tok_s}, {"accepted_per_round", e.stats.accepted_per_round},
                             {"tokens", e.stats.tokens}, {"rounds", e.stats.rounds}, {"fallbacks", e.stats.fallbacks},
                             {"final_tier", e.stats.final_tier}, {"final_model", e.stats.final_model}, {"answered_by", segs},
                             {"provenance", "observed on this run"}}}};
        } else if constexpr (std::is_same_v<T, ErrorEvent>)
          return {{"event", "error"}, {"request_id", e.request}, {"code", std::string(to_string(e.code))}, {"message", e.message}};
        else
          return {{"event", "released"}, {"message", e.message}};
      },
      event);
}

}  // namespace

Result<std::unique_ptr<FatherServiceApi>> FatherServiceApi::create(FatherApiConfig config) {
  if (!config.settings || !config.identity || !config.readiness || !config.deployments)
    return make_error(ErrorCode::kInvalidArgument, "FatherServiceApi needs settings, identity, readiness and deployments");
  if (!config.tokenizer) config.tokenizer = std::make_shared<NoTokenizer>();
  std::unique_ptr<FatherServiceApi> api(new FatherServiceApi(std::move(config)));
  CLM_RETURN_IF_ERROR(api->rebuild_service());
  api->touch();
  api->watchdog_ = std::thread([p = api.get()] { p->watchdog(); });
  return api;
}

FatherServiceApi::~FatherServiceApi() {
  stopping_.store(true);
  if (watchdog_.joinable()) watchdog_.join();
  std::shared_ptr<FatherService> s;
  {
    std::lock_guard lk(mu_);
    s = std::move(service_);
  }
  if (s) {
    s->unsubscribe(subscription_);
    s.reset();  // joins the job, releases leases
  }
}

void FatherServiceApi::set_event_push(std::function<void(Bytes)> p) {
  std::lock_guard lk(mu_);
  push_ = std::move(p);
}

std::shared_ptr<FatherService> FatherServiceApi::service() const {
  std::lock_guard lk(mu_);
  return service_;
}

SessionPhase FatherServiceApi::session_phase() const {
  if (job_running_.load()) return tier_ready_.load() ? SessionPhase::kInferencing : SessionPhase::kPreparing;
  return tier_ready_.load() ? SessionPhase::kReadyIdle : SessionPhase::kNone;
}

bool FatherServiceApi::session_active() const { return session_phase() != SessionPhase::kNone; }

void FatherServiceApi::push(const std::string& text) {
  std::function<void(Bytes)> p;
  {
    std::lock_guard lk(mu_);
    p = push_;
  }
  if (p) p(Bytes(text.begin(), text.end()));
}

void FatherServiceApi::on_event(const Event& event) {
  touch();
  if (std::holds_alternative<TierReadyEvent>(event)) {
    tier_ready_.store(true);
    job_running_.store(false);
  } else if (std::holds_alternative<FinishedEvent>(event)) {
    job_running_.store(false);
  } else if (std::holds_alternative<ErrorEvent>(event)) {
    job_running_.store(false);
    tier_ready_.store(false);  // conservative: observation resumes and tells the truth
  } else if (std::holds_alternative<ReleasedEvent>(event)) {
    job_running_.store(false);
    tier_ready_.store(false);
  }
  push(dump_safe(event_json(event)));
}

Status FatherServiceApi::rebuild_service() {
  std::shared_ptr<FatherService> old;
  SubscriptionId old_sub = 0;
  {
    std::lock_guard lk(mu_);
    old = std::move(service_);
    old_sub = subscription_;
  }
  if (old) {
    old->unsubscribe(old_sub);
    (void)old->release();
    old.reset();
  }
  job_running_.store(false);
  tier_ready_.store(false);
  const auto s = cfg_.settings->get();
  ServiceDeps deps;
  deps.catalog = cfg_.catalog;
  deps.assignment.bind(std::string(catalog::kRoleFather), cfg_.father_name);
  for (const auto& [role, fp] : s.assignments) deps.assignment.bind(role, fp);
  deps.tokenizer = cfg_.tokenizer;
  deps.readiness = cfg_.readiness;
  deps.deployments = cfg_.deployments;
  deps.options = cfg_.options;
  deps.options.default_context_tokens = s.context_tokens;
  CLM_ASSIGN_OR_RETURN(auto svc, make_father_service(std::move(deps)));
  std::shared_ptr<FatherService> shared(std::move(svc));
  if (cfg_.catalog.find(s.selected_tier)) (void)shared->select_tier(s.selected_tier);
  const auto sub = shared->subscribe([this](const Event& e) { on_event(e); });
  std::lock_guard lk(mu_);
  service_ = std::move(shared);
  subscription_ = sub;
  return Status::ok();
}

// keep-ready policy: leases are held for others' benefit too, so an idle session is released.
//   enabled  -> after release_after_idle_minutes (0 = never)
//   disabled -> after 60 s without activity
void FatherServiceApi::watchdog() {
  while (!stopping_.load()) {
    for (int i = 0; i < 20 && !stopping_.load(); ++i) std::this_thread::sleep_for(100ms);
    auto svc = service();
    if (!svc) continue;
    if (job_running_.load() || !tier_ready_.load()) {
      touch();
      continue;
    }
    const auto policy = cfg_.settings->get().keep_ready;
    std::chrono::seconds limit = 60s;
    if (policy.enabled) {
      if (policy.release_after_idle_minutes == 0) continue;
      limit = std::chrono::minutes(policy.release_after_idle_minutes);
    }
    const auto idle = std::chrono::steady_clock::now().time_since_epoch() - std::chrono::steady_clock::duration(last_activity_.load());
    if (idle >= limit) {
      log::info("keep_ready_release_idle");
      (void)svc->release();
      touch();
    }
  }
}

Result<ipc::Envelope> FatherServiceApi::handle(const ipc::Envelope& request, const ipc::PeerCredentials&) {
  json req = json::parse(request.payload.begin(), request.payload.end(), nullptr, false);
  std::string op;
  if (req.is_discarded() || !req.is_object() || !get_opt(req, "op", op))
    return make_error(ErrorCode::kProtocolError, "request must be a JSON object with an 'op'");
  touch();
  json reply = {{"id", req.value("id", json(nullptr))}};
  auto fail = [&](const Status& s) {
    reply["ok"] = false;
    reply["error"] = error_json(s);
  };
  auto succeed = [&](json result = json::object()) {
    reply["ok"] = true;
    reply["result"] = std::move(result);
  };
  auto svc = service();
  if (!svc) {
    fail(make_error(ErrorCode::kUnavailable, "service is restarting"));
  } else if (op == "hello") {
    succeed({{"api_version", kFatherIpcApiVersion}, {"dev_fixture_model", cfg_.dev_fixture_model},
             {"device_fingerprint", cfg_.identity->fingerprint()}, {"device_short", pairing::short_fingerprint(cfg_.identity->fingerprint())},
             {"settings_note", cfg_.settings->report().note}});
  } else if (op == "tiers.list") {
    std::uint32_t ctx = 0;
    get_opt(req, "context_tokens", ctx);
    json tiers = json::array();
    for (const auto& r : svc->list_tiers(ctx)) {
      json t = {{"tier_id", r.tier_id}, {"model_name", r.model_name}, {"state", std::string(catalog::to_string(r.state))},
                {"reasons", r.reasons}, {"notes", r.notes}, {"suggested_fallback", r.suggested_fallback},
                {"details", cfg_.details->get(r.tier_id)}};
      if (const auto* entry = cfg_.catalog.find(r.tier_id)) {
        t["display_name"] = entry->display_name;
        t["roles"] = entry->roles;
      }
      if (r.progress) t["progress"] = {{"percent", r.progress->percent}, {"eta_seconds", r.progress->eta_seconds ? json(*r.progress->eta_seconds) : json(nullptr)}, {"eta_is_estimate", true}};
      tiers.push_back(std::move(t));
    }
    succeed({{"tiers", tiers}, {"selected", svc->selected_tier()}});
  } else if (op == "tiers.select") {
    std::string tier;
    if (auto st = need_string(req, "tier_id", tier); !st.is_ok()) { fail(st); }
    else if (auto s2 = svc->select_tier(tier); !s2.is_ok()) fail(s2);
    else {
      (void)cfg_.settings->update([&](config::FatherSettings& s) { s.selected_tier = tier; return Status::ok(); });
      succeed({{"selected", tier}});
    }
  } else if (op == "tiers.prepare") {
    std::string tier;
    std::uint32_t ctx = cfg_.settings->get().context_tokens;
    get_opt(req, "context_tokens", ctx);
    if (auto st = need_string(req, "tier_id", tier); !st.is_ok()) fail(st);
    else {
      tier_ready_.store(false);
      job_running_.store(true);
      auto s2 = svc->prepare_tier(tier, ctx);
      if (s2.code() == ErrorCode::kFailedPrecondition && svc->wait_idle(2s)) s2 = svc->prepare_tier(tier, ctx);
      if (!s2.is_ok()) {
        job_running_.store(false);
        fail(s2);
      } else {
        succeed({{"accepted", true}});
      }
    }
  } else if (op == "chat.send") {
    ChatRequest cr;
    if (auto st = need_string(req, "message", cr.user_message); !st.is_ok()) { fail(st); }
    else {
      std::string sys;
      if (get_opt(req, "system_prompt", sys)) cr.system_prompt = sys;
      const auto s = cfg_.settings->get();
      cr.max_new_tokens = 32;
      cr.context_tokens = s.context_tokens;
      cr.q = s.advanced.default_q;
      get_opt(req, "max_new_tokens", cr.max_new_tokens);
      get_opt(req, "context_tokens", cr.context_tokens);
      get_opt(req, "q", cr.q);
      job_running_.store(true);
      // A job emits its final event slightly before it is idle; give a just-finished one a moment to settle.
      auto id = svc->chat(cr);
      if (!id.is_ok() && id.status().code() == ErrorCode::kFailedPrecondition && svc->wait_idle(2s)) id = svc->chat(cr);
      if (!id.is_ok()) {
        job_running_.store(false);
        fail(id.status());
      }
      else succeed({{"request_id", id.value()}});
    }
  } else if (op == "chat.cancel") {
    RequestId id = 0;
    if (!get_opt(req, "request_id", id)) fail(make_error(ErrorCode::kInvalidArgument, "missing 'request_id'"));
    else if (auto st = svc->cancel(id); !st.is_ok()) fail(st);
    else succeed();
  } else if (op == "session.release") {
    if (auto st = svc->release(); !st.is_ok()) fail(st);
    else {
      tier_ready_.store(false);
      succeed();
    }
  } else if (op == "conversation.reset") {
    if (auto st = svc->reset_conversation(); !st.is_ok()) fail(st);
    else succeed();
  } else if (op == "diagnostics.get") {
    bool include_text = false;
    get_opt(req, "include_text", include_text);
    const auto d = svc->diagnostics(include_text);
    json tiers = json::array();
    for (const auto& t : d.tiers) tiers.push_back({{"tier_id", t.tier_id}, {"model_name", t.model_name}, {"state", t.state}, {"headline", t.headline}});
    json out = {{"text", d.to_string()}, {"selected_tier", d.selected_tier}, {"active_tier", d.active_tier}, {"active_ready", d.active_ready},
                {"job_running", d.job_running}, {"requests_started", d.requests_started}, {"requests_completed", d.requests_completed},
                {"requests_cancelled", d.requests_cancelled}, {"requests_failed", d.requests_failed}, {"fallbacks", d.fallbacks},
                {"tokens_emitted", d.tokens_emitted}, {"conversation_messages", d.conversation_messages}, {"tiers", tiers},
                {"last_error", d.last_error}, {"text_included", include_text},
                {"paired_nodes", cfg_.settings->get().paired_nodes.size()}, {"dev_fixture_model", cfg_.dev_fixture_model},
                {"settings_load", cfg_.settings->report().note}};
    if (include_text) {
      json conv = json::array();
      for (const auto& m : d.conversation) conv.push_back({{"role", std::string(to_string(m.role))}, {"content", m.content}});
      out["conversation"] = conv;
    }
    succeed(std::move(out));
  } else if (op == "pairing.start") {
    std::string address, code;
    if (auto st = need_string(req, "address", address); !st.is_ok()) { fail(st); }
    else if (auto s2 = need_string(req, "code", code); !s2.is_ok()) fail(s2);
    else {
      std::lock_guard pl(pairing_mu_);
      auto ep = transport::Endpoint::parse(address);
      if (!ep.is_ok()) { fail(ep.status()); }
      else {
        auto peer = pairing::pair_with(ep.value(), cfg_.identity, code, {cfg_.father_name, "father", 0}, 20s);
        if (!peer.is_ok()) fail(peer.status());
        else if (peer->info.data_port == 0) fail(make_error(ErrorCode::kProtocolError, "the Node reported no data port"));
        else {
          config::PairedDevice d;
          d.fingerprint = peer->fingerprint;
          d.name = peer->info.name;
          std::string label;
          if (get_opt(req, "name", label) && !label.empty()) d.name = label;
          d.address = ep->host.find(':') != std::string::npos ? "[" + ep->host + "]:" + std::to_string(peer->info.data_port)
                                                              : ep->host + ":" + std::to_string(peer->info.data_port);
          d.role = "node";
          d.paired_at_unix = now_unix();
          if (auto ust = cfg_.settings->update([&](config::FatherSettings& s) { s.upsert_node(d); return Status::ok(); }); !ust.is_ok()) fail(ust);
          else succeed({{"device", device_json(d)}});
        }
      }
    }
  } else if (op == "pairing.list") {
    const auto s = cfg_.settings->get();
    json devices = json::array();
    for (const auto& d : s.paired_nodes) devices.push_back(device_json(d));
    succeed({{"devices", devices}, {"assignments", s.assignments}});
  } else if (op == "pairing.unpair") {
    std::string fp;
    if (auto st = need_string(req, "fingerprint", fp); !st.is_ok()) { fail(st); }
    else {
      bool removed = false;
      auto ust = cfg_.settings->update([&](config::FatherSettings& s) {
        removed = s.remove_node(fp);
        return removed ? Status::ok() : make_error(ErrorCode::kNotFound, "no such paired device");
      });
      if (!ust.is_ok()) fail(ust);
      else if (auto rb = rebuild_service(); !rb.is_ok()) fail(rb);  // releases any lease that used the device
      else succeed();
    }
  } else if (op == "assign.set") {
    if (!req.contains("assignments") || !req["assignments"].is_object()) {
      fail(make_error(ErrorCode::kInvalidArgument, "missing 'assignments' object"));
    } else if (session_active()) {
      fail(make_error(ErrorCode::kFailedPrecondition, "release the active session before changing assignments"));
    } else {
      std::map<std::string, std::string> a;
      bool ok = true;
      for (auto it = req["assignments"].begin(); it != req["assignments"].end(); ++it) {
        if (!it.value().is_string()) { ok = false; break; }
        a[it.key()] = it.value().get<std::string>();
      }
      if (!ok) fail(make_error(ErrorCode::kInvalidArgument, "assignment values must be fingerprints"));
      else if (auto st = cfg_.settings->update([&](config::FatherSettings& s) { s.assignments = a; return Status::ok(); }); !st.is_ok()) fail(st);
      else if (auto rb = rebuild_service(); !rb.is_ok()) fail(rb);
      else succeed({{"assignments", a}});
    }
  } else if (op == "model.confirm") {
    std::string tier;
    if (auto st = need_string(req, "tier_id", tier); !st.is_ok()) { fail(st); }
    else {
      const auto s = cfg_.settings->get();
      auto dir = s.model_dirs.find(tier);
      if (dir == s.model_dirs.end()) fail(make_error(ErrorCode::kFailedPrecondition, "no model directory configured for this tier"));
      else if (auto store = objects::CanonicalModelStore::open(dir->second); !store.is_ok()) fail(store.status());
      else {
        const std::string root = store.value()->manifest().root_hash().hex();
        if (auto ust = cfg_.settings->update([&](config::FatherSettings& x) { x.confirmed_model_roots[tier] = root; return Status::ok(); }); !ust.is_ok()) fail(ust);
        else succeed({{"tier_id", tier}, {"manifest_root", root}});
      }
    }
  } else if (op == "settings.get") {
    json doc = json::parse(config::to_json(cfg_.settings->get()));
    succeed(std::move(doc));
  } else if (op == "settings.set") {
    if (!req.contains("patch") || !req["patch"].is_object()) {
      fail(make_error(ErrorCode::kInvalidArgument, "missing 'patch' object"));
    } else {
      static const char* kAllowed[] = {"selected_tier", "context_tokens", "keep_ready", "model_dirs", "advanced"};
      auto st = cfg_.settings->update([&](config::FatherSettings& s) -> Status {
        json doc = json::parse(config::to_json(s));
        for (auto it = req["patch"].begin(); it != req["patch"].end(); ++it) {
          bool allowed = false;
          for (const char* k : kAllowed) allowed = allowed || it.key() == k;
          if (!allowed) return make_error(ErrorCode::kInvalidArgument, "setting '" + it.key() + "' cannot be set here");
          if (it.value().is_object() && doc[it.key()].is_object()) {
            for (auto sub = it.value().begin(); sub != it.value().end(); ++sub) {
              if (sub.value().is_null()) doc[it.key()].erase(sub.key());
              else doc[it.key()][sub.key()] = sub.value();
            }
          } else {
            doc[it.key()] = it.value();
          }
        }
        auto parsed = config::father_settings_from_json(doc.dump());
        if (!parsed.is_ok()) return parsed.status();
        s = std::move(parsed).value();
        return Status::ok();
      });
      if (!st.is_ok()) fail(st);
      else succeed(json::parse(config::to_json(cfg_.settings->get())));
    }
  } else {
    fail(make_error(ErrorCode::kUnimplemented, "unknown op"));
  }
  ipc::Envelope out;
  out.payload = to_bytes(reply);
  return out;
}

}  // namespace clusterlm::father
