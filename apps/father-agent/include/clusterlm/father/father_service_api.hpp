#pragma once
// FatherServiceApi: the real FatherApiHandler. It maps the Father UI IPC (docs/father-ipc.md) onto FatherService,
// the persisted settings and the pairing client.
//
// Wire: every kFatherRequest payload is one UTF-8 JSON object {"op": "...", "id": <number>, ...args}; the
// kFatherReply payload is {"id", "ok": true, "result": {...}} or {"id", "ok": false, "error": {"code", "message"}}.
// A payload that is not such an object is a protocol error (the agent answers with an Ack, no detail). Events
// (prepare progress, streamed tokens, fallbacks, finish, errors) are pushed as kFatherEvent payloads
// {"event": "...", ...} through the push callback (FatherAgent::broadcast) on the same pipe connections.
//
// Privacy: requests and token events carry the user's text (Father-local IPC between the user's own processes);
// nothing in this file logs it. Diagnostics are redacted unless the caller asks for the conversation text.
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "clusterlm/config/store.hpp"
#include "clusterlm/father/father_agent.hpp"
#include "clusterlm/father/node_notify.hpp"
#include "clusterlm/father/production.hpp"
#include "clusterlm/father/service.hpp"

namespace clusterlm::father {

inline constexpr int kFatherIpcApiVersion = 1;

struct FatherApiConfig {
  catalog::Catalog catalog;
  std::shared_ptr<config::FatherSettingsStore> settings;
  std::shared_ptr<const transport::DeviceIdentity> identity;
  std::shared_ptr<TierTokenizerProvider> tokenizers;  // per-tier real tokenizers (production); overrides `tokenizer`
  std::shared_ptr<Tokenizer> tokenizer;  // null: chat fails with "no tokenizer available in this build"
  std::shared_ptr<ReadinessSource> readiness;
  std::shared_ptr<DeploymentProvider> deployments;
  std::shared_ptr<DetailsBoard> details = std::make_shared<DetailsBoard>();
  std::string father_name = "Father";
  bool dev_fixture_model = false;  // reported to the UI so it can label the development override
  // Tells a just-unpaired Node to drop trust in this Father (docs/pairing.md). Null = the real notifier over mutual
  // TLS (notify_node_unpaired). Tests inject a recorder.
  std::function<UnpairNotifyResult(const config::PairedDevice&)> notify_unpair;
  PrepareObserver prepare_observer;  // Coordinator preparation progress (ProvisioningBoard::observer() in production)
  ServiceOptions options;
};

class FatherServiceApi final : public FatherApiHandler {
 public:
  static Result<std::unique_ptr<FatherServiceApi>> create(FatherApiConfig config);
  ~FatherServiceApi() override;

  // Where events go (FatherAgent::broadcast). Set before the first request.
  void set_event_push(std::function<void(Bytes)> push);
  bool session_active() const;
  SessionPhase session_phase() const;

  Result<ipc::Envelope> handle(const ipc::Envelope& request, const ipc::PeerCredentials& peer) override;

 private:
  explicit FatherServiceApi(FatherApiConfig config) : cfg_(std::move(config)) {}
  Status rebuild_service();  // after assignment changes; releases the old service first
  std::shared_ptr<FatherService> service() const;
  void push(const std::string& json_text);
  void on_event(const Event& event);
  void watchdog();
  void touch() { last_activity_.store(std::chrono::steady_clock::now().time_since_epoch().count()); }

  FatherApiConfig cfg_;
  mutable std::mutex mu_;
  std::shared_ptr<FatherService> service_;
  SubscriptionId subscription_ = 0;
  std::function<void(Bytes)> push_;
  std::mutex pairing_mu_;  // one pairing operation at a time
  std::atomic<std::int64_t> last_activity_{0};
  // Session state tracked from the service's own events (diagnostics() re-observes readiness, so it must not be
  // used to answer "is a session active?" from inside an observation).
  std::atomic<bool> job_running_{false};
  std::atomic<bool> tier_ready_{false};
  std::atomic<bool> stopping_{false};
  std::thread watchdog_;
};

}  // namespace clusterlm::father
