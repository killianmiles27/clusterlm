#pragma once
// Concrete FatherClient / NodeClient implementations.
//
//   InProcessFatherClient  wraps a father::FatherService living in this process (tests, dev mode).
//   IpcFatherClient        the documented SEAM for the Father agent pipe (see below). Not functional yet.
//   ScriptedFatherClient   controllable fake: unit tests drive events by hand, `--demo` installs a scripted run.
//   IpcNodeClient          talks to the Node service over the existing helper pipe messages.
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "clusterlm/father/tokenizer.hpp"
#include "clusterlm/platform/ipc.hpp"
#include "clusterlm/ui/father_client.hpp"
#include "clusterlm/ui/node_client.hpp"

namespace clusterlm::ui {

// ---- in-process --------------------------------------------------------------------------------------------

class InProcessFatherClient final : public FatherClient {
 public:
  struct Options {
    catalog::Catalog catalog;           // for participants(); may be default-constructed (no participants)
    catalog::TierAssignment assignment; // role -> paired machine
    std::shared_ptr<father::Tokenizer> tokenizer;  // optional: enables exact context_use()
    std::vector<PairedMachine> paired;  // what paired_machines() reports (the service has no pairing store)
  };
  // `service` must outlive the client. Settings live in memory (the service itself has no settings store).
  InProcessFatherClient(father::FatherService& service, Options options);

  Result<std::vector<catalog::TierReadiness>> list_tiers(std::uint32_t context_tokens) override;
  Result<std::vector<TierParticipant>> participants(std::string_view tier_id) override;
  std::string selected_tier() override;
  Status select_tier(std::string_view tier_id) override;
  Status prepare_tier(std::string_view tier_id, std::uint32_t context_tokens) override;
  Result<father::RequestId> send_chat(father::ChatRequest request) override;
  Status cancel(father::RequestId request) override;
  Status release() override;
  Status reset_conversation() override;
  Result<ContextUse> context_use() override;
  Result<DiagnosticsExport> export_diagnostics(bool include_text) override;
  Result<FatherSettings> get_settings() override;
  Status set_settings(const FatherSettings& settings) override;
  Result<std::vector<PairedMachine>> paired_machines() override;
  Status start_pairing(const PairingRequest& request) override;
  Status unpair(std::string_view machine_id) override;
  father::SubscriptionId subscribe(father::EventSink sink) override { return svc_.subscribe(std::move(sink)); }
  void unsubscribe(father::SubscriptionId id) override { svc_.unsubscribe(id); }
  std::string_view stats_provenance() const override { return "Measured"; }

 private:
  father::FatherService& svc_;
  Options opts_;
  std::mutex mu_;
  FatherSettings settings_;
};

// ---- Father agent over IPC -----------------------------------------------------------------------------------

// Client of the Father agent's local pipe API (docs/father-ipc.md, FatherServiceApi): JSON requests in
// kFatherRequest envelopes, replies matched by id, events pushed as kFatherEvent. One reader thread demultiplexes
// replies and events; calls block with a timeout. The connection is made lazily and re-made after loss: an absent
// agent is reported as kUnavailable in words, never as a crash. Token IDs never arrive (the agent sends only
// counts), so TokensEvent::tokens holds placeholders of the right length.
class IpcFatherClient final : public FatherClient {
 public:
  struct Options {
    ipc::Endpoint endpoint;  // name = ipc::father_ui_pipe_name(tag); socket_dir for POSIX
    ipc::ClientOptions client;
    std::chrono::milliseconds connect_timeout{1000};
    std::chrono::milliseconds request_timeout{15000};
    std::chrono::milliseconds pairing_timeout{25000};  // the agent blocks up to 20 s on pairing
  };
  explicit IpcFatherClient(Options options);
  ~IpcFatherClient() override;
  static std::string_view not_running_message();

  Result<std::vector<catalog::TierReadiness>> list_tiers(std::uint32_t context_tokens) override;
  Result<std::vector<TierParticipant>> participants(std::string_view tier_id) override;
  std::string selected_tier() override;
  Status select_tier(std::string_view tier_id) override;
  Status prepare_tier(std::string_view tier_id, std::uint32_t context_tokens) override;
  Result<father::RequestId> send_chat(father::ChatRequest request) override;
  Status cancel(father::RequestId request) override;
  Status release() override;
  Status reset_conversation() override;
  Result<ContextUse> context_use() override;
  Result<DiagnosticsExport> export_diagnostics(bool include_text) override;
  Result<FatherSettings> get_settings() override;
  Status set_settings(const FatherSettings& settings) override;
  Result<std::vector<PairedMachine>> paired_machines() override;
  Status start_pairing(const PairingRequest& request) override;
  Status unpair(std::string_view machine_id) override;
  father::SubscriptionId subscribe(father::EventSink sink) override;
  void unsubscribe(father::SubscriptionId id) override;
  // "Synthetic" when the agent runs the development fixture model (its numbers are not a measurement of anything).
  std::string_view stats_provenance() const override { return dev_fixture_.load() ? "Synthetic" : "Measured"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::atomic<bool> dev_fixture_{false};
};

// ---- scripted ----------------------------------------------------------------------------------------------

class ScriptedFatherClient : public FatherClient {
 public:
  explicit ScriptedFatherClient(std::string provenance = "Synthetic") : provenance_(std::move(provenance)) {}
  ~ScriptedFatherClient() override;

  // -- test control
  void set_tiers(std::vector<catalog::TierReadiness> tiers);
  void set_participants(std::string tier_id, std::vector<TierParticipant> p);
  void set_context_use(std::optional<ContextUse> use);
  void set_paired(std::vector<PairedMachine> paired);
  void set_diagnostics(DiagnosticsExport d);
  // The next call of `method` (the FatherClient method name, e.g. "send_chat") fails with this status.
  void fail_next(const std::string& method, Status status);
  // Delivers an event to every subscriber, in the caller's thread.
  void emit(const father::Event& event);
  std::vector<std::string> calls() const;   // "method" or "method:arg" in call order
  std::vector<father::ChatRequest> chats() const;
  std::optional<bool> last_export_include_text() const;
  // Hooks run on the caller's thread right after the call is recorded and accepted.
  std::function<void(ScriptedFatherClient&, father::RequestId, const father::ChatRequest&)> on_chat;
  std::function<void(ScriptedFatherClient&, const std::string& tier, std::uint32_t ctx)> on_prepare;
  std::function<void(ScriptedFatherClient&, father::RequestId)> on_cancel;
  // Keeps a thread joined at destruction (used by the demo hooks). Returns false once stopping.
  void spawn(std::function<void(const std::atomic<bool>& stop)> body);
  bool stopping() const { return stop_.load(); }
  // Updates one tier's state in place (convenience for scripted flows).
  void patch_tier(const std::string& tier_id, const std::function<void(catalog::TierReadiness&)>& fn);

  Result<std::vector<catalog::TierReadiness>> list_tiers(std::uint32_t context_tokens) override;
  Result<std::vector<TierParticipant>> participants(std::string_view tier_id) override;
  std::string selected_tier() override;
  Status select_tier(std::string_view tier_id) override;
  Status prepare_tier(std::string_view tier_id, std::uint32_t context_tokens) override;
  Result<father::RequestId> send_chat(father::ChatRequest request) override;
  Status cancel(father::RequestId request) override;
  Status release() override;
  Status reset_conversation() override;
  Result<ContextUse> context_use() override;
  Result<DiagnosticsExport> export_diagnostics(bool include_text) override;
  Result<FatherSettings> get_settings() override;
  Status set_settings(const FatherSettings& settings) override;
  Result<std::vector<PairedMachine>> paired_machines() override;
  Status start_pairing(const PairingRequest& request) override;
  Status unpair(std::string_view machine_id) override;
  father::SubscriptionId subscribe(father::EventSink sink) override;
  void unsubscribe(father::SubscriptionId id) override;
  std::string_view stats_provenance() const override { return provenance_; }

 private:
  Status take_failure(const std::string& method);  // under mu_
  void record(std::string call);                   // under mu_

  std::string provenance_;
  mutable std::mutex mu_;
  std::vector<catalog::TierReadiness> tiers_;
  std::map<std::string, std::vector<TierParticipant>> participants_;
  std::string selected_;
  std::optional<ContextUse> context_use_;
  std::vector<PairedMachine> paired_;
  DiagnosticsExport diag_;
  std::optional<bool> last_export_include_text_;
  FatherSettings settings_;
  std::map<std::string, Status> failures_;
  std::vector<std::string> calls_;
  std::vector<father::ChatRequest> chats_;
  father::RequestId next_request_ = 1;
  std::map<father::SubscriptionId, father::EventSink> sinks_;
  father::SubscriptionId next_sub_ = 1;
  std::atomic<bool> stop_{false};
  std::vector<std::thread> threads_;
};

// `--demo`: a scripted client whose data is labelled Synthetic everywhere: prepare animates progress, chat streams a
// canned answer, and a message containing "fallback" triggers a scripted downgrade. No model runs.
std::unique_ptr<ScriptedFatherClient> make_demo_father_client();

// ---- Node over the helper pipe ----------------------------------------------------------------------------

class IpcNodeClient final : public NodeClient {
 public:
  struct Options {
    ipc::Endpoint endpoint{ipc::kNodeHelperPipeName, {}};
    ipc::ClientOptions client;
    std::chrono::milliseconds connect_timeout{300};
    std::chrono::milliseconds io_timeout{2000};
  };
  explicit IpcNodeClient(Options options) : opts_(std::move(options)) {}
  ~IpcNodeClient() override;

  // An unreachable service is NOT an error: it is a state the UI shows (kUnreachable + detail).
  Result<NodeStatus> status() override;
  Status pause() override;
  Status resume() override;
  // Settings are read and saved through the service (SettingsRequest / SettingsUpdate on the helper pipe): the
  // service validates, persists and applies them; a save returns only after it did. set_settings keeps the service
  // fields this window does not edit (idle time) by re-reading them first.
  Result<NodeSettings> get_settings() override;
  Status set_settings(const NodeSettings&) override;
  Result<NodePairingInfo> enter_pairing_mode() override;

 private:
  Result<ipc::Envelope> call(const ipc::Envelope& request);
  Status expect_ack(const ipc::Envelope& reply);
  Result<ipc::NodeSettingsView> read_view();

  Options opts_;
  std::mutex mu_;
  std::unique_ptr<ipc::Connection> conn_;
};

}  // namespace clusterlm::ui
