#pragma once
// FatherClient: everything the Father UI may ask of the Father side, as an abstract interface.
//
// The view-models depend on this interface only. Implementations:
//   * InProcessFatherClient (in_process_father_client.hpp): wraps a father::FatherService; tests + dev mode.
//   * IpcFatherClient (ipc_father_client.hpp): the seam for the Father agent pipe. Its wire layout belongs to the
//     Father agent workstream (WP14); until that lands every call fails with kUnavailable/kUnimplemented and the UI
//     says so in words.
//   * ScriptedFatherClient (scripted_father_client.hpp): a controllable fake (unit tests, `--demo` mode).
//
// Events are the father::Event variants. Only TokensEvent ever carries answer text: the UI never displays
// anything speculative. Events may arrive on any thread; sinks must not block.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/catalog/readiness.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/father/service.hpp"

namespace clusterlm::ui {

// A machine that takes part in a tier (from the user's tier assignment). State is not known at this layer.
struct TierParticipant {
  std::string role;        // catalog role ("father", "node:laptop-class", ...)
  std::string machine_id;  // user-facing machine name
};

struct PairedMachine {
  std::string machine_id;   // user-facing name
  std::string role_hint;    // "Father", "Node", ...
  std::string fingerprint;  // short device-id fingerprint, may be empty
};

struct FatherSettings {
  std::uint32_t context_tokens = 4096;  // a catalog context profile
  std::uint32_t max_new_tokens = 256;
  std::string system_prompt;            // applied only to an empty conversation
};
Status validate(const FatherSettings& s);

// How many tokens the conversation occupies. `exact` false means a lower bound (only generated tokens known).
struct ContextUse {
  std::uint32_t tokens_used = 0;
  bool exact = false;
};

struct StageTiming {
  std::string stage;  // e.g. "G14 layers 4-9"
  double ms = 0;
};

struct DiagnosticsExport {
  std::string text;                        // the redacted snapshot, ready to save
  bool includes_conversation = false;      // true only if the caller asked for text
  std::string plan_summary;                // empty when the service does not report a plan
  std::vector<StageTiming> stage_timings;  // empty when the service does not report them
};

class FatherClient {
 public:
  virtual ~FatherClient() = default;

  virtual Result<std::vector<catalog::TierReadiness>> list_tiers(std::uint32_t context_tokens) = 0;
  virtual Result<std::vector<TierParticipant>> participants(std::string_view tier_id) = 0;
  virtual std::string selected_tier() = 0;
  virtual Status select_tier(std::string_view tier_id) = 0;
  virtual Status prepare_tier(std::string_view tier_id, std::uint32_t context_tokens) = 0;
  virtual Result<father::RequestId> send_chat(father::ChatRequest request) = 0;
  virtual Status cancel(father::RequestId request) = 0;
  virtual Status release() = 0;
  virtual Status reset_conversation() = 0;
  virtual Result<ContextUse> context_use() = 0;
  virtual Result<DiagnosticsExport> export_diagnostics(bool include_text) = 0;

  virtual Result<FatherSettings> get_settings() = 0;
  virtual Status set_settings(const FatherSettings& settings) = 0;

  // Pairing may be Unimplemented; the UI then explains that pairing is done elsewhere.
  virtual Result<std::vector<PairedMachine>> paired_machines() = 0;
  virtual Status start_pairing() = 0;
  virtual Status unpair(std::string_view machine_id) = 0;

  virtual father::SubscriptionId subscribe(father::EventSink sink) = 0;
  virtual void unsubscribe(father::SubscriptionId id) = 0;

  // Label for GenerationStats shown in diagnostics: "Measured" (observed on this run) or "Synthetic" (scripted).
  // Never "Qualified": qualification is a bench-suite verdict, not something a UI can assert.
  virtual std::string_view stats_provenance() const = 0;
};

}  // namespace clusterlm::ui
