#pragma once
// FatherViewModel: all behaviour and state of the Father chat UI, with no rendering and no platform code.
//
// The draw layer (father_draw.hpp) reads `snapshot()` once per frame and calls the command methods below; tests
// drive the same surface with a fake FatherClient. Rules the view-model enforces:
//   * A tier is shown Ready only when the service's own list_tiers() says Ready, never from an event alone.
//   * Only TokensEvent text ever reaches the transcript: nothing speculative can be displayed.
//   * Fallbacks, errors and model changes are written into the transcript in plain words; the conversation is kept.
//   * Diagnostics are redacted unless the user ticks "include conversation text" for that export.
//
// Threading: events arrive on any thread; state is mutex-guarded. Command methods and tick() are called from the UI
// thread and never hold the lock across a client call (a client may deliver events synchronously). tick() is what
// re-reads tier readiness and context use, so it is the only place a slow client can stall a frame.
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "clusterlm/ui/father_client.hpp"

namespace clusterlm::ui {

struct TierRow {
  std::string id;
  std::string name;   // "Fast"
  std::string model;  // catalog display name of the model this tier would run
  catalog::TierState state = catalog::TierState::kUnavailable;
  std::string state_label;  // "Unavailable" | "Available" | "Preparing 42%" | "Ready"
  std::string headline;     // the service's one-line reason
  std::vector<std::string> notes;
  std::optional<double> progress_percent;  // Preparing only
  std::string eta;                         // "about 3 min (estimate)"; empty when unknown
  std::vector<std::string> progress_lines; // Preparing only: per machine "G14: 1.2 GB of 4.5 GB (3 of 12 parts)" 
  std::vector<std::string> participants;   // machines that take part ("This PC", "G14")
  bool selected = false;
  bool can_prepare = false;                // Available, and nothing else is running
};

enum class EntryKind : std::uint8_t { kUser, kAssistant, kNotice, kError };

struct AnswerSpan {
  std::string tier_id, model;
  std::string text;
};

struct ChatEntry {
  EntryKind kind = EntryKind::kNotice;
  father::RequestId request = 0;
  std::string text;                 // user / notice / error body
  std::vector<AnswerSpan> spans;    // assistant: text per producing model, in order
  bool streaming = false;
  bool cancelled = false;
  bool continued = false;           // assistant: continues an answer after a model change (a notice sits above it)
  std::string attribution;          // assistant: "Answered by A" / "Answered by A, then B"
  std::string full_text() const;    // assistant: the spans concatenated
};

struct PrepareView {
  bool active = false;
  std::string tier_id, model;
  std::optional<double> percent;
  std::string eta;      // "... (estimate)"
  std::string message;  // what is happening, from the service
  std::vector<std::string> lines;  // per machine: bytes sent of total, parts sealed of total, or what it is doing now
};

struct StatsView {
  bool valid = false;
  std::string ttft, tok_s, accepted_per_round;  // formatted
  std::uint32_t tokens = 0, rounds = 0, fallbacks = 0;
  std::string final_model;
  std::string reason;  // "completed" | "cancelled" | "failed"
  std::string provenance;
};

struct ContextView {
  std::uint32_t context_tokens = 4096;
  std::uint32_t tokens_used = 0;
  bool exact = false;  // false: a lower bound (only generated tokens are known)
  std::string label;   // "812 / 4096 tokens" or "at least 96 / 4096 tokens"
};

struct DiagRow {
  std::string label, value, provenance;
};

struct DiagnosticsView {
  bool open = false;
  bool include_conversation = false;  // the redaction flag: false unless the user opts in for this export
  std::vector<DiagRow> rows;
  std::string plan_summary;           // "" = not reported by the service
  std::vector<StageTiming> stage_timings;
  std::string snapshot_text;          // redacted snapshot (Advanced panel preview)
  std::string last_export_path;
  std::string export_message;
};

struct Banner {
  enum class Kind : std::uint8_t { kNone, kInfo, kError } kind = Kind::kNone;
  std::string text;
};

struct PairingView {
  std::vector<PairedMachine> machines;
  std::string message;  // e.g. "Pairing is done outside this window in this version."
};

struct FatherViewState {
  std::vector<TierRow> tiers;
  std::string selected_tier;
  std::string active_tier, active_model;  // what is answering (or would answer) right now
  std::vector<ChatEntry> entries;
  bool chat_busy = false;
  PrepareView prepare;
  StatsView last_answer;
  ContextView context;
  std::vector<std::string> participants;  // for the selected tier
  Banner banner;
  DiagnosticsView diagnostics;
  FatherSettings settings;
  PairingView pairing;
  bool can_send = false;
  std::string send_hint;  // why sending is blocked, or "will prepare first"
  bool demo = false;
  std::string connection_message;  // non-empty when the Father side cannot be reached
};

// Saves an exported diagnostics text; returns where it went.
using DiagnosticsExporter = std::function<Result<std::string>(const std::string& text)>;
// Writes "clusterlm-diagnostics-<n>.txt" into `dir` (created if missing).
DiagnosticsExporter make_file_exporter(std::string dir);

class FatherViewModel {
 public:
  explicit FatherViewModel(FatherClient& client, bool demo = false);
  ~FatherViewModel();
  FatherViewModel(const FatherViewModel&) = delete;
  FatherViewModel& operator=(const FatherViewModel&) = delete;

  void set_exporter(DiagnosticsExporter e);

  FatherViewState snapshot() const;

  // ---- commands (UI thread) ----
  // Re-reads tiers/participants/context when events asked for it or `force`/the poll interval says so.
  void tick(bool force = false);
  Status select_tier(const std::string& tier_id);
  Status prepare_tier(const std::string& tier_id);
  Status send(const std::string& text);
  Status cancel();
  Status release();
  Status new_conversation();
  Status apply_settings(const FatherSettings& s);
  void set_diagnostics_open(bool open);
  void set_include_conversation(bool include);
  void refresh_diagnostics();
  Status export_diagnostics();
  Status start_pairing(const PairingRequest& request);
  Status unpair(const std::string& machine_id);
  void dismiss_banner();

  // Delivered by the client subscription; public so tests can inject events without a client emitting them.
  void on_event(const father::Event& e);

 private:
  struct Gate;
  void set_banner(Banner::Kind k, std::string text);  // under mu_
  void rebuild_attribution(ChatEntry& e) const;  // under mu_
  std::string tier_label(const std::string& tier_id, const std::string& model) const;  // under mu_
  ChatEntry& assistant_entry(father::RequestId id);   // under mu_
  void recompute_send_state();                        // under mu_
  void reload_tiers();
  void reload_context();

  FatherClient& client_;
  std::shared_ptr<Gate> gate_;
  father::SubscriptionId sub_ = 0;

  mutable std::mutex mu_;
  FatherViewState st_;
  DiagnosticsExporter exporter_;
  father::RequestId current_request_ = 0;
  std::set<father::RequestId> finished_;
  std::uint32_t generated_tokens_ = 0;  // lower bound for context when the client cannot tell
  bool refresh_pending_ = true;
  bool context_pending_ = true;
  bool tiers_loaded_ = false;
  std::chrono::steady_clock::time_point last_poll_{};
  bool active_from_answer_ = false;  // active model comes from what actually answered, not from the selection
};

}  // namespace clusterlm::ui
