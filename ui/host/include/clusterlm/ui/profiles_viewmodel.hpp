#pragma once
// ProfilesViewModel: execution profiles (list, readiness, editor, dry-run placement, export/import).
//
// Rules: the readiness shown is the Host's seven-state answer, never derived here ("ready" appears only when the Host
// says ready or busy); when the Host cannot be reached the state is "Status unknown", not the last known one; a dry run
// is shown with the provenance the Host gave and, when any number is Synthetic, the fixed sentence that it is an
// estimate and not a benchmark; layer numbers and GGUF details never appear.
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/ui/host_admin.hpp"
#include "clusterlm/ui/host_common.hpp"

namespace clusterlm::ui {

struct ProfileRow {
  std::string id, name;
  std::string subtitle;      // "Qwen3 30B, IQ3_S - runs on: This PC + G14"
  ReadyState state = ReadyState::kUnavailable;
  std::string state_label;   // describe_ready_state, or "Status unknown"
  std::string headline;      // first reason, or empty when ready
  std::vector<std::string> reasons;
  std::optional<double> percent;
  std::string eta;           // "about 3 min (estimate)" or empty
  std::string last_error;
  std::string machines_text; // "This PC, G14" for filled slots, "Waiting for: Worker slot w1" otherwise
  std::string exposure_text; // "Not available to other apps" | "Apps on this PC can use it as 'x'" | "... and other PCs"
  std::vector<std::string> goals;  // always carries the "pending qualification" wording
  bool is_example = false;
  bool selected = false;
  bool can_prepare = false, can_release = false, can_edit = true;
  std::string edit_hint;     // why can_edit is false
  bool stale = false;
};

struct EditorState {
  bool open = false;
  std::string editing_id;    // empty = creating
  ProfileDraft draft;
  std::vector<Finding> findings;
  bool can_save = false;
  std::string save_error;
};

struct DryRunLine {
  std::string machine;
  std::string cpu, gpu, state, scratch;  // formatted bytes
};
struct DryRunView {
  bool shown = false;
  std::string profile_id, profile_name;
  bool feasible = false;
  std::string headline;       // "This should fit" / "This will not fit"
  std::vector<DryRunLine> lines;
  std::string preparation;    // "Data sent to other PCs: about 11.2 GB"
  std::vector<std::string> bottlenecks;
  std::vector<std::string> quantities;  // "Predicted speed: 14.0 tok/s (Synthetic: model of the hardware, not a benchmark)"
  std::string provenance_note;          // non-empty whenever any number is Synthetic
  std::string plan_summary;
  std::uint32_t context_tokens = 0;
};

struct ImportState {
  bool conflict = false;      // same id, different content: waiting for Replace / Keep both
  std::vector<std::string> todo;
};

struct ProfilesState {
  bool supported = true;
  std::string unsupported_text;
  bool stale = false;
  std::vector<ProfileRow> rows;
  std::string selected;
  std::vector<std::pair<std::string, std::string>> model_choices;    // library id, label
  std::vector<std::pair<std::string, std::string>> backend_choices;  // backend id, label
  std::vector<std::pair<std::string, std::string>> binding_choices;  // binding name, label
  EditorState editor;
  DryRunView dry_run;
  std::string export_text;
  std::string export_name;
  ImportState import;
  Notice notice;
  std::uint32_t context_tokens = 4096;  // the context class used for readiness and dry runs
};

inline constexpr const char* kSyntheticNote =
    "These numbers are an estimate from a model of your hardware. Nothing was measured, and they are not a benchmark.";

class ProfilesViewModel {
 public:
  explicit ProfilesViewModel(HostAdminClient& client) : client_(client) {}
  const ProfilesState& state() const { return st_; }

  void refresh();
  void set_context(std::uint32_t tokens);
  void select(const std::string& id);
  Status prepare(const std::string& id);
  Status release(const std::string& id);
  Status duplicate(const std::string& id);
  Status remove(const std::string& id, bool i_confirm);
  Status run_dry_run(const std::string& id);
  void close_dry_run() { st_.dry_run = {}; }

  // editor
  void begin_create();
  Status begin_edit(const std::string& id);
  ProfileDraft& draft() { return st_.editor.draft; }   // the draw layer edits this in place, then calls revalidate()
  void revalidate();
  Status save();
  void cancel_edit() { st_.editor = {}; }

  // export / import (text goes to the clipboard or a file chosen by the shell; paths never travel in it)
  Status export_profile(const std::string& id);
  Status import_text(const std::string& text, ImportChoice choice = ImportChoice::kAsk);
  Status resolve_import(ImportChoice choice);  // Replace or Duplicate after a conflict
  void dismiss_notice() { st_.notice = {}; }

 private:
  void fail(const Status& s);
  void ok(std::string text);
  const ProfileView* find(const std::string& id) const;
  HostAdminClient& client_;
  ProfilesState st_;
  std::vector<ProfileView> profiles_;
  std::vector<ModelView> models_;
  std::string pending_import_;
};

}  // namespace clusterlm::ui
