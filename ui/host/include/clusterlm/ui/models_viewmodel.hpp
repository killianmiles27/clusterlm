#pragma once
// ModelsViewModel: the model library as an ordinary user sees it. Single-threaded (UI thread); every client call is
// made without holding any lock because there is none.
//
// Rules: compatibility labels come from the Host and are printed verbatim (never computed or softened here); a model
// whose compatibility has not been checked says so; a record is never shown "verified" unless the Host reports a
// verified root; removing a record never deletes files and is refused while a profile uses it.
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/ui/host_admin.hpp"
#include "clusterlm/ui/host_common.hpp"

namespace clusterlm::ui {

struct CompatRow {
  std::string backend;       // "llama.cpp on this PC"
  std::string label;         // exact spec wording, or "Not checked yet"
  CompatLabel kind = CompatLabel::kUnsupported;
  bool checked = false;
  std::vector<std::string> notes;  // why; plus "backend not installed" when the build/runtime is missing
  std::string reach;               // "This PC only" | "Up to 2 other PCs"
};

struct ModelRow {
  std::string id;
  std::string title;      // name
  std::string subtitle;   // "Qwen3 30B-A3B - IQ3_S - 12.4 GB - 3 files"
  std::string quant_note; // "Mixed precision: Q4_K, Q6_K" or empty for a single type
  std::string integrity;  // "Not verified yet" | "Verified" | "Confirmed by you" | "Verified, but differs from what you confirmed"
  bool verified = false;
  std::string location;   // Host-local folder; owner-only display
  std::vector<std::string> facts;  // "Trained for up to 32768 tokens", "48 layers", "128 experts"
  std::vector<CompatRow> compat;
  std::vector<std::string> used_by;
  bool can_remove = false;
  std::string remove_hint;  // why not
  bool can_confirm = false; // Host reports a verified root that the user has not confirmed yet
  std::string root_short;   // first 12 hex of the verified root
};

struct ModelsState {
  bool supported = true;     // false: this Host has no model library yet
  std::string unsupported_text;
  bool stale = false;        // the last refresh failed; rows are from before
  std::vector<ModelRow> rows;
  std::vector<std::string> scan_roots;
  std::string selected;
  std::string last_scan;     // "Found 3 models. 1 file was skipped."
  std::vector<ScanIssue> scan_issues;
  Notice notice;
};

class ModelsViewModel {
 public:
  explicit ModelsViewModel(HostAdminClient& client) : client_(client) {}
  const ModelsState& state() const { return st_; }

  void refresh();
  void select(const std::string& id);
  Status add_scan_root(const std::string& dir);
  Status remove_scan_root(const std::string& dir);
  Status rescan();
  Status import_path(const std::string& path);
  // Confirms the Host-verified root of this record as the one the user expects. `i_confirm` must be true: the draw
  // layer sets it only from an explicit checkbox/button, never as a default.
  Status confirm_verified_root(const std::string& id, bool i_confirm);
  Status remove(const std::string& id);
  void dismiss_notice() { st_.notice = {}; }

 private:
  void fail(const Status& s);
  void ok(std::string text);
  HostAdminClient& client_;
  ModelsState st_;
  std::vector<ModelView> models_;
};

}  // namespace clusterlm::ui
