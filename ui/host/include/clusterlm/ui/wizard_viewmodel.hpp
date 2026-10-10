#pragma once
// WizardViewModel: the first-run guide. Host or Worker -> explain -> detect -> pair or skip -> import/choose a model ->
// suggest a profile -> optional app access -> validate -> open Chat. No GGUF, layer or quantization knowledge needed.
//
// Rules: nothing is turned on by default (app access starts off and, when chosen, is this-PC-only with a chat-only
// key); every step can be left and the wizard can be skipped at any time; the wizard creates at most one profile and
// edits that same one when the user goes back; the last step reports the Host's own readiness word for word and says
// plainly when the profile is not prepared yet. It never claims anything works that the Host did not report.
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/ui/host_admin.hpp"
#include "clusterlm/ui/host_common.hpp"
#include "clusterlm/ui/machines_viewmodel.hpp"

namespace clusterlm::ui {

enum class WizardStep : std::uint8_t { kRole, kExplain, kDetect, kPair, kModel, kProfile, kApi, kValidate, kDone };
std::string_view step_title(WizardStep s);

enum class WizardRole : std::uint8_t { kNone, kHost, kWorker };

struct WizardState {
  WizardStep step = WizardStep::kRole;
  WizardRole role = WizardRole::kNone;
  int step_number = 1, step_count = 1;
  std::vector<std::string> paragraphs;   // what to read on this step
  bool can_next = false;
  std::string next_hint;                 // why Next is disabled
  std::string next_label = "Next";
  bool can_back = false;
  // detect
  std::vector<std::string> host_facts;
  // pair
  std::vector<std::string> paired;       // names
  // model
  std::vector<std::pair<std::string, std::string>> model_choices;  // id, label
  std::string model_id;
  std::vector<std::string> model_notes;  // compatibility result for the chosen model
  // profile
  ProfileDraft draft;
  bool draft_ok = false;
  std::vector<std::string> draft_notes;
  // api
  bool api_enabled = false;
  std::string api_model_id;
  std::string key_secret;                // shown once on the Api step
  // validate / done
  std::vector<std::string> validation;
  std::string readiness_text;
  std::string created_profile_id;
  bool finished = false;
  Notice notice;
};

class WizardViewModel {
 public:
  WizardViewModel(HostAdminClient& client, PairingPort* pairing) : client_(client), pairing_(pairing) {}
  const WizardState& state() const { return st_; }
  PairForm& pair_form() { return pair_; }

  void start();   // loads what the Host knows
  void choose_role(WizardRole r);
  void next();
  void back();
  Status pair();                      // pair step
  Status import_model_path(const std::string& path);
  Status scan_folder(const std::string& dir);
  void choose_model(const std::string& id);
  void set_profile_name(const std::string& name);
  void set_api_enabled(bool on);
  void dismiss_secret();
  void dismiss_notice() { st_.notice = {}; }
  void skip();                        // leave the wizard without finishing

 private:
  void enter(WizardStep s);
  void load_models();
  void suggest();
  void recompute();
  Status create_or_update_profile();
  Status apply_api();
  void validate();

  HostAdminClient& client_;
  PairingPort* pairing_;
  WizardState st_;
  PairForm pair_;
  std::vector<ModelView> models_;
};

}  // namespace clusterlm::ui
