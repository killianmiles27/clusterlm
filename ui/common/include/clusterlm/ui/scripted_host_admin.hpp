#pragma once
// ScriptedHostAdminClient: an in-memory HostAdminClient for unit tests and `--demo`. It implements the semantics a
// real Host would (ids, revisions, conflicts, one-time key secrets, a dry run that is always Synthetic) over data the
// test sets. It is NOT the Host: nothing it returns is a measurement, and `summary().dev_fixture` is true so every
// view says so.
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "clusterlm/library/model_record.hpp"
#include "clusterlm/ui/host_admin.hpp"

namespace clusterlm::ui {

// A library record as the Models view shows it. `compat` is whatever the Host's backend descriptors computed: with
// none given the view says compatibility has not been checked, it never invents a label.
ModelView model_view_from_record(const library::ModelRecord& r, std::vector<ModelCompat> compat = {},
                                 std::vector<std::string> used_by = {});

class ScriptedHostAdminClient final : public HostAdminClient {
 public:
  ScriptedHostAdminClient();

  // ---- test control (all thread-safe)
  void set_capabilities(HostCapabilities c);
  void set_models(std::vector<ModelView> m);
  void set_profiles(std::vector<ProfileView> p);
  void set_machines(std::vector<MachineView> m);
  void set_bindings(std::vector<BindingView> b);
  void set_runs(std::vector<RunStat> r);
  void set_server(ServerStatus s);
  void set_dry_run(DryRunReport r);  // returned for every dry_run call
  void set_scan_result(ScanReport r, std::vector<ModelView> found);  // the next rescan() adds `found`
  void fail_next(const std::string& method, Status status);
  std::vector<std::string> calls() const;  // "method" or "method:arg", in call order
  // Seeds the three example profiles (migrated tiers) as the Host would after migration: model missing, so every one
  // is `unavailable` with a blocker, and no API exposure.
  void seed_example_profiles();
  // A fuller, clearly synthetic installation for `--demo`.
  void seed_demo();

  Result<HostSummary> summary() override;
  HostCapabilities capabilities() override;
  Result<std::vector<ModelView>> list_models() override;
  Result<std::vector<std::string>> scan_roots() override;
  Status add_scan_root(std::string_view dir) override;
  Status remove_scan_root(std::string_view dir) override;
  Result<ScanReport> rescan() override;
  Result<ModelView> import_model(std::string_view path) override;
  Status pin_model_root(std::string_view model_id, std::string_view root_hash) override;
  Status remove_model(std::string_view model_id) override;
  Result<std::vector<ProfileView>> list_profiles(std::uint32_t context_tokens) override;
  Result<std::string> create_profile(const ProfileDraft& d) override;
  Status update_profile(std::string_view id, const ProfileDraft& d) override;
  Status delete_profile(std::string_view id) override;
  Result<std::string> duplicate_profile(std::string_view id) override;
  Result<std::string> export_profile(std::string_view id) override;
  Result<ImportResult> import_profile(std::string_view document, ImportChoice choice) override;
  Result<std::vector<Finding>> validate_profile(std::string_view id) override;
  Result<std::vector<Finding>> validate_draft(const ProfileDraft& d) override;
  Result<DryRunReport> dry_run(std::string_view id, std::uint32_t context_tokens) override;
  Status select_profile(std::string_view id) override;
  Status prepare_profile(std::string_view id, std::uint32_t context_tokens) override;
  Status release_profile(std::string_view id) override;
  Result<std::vector<MachineView>> list_machines() override;
  Result<std::vector<BindingView>> list_bindings() override;
  Status set_binding(std::string_view name, std::string_view machine_id) override;
  Result<ServerStatus> server_status() override;
  Status set_server(const ServerSettings& s) override;
  Result<std::vector<ApiKeyView>> list_keys() override;
  Result<NewKey> create_key(std::string_view name, const std::vector<std::string>& scopes,
                            const std::vector<std::string>& allowed_models, bool lan_allowed) override;
  Status revoke_key(std::string_view key_id) override;
  Result<std::vector<RunStat>> recent_runs() override;
  Result<std::string> export_backup() override;
  Status import_backup(std::string_view document) override;

 private:
  Status enter(const std::string& call, bool needs_cap);  // records the call; consumes fail_next; checks capability
  ProfileView* find(std::string_view id);
  std::string next_id(std::string_view prefix);

  mutable std::mutex mu_;
  HostCapabilities caps_{true, true, true, true, true, true, true};
  std::vector<ModelView> models_;
  std::vector<std::string> roots_;
  std::vector<ProfileView> profiles_;
  std::vector<MachineView> machines_;
  std::vector<BindingView> bindings_;
  std::vector<RunStat> runs_;
  ServerStatus server_;
  std::vector<ApiKeyView> keys_;
  DryRunReport dry_;
  ScanReport scan_report_;
  std::vector<ModelView> scan_found_;
  std::map<std::string, Status> failures_;
  std::vector<std::string> calls_;
  std::uint32_t counter_ = 0;
  std::string selected_;
};

}  // namespace clusterlm::ui
