#pragma once
// AppViewModel: the Host window as a whole. Owns the page view-models, the current page, first-run state and the
// refresh schedule (the visible page only; hidden pages are not polled). The Chat page stays the existing
// FatherViewModel, owned by the shell; this class only decides which page is showing.
//
// Single-threaded: every method is called from the UI thread.
#include <chrono>
#include <memory>
#include <optional>

#include "clusterlm/ui/connections_viewmodel.hpp"
#include "clusterlm/ui/machines_viewmodel.hpp"
#include "clusterlm/ui/models_viewmodel.hpp"
#include "clusterlm/ui/performance_viewmodel.hpp"
#include "clusterlm/ui/profiles_viewmodel.hpp"
#include "clusterlm/ui/settings_viewmodel.hpp"
#include "clusterlm/ui/wizard_viewmodel.hpp"

namespace clusterlm::ui {

struct AppDeps {
  HostAdminClient* admin = nullptr;   // required
  PairingPort* pairing = nullptr;     // optional: without it pairing says "not available here"
  std::filesystem::path prefs_file;   // UiPrefsStore location
  FileSink files;
  bool demo = false;
};

class AppViewModel {
 public:
  explicit AppViewModel(AppDeps deps);

  Page page() const { return page_; }
  bool wizard_active() const { return wizard_active_; }
  bool demo() const { return demo_; }
  void go(Page p);
  // Ctrl+1 .. Ctrl+7 map to pages in nav order.
  static std::optional<Page> page_for_shortcut(int digit);

  // Called every frame. Refreshes the visible page when `now` is at least `interval` after its last refresh or when a
  // command invalidated it. Never refreshes the Chat page (the Father view-model has its own tick).
  void tick(std::chrono::steady_clock::time_point now, std::chrono::milliseconds interval = std::chrono::milliseconds(2000));
  void invalidate() { stale_ = true; }

  void open_wizard();
  void close_wizard(bool completed);   // completed or skipped: either way the first-run guide is not shown again

  ModelsViewModel& models() { return models_; }
  ProfilesViewModel& profiles() { return profiles_; }
  MachinesViewModel& machines() { return machines_; }
  ConnectionsViewModel& connections() { return connections_; }
  PerformanceViewModel& performance() { return performance_; }
  SettingsViewModel& settings() { return settings_; }
  WizardViewModel& wizard() { return wizard_; }
  HostSummary host_summary() const { return summary_; }

 private:
  void refresh_page(Page p);
  HostAdminClient& admin_;
  bool demo_;
  ModelsViewModel models_;
  ProfilesViewModel profiles_;
  MachinesViewModel machines_;
  ConnectionsViewModel connections_;
  PerformanceViewModel performance_;
  SettingsViewModel settings_;
  WizardViewModel wizard_;
  HostSummary summary_;
  Page page_ = Page::kChat;
  bool wizard_active_ = false;
  bool stale_ = true;
  std::chrono::steady_clock::time_point last_refresh_{};
  bool refreshed_once_ = false;
};

}  // namespace clusterlm::ui
