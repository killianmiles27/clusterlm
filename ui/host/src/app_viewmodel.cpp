#include "clusterlm/ui/app_viewmodel.hpp"

namespace clusterlm::ui {

AppViewModel::AppViewModel(AppDeps deps)
    : admin_(*deps.admin),
      demo_(deps.demo),
      models_(*deps.admin),
      profiles_(*deps.admin),
      machines_(*deps.admin, deps.pairing),
      connections_(*deps.admin),
      performance_(*deps.admin),
      settings_(*deps.admin, UiPrefsStore(deps.prefs_file), std::move(deps.files)),
      wizard_(*deps.admin, deps.pairing) {
  if (auto s = admin_.summary(); s.is_ok()) summary_ = s.value();
  settings_.refresh();
  page_ = settings_.prefs().last_page;
  if (!settings_.prefs().wizard_done) open_wizard();
}

void AppViewModel::go(Page p) {
  if (p == page_) return;
  page_ = p;
  (void)settings_.set_last_page(p);
  stale_ = true;  // refresh at the next tick, not later: a page never opens on old numbers
}

std::optional<Page> AppViewModel::page_for_shortcut(int digit) {
  if (digit < 1 || digit > static_cast<int>(kAllPages.size())) return std::nullopt;
  return kAllPages[static_cast<std::size_t>(digit - 1)];
}

void AppViewModel::refresh_page(Page p) {
  switch (p) {
    case Page::kChat: break;
    case Page::kModels: models_.refresh(); break;
    case Page::kProfiles: profiles_.refresh(); break;
    case Page::kMachines: machines_.refresh(); break;
    case Page::kConnections: connections_.refresh(); break;
    case Page::kPerformance: performance_.refresh(); break;
    case Page::kSettings: settings_.refresh(); break;
  }
}

void AppViewModel::tick(std::chrono::steady_clock::time_point now, std::chrono::milliseconds interval) {
  if (wizard_active_) return;  // the wizard drives its own calls
  if (page_ == Page::kChat) return;
  if (!stale_ && refreshed_once_ && now - last_refresh_ < interval) return;
  refresh_page(page_);
  last_refresh_ = now;
  refreshed_once_ = true;
  stale_ = false;
}

void AppViewModel::open_wizard() {
  wizard_.start();
  wizard_active_ = true;
}

void AppViewModel::close_wizard(bool) {
  wizard_.dismiss_secret();
  wizard_active_ = false;
  (void)settings_.set_wizard_done(true);
  page_ = Page::kChat;
  stale_ = true;
}

}  // namespace clusterlm::ui
