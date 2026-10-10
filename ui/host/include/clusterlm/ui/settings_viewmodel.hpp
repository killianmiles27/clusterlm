#pragma once
// UiPrefs + SettingsViewModel: this window's own preferences (theme, size, advanced view, first-run state) and the
// configuration backup.
//
// UiPrefs live in one small JSON file next to the Host's settings. Loading is bounded (64 KiB) and strict about
// types; a damaged file is set aside as "<name>.bad" and defaults are used, with a notice, never a crash. Saving
// writes a temporary file and renames it over the old one, so a power cut leaves either the old or the new file.
#include <filesystem>
#include <functional>
#include <string>

#include "clusterlm/ui/host_admin.hpp"
#include "clusterlm/ui/host_common.hpp"

namespace clusterlm::ui {

enum class Theme : std::uint8_t { kSystem, kDark, kLight };
std::string_view to_string(Theme t);

struct UiPrefs {
  Theme theme = Theme::kSystem;
  std::uint32_t ui_scale_percent = 100;  // extra scale on top of the monitor's DPI scale; 75..200
  bool show_advanced = false;
  bool wizard_done = false;
  Page last_page = Page::kChat;
  friend bool operator==(const UiPrefs&, const UiPrefs&) = default;
};
inline constexpr std::uint32_t kMinUiScale = 75, kMaxUiScale = 200;
UiPrefs clamp(UiPrefs p);

class UiPrefsStore {
 public:
  explicit UiPrefsStore(std::filesystem::path file) : file_(std::move(file)) {}
  // Never fails: returns defaults and sets `note` when the file is missing-but-expected or damaged.
  UiPrefs load(std::string* note = nullptr) const;
  Status save(const UiPrefs& p) const;
  const std::filesystem::path& file() const { return file_; }
 private:
  std::filesystem::path file_;
};

// Where a backup or an export goes / comes from; the shell shows the file dialog.
struct FileSink {
  std::function<Result<std::string>(const std::string& suggested_name, const std::string& text)> save;  // returns where
  std::function<Result<std::string>()> open;                                                         // returns the text
};

struct SettingsState {
  UiPrefs prefs;
  std::string prefs_note;  // e.g. "Your saved window settings were damaged and have been reset."
  bool backup_available = false;
  std::string last_backup_path;
  std::string unavailable_text;
  Notice notice;
};

class SettingsViewModel {
 public:
  SettingsViewModel(HostAdminClient& client, UiPrefsStore store, FileSink files);
  const SettingsState& state() const { return st_; }
  const UiPrefs& prefs() const { return st_.prefs; }

  void refresh();
  Status set_theme(Theme t);
  Status set_scale(std::uint32_t percent);
  Status set_show_advanced(bool on);
  Status set_wizard_done(bool done);
  Status set_last_page(Page p);
  Status backup();
  Status restore(bool i_confirm);
  void dismiss_notice() { st_.notice = {}; }

 private:
  Status persist();
  HostAdminClient& client_;
  UiPrefsStore store_;
  FileSink files_;
  SettingsState st_;
};

}  // namespace clusterlm::ui
