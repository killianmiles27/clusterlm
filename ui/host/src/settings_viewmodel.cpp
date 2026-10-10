#include "clusterlm/ui/settings_viewmodel.hpp"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

namespace {
constexpr std::uintmax_t kMaxPrefsBytes = 64 * 1024;
const char* theme_name(Theme t) { return t == Theme::kDark ? "dark" : t == Theme::kLight ? "light" : "system"; }
}  // namespace

std::string_view to_string(Theme t) { return theme_name(t); }

UiPrefs clamp(UiPrefs p) {
  p.ui_scale_percent = std::clamp(p.ui_scale_percent, kMinUiScale, kMaxUiScale);
  return p;
}

UiPrefs UiPrefsStore::load(std::string* note) const {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::exists(file_, ec)) return {};
  auto damaged = [&](const char* why) {
    fs::path bad = file_;
    bad += ".bad";
    fs::rename(file_, bad, ec);  // keep it for the owner to look at; defaults are used
    if (note) *note = std::string("Your saved window settings could not be read (") + why + ") and were reset. The old file was kept as " + bad.filename().string() + ".";
    return UiPrefs{};
  };
  const auto size = fs::file_size(file_, ec);
  if (ec || size > kMaxPrefsBytes) return damaged("it is too large");
  std::ifstream in(file_, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  if (j.is_discarded() || !j.is_object()) return damaged("it is not valid");
  UiPrefs p;
  try {
    if (j.contains("theme")) {
      const auto t = j.at("theme").get<std::string>();
      if (t == "dark") p.theme = Theme::kDark;
      else if (t == "light") p.theme = Theme::kLight;
      else if (t == "system") p.theme = Theme::kSystem;
      else return damaged("unknown theme");
    }
    if (j.contains("ui_scale_percent")) p.ui_scale_percent = j.at("ui_scale_percent").get<std::uint32_t>();
    if (j.contains("show_advanced")) p.show_advanced = j.at("show_advanced").get<bool>();
    if (j.contains("wizard_done")) p.wizard_done = j.at("wizard_done").get<bool>();
    if (j.contains("last_page")) {
      const auto name = j.at("last_page").get<std::string>();
      for (auto pg : kAllPages) if (page_label(pg) == name) p.last_page = pg;
    }
  } catch (const std::exception&) {
    return damaged("a value has the wrong type");
  }
  return clamp(p);
}

Status UiPrefsStore::save(const UiPrefs& p) const {
  namespace fs = std::filesystem;
  nlohmann::json j;
  j["theme"] = theme_name(p.theme);
  j["ui_scale_percent"] = p.ui_scale_percent;
  j["show_advanced"] = p.show_advanced;
  j["wizard_done"] = p.wizard_done;
  j["last_page"] = std::string(page_label(p.last_page));
  std::error_code ec;
  if (file_.has_parent_path()) fs::create_directories(file_.parent_path(), ec);
  fs::path tmp = file_;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return make_error(ErrorCode::kUnavailable, "Could not write the window settings file.");
    out << j.dump(2);
    out.flush();
    if (!out) return make_error(ErrorCode::kUnavailable, "Could not write the window settings file.");
  }
  fs::rename(tmp, file_, ec);
  if (ec) {
    fs::remove(tmp, ec);
    return make_error(ErrorCode::kUnavailable, "Could not replace the window settings file.");
  }
  return Status::ok();
}

SettingsViewModel::SettingsViewModel(HostAdminClient& client, UiPrefsStore store, FileSink files)
    : client_(client), store_(std::move(store)), files_(std::move(files)) {
  st_.prefs = store_.load(&st_.prefs_note);
}

void SettingsViewModel::refresh() {
  st_.backup_available = client_.capabilities().backup;
  st_.unavailable_text = st_.backup_available ? "" : "This Host cannot back up its configuration yet.";
}

Status SettingsViewModel::persist() {
  auto s = store_.save(st_.prefs);
  if (!s.is_ok()) st_.notice = Notice{Notice::Kind::kError, "Could not remember that setting: " + ascii_display(s.message())};
  return s;
}

Status SettingsViewModel::set_theme(Theme t) { st_.prefs.theme = t; return persist(); }
Status SettingsViewModel::set_scale(std::uint32_t percent) { st_.prefs.ui_scale_percent = std::clamp(percent, kMinUiScale, kMaxUiScale); return persist(); }
Status SettingsViewModel::set_show_advanced(bool on) { st_.prefs.show_advanced = on; return persist(); }
Status SettingsViewModel::set_wizard_done(bool done) { st_.prefs.wizard_done = done; return persist(); }
Status SettingsViewModel::set_last_page(Page p) {
  if (st_.prefs.last_page == p) return Status::ok();
  st_.prefs.last_page = p;
  return persist();
}

Status SettingsViewModel::backup() {
  auto doc = client_.export_backup();
  if (!doc.is_ok()) { st_.notice = Notice{Notice::Kind::kError, ascii_display(doc.status().message())}; return doc.status(); }
  if (!files_.save) return make_error(ErrorCode::kUnimplemented, "Saving files is not available here.");
  auto where = files_.save("clusterlm-backup.json", doc.value());
  if (!where.is_ok()) { st_.notice = Notice{Notice::Kind::kError, ascii_display(where.status().message())}; return where.status(); }
  st_.last_backup_path = where.value();
  st_.notice = Notice{Notice::Kind::kSuccess, "Saved. The backup holds profiles, library records and settings. It does not hold models, keys or conversations."};
  return Status::ok();
}

Status SettingsViewModel::restore(bool i_confirm) {
  if (!i_confirm) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "Confirm that restoring may replace your current profiles and settings.");
    st_.notice = Notice{Notice::Kind::kError, s.message()};
    return s;
  }
  if (!files_.open) return make_error(ErrorCode::kUnimplemented, "Opening files is not available here.");
  auto text = files_.open();
  if (!text.is_ok()) { st_.notice = Notice{Notice::Kind::kError, ascii_display(text.status().message())}; return text.status(); }
  auto s = client_.import_backup(text.value());
  if (!s.is_ok()) { st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())}; return s; }
  st_.notice = Notice{Notice::Kind::kSuccess, "Restored."};
  return s;
}

}  // namespace clusterlm::ui
