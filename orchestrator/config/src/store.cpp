#include "clusterlm/config/store.hpp"

#include <chrono>
#include <system_error>

#include <nlohmann/json.hpp>

#include "clusterlm/common/log.hpp"
#include "clusterlm/platform/durable_file.hpp"

namespace clusterlm::config {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr std::uintmax_t kMaxDocumentBytes = 1u << 20;

template <typename T> struct Traits;
template <> struct Traits<FatherSettings> {
  static constexpr int kVersion = kFatherSettingsVersion;
  static Result<FatherSettings> parse(std::string_view t) { return father_settings_from_json(t); }
  static std::string dump(const FatherSettings& s) { return to_json(s); }
  static Status check(const FatherSettings& s) { return validate(s); }
};
template <> struct Traits<NodeSettings> {
  static constexpr int kVersion = kNodeSettingsVersion;
  static Result<NodeSettings> parse(std::string_view t) { return node_settings_from_json(t); }
  static std::string dump(const NodeSettings& s) { return to_json(s); }
  static Status check(const NodeSettings& s) { return validate(s); }
};

Status io_error(const std::string& what, const std::error_code& ec) {
  return make_error(ErrorCode::kUnavailable, what + ": " + ec.message());
}

fs::path unique_sibling(const fs::path& file, const std::string& infix) {
  for (int i = 0; i < 1000; ++i) {
    fs::path p = file;
    p += "." + infix + (i == 0 ? std::string() : "-" + std::to_string(i));
    std::error_code ec;
    if (!fs::exists(p, ec)) return p;
  }
  fs::path p = file;
  p += "." + infix + "-overflow";
  return p;
}

// Moves the damaged file aside; a rename keeps its (owner-only) permissions.
Result<fs::path> set_aside(const fs::path& file) {
  const auto stamp =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  fs::path backup = unique_sibling(file, "corrupt-" + std::to_string(stamp));
  std::error_code ec;
  fs::rename(file, backup, ec);
  if (ec) return io_error("could not preserve the damaged settings file", ec);
  return backup;
}

}  // namespace

template <typename T>
Result<std::unique_ptr<SettingsStore<T>>> SettingsStore<T>::open(fs::path file, const MigrationHook& migrate) {
  using Tr = Traits<T>;
  std::unique_ptr<SettingsStore<T>> store(new SettingsStore<T>(std::move(file)));
  const fs::path f = store->file_;
  if (f.has_parent_path()) CLM_RETURN_IF_ERROR(platform::create_owner_only_directory(f.parent_path()));

  std::error_code ec;
  if (!fs::exists(f, ec)) {
    store->report_.outcome = LoadReport::Outcome::kDefaultsNoFile;
    return store;
  }
  // Permissions first: a settings file other accounts can read or write is tightened before it is trusted.
  auto owner_only = platform::is_owner_only(f);
  if (!owner_only.is_ok() || !owner_only.value()) {
    if (platform::restrict_to_owner(f).is_ok()) {
      store->report_.permissions_tightened = true;
      store->report_.note = "settings file permissions were too open and have been tightened";
    }
  }

  auto recover = [&](const std::string& why) -> Result<std::unique_ptr<SettingsStore<T>>> {
    CLM_ASSIGN_OR_RETURN(auto backup, set_aside(f));
    log::warn("settings_recovered", {{"why", why}, {"backup", backup.filename().string()}});
    store->report_.outcome = LoadReport::Outcome::kRecoveredCorrupt;
    store->report_.backup = backup;
    store->report_.note = "settings file was unreadable (" + why + "); defaults are in use and the original was kept";
    store->value_ = T{};
    return std::move(store);
  };

  const auto size = fs::file_size(f, ec);
  if (ec || size > kMaxDocumentBytes) return recover("size");
  auto bytes = platform::read_file_bytes(f);
  if (!bytes.is_ok()) return bytes.status();
  const std::string text(bytes->begin(), bytes->end());

  json doc = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object() || !doc.contains("version") || !doc["version"].is_number_integer())
    return recover("not a versioned JSON document");
  const int version = doc["version"].get<int>();
  if (version > Tr::kVersion)
    return make_error(ErrorCode::kVersionMismatch,
                      "settings file was written by a newer ClusterLM (version " + std::to_string(version) + ")");

  if (version < Tr::kVersion) {
    if (!migrate || version < 0) return recover("unsupported old version");
    fs::path backup = unique_sibling(f, "v" + std::to_string(version) + ".bak");
    fs::copy_file(f, backup, fs::copy_options::none, ec);
    if (ec) return io_error("could not back up the settings file before migration", ec);
    (void)platform::restrict_to_owner(backup);
    std::string current = text;
    for (int v = version; v < Tr::kVersion; ++v) {
      auto next = migrate(current, v);
      if (!next.is_ok()) return recover("migration failed");
      current = std::move(next).value();
    }
    auto parsed = Tr::parse(current);
    if (!parsed.is_ok()) return recover("migrated document invalid");
    store->value_ = std::move(parsed).value();
    {
      std::lock_guard lk(store->mu_);
      CLM_RETURN_IF_ERROR(store->persist_locked(store->value_));
    }
    store->report_.outcome = LoadReport::Outcome::kMigrated;
    store->report_.backup = backup;
    store->report_.note = "migrated from version " + std::to_string(version);
    return store;
  }

  auto parsed = Tr::parse(text);
  if (!parsed.is_ok()) return recover("failed validation");
  store->value_ = std::move(parsed).value();
  store->report_.outcome = LoadReport::Outcome::kLoaded;
  return store;
}

template <typename T>
T SettingsStore<T>::get() const {
  std::lock_guard lk(mu_);
  return value_;
}

template <typename T>
Status SettingsStore<T>::persist_locked(const T& value) {
  using Tr = Traits<T>;
  CLM_RETURN_IF_ERROR(Tr::check(value));
  const std::string text = Tr::dump(value);
  fs::path tmp = file_;
  tmp += ".tmp";
  const ByteSpan span(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
  CLM_RETURN_IF_ERROR(platform::write_owner_only_file(tmp, span));
  std::error_code ec;
  fs::rename(tmp, file_, ec);
  if (ec) {
    std::error_code ignore;
    fs::remove(tmp, ignore);
    return io_error("could not replace the settings file", ec);
  }
  if (file_.has_parent_path()) (void)platform::sync_directory(file_.parent_path());
  return Status::ok();
}

template <typename T>
Status SettingsStore<T>::update(const std::function<Status(T&)>& mutate) {
  std::lock_guard lk(mu_);
  T copy = value_;
  CLM_RETURN_IF_ERROR(mutate(copy));
  CLM_RETURN_IF_ERROR(persist_locked(copy));
  value_ = std::move(copy);
  return Status::ok();
}

template <typename T>
Status SettingsStore<T>::replace(T value) {
  return update([&](T& v) {
    v = std::move(value);
    return Status::ok();
  });
}

template class SettingsStore<FatherSettings>;
template class SettingsStore<NodeSettings>;

fs::path father_settings_path(const platform::DefaultPaths& p) { return p.father_root / "father-settings.json"; }
fs::path node_settings_path(const platform::DefaultPaths& p) { return p.node_root / "node-settings.json"; }

}  // namespace clusterlm::config
