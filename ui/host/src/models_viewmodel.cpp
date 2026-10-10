#include "clusterlm/ui/models_viewmodel.hpp"

#include <algorithm>

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

namespace {
std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) out += (i ? sep : "") + v[i];
  return out;
}

std::string backend_title(const ModelCompat& c) {
  if (!c.backend_name.empty()) return ascii_display(c.backend_name);
  return c.backend_id;
}

CompatRow compat_row(const ModelCompat& c) {
  CompatRow r;
  r.backend = backend_title(c);
  r.kind = c.label;
  r.checked = true;
  r.label = std::string(to_string(c.label));
  for (const auto& f : c.findings) r.notes.push_back(ascii_display(f));
  if (!c.backend_built) r.notes.push_back(r.backend + " is not part of this installation.");
  else if (!c.runtime_present) r.notes.push_back("The software " + r.backend + " needs is not installed on this PC.");
  r.reach = c.max_workers == 0 ? "This PC only" : "Up to " + std::to_string(c.max_workers) + (c.max_workers == 1 ? " other PC" : " other PCs");
  return r;
}

ModelRow row_for(const ModelView& m) {
  ModelRow r;
  r.id = m.id;
  r.title = ascii_display(m.name.empty() ? m.family : m.name);
  std::string sub = ascii_display(m.family);
  if (!m.quant.empty()) sub += (sub.empty() ? "" : " - ") + m.quant;
  sub += (sub.empty() ? "" : " - ") + format_bytes(m.total_bytes);
  if (m.file_count > 1) sub += " - " + std::to_string(m.file_count) + " files";
  r.subtitle = sub;
  if (m.quant_mix.size() > 1) r.quant_note = "Mixed precision: " + join(m.quant_mix, ", ");
  r.location = m.dir;
  if (m.context_length) r.facts.push_back("Trained for up to " + std::to_string(*m.context_length) + " tokens of context");
  if (m.block_count) r.facts.push_back(std::to_string(*m.block_count) + " layers");
  if (m.expert_count && *m.expert_count > 0) r.facts.push_back(std::to_string(*m.expert_count) + " experts");
  r.verified = m.root_hash.has_value();
  if (m.root_hash) {
    r.root_short = m.root_hash->substr(0, 12);
    if (!m.pinned_root) { r.integrity = "Verified"; r.can_confirm = true; }
    else if (*m.pinned_root == *m.root_hash) r.integrity = "Confirmed by you";
    else r.integrity = "Verified, but it differs from the file you confirmed";
  } else {
    r.integrity = m.pinned_root ? "Confirmed by you, not re-verified yet" : "Not verified yet";
  }
  if (m.compat.empty()) {
    CompatRow c;
    c.backend = "Any engine";
    c.label = "Not checked yet";
    c.notes.push_back("ClusterLM has not worked out which engines can run this model.");
    r.compat.push_back(std::move(c));
  } else {
    for (const auto& c : m.compat) r.compat.push_back(compat_row(c));
  }
  r.used_by = m.used_by;
  r.can_remove = m.used_by.empty();
  if (!r.can_remove) r.remove_hint = "Used by: " + join(m.used_by, ", ") + ". Change or delete those profiles first.";
  return r;
}
}  // namespace

void ModelsViewModel::fail(const Status& s) {
  st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())};
}
void ModelsViewModel::ok(std::string text) { st_.notice = Notice{Notice::Kind::kSuccess, std::move(text)}; }

void ModelsViewModel::refresh() {
  if (!client_.capabilities().models) {
    st_.supported = false;
    st_.unsupported_text = "This Host does not have a model library yet, so models cannot be listed here.";
    st_.rows.clear();
    return;
  }
  st_.supported = true;
  auto list = client_.list_models();
  if (!list.is_ok()) {
    st_.stale = true;
    fail(list.status());
    return;  // keep the previous rows, flagged stale
  }
  st_.stale = false;
  models_ = std::move(list.value());
  st_.rows.clear();
  for (const auto& m : models_) st_.rows.push_back(row_for(m));
  if (auto roots = client_.scan_roots(); roots.is_ok()) st_.scan_roots = std::move(roots.value());
  if (std::none_of(st_.rows.begin(), st_.rows.end(), [&](const ModelRow& r) { return r.id == st_.selected; })) {
    st_.selected = st_.rows.empty() ? std::string() : st_.rows.front().id;
  }
}

void ModelsViewModel::select(const std::string& id) {
  if (std::any_of(st_.rows.begin(), st_.rows.end(), [&](const ModelRow& r) { return r.id == id; })) st_.selected = id;
}

Status ModelsViewModel::add_scan_root(const std::string& dir) {
  auto s = client_.add_scan_root(dir);
  if (!s.is_ok()) { fail(s); return s; }
  ok("Watching that folder for models.");
  refresh();
  return s;
}
Status ModelsViewModel::remove_scan_root(const std::string& dir) {
  auto s = client_.remove_scan_root(dir);
  if (!s.is_ok()) { fail(s); return s; }
  ok("Stopped watching that folder. Models already in the library stay.");
  refresh();
  return s;
}
Status ModelsViewModel::rescan() {
  auto r = client_.rescan();
  if (!r.is_ok()) { fail(r.status()); return r.status(); }
  st_.scan_issues = r->issues;
  st_.last_scan = "Found " + std::to_string(r->found) + (r->found == 1 ? " model." : " models.");
  if (!r->issues.empty()) st_.last_scan += " " + std::to_string(r->issues.size()) + (r->issues.size() == 1 ? " file was skipped." : " files were skipped.");
  ok(st_.last_scan);
  refresh();
  return Status::ok();
}
Status ModelsViewModel::import_path(const std::string& path) {
  auto r = client_.import_model(path);
  if (!r.is_ok()) { fail(r.status()); return r.status(); }
  st_.selected = r->id;
  ok("Added \"" + ascii_display(r->name) + "\" to the library. The file stays where it is.");
  refresh();
  return Status::ok();
}
Status ModelsViewModel::confirm_verified_root(const std::string& id, bool i_confirm) {
  if (!i_confirm) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "Tick the box to confirm this is the exact file you expect.");
    fail(s);
    return s;
  }
  auto it = std::find_if(models_.begin(), models_.end(), [&](const ModelView& m) { return m.id == id; });
  if (it == models_.end() || !it->root_hash) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "ClusterLM has not verified this model's files yet, so there is nothing to confirm.");
    fail(s);
    return s;
  }
  auto s = client_.pin_model_root(id, *it->root_hash);
  if (!s.is_ok()) { fail(s); return s; }
  ok("Confirmed. Profiles that name this exact model can now use it.");
  refresh();
  return s;
}
Status ModelsViewModel::remove(const std::string& id) {
  auto it = std::find_if(st_.rows.begin(), st_.rows.end(), [&](const ModelRow& r) { return r.id == id; });
  if (it != st_.rows.end() && !it->can_remove) {
    Status s = make_error(ErrorCode::kFailedPrecondition, it->remove_hint);
    fail(s);
    return s;
  }
  auto s = client_.remove_model(id);
  if (!s.is_ok()) { fail(s); return s; }
  ok("Removed from the library. The files on disk were not touched.");
  refresh();
  return s;
}

}  // namespace clusterlm::ui
