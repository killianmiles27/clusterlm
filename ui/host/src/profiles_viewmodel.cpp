#include "clusterlm/ui/profiles_viewmodel.hpp"

#include <algorithm>
#include <set>

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

namespace {
std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) out += (i ? sep : "") + v[i];
  return out;
}

std::string quantity_text(const Quantity& q) {
  std::string v;
  if (q.unit == "bytes") v = format_bytes(static_cast<std::uint64_t>(std::max(0.0, q.value)));
  else if (q.unit == "tok/s") v = format_rate(q.value);
  else {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.1f %s", q.value, q.unit.c_str());
    v = buf;
  }
  return v;
}

std::string provenance_text(const Quantity& q) {
  std::string p(to_string(q.provenance));
  if (q.provenance == Provenance::kSynthetic) p += ": estimate, not a benchmark";
  return p;
}

ProfileRow row_for(const ProfileView& p, bool stale) {
  ProfileRow r;
  r.id = p.id;
  r.name = ascii_display(p.name);
  r.state = p.readiness.state;
  r.stale = stale;
  r.state_label = stale ? "Status unknown" : describe_ready_state(p.readiness.state);
  r.reasons = p.readiness.reasons;
  for (auto& s : r.reasons) s = ascii_display(s);
  if (!stale && p.readiness.state != ReadyState::kReady && p.readiness.state != ReadyState::kBusy && !r.reasons.empty()) r.headline = r.reasons.front();
  if (stale) r.headline = "ClusterLM could not reach the Host, so this status may be out of date.";
  r.last_error = ascii_display(p.readiness.last_error);
  if (!stale && p.readiness.state == ReadyState::kPreparing) {
    r.percent = p.readiness.percent;
    if (p.readiness.eta_seconds) r.eta = format_eta(p.readiness.eta_seconds);  // empty when there is no rate
  }
  std::vector<std::string> filled, waiting;
  for (const auto& s : p.slots) {
    if (s.host) filled.push_back("This PC");
    else if (!s.bound_machine.empty()) filled.push_back(s.bound_machine);
    else waiting.push_back(s.selector.empty() ? s.name : s.selector);
  }
  r.machines_text = join(filled, ", ");
  if (!waiting.empty()) r.machines_text += std::string(filled.empty() ? "" : "; ") + "waiting for: " + join(waiting, ", ");
  r.subtitle = ascii_display(p.model_text);
  if (!p.backend_id.empty()) r.subtitle += (r.subtitle.empty() ? "" : " - ") + p.backend_id;
  if (!p.api_enabled) r.exposure_text = "Not offered to other apps";
  else if (p.allow_lan) r.exposure_text = "Apps can use it as \"" + p.api_model_id + "\", including apps on other PCs";
  else r.exposure_text = "Apps on this PC can use it as \"" + p.api_model_id + "\"";
  r.goals = p.goals;
  r.is_example = !p.example_of.empty();
  r.selected = p.selected;
  r.can_prepare = !stale && p.readiness.state == ReadyState::kLoadable;
  r.can_release = !stale && (p.readiness.state == ReadyState::kReady || p.readiness.state == ReadyState::kPreparing || p.readiness.state == ReadyState::kBusy);
  for (const auto& s : p.slots) {
    if (!s.host && s.binding.empty()) {
      r.can_edit = false;
      r.edit_hint = "This profile picks Workers by their abilities. That can only be edited in its file for now.";
    }
  }
  return r;
}
}  // namespace

void ProfilesViewModel::fail(const Status& s) { st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())}; }
void ProfilesViewModel::ok(std::string text) { st_.notice = Notice{Notice::Kind::kSuccess, std::move(text)}; }

const ProfileView* ProfilesViewModel::find(const std::string& id) const {
  for (const auto& p : profiles_) if (p.id == id) return &p;
  return nullptr;
}

void ProfilesViewModel::refresh() {
  if (!client_.capabilities().profiles) {
    st_.supported = false;
    st_.unsupported_text = "This Host does not have execution profiles yet. Models are still chosen from the Chat page.";
    st_.rows.clear();
    return;
  }
  st_.supported = true;
  auto list = client_.list_profiles(st_.context_tokens);
  if (!list.is_ok()) {
    st_.stale = true;
    for (auto& r : st_.rows) {  // keep the rows but never keep a remembered state
      r.stale = true;
      r.state_label = "Status unknown";
      r.can_prepare = r.can_release = false;
      r.percent.reset();
      r.eta.clear();
      r.headline = "ClusterLM could not reach the Host, so this status may be out of date.";
    }
    fail(list.status());
    return;
  }
  st_.stale = false;
  profiles_ = std::move(list.value());
  st_.rows.clear();
  for (const auto& p : profiles_) st_.rows.push_back(row_for(p, false));
  if (std::none_of(st_.rows.begin(), st_.rows.end(), [&](const ProfileRow& r) { return r.id == st_.selected; }))
    st_.selected = st_.rows.empty() ? std::string() : st_.rows.front().id;

  // Choices for the editor.
  st_.model_choices.clear();
  if (auto m = client_.list_models(); m.is_ok()) {
    models_ = std::move(m.value());
    for (const auto& mv : models_) st_.model_choices.emplace_back(mv.id, ascii_display(mv.name) + (mv.quant.empty() ? "" : " (" + mv.quant + ")"));
  }
  std::set<std::string> backends;
  for (const auto& mv : models_) for (const auto& c : mv.compat) if (c.label != CompatLabel::kUnsupported) backends.insert(c.backend_id);
  for (const auto& p : profiles_) if (!p.backend_id.empty()) backends.insert(p.backend_id);
  st_.backend_choices.clear();
  for (const auto& b : backends) st_.backend_choices.emplace_back(b, b);
  st_.binding_choices.clear();
  if (auto b = client_.list_bindings(); b.is_ok()) for (const auto& x : b.value()) st_.binding_choices.emplace_back(x.name, x.name);
}

void ProfilesViewModel::set_context(std::uint32_t tokens) {
  if (tokens == 0) return;
  st_.context_tokens = tokens;
  refresh();
}

void ProfilesViewModel::select(const std::string& id) {
  if (!find(id)) return;
  st_.selected = id;
}

Status ProfilesViewModel::prepare(const std::string& id) {
  auto s = client_.prepare_profile(id, st_.context_tokens);
  if (!s.is_ok()) { fail(s); return s; }
  ok("Preparing. This can take a while; the profile shows Ready only when the Host says so.");
  refresh();
  return s;
}
Status ProfilesViewModel::release(const std::string& id) {
  auto s = client_.release_profile(id);
  if (!s.is_ok()) { fail(s); return s; }
  ok("Released. Other PCs will delete their temporary files.");
  refresh();
  return s;
}
Status ProfilesViewModel::duplicate(const std::string& id) {
  auto r = client_.duplicate_profile(id);
  if (!r.is_ok()) { fail(r.status()); return r.status(); }
  ok("Made a copy. App access is off on the copy.");
  refresh();
  st_.selected = r.value();
  return Status::ok();
}
Status ProfilesViewModel::remove(const std::string& id, bool i_confirm) {
  if (!i_confirm) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "Confirm to delete this profile.");
    fail(s);
    return s;
  }
  auto s = client_.delete_profile(id);
  if (!s.is_ok()) { fail(s); return s; }
  ok("Profile deleted. Your models and files are untouched.");
  refresh();
  return s;
}

Status ProfilesViewModel::run_dry_run(const std::string& id) {
  auto r = client_.dry_run(id, st_.context_tokens);
  if (!r.is_ok()) { fail(r.status()); return r.status(); }
  DryRunView v;
  v.shown = true;
  v.profile_id = id;
  if (const auto* p = find(id)) v.profile_name = ascii_display(p->name);
  v.feasible = r->feasible;
  v.context_tokens = r->context_tokens;
  v.headline = r->feasible ? "This should fit on your machines" : "This will not fit as things are now";
  bool synthetic = false;
  auto track = [&](const Quantity& q) { if (q.provenance == Provenance::kSynthetic) synthetic = true; };
  for (const auto& m : r->per_machine) {
    DryRunLine l;
    l.machine = ascii_display(m.machine);
    l.cpu = format_bytes(static_cast<std::uint64_t>(std::max(0.0, m.cpu_bytes.value)));
    l.gpu = format_bytes(static_cast<std::uint64_t>(std::max(0.0, m.gpu_bytes.value)));
    l.state = format_bytes(static_cast<std::uint64_t>(std::max(0.0, m.state_bytes.value)));
    l.scratch = format_bytes(static_cast<std::uint64_t>(std::max(0.0, m.scratch_bytes.value)));
    track(m.cpu_bytes); track(m.gpu_bytes); track(m.state_bytes); track(m.scratch_bytes);
    v.lines.push_back(std::move(l));
  }
  track(r->preparation_bytes);
  v.preparation = "Data sent to other PCs: about " + quantity_text(r->preparation_bytes) + " (" + provenance_text(r->preparation_bytes) + ")";
  for (const auto& b : r->bottlenecks) v.bottlenecks.push_back(ascii_display(b));
  for (const auto& q : r->quantities) {
    track(q);
    v.quantities.push_back(ascii_display(q.source.empty() ? q.unit : q.source) + ": " + quantity_text(q) + " (" + provenance_text(q) + ")");
  }
  if (synthetic) v.provenance_note = kSyntheticNote;
  v.plan_summary = ascii_display(r->plan_summary);
  st_.dry_run = std::move(v);
  return Status::ok();
}

// ---- editor -----------------------------------------------------------------------------------------------------

void ProfilesViewModel::begin_create() {
  EditorState e;
  e.open = true;
  if (!models_.empty()) {
    e.draft.model_library_id = models_.front().id;
    e.draft.name = ascii_display(models_.front().name);
    for (const auto& c : models_.front().compat) {
      if (c.label != CompatLabel::kUnsupported) { e.draft.backend_id = c.backend_id; break; }
    }
  }
  st_.editor = std::move(e);
  revalidate();
}

Status ProfilesViewModel::begin_edit(const std::string& id) {
  const ProfileView* p = find(id);
  if (!p) {
    Status s = make_error(ErrorCode::kNotFound, "That profile no longer exists.");
    fail(s);
    return s;
  }
  for (const auto& r : st_.rows) {
    if (r.id == id && !r.can_edit) {
      Status s = make_error(ErrorCode::kFailedPrecondition, r.edit_hint);
      fail(s);
      return s;
    }
  }
  EditorState e;
  e.open = true;
  e.editing_id = id;
  e.draft.name = p->name;
  e.draft.model_library_id = p->model_library_id;
  e.draft.backend_id = p->backend_id;
  e.draft.default_context = p->default_context;
  e.draft.max_context = p->max_context;
  for (const auto& s : p->slots) if (!s.host) e.draft.worker_bindings.push_back(s.binding);
  e.draft.min_workers = p->min_workers;
  e.draft.preparation = p->preparation;
  e.draft.release_after_idle_seconds = p->release_after_idle_seconds;
  e.draft.fallback_profile_id = p->fallback_profile_id;
  e.draft.api_enabled = p->api_enabled;
  e.draft.allow_lan = p->allow_lan;
  e.draft.api_model_id = p->api_model_id;
  st_.editor = std::move(e);
  revalidate();
  return Status::ok();
}

void ProfilesViewModel::revalidate() {
  auto& e = st_.editor;
  if (!e.open) return;
  e.save_error.clear();
  auto f = client_.validate_draft(e.draft);
  if (!f.is_ok()) {
    e.findings.clear();
    e.can_save = false;
    e.save_error = ascii_display(f.status().message());
    return;
  }
  e.findings = std::move(f.value());
  for (auto& x : e.findings) x.message = ascii_display(x.message);
  e.can_save = std::none_of(e.findings.begin(), e.findings.end(), [](const Finding& x) { return x.blocking; });
}

Status ProfilesViewModel::save() {
  auto& e = st_.editor;
  if (!e.open) return make_error(ErrorCode::kFailedPrecondition, "Nothing is being edited.");
  revalidate();
  if (!e.can_save) {
    Status s = make_error(ErrorCode::kInvalidArgument, e.save_error.empty() ? "Fix the problems listed first." : e.save_error);
    e.save_error = s.message();
    return s;
  }
  Status s = Status::ok();
  std::string new_id;
  if (e.editing_id.empty()) {
    auto r = client_.create_profile(e.draft);
    if (!r.is_ok()) s = r.status(); else new_id = r.value();
  } else {
    s = client_.update_profile(e.editing_id, e.draft);
    new_id = e.editing_id;
  }
  if (!s.is_ok()) {
    e.save_error = ascii_display(s.message());  // the editor stays open with the user's work
    return s;
  }
  st_.editor = {};
  ok("Saved. The Host is checking it now.");
  refresh();
  st_.selected = new_id;
  return Status::ok();
}

// ---- export / import ---------------------------------------------------------------------------------------------

Status ProfilesViewModel::export_profile(const std::string& id) {
  auto r = client_.export_profile(id);
  if (!r.is_ok()) { fail(r.status()); return r.status(); }
  st_.export_text = r.value();
  if (const auto* p = find(id)) st_.export_name = ascii_display(p->name);
  ok("Ready to copy or save. It contains no machine names, folders or keys.");
  return Status::ok();
}

Status ProfilesViewModel::import_text(const std::string& text, ImportChoice choice) {
  auto r = client_.import_profile(text, choice);
  if (!r.is_ok()) { fail(r.status()); return r.status(); }
  st_.import = {};
  if (r->conflict) {
    pending_import_ = text;
    st_.import.conflict = true;
    st_.notice = Notice{Notice::Kind::kInfo, "You already have a different profile with this identity. Replace it, or keep both."};
    return Status::ok();
  }
  pending_import_.clear();
  st_.import.todo = r->todo;
  ok(r->todo.empty() ? "That profile was already here." : "Imported. A few things still need you; see the list below.");
  refresh();
  st_.selected = r->profile_id;
  return Status::ok();
}

Status ProfilesViewModel::resolve_import(ImportChoice choice) {
  if (pending_import_.empty() || choice == ImportChoice::kAsk) {
    Status s = make_error(ErrorCode::kFailedPrecondition, "There is no import waiting for a decision.");
    fail(s);
    return s;
  }
  const std::string text = pending_import_;
  return import_text(text, choice);
}

}  // namespace clusterlm::ui
