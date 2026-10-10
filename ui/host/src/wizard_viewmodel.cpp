#include "clusterlm/ui/wizard_viewmodel.hpp"

#include <algorithm>
#include <cctype>

#include "clusterlm/ui/format.hpp"

namespace clusterlm::ui {

std::string_view step_title(WizardStep s) {
  switch (s) {
    case WizardStep::kRole: return "What should this PC do?";
    case WizardStep::kExplain: return "How ClusterLM works";
    case WizardStep::kDetect: return "Your PC";
    case WizardStep::kPair: return "Other PCs";
    case WizardStep::kModel: return "Choose a model";
    case WizardStep::kProfile: return "Your first profile";
    case WizardStep::kApi: return "Use it from other apps";
    case WizardStep::kValidate: return "Check";
    case WizardStep::kDone: return "Done";
  }
  return "";
}

namespace {
std::string slug(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c))) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    else if (!out.empty() && out.back() != '-') out.push_back('-');
  }
  while (!out.empty() && out.back() == '-') out.pop_back();
  if (out.size() > 48) out.resize(48);
  return out.empty() ? std::string("my-model") : out;
}

WizardStep next_of(WizardRole role, WizardStep s) {
  if (role == WizardRole::kWorker) return s == WizardStep::kRole ? WizardStep::kExplain : WizardStep::kDone;
  return static_cast<WizardStep>(static_cast<int>(s) + 1);
}
WizardStep prev_of(WizardRole role, WizardStep s) {
  if (role == WizardRole::kWorker) return s == WizardStep::kDone ? WizardStep::kExplain : WizardStep::kRole;
  return static_cast<WizardStep>(std::max(0, static_cast<int>(s) - 1));
}
}  // namespace

void WizardViewModel::start() {
  st_ = {};
  pair_ = {};
  load_models();
  enter(WizardStep::kRole);
}

void WizardViewModel::load_models() {
  st_.model_choices.clear();
  models_.clear();
  if (!client_.capabilities().models) return;
  if (auto m = client_.list_models(); m.is_ok()) {
    models_ = std::move(m.value());
    for (const auto& x : models_) st_.model_choices.emplace_back(x.id, ascii_display(x.name) + (x.quant.empty() ? "" : " (" + x.quant + ")") + " - " + format_bytes(x.total_bytes));
  }
}

void WizardViewModel::choose_role(WizardRole r) {
  st_.role = r;
  recompute();
}

void WizardViewModel::enter(WizardStep s) {
  st_.step = s;
  st_.notice = {};
  st_.paragraphs.clear();
  st_.next_label = "Next";
  switch (s) {
    case WizardStep::kRole:
      st_.paragraphs = {"ClusterLM runs AI models on PCs you own. One PC is the Host: it keeps your models and answers questions. Other PCs can optionally be Workers that lend spare memory and speed while nobody is using them.",
                        "Pick what this PC should be. You can add Workers any time."};
      break;
    case WizardStep::kExplain:
      if (st_.role == WizardRole::kWorker) {
        st_.paragraphs = {"A Worker helps a Host while you are away from this PC. It never keeps model files: anything it receives is temporary and is deleted when the Host is done or when you come back to this PC.",
                          "Open the ClusterLM Worker app on this PC. It shows an address and a one-time code. Type them on the Host (Machines page, Pair a Worker). You can stop helping at any time from the Worker's tray icon."};
      } else {
        st_.paragraphs = {"Models live only on this Host. Your Workers never keep copies; the pieces they need are sent when you prepare a profile and removed afterwards.",
                          "A profile says which model to run, how, and on which PCs. Chat uses the profile you pick. Nothing leaves your network, and there are no accounts."};
      }
      break;
    case WizardStep::kDetect: {
      st_.host_facts.clear();
      if (auto ms = client_.list_machines(); ms.is_ok()) {
        for (const auto& m : ms.value()) {
          if (!m.is_host) continue;
          if (!m.os.empty()) st_.host_facts.push_back(ascii_display(m.os));
          if (!m.cpu.empty()) st_.host_facts.push_back(ascii_display(m.cpu));
          if (m.ram_bytes) st_.host_facts.push_back(format_bytes(m.ram_bytes) + " memory");
          for (const auto& g : m.gpus) st_.host_facts.push_back(ascii_display(g));
          if (m.vram_bytes) st_.host_facts.push_back(format_bytes(m.vram_bytes) + " graphics memory");
        }
      }
      if (st_.host_facts.empty()) st_.host_facts.push_back("ClusterLM could not read this PC's details yet. You can continue; profiles check what fits before anything is prepared.");
      st_.paragraphs = {"This is what ClusterLM found on this PC."};
      break;
    }
    case WizardStep::kPair:
      st_.paired.clear();
      if (pairing_) if (auto p = pairing_->paired(); p.is_ok()) for (const auto& m : p.value()) st_.paired.push_back(ascii_display(m.machine_id));
      st_.paragraphs = {"Optional. To use other PCs, open the ClusterLM Worker app on each one, then enter the address and code it shows. You can skip this and do it later in Machines."};
      st_.next_label = st_.paired.empty() ? "Skip for now" : "Next";
      break;
    case WizardStep::kModel:
      load_models();
      st_.paragraphs = {"Pick a model file you already have, or tell ClusterLM where to look. Models are .gguf files. The file stays where it is; ClusterLM only remembers where."};
      break;
    case WizardStep::kProfile:
      suggest();
      break;
    case WizardStep::kApi:
      st_.paragraphs = {"Apps such as coding assistants can use your model through a local connection. This is off unless you turn it on, and it only works from this PC with a key you control."};
      break;
    case WizardStep::kValidate:
      validate();
      break;
    case WizardStep::kDone:
      if (st_.role == WizardRole::kWorker) st_.paragraphs = {"All set. Open the ClusterLM Worker app to get your pairing code."};
      else st_.paragraphs = {st_.readiness_text.empty() ? "Setup is finished." : st_.readiness_text};
      st_.next_label = "Open Chat";
      break;
  }
  recompute();
}

void WizardViewModel::recompute() {
  st_.can_back = st_.step != WizardStep::kRole;
  st_.can_next = true;
  st_.next_hint.clear();
  switch (st_.step) {
    case WizardStep::kRole:
      st_.can_next = st_.role != WizardRole::kNone;
      if (!st_.can_next) st_.next_hint = "Choose Host or Worker.";
      break;
    case WizardStep::kModel:
      st_.can_next = !st_.model_id.empty();
      if (!st_.can_next) st_.next_hint = "Choose a model, or use Skip to finish without one.";
      break;
    case WizardStep::kProfile:
      st_.can_next = st_.draft_ok;
      if (!st_.can_next) st_.next_hint = "Fix the problems listed, or go back and choose another model.";
      break;
    default:
      break;
  }
  const int total = st_.role == WizardRole::kWorker ? 3 : 9;
  st_.step_count = total;
  st_.step_number = st_.role == WizardRole::kWorker ? (st_.step == WizardStep::kRole ? 1 : st_.step == WizardStep::kExplain ? 2 : 3) : static_cast<int>(st_.step) + 1;
}

void WizardViewModel::suggest() {
  st_.draft_ok = false;
  st_.draft_notes.clear();
  st_.paragraphs = {"Here is a profile ClusterLM suggests for this model, using only this PC. You can spread a model across other PCs later in Profiles."};
  auto it = std::find_if(models_.begin(), models_.end(), [&](const ModelView& m) { return m.id == st_.model_id; });
  if (it == models_.end()) {
    st_.draft_notes.push_back("Choose a model first.");
    return;
  }
  // Prefer an engine the Host labels Supported; never pick Experimental or Unsupported on the user's behalf.
  const ModelCompat* best = nullptr;
  for (const auto& c : it->compat) {
    if (!c.backend_built || !c.runtime_present) continue;
    if (c.label == CompatLabel::kSupportedQualified) { best = &c; break; }
    if (c.label == CompatLabel::kSupportedAwaitingQualification && !best) best = &c;
  }
  if (it->compat.empty()) {
    st_.draft_notes.push_back("ClusterLM has not checked which engines can run this model, so it cannot suggest a profile yet.");
    return;
  }
  if (!best) {
    st_.draft_notes.push_back("None of the installed engines supports this model for normal use.");
    for (const auto& c : it->compat) for (const auto& f : c.findings) st_.draft_notes.push_back(ascii_display(f));
    return;
  }
  ProfileDraft d = st_.draft;
  d.model_library_id = it->id;
  d.backend_id = best->backend_id;
  if (d.name.empty() || st_.created_profile_id.empty()) d.name = ascii_display(it->name);
  d.max_context = std::min<std::uint32_t>(32768, it->context_length.value_or(32768));
  d.default_context = std::min<std::uint32_t>(4096, d.max_context);
  d.worker_bindings.clear();
  d.min_workers = 0;
  d.preparation = "on-demand";
  d.release_after_idle_seconds = 60;
  d.api_enabled = false;
  d.allow_lan = false;
  st_.draft = d;
  st_.draft_notes.push_back("Engine: " + (best->backend_name.empty() ? best->backend_id : ascii_display(best->backend_name)) + " - " + std::string(to_string(best->label)) + ".");
  if (best->label == CompatLabel::kSupportedAwaitingQualification)
    st_.draft_notes.push_back("This combination has not been tested on real hardware yet, so speed is unknown.");
  if (auto f = client_.validate_draft(d); f.is_ok()) {
    bool blocked = false;
    for (const auto& x : f.value()) {
      st_.draft_notes.push_back(ascii_display(x.message));
      blocked = blocked || x.blocking;
    }
    st_.draft_ok = !blocked;
  } else {
    st_.draft_notes.push_back(ascii_display(f.status().message()));
  }
}

Status WizardViewModel::create_or_update_profile() {
  Status s = Status::ok();
  if (st_.created_profile_id.empty()) {
    auto r = client_.create_profile(st_.draft);
    if (!r.is_ok()) s = r.status(); else st_.created_profile_id = r.value();
  } else {
    s = client_.update_profile(st_.created_profile_id, st_.draft);
  }
  if (!s.is_ok()) st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())};
  return s;
}

Status WizardViewModel::apply_api() {
  if (!st_.api_enabled) return Status::ok();
  st_.draft.api_enabled = true;
  st_.draft.allow_lan = false;
  st_.draft.api_model_id = st_.api_model_id.empty() ? slug(st_.draft.name) : st_.api_model_id;
  st_.api_model_id = st_.draft.api_model_id;
  if (auto s = client_.update_profile(st_.created_profile_id, st_.draft); !s.is_ok()) {
    st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())};
    return s;
  }
  ServerSettings ss;
  if (auto cur = client_.server_status(); cur.is_ok()) ss = cur->settings;
  ss.enabled = true;
  ss.allow_lan = false;  // this PC only; other PCs are a deliberate step in Connections
  if (auto s = client_.set_server(ss); !s.is_ok()) {
    st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())};
    return s;
  }
  if (st_.key_secret.empty()) {
    auto k = client_.create_key("First app", {"inference"}, {st_.api_model_id}, false);
    if (!k.is_ok()) {
      st_.notice = Notice{Notice::Kind::kError, ascii_display(k.status().message())};
      return k.status();
    }
    st_.key_secret = k->secret;
    st_.notice = Notice{Notice::Kind::kInfo, "Copy your key now. It is shown only once."};
  }
  return Status::ok();
}

void WizardViewModel::validate() {
  st_.validation.clear();
  st_.readiness_text.clear();
  st_.paragraphs = {"ClusterLM is checking the profile with the Host."};
  if (st_.created_profile_id.empty()) {
    st_.validation.push_back("No profile was created.");
    return;
  }
  if (auto f = client_.validate_profile(st_.created_profile_id); f.is_ok()) {
    for (const auto& x : f.value()) st_.validation.push_back(ascii_display(x.message));
  } else {
    st_.validation.push_back(ascii_display(f.status().message()));
  }
  if (auto d = client_.dry_run(st_.created_profile_id, st_.draft.default_context); d.is_ok()) {
    st_.validation.push_back(d->feasible ? "It should fit on this PC." : "It may not fit on this PC as things are now.");
    bool synthetic = false;
    for (const auto& m : d->per_machine) synthetic = synthetic || m.gpu_bytes.provenance == Provenance::kSynthetic;
    if (synthetic) st_.validation.push_back("That is an estimate, not a measurement.");
  }
  auto ps = client_.list_profiles(st_.draft.default_context);
  if (ps.is_ok()) {
    for (const auto& p : ps.value()) {
      if (p.id != st_.created_profile_id) continue;
      const auto& r = p.readiness;
      st_.readiness_text = "The Host reports this profile as: " + describe_ready_state(r.state) + ".";
      if (r.state != ReadyState::kReady && r.state != ReadyState::kBusy) {
        st_.readiness_text += " It is not ready yet; ClusterLM prepares it the first time you chat";
        st_.readiness_text += r.state == ReadyState::kUnavailable || r.state == ReadyState::kCompatible ? ", once the problem below is fixed." : ".";
        if (!r.reasons.empty()) st_.readiness_text += " " + ascii_display(r.reasons.front());
      }
    }
  }
  st_.paragraphs = {st_.readiness_text.empty() ? "The Host did not report the profile's readiness." : st_.readiness_text};
}

void WizardViewModel::next() {
  if (!st_.can_next) return;
  if (st_.step == WizardStep::kDone) {
    st_.finished = true;
    return;
  }
  WizardStep to = next_of(st_.role, st_.step);
  if (st_.step == WizardStep::kProfile) {
    if (!create_or_update_profile().is_ok()) return;
  }
  if (st_.step == WizardStep::kApi) {
    if (!apply_api().is_ok()) return;
  }
  enter(to);
}

void WizardViewModel::back() {
  if (!st_.can_back) return;
  enter(prev_of(st_.role, st_.step));
}

void WizardViewModel::skip() { st_.finished = true; }

Status WizardViewModel::pair() {
  pair_.error.clear();
  if (!pairing_) {
    pair_.error = "Pairing is not available from this window in this build. Use the Worker app and the command line for now.";
    return make_error(ErrorCode::kUnimplemented, pair_.error);
  }
  if (pair_.address.empty() || pair_.code.empty()) {
    pair_.error = "Enter the address and the code shown on the Worker.";
    return make_error(ErrorCode::kInvalidArgument, pair_.error);
  }
  auto s = pairing_->pair(PairingRequest{pair_.address, pair_.code, pair_.name});
  pair_.code.clear();
  if (!s.is_ok()) {
    pair_.error = ascii_display(user_words(s.message()));
    return s;
  }
  pair_ = {};
  enter(WizardStep::kPair);
  st_.notice = Notice{Notice::Kind::kSuccess, "Paired."};
  return s;
}

Status WizardViewModel::import_model_path(const std::string& path) {
  auto r = client_.import_model(path);
  if (!r.is_ok()) {
    st_.notice = Notice{Notice::Kind::kError, ascii_display(r.status().message())};
    return r.status();
  }
  load_models();
  st_.model_id = r->id;
  st_.notice = Notice{Notice::Kind::kSuccess, "Added. The file stays where it is."};
  recompute();
  return Status::ok();
}

Status WizardViewModel::scan_folder(const std::string& dir) {
  if (auto s = client_.add_scan_root(dir); !s.is_ok() && s.code() != ErrorCode::kAlreadyExists) {
    st_.notice = Notice{Notice::Kind::kError, ascii_display(s.message())};
    return s;
  }
  auto r = client_.rescan();
  if (!r.is_ok()) {
    st_.notice = Notice{Notice::Kind::kError, ascii_display(r.status().message())};
    return r.status();
  }
  load_models();
  st_.notice = Notice{Notice::Kind::kSuccess, "Found " + std::to_string(r->found) + (r->found == 1 ? " model." : " models.")};
  recompute();
  return Status::ok();
}

void WizardViewModel::choose_model(const std::string& id) {
  if (std::none_of(models_.begin(), models_.end(), [&](const ModelView& m) { return m.id == id; })) return;
  st_.model_id = id;
  st_.model_notes.clear();
  for (const auto& m : models_) {
    if (m.id != id) continue;
    if (m.compat.empty()) st_.model_notes.push_back("Not checked yet.");
    for (const auto& c : m.compat) st_.model_notes.push_back((c.backend_name.empty() ? c.backend_id : ascii_display(c.backend_name)) + ": " + std::string(to_string(c.label)));
  }
  recompute();
}

void WizardViewModel::set_profile_name(const std::string& name) {
  st_.draft.name = name.substr(0, 64);
  if (auto f = client_.validate_draft(st_.draft); f.is_ok()) {
    st_.draft_ok = std::none_of(f->begin(), f->end(), [](const Finding& x) { return x.blocking; });
  }
  recompute();
}

void WizardViewModel::set_api_enabled(bool on) {
  st_.api_enabled = on;
  if (on && st_.api_model_id.empty()) st_.api_model_id = slug(st_.draft.name);
}

void WizardViewModel::dismiss_secret() {
  st_.key_secret.assign(st_.key_secret.size(), '\0');
  st_.key_secret.clear();
}

}  // namespace clusterlm::ui
