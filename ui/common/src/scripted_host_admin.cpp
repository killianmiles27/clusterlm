#include "clusterlm/ui/scripted_host_admin.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace clusterlm::ui {

namespace {
constexpr std::size_t kMaxImportBytes = 1024 * 1024;

Status unsupported(const char* what) {
  return make_error(ErrorCode::kUnimplemented, std::string(what) + " is not available from this Host yet.");
}

Quantity synth(double v, const char* unit, const char* source) { return Quantity{v, unit, Provenance::kSynthetic, source}; }

ProfileView example(const char* id, const char* name, const char* tier, const char* model, const char* backend,
                    std::vector<std::string> bindings, const char* fallback, const char* api_id) {
  ProfileView p;
  p.id = id;
  p.name = name;
  p.example_of = std::string("tier:") + tier;
  p.model_text = model;
  p.backend_id = backend;
  p.offered_contexts = {4096, 8192, 16384, 32768};
  p.default_context = 4096;
  p.max_context = 32768;
  p.preparation = "on-demand";
  p.release_after_idle_seconds = 60;
  p.fallback_profile_id = fallback;
  p.worker_loss_text = *fallback ? "use another profile (you will see a notice)" : "stop";
  p.api_model_id = api_id;
  SlotView host;
  host.name = "host";
  host.host = true;
  host.selector = "This PC";
  p.slots.push_back(host);
  int n = 1;
  for (auto& b : bindings) {
    SlotView s;
    s.name = "w" + std::to_string(n++);
    s.binding = b;
    s.selector = "not assigned";
    p.slots.push_back(s);
  }
  p.min_workers = p.max_workers = static_cast<std::uint32_t>(bindings.size());
  p.goals = {"Target speed: pending qualification"};
  p.readiness.state = ReadyState::kUnavailable;
  p.readiness.reasons = {std::string(model) + " is not in the model library"};
  p.readiness.blockers = {"model_missing"};
  return p;
}

}  // namespace

ModelView model_view_from_record(const library::ModelRecord& r, std::vector<ModelCompat> compat,
                                 std::vector<std::string> used_by) {
  ModelView v;
  v.id = r.id;
  v.name = r.name.empty() ? r.family : r.name;
  v.family = r.family;
  v.architecture = r.architecture;
  v.quant = r.quant;
  for (const auto& t : r.tensor_types) v.quant_mix.push_back(t.type);
  v.dir = r.dir;
  v.file_count = static_cast<std::uint32_t>(r.files.size());
  v.split = r.split;
  v.total_bytes = r.total_bytes;
  v.block_count = r.block_count;
  v.context_length = r.context_length;
  v.expert_count = r.expert_count;
  v.root_hash = r.root_hash;
  v.pinned_root = r.pinned_root;
  v.compat = std::move(compat);
  v.used_by = std::move(used_by);
  return v;
}

ScriptedHostAdminClient::ScriptedHostAdminClient() {
  MachineView host;
  host.id = "host";
  host.name = "This PC";
  host.is_host = true;
  host.state = "Available";
  machines_.push_back(host);
  server_.available = true;
}

// ---- test control ---------------------------------------------------------------------------------------------

void ScriptedHostAdminClient::set_capabilities(HostCapabilities c) { std::scoped_lock l(mu_); caps_ = c; }
void ScriptedHostAdminClient::set_models(std::vector<ModelView> m) { std::scoped_lock l(mu_); models_ = std::move(m); }
void ScriptedHostAdminClient::set_profiles(std::vector<ProfileView> p) { std::scoped_lock l(mu_); profiles_ = std::move(p); }
void ScriptedHostAdminClient::set_machines(std::vector<MachineView> m) { std::scoped_lock l(mu_); machines_ = std::move(m); }
void ScriptedHostAdminClient::set_bindings(std::vector<BindingView> b) { std::scoped_lock l(mu_); bindings_ = std::move(b); }
void ScriptedHostAdminClient::set_runs(std::vector<RunStat> r) { std::scoped_lock l(mu_); runs_ = std::move(r); }
void ScriptedHostAdminClient::set_server(ServerStatus s) { std::scoped_lock l(mu_); server_ = std::move(s); }
void ScriptedHostAdminClient::set_dry_run(DryRunReport r) { std::scoped_lock l(mu_); dry_ = std::move(r); }
void ScriptedHostAdminClient::set_scan_result(ScanReport r, std::vector<ModelView> found) {
  std::scoped_lock l(mu_);
  scan_report_ = std::move(r);
  scan_found_ = std::move(found);
}
void ScriptedHostAdminClient::fail_next(const std::string& method, Status status) {
  std::scoped_lock l(mu_);
  failures_[method] = std::move(status);
}
std::vector<std::string> ScriptedHostAdminClient::calls() const { std::scoped_lock l(mu_); return calls_; }

void ScriptedHostAdminClient::seed_example_profiles() {
  std::scoped_lock l(mu_);
  profiles_.clear();
  profiles_.push_back(example("prof_example_fast", "Fast", "fast", "Swift-1.5-Qwen3.8-27B IQ3_S", "llama-local", {}, "", "example-fast"));
  profiles_.push_back(example("prof_example_strong", "Strong", "strong", "Qwen3.8 Coder 80B-A3B", "strata-hybrid",
                              {"node:laptop-class"}, "prof_example_fast", "example-strong"));
  profiles_.push_back(example("prof_example_ultra", "Ultra", "ultra", "Qwen3.5 397B-A17B IQ3_S", "strata-hybrid",
                              {"node:laptop-class", "node:designated-3060"}, "prof_example_strong", "example-ultra"));
  bindings_ = {{"node:laptop-class", "", {"Strong", "Ultra"}}, {"node:designated-3060", "", {"Ultra"}}};
}

void ScriptedHostAdminClient::seed_demo() {
  seed_example_profiles();
  std::scoped_lock l(mu_);
  MachineView host = machines_.front();
  host.os = "Windows 11 (demo data)";
  host.cpu = "Demo CPU, 16 threads";
  host.ram_bytes = 64ULL << 30;
  host.free_ram_bytes = 40ULL << 30;
  host.gpus = {"Demo GPU, 24 GB"};
  host.vram_bytes = 24ULL << 30;
  host.free_vram_bytes = 22ULL << 30;
  host.last_seen = "just now";
  MachineView g14;
  g14.id = "ab12cd34";
  g14.name = "G14";
  g14.state = "Available";
  g14.os = "Windows 11 (demo data)";
  g14.cpu = "Demo laptop CPU, 16 threads";
  g14.ram_bytes = 32ULL << 30;
  g14.free_ram_bytes = 20ULL << 30;
  g14.gpus = {"Demo laptop GPU, 8 GB"};
  g14.vram_bytes = 8ULL << 30;
  g14.free_vram_bytes = 7ULL << 30;
  g14.last_seen = "just now";
  g14.bindings = {"node:laptop-class"};
  MachineView pc3060;
  pc3060.id = "ef56ab78";
  pc3060.name = "3060 PC";
  pc3060.state = "Someone is using this PC";
  pc3060.state_detail = "ClusterLM stays out of the way until they are done.";
  pc3060.last_seen = "just now";
  pc3060.vram_bytes = 12ULL << 30;
  pc3060.gpus = {"Demo GPU, 12 GB"};
  machines_ = {host, g14, pc3060};
  bindings_[0].machine = "ab12cd34";
  bindings_[1].machine = "ef56ab78";
  server_ = ServerStatus{};
  server_.available = true;

  // Two demo models with honest, hand-written compatibility (this is demo data, labelled as such in the UI).
  ModelView swift;
  swift.id = "mdl_demo_swift";
  swift.name = "Demo 27B model";
  swift.family = "Swift-1.5-Qwen3.8-27B";
  swift.quant = "IQ3_S";
  swift.quant_mix = {"IQ3_S", "F32"};
  swift.file_count = 1;
  swift.total_bytes = 11ULL << 30;
  swift.block_count = 64;
  swift.context_length = 32768;
  swift.dir = "D:\\Models\\demo";
  ModelCompat llama;
  llama.backend_id = "llama-local";
  llama.backend_name = "llama.cpp (this PC)";
  llama.label = CompatLabel::kSupportedAwaitingQualification;
  llama.max_workers = 0;
  llama.findings = {"Runs on this PC. Not yet tested on real hardware."};
  swift.compat = {llama};
  models_ = {swift};
  roots_ = {"D:\\Models\\demo"};
  auto& fast = profiles_[0];
  fast.model_library_id = swift.id;
  fast.model_text = "Demo 27B model, IQ3_S";
  fast.readiness = ReadinessView{};
  fast.readiness.state = ReadyState::kLoadable;
  fast.readiness.reasons = {"Ready to prepare"};
  profiles_[1].readiness.reasons = {"Qwen3.8 Coder 80B-A3B is not in the model library"};

  dry_.feasible = true;
  dry_.plan_summary = "Whole model on this PC (demo data).";
  dry_.per_machine = {MachineShare{"This PC", synth(1.0 * (1ULL << 30), "bytes", "demo"), synth(10.0 * (1ULL << 30), "bytes", "demo"),
                                   synth(0.5 * (1ULL << 30), "bytes", "demo"), synth(0.3 * (1ULL << 30), "bytes", "demo")}};
  dry_.preparation_bytes = synth(0, "bytes", "demo");
  dry_.quantities = {synth(14.0, "tok/s", "Predicted speed")};
  dry_.bottlenecks = {"Graphics memory on This PC (demo data)"};

  RunStat run;
  run.profile_id = "prof_example_fast";
  run.profile_name = "Fast";
  run.model = "Demo 27B model";
  run.when = "2 min ago";
  run.tok_s = 13.2;
  run.ttft_ms = 640;
  run.tokens = 180;
  run.reason = "completed";
  run.provenance = Provenance::kSynthetic;
  run.machines = {"This PC"};
  runs_ = {run};
}

std::string ScriptedHostAdminClient::next_id(std::string_view prefix) {
  return std::string(prefix) + "_" + std::to_string(++counter_);
}

Status ScriptedHostAdminClient::enter(const std::string& call, bool needs_cap) {
  calls_.push_back(call);
  const std::string method = call.substr(0, call.find(':'));
  if (auto it = failures_.find(method); it != failures_.end()) {
    Status s = std::move(it->second);
    failures_.erase(it);
    return s;
  }
  if (!needs_cap) return Status::ok();
  return Status::ok();
}

ProfileView* ScriptedHostAdminClient::find(std::string_view id) {
  for (auto& p : profiles_) {
    if (p.id == id) return &p;
  }
  return nullptr;
}

// ---- summary ---------------------------------------------------------------------------------------------------

Result<HostSummary> ScriptedHostAdminClient::summary() {
  std::scoped_lock l(mu_);
  if (auto s = enter("summary", false); !s.is_ok()) return s;
  return HostSummary{"This PC", true};
}
HostCapabilities ScriptedHostAdminClient::capabilities() { std::scoped_lock l(mu_); return caps_; }

// ---- models -----------------------------------------------------------------------------------------------------

Result<std::vector<ModelView>> ScriptedHostAdminClient::list_models() {
  std::scoped_lock l(mu_);
  if (auto s = enter("list_models", true); !s.is_ok()) return s;
  if (!caps_.models) return unsupported("The model library");
  auto out = models_;
  for (auto& m : out) {
    m.used_by.clear();
    for (const auto& p : profiles_) {
      if (p.model_library_id == m.id) m.used_by.push_back(p.name);
    }
  }
  return out;
}
Result<std::vector<std::string>> ScriptedHostAdminClient::scan_roots() {
  std::scoped_lock l(mu_);
  if (auto s = enter("scan_roots", true); !s.is_ok()) return s;
  return roots_;
}
Status ScriptedHostAdminClient::add_scan_root(std::string_view dir) {
  std::scoped_lock l(mu_);
  if (auto s = enter("add_scan_root:" + std::string(dir), true); !s.is_ok()) return s;
  if (dir.empty()) return make_error(ErrorCode::kInvalidArgument, "Choose a folder first.");
  if (std::find(roots_.begin(), roots_.end(), dir) != roots_.end()) return make_error(ErrorCode::kAlreadyExists, "That folder is already watched.");
  roots_.emplace_back(dir);
  return Status::ok();
}
Status ScriptedHostAdminClient::remove_scan_root(std::string_view dir) {
  std::scoped_lock l(mu_);
  if (auto s = enter("remove_scan_root:" + std::string(dir), true); !s.is_ok()) return s;
  auto it = std::find(roots_.begin(), roots_.end(), dir);
  if (it == roots_.end()) return make_error(ErrorCode::kNotFound, "That folder is not in the list.");
  roots_.erase(it);
  return Status::ok();
}
Result<ScanReport> ScriptedHostAdminClient::rescan() {
  std::scoped_lock l(mu_);
  if (auto s = enter("rescan", true); !s.is_ok()) return s;
  for (auto& m : scan_found_) {
    auto it = std::find_if(models_.begin(), models_.end(), [&](const ModelView& x) { return x.id == m.id; });
    if (it == models_.end()) models_.push_back(m);
  }
  scan_found_.clear();
  return scan_report_;
}
Result<ModelView> ScriptedHostAdminClient::import_model(std::string_view path) {
  std::scoped_lock l(mu_);
  if (auto s = enter("import_model:" + std::string(path), true); !s.is_ok()) return s;
  if (!caps_.models) return unsupported("Importing models");
  if (path.empty()) return make_error(ErrorCode::kInvalidArgument, "Enter the path of a .gguf file.");
  if (path.size() < 5 || path.substr(path.size() - 5) != ".gguf") return make_error(ErrorCode::kInvalidArgument, "That file is not a .gguf model.");
  ModelView v;
  v.id = next_id("mdl");
  v.name = std::string(path.substr(path.find_last_of("/\\") == std::string_view::npos ? 0 : path.find_last_of("/\\") + 1));
  v.dir = std::string(path.substr(0, path.find_last_of("/\\") == std::string_view::npos ? 0 : path.find_last_of("/\\")));
  v.file_count = 1;
  models_.push_back(v);
  return v;
}
Status ScriptedHostAdminClient::pin_model_root(std::string_view model_id, std::string_view root_hash) {
  std::scoped_lock l(mu_);
  if (auto s = enter("pin_model_root:" + std::string(model_id), true); !s.is_ok()) return s;
  if (root_hash.size() != 64 || !std::all_of(root_hash.begin(), root_hash.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); }))
    return make_error(ErrorCode::kInvalidArgument, "A confirmed root is 64 hexadecimal characters.");
  for (auto& m : models_) {
    if (m.id == model_id) { m.pinned_root = std::string(root_hash); return Status::ok(); }
  }
  return make_error(ErrorCode::kNotFound, "That model is not in the library.");
}
Status ScriptedHostAdminClient::remove_model(std::string_view model_id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("remove_model:" + std::string(model_id), true); !s.is_ok()) return s;
  for (const auto& p : profiles_) {
    if (p.model_library_id == model_id) return make_error(ErrorCode::kFailedPrecondition, "Profile \"" + p.name + "\" uses this model. Change or delete it first.");
  }
  auto it = std::find_if(models_.begin(), models_.end(), [&](const ModelView& x) { return x.id == model_id; });
  if (it == models_.end()) return make_error(ErrorCode::kNotFound, "That model is not in the library.");
  models_.erase(it);
  return Status::ok();
}

// ---- profiles -----------------------------------------------------------------------------------------------------

Result<std::vector<ProfileView>> ScriptedHostAdminClient::list_profiles(std::uint32_t) {
  std::scoped_lock l(mu_);
  if (auto s = enter("list_profiles", true); !s.is_ok()) return s;
  if (!caps_.profiles) return unsupported("Profiles");
  auto out = profiles_;
  for (auto& p : out) p.selected = (p.id == selected_);
  return out;
}

namespace {
std::vector<Finding> check_draft(const ProfileDraft& d, const std::vector<ModelView>& models,
                                 const std::vector<ProfileView>& profiles, std::string_view self_id) {
  std::vector<Finding> f;
  auto add = [&](std::uint8_t level, const char* code, std::string msg, bool blocking = true) {
    f.push_back(Finding{level, code, std::move(msg), blocking});
  };
  if (d.name.empty()) add(1, "name_required", "Give the profile a name.");
  if (d.name.size() > 64) add(1, "name_too_long", "The name can be at most 64 characters.");
  if (d.default_context == 0 || d.default_context > d.max_context)
    add(1, "context_range", "The starting context must be larger than zero and no larger than the maximum.");
  if (d.model_library_id.empty()) add(2, "model_required", "Choose a model from the library.");
  const ModelView* model = nullptr;
  for (const auto& m : models) if (m.id == d.model_library_id) model = &m;
  if (!d.model_library_id.empty() && !model) add(2, "model_unknown", "That model is no longer in the library.");
  if (d.backend_id.empty()) add(2, "backend_required", "Choose how the model is run.");
  if (model && !d.backend_id.empty()) {
    const ModelCompat* c = nullptr;
    for (const auto& x : model->compat) if (x.backend_id == d.backend_id) c = &x;
    if (!c) add(2, "compat_unchecked", "Compatibility of this model with that backend has not been checked.", false);
    else if (c->label == CompatLabel::kUnsupported) add(2, "unsupported", "This model is not supported by that backend: " + (c->findings.empty() ? std::string("no reason given") : c->findings.front()));
    else if (!c->backend_built || !c->runtime_present) add(2, "backend_missing", "That backend is not available on this PC.");
    else if (d.worker_bindings.size() > c->max_workers) add(2, "too_many_workers", "This model can be spread over at most " + std::to_string(c->max_workers) + " other PC(s) with that backend.");
    if (model->context_length && d.max_context > *model->context_length) add(2, "context_beyond_model", "The model was trained for " + std::to_string(*model->context_length) + " tokens at most.", false);
  }
  if (d.min_workers > d.worker_bindings.size()) add(1, "min_workers", "Cannot require more PCs than there are worker slots.");
  if (d.api_enabled && d.api_model_id.empty()) add(1, "api_id_required", "Give the model a name for apps to use.");
  if (!d.api_model_id.empty()) {
    if (d.api_model_id.rfind("clusterlm-", 0) == 0) add(1, "api_id_reserved", "Names starting with \"clusterlm-\" are reserved.");
    if (d.api_model_id == "auto" || d.api_model_id == "default") add(1, "api_id_reserved", "\"auto\" and \"default\" are reserved for routing aliases.");
    for (const auto& p : profiles) if (p.id != self_id && p.api_model_id == d.api_model_id) add(1, "api_id_taken", "Another profile already uses that name.");
  }
  if (d.allow_lan && !d.api_enabled) add(1, "lan_without_api", "Turn on app access before allowing other PCs to see this profile.");
  return f;
}
}  // namespace

namespace {
void apply_draft(ProfileView& p, const ProfileDraft& d, const std::vector<ModelView>& models) {
  p.name = d.name;
  p.model_library_id = d.model_library_id;
  for (const auto& m : models) {
    if (m.id == d.model_library_id) p.model_text = m.name + (m.quant.empty() ? "" : ", " + m.quant);
  }
  p.backend_id = d.backend_id;
  p.default_context = d.default_context;
  p.max_context = d.max_context;
  p.offered_contexts.clear();
  for (std::uint32_t c = 4096; c < d.max_context; c *= 2) p.offered_contexts.push_back(c);
  p.offered_contexts.push_back(d.max_context);
  p.slots.clear();
  SlotView host;
  host.name = "host";
  host.host = true;
  host.selector = "This PC";
  p.slots.push_back(host);
  int n = 1;
  for (const auto& b : d.worker_bindings) {
    SlotView s;
    s.name = "w" + std::to_string(n++);
    s.binding = b;
    s.selector = "not assigned";
    p.slots.push_back(s);
  }
  p.min_workers = d.min_workers;
  p.max_workers = static_cast<std::uint32_t>(d.worker_bindings.size());
  p.preparation = d.preparation;
  p.release_after_idle_seconds = d.release_after_idle_seconds;
  p.fallback_profile_id = d.fallback_profile_id;
  p.worker_loss_text = d.fallback_profile_id.empty() ? "stop" : "use another profile (you will see a notice)";
  p.api_enabled = d.api_enabled;
  p.allow_lan = d.allow_lan && d.api_enabled;
  p.api_model_id = d.api_model_id;
  ++p.revision;
  p.readiness = ReadinessView{};
  p.readiness.state = ReadyState::kCompatible;
  p.readiness.reasons = {"Saved. Waiting for the Host to check it."};
}
}  // namespace

Result<std::string> ScriptedHostAdminClient::create_profile(const ProfileDraft& d) {
  std::scoped_lock l(mu_);
  if (auto s = enter("create_profile", true); !s.is_ok()) return s;
  if (!caps_.profiles) return unsupported("Profiles");
  for (const auto& f : check_draft(d, models_, profiles_, "")) {
    if (f.blocking) return make_error(ErrorCode::kInvalidArgument, f.message);
  }
  ProfileView p;
  p.id = next_id("prof");
  p.revision = 0;
  apply_draft(p, d, models_);
  profiles_.push_back(p);
  return p.id;
}
Status ScriptedHostAdminClient::update_profile(std::string_view id, const ProfileDraft& d) {
  std::scoped_lock l(mu_);
  if (auto s = enter("update_profile:" + std::string(id), true); !s.is_ok()) return s;
  auto* p = find(id);
  if (!p) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  for (const auto& f : check_draft(d, models_, profiles_, id)) {
    if (f.blocking) return make_error(ErrorCode::kInvalidArgument, f.message);
  }
  apply_draft(*p, d, models_);
  return Status::ok();
}
Status ScriptedHostAdminClient::delete_profile(std::string_view id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("delete_profile:" + std::string(id), true); !s.is_ok()) return s;
  auto it = std::find_if(profiles_.begin(), profiles_.end(), [&](const ProfileView& p) { return p.id == id; });
  if (it == profiles_.end()) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  for (const auto& p : profiles_) {
    if (p.fallback_profile_id == id) return make_error(ErrorCode::kFailedPrecondition, "Profile \"" + p.name + "\" falls back to this one. Change that first.");
  }
  if (selected_ == id) selected_.clear();
  profiles_.erase(it);
  return Status::ok();
}
Result<std::string> ScriptedHostAdminClient::duplicate_profile(std::string_view id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("duplicate_profile:" + std::string(id), true); !s.is_ok()) return s;
  auto* p = find(id);
  if (!p) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  ProfileView c = *p;
  c.id = next_id("prof");
  c.name += " (copy)";
  c.revision = 1;
  c.example_of.clear();
  c.api_enabled = false;
  c.allow_lan = false;
  c.api_model_id = c.id;
  c.selected = false;
  profiles_.push_back(c);
  return c.id;
}
Result<std::string> ScriptedHostAdminClient::export_profile(std::string_view id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("export_profile:" + std::string(id), true); !s.is_ok()) return s;
  auto* p = find(id);
  if (!p) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  nlohmann::json j;
  j["schema"] = "clusterlm.profile.scripted";
  j["id"] = p->id;
  j["name"] = p->name;
  j["revision"] = p->revision;
  j["model"] = p->model_text;
  j["backend"] = p->backend_id;
  j["default_context"] = p->default_context;
  j["max_context"] = p->max_context;
  nlohmann::json bindings = nlohmann::json::array();
  for (const auto& s : p->slots) if (!s.host) bindings.push_back(s.binding);
  j["worker_bindings"] = bindings;
  j["preparation"] = p->preparation;
  j["api_model_id"] = p->api_model_id;
  // Rules of profile-schema-v1 section 7: no library id, no machine, no path, no exposure.
  return j.dump(2);
}
Result<ImportResult> ScriptedHostAdminClient::import_profile(std::string_view document, ImportChoice choice) {
  std::scoped_lock l(mu_);
  if (auto s = enter("import_profile", true); !s.is_ok()) return s;
  if (document.size() > kMaxImportBytes) return make_error(ErrorCode::kInvalidArgument, "That file is larger than 1 MB, so it is not a profile.");
  auto j = nlohmann::json::parse(document, nullptr, false);
  if (j.is_discarded() || !j.is_object() || j.value("schema", "") != "clusterlm.profile.scripted" || !j.contains("name"))
    return make_error(ErrorCode::kInvalidArgument, "That is not a ClusterLM profile.");
  ImportResult r;
  const std::string id = j.value("id", "");
  ProfileView p;
  p.name = j.value("name", "");
  p.model_text = j.value("model", "");
  p.backend_id = j.value("backend", "");
  p.default_context = j.value("default_context", 4096u);
  p.max_context = j.value("max_context", 4096u);
  p.preparation = j.value("preparation", "on-demand");
  p.api_model_id = j.value("api_model_id", "");
  p.revision = j.value("revision", 1u);
  SlotView host;
  host.name = "host";
  host.host = true;
  host.selector = "This PC";
  p.slots.push_back(host);
  int n = 1;
  for (const auto& b : j.value("worker_bindings", nlohmann::json::array())) {
    if (!b.is_string()) continue;
    SlotView s;
    s.name = "w" + std::to_string(n++);
    s.binding = b.get<std::string>();
    s.selector = "not assigned";
    p.slots.push_back(s);
    r.todo.push_back("Assign a PC to \"" + s.binding + "\" in Machines.");
  }
  r.todo.push_back("Choose the model \"" + p.model_text + "\" from your library (import it first if you do not have it).");
  auto* existing = id.empty() ? nullptr : find(id);
  if (existing) {
    if (existing->name == p.name && existing->revision == p.revision && existing->model_text == p.model_text) {
      r.profile_id = existing->id;
      r.todo.clear();
      return r;  // same id, same content: nothing to do
    }
    if (choice == ImportChoice::kAsk) {
      r.conflict = true;
      r.profile_id = existing->id;
      return r;
    }
    if (choice == ImportChoice::kReplace) {
      p.id = existing->id;
      p.revision = existing->revision + 1;
      *existing = p;
      r.profile_id = p.id;
      return r;
    }
  }
  p.id = next_id("prof");
  p.api_enabled = false;  // an import never turns exposure on
  p.allow_lan = false;
  p.readiness.state = ReadyState::kUnavailable;
  p.readiness.reasons = {"Choose a model for this profile"};
  p.readiness.blockers = {"model_missing"};
  profiles_.push_back(p);
  r.profile_id = p.id;
  return r;
}
Result<std::vector<Finding>> ScriptedHostAdminClient::validate_profile(std::string_view id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("validate_profile:" + std::string(id), true); !s.is_ok()) return s;
  auto* p = find(id);
  if (!p) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  std::vector<Finding> f;
  for (const auto& r : p->readiness.reasons) {
    if (p->readiness.state == ReadyState::kUnavailable) f.push_back(Finding{2, p->readiness.blockers.empty() ? "blocked" : p->readiness.blockers.front(), r, true});
  }
  return f;
}
Result<std::vector<Finding>> ScriptedHostAdminClient::validate_draft(const ProfileDraft& d) {
  std::scoped_lock l(mu_);
  if (auto s = enter("validate_draft", true); !s.is_ok()) return s;
  return check_draft(d, models_, profiles_, "");
}
Result<DryRunReport> ScriptedHostAdminClient::dry_run(std::string_view id, std::uint32_t context_tokens) {
  std::scoped_lock l(mu_);
  if (auto s = enter("dry_run:" + std::string(id), true); !s.is_ok()) return s;
  if (!caps_.dry_run) return unsupported("The placement preview");
  if (!find(id)) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  DryRunReport r = dry_;
  r.context_tokens = context_tokens;
  return r;
}
Status ScriptedHostAdminClient::select_profile(std::string_view id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("select_profile:" + std::string(id), true); !s.is_ok()) return s;
  if (!find(id)) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  selected_ = std::string(id);
  return Status::ok();
}
Status ScriptedHostAdminClient::prepare_profile(std::string_view id, std::uint32_t context_tokens) {
  std::scoped_lock l(mu_);
  if (auto s = enter("prepare_profile:" + std::string(id) + ":" + std::to_string(context_tokens), true); !s.is_ok()) return s;
  auto* p = find(id);
  if (!p) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  if (p->readiness.state != ReadyState::kLoadable) return make_error(ErrorCode::kFailedPrecondition, p->readiness.reasons.empty() ? "This profile cannot be prepared yet." : p->readiness.reasons.front());
  p->readiness.state = ReadyState::kPreparing;
  p->readiness.percent = 0.0;
  return Status::ok();
}
Status ScriptedHostAdminClient::release_profile(std::string_view id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("release_profile:" + std::string(id), true); !s.is_ok()) return s;
  auto* p = find(id);
  if (!p) return make_error(ErrorCode::kNotFound, "That profile no longer exists.");
  if (p->readiness.state == ReadyState::kReady || p->readiness.state == ReadyState::kPreparing) p->readiness.state = ReadyState::kLoadable;
  return Status::ok();
}

// ---- machines -----------------------------------------------------------------------------------------------------

Result<std::vector<MachineView>> ScriptedHostAdminClient::list_machines() {
  std::scoped_lock l(mu_);
  if (auto s = enter("list_machines", true); !s.is_ok()) return s;
  if (!caps_.machines) return unsupported("The machine list");
  auto out = machines_;
  for (auto& m : out) {
    m.bindings.clear();
    for (const auto& b : bindings_) if (b.machine == m.id) m.bindings.push_back(b.name);
  }
  return out;
}
Result<std::vector<BindingView>> ScriptedHostAdminClient::list_bindings() {
  std::scoped_lock l(mu_);
  if (auto s = enter("list_bindings", true); !s.is_ok()) return s;
  return bindings_;
}
Status ScriptedHostAdminClient::set_binding(std::string_view name, std::string_view machine_id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("set_binding:" + std::string(name) + ":" + std::string(machine_id), true); !s.is_ok()) return s;
  if (!machine_id.empty()) {
    auto it = std::find_if(machines_.begin(), machines_.end(), [&](const MachineView& m) { return m.id == machine_id && !m.is_host; });
    if (it == machines_.end()) return make_error(ErrorCode::kNotFound, "That PC is not paired.");
  }
  for (auto& b : bindings_) {
    if (b.name == name) { b.machine = std::string(machine_id); return Status::ok(); }
  }
  bindings_.push_back(BindingView{std::string(name), std::string(machine_id), {}});
  return Status::ok();
}

// ---- connections -------------------------------------------------------------------------------------------------

Result<ServerStatus> ScriptedHostAdminClient::server_status() {
  std::scoped_lock l(mu_);
  if (auto s = enter("server_status", true); !s.is_ok()) return s;
  ServerStatus out = server_;
  out.exposed_models.clear();
  for (const auto& p : profiles_) if (p.api_enabled && !p.api_model_id.empty()) out.exposed_models.push_back(p.api_model_id);
  return out;
}
Status ScriptedHostAdminClient::set_server(const ServerSettings& s) {
  std::scoped_lock l(mu_);
  if (auto st = enter("set_server", true); !st.is_ok()) return st;
  if (!server_.available) return unsupported("The app connection server");
  if (s.port < 1024) return make_error(ErrorCode::kInvalidArgument, "Use a port number of 1024 or higher.");
  if (s.allow_lan && !s.tls) return make_error(ErrorCode::kFailedPrecondition, "Other PCs can only connect over an encrypted connection. Turn on encryption first.");
  server_.settings = s;
  server_.listening = s.enabled;
  server_.address = s.enabled ? (s.allow_lan ? "0.0.0.0:" : "127.0.0.1:") + std::to_string(s.port) : "";
  server_.tls_fingerprint = s.tls ? "SCRIPTED:00:00:00" : "";
  return Status::ok();
}
Result<std::vector<ApiKeyView>> ScriptedHostAdminClient::list_keys() {
  std::scoped_lock l(mu_);
  if (auto s = enter("list_keys", true); !s.is_ok()) return s;
  return keys_;
}
Result<NewKey> ScriptedHostAdminClient::create_key(std::string_view name, const std::vector<std::string>& scopes,
                                                    const std::vector<std::string>& allowed_models, bool lan_allowed) {
  std::scoped_lock l(mu_);
  if (auto s = enter("create_key", true); !s.is_ok()) return s;
  if (!caps_.keys) return unsupported("App keys");
  if (name.empty()) return make_error(ErrorCode::kInvalidArgument, "Name the app this key is for.");
  if (scopes.empty()) return make_error(ErrorCode::kInvalidArgument, "Choose what the key may do.");
  NewKey k;
  k.key.id = next_id("key");
  k.key.name = std::string(name);
  k.key.scopes = scopes;
  k.key.allowed_models = allowed_models.empty() ? std::vector<std::string>{"*"} : allowed_models;
  k.key.lan_allowed = lan_allowed;
  k.key.created = "just now";
  k.secret = "clk_" + k.key.id + "_scriptednotarealsecret";
  keys_.push_back(k.key);
  return k;
}
Status ScriptedHostAdminClient::revoke_key(std::string_view key_id) {
  std::scoped_lock l(mu_);
  if (auto s = enter("revoke_key:" + std::string(key_id), true); !s.is_ok()) return s;
  for (auto& k : keys_) if (k.id == key_id) { k.revoked = true; return Status::ok(); }
  return make_error(ErrorCode::kNotFound, "That key no longer exists.");
}

// ---- performance / settings ------------------------------------------------------------------------------------

Result<std::vector<RunStat>> ScriptedHostAdminClient::recent_runs() {
  std::scoped_lock l(mu_);
  if (auto s = enter("recent_runs", true); !s.is_ok()) return s;
  return runs_;
}
Result<std::string> ScriptedHostAdminClient::export_backup() {
  std::scoped_lock l(mu_);
  if (auto s = enter("export_backup", true); !s.is_ok()) return s;
  if (!caps_.backup) return unsupported("Backup");
  nlohmann::json j;
  j["schema"] = "clusterlm.backup.scripted";
  nlohmann::json ps = nlohmann::json::array();
  for (const auto& p : profiles_) ps.push_back({{"id", p.id}, {"name", p.name}});
  j["profiles"] = ps;
  nlohmann::json roots = nlohmann::json::array();
  for (const auto& r : roots_) roots.push_back(r);
  j["scan_roots"] = roots;
  return j.dump(2);  // configuration and library metadata only: no weights, keys or conversations
}
Status ScriptedHostAdminClient::import_backup(std::string_view document) {
  std::scoped_lock l(mu_);
  if (auto s = enter("import_backup", true); !s.is_ok()) return s;
  if (!caps_.backup) return unsupported("Restore");
  if (document.size() > 4 * kMaxImportBytes) return make_error(ErrorCode::kInvalidArgument, "That file is too large to be a ClusterLM backup.");
  auto j = nlohmann::json::parse(document, nullptr, false);
  if (j.is_discarded() || !j.is_object() || j.value("schema", "") != "clusterlm.backup.scripted")
    return make_error(ErrorCode::kInvalidArgument, "That is not a ClusterLM backup.");
  return Status::ok();
}

}  // namespace clusterlm::ui
