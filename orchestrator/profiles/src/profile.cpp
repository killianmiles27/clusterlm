#include "clusterlm/profiles/profile.hpp"

#include <algorithm>

#include "strict_json.hpp"

namespace clusterlm::profiles {

using detail::bad;
using detail::Obj;
using nlohmann::json;

namespace {

constexpr std::uint64_t kMaxJsInt = 9007199254740991ull;
constexpr std::uint32_t kMaxMib = 16777216;

const std::vector<std::string_view> kDevice = {"auto", "gpu-first", "gpu-only", "cpu-only"};
const std::vector<std::string_view> kPlaceMode = {"auto", "manual"};
const std::vector<std::string_view> kObjective = {"balanced", "latency", "prep-cost"};
const std::vector<std::string_view> kPrep = {"on-demand", "keep-ready", "manual"};
const std::vector<std::string_view> kThen = {"stop", "replan", "fallback-profile"};
const std::vector<std::string_view> kPolicy = {"first-viable", "best-ready"};
const std::vector<std::string_view> kSelMode = {"binding", "machine", "requirements"};
const std::vector<std::string_view> kKind = {"host", "worker"};

template <typename E>
Status read_enum(Obj& o, const char* key, const std::vector<std::string_view>& names, E& out) {
  std::size_t idx = static_cast<std::size_t>(out);
  CLM_RETURN_IF_ERROR(o.opt_enum(key, names, idx));
  out = static_cast<E>(idx);
  return Status::ok();
}

// ---- readers ----------------------------------------------------------------------------------------------------

Result<Requirements> read_requirements(Obj& o) {
  Requirements r;
  CLM_RETURN_IF_ERROR(o.opt_string_array("gpu_vendor", r.gpu_vendor, 3, 16));
  CLM_RETURN_IF_ERROR(o.opt_opt_u32("min_vram_mib", r.min_vram_mib, 0, kMaxMib));
  CLM_RETURN_IF_ERROR(o.opt_opt_u32("min_ram_mib", r.min_ram_mib, 0, kMaxMib));
  CLM_RETURN_IF_ERROR(o.opt_nullable_str("min_compute_capability", r.min_compute_capability, 8));
  CLM_RETURN_IF_ERROR(o.opt_opt_u32("min_link_mbit", r.min_link_mbit, 0, 1000000));
  CLM_RETURN_IF_ERROR(o.opt_bool("cpu_only_ok", r.cpu_only_ok));
  CLM_RETURN_IF_ERROR(o.finish());
  return r;
}

Result<Selector> read_selector(Obj& o) {
  Selector s;
  std::size_t mode = 0;
  if (!o.has("mode")) return bad(o.path() + ".mode is required");
  CLM_RETURN_IF_ERROR(o.opt_enum("mode", kSelMode, mode));
  s.mode = static_cast<Selector::Mode>(mode);
  CLM_RETURN_IF_ERROR(o.opt_str("binding", s.binding, 48));
  CLM_RETURN_IF_ERROR(o.opt_str("machine", s.machine, 64));
  CLM_ASSIGN_OR_RETURN(auto req, o.opt_obj("requirements"));
  if (req) {
    CLM_ASSIGN_OR_RETURN(auto r, read_requirements(*req));
    s.requirements = std::move(r);
  }
  CLM_RETURN_IF_ERROR(o.finish());
  return s;
}

Result<Profile> read_profile(const json& doc) {
  CLM_ASSIGN_OR_RETURN(Obj o, Obj::of(doc, "profile"));
  Profile p;
  std::string schema;
  CLM_RETURN_IF_ERROR(o.req_str("schema", schema, 32));
  if (schema != "clusterlm.profile") return bad("profile.schema must be 'clusterlm.profile'");
  std::uint64_t ver = 0;
  CLM_RETURN_IF_ERROR(o.req_u64("schema_version", ver, 0, 1u << 30));
  if (ver != 1)
    return make_error(ErrorCode::kVersionMismatch, "profile schema_version " + std::to_string(ver) + " needs a newer ClusterLM");
  CLM_RETURN_IF_ERROR(o.req_str("id", p.id, 64));
  CLM_RETURN_IF_ERROR(o.req_str("name", p.name, 80));
  CLM_RETURN_IF_ERROR(o.opt_str("description", p.description, 1000));
  CLM_RETURN_IF_ERROR(o.req_u64("revision", p.revision, 1, kMaxJsInt));
  CLM_RETURN_IF_ERROR(o.opt_nullable_str("example_of", p.example_of, 64));
  CLM_RETURN_IF_ERROR(o.opt_string_array("qualification_experiments", p.qualification_experiments, 16, 32));

  {  // model
    CLM_ASSIGN_OR_RETURN(auto m, o.opt_obj("model"));
    if (!m) return bad("profile.model is required");
    CLM_ASSIGN_OR_RETURN(auto id, m->opt_obj("identity"));
    if (!id) return bad("profile.model.identity is required");
    auto& i = p.model.identity;
    CLM_RETURN_IF_ERROR(id->req_str("family", i.family, 120));
    CLM_RETURN_IF_ERROR(id->opt_str("display_name", i.display_name, 200));
    CLM_RETURN_IF_ERROR(id->req_str("quant", i.quant, 32));
    CLM_RETURN_IF_ERROR(id->opt_nullable_str("artifact_id", i.artifact_id, 200));
    CLM_RETURN_IF_ERROR(id->opt_nullable_str("expected_root_hash", i.expected_root_hash, 64));
    CLM_ASSIGN_OR_RETURN(const json* files, id->opt_array("expected_files", 64));
    if (files != nullptr) {
      for (const auto& fj : *files) {
        CLM_ASSIGN_OR_RETURN(Obj fo, Obj::of(fj, "profile.model.identity.expected_files[]"));
        ExpectedFile f;
        CLM_RETURN_IF_ERROR(fo.req_str("role", f.role, 32));
        CLM_RETURN_IF_ERROR(fo.opt_nullable_str("name", f.name, 255));
        CLM_RETURN_IF_ERROR(fo.opt_nullable_u64("approx_bytes", f.approx_bytes, kMaxJsInt));
        CLM_RETURN_IF_ERROR(fo.finish());
        i.expected_files.push_back(std::move(f));
      }
    }
    CLM_RETURN_IF_ERROR(id->finish());
    CLM_RETURN_IF_ERROR(m->opt_nullable_str("library_id", p.model.library_id, 40));
    CLM_RETURN_IF_ERROR(m->finish());
  }
  {  // backend
    CLM_ASSIGN_OR_RETURN(auto b, o.opt_obj("backend"));
    if (!b) return bad("profile.backend is required");
    CLM_RETURN_IF_ERROR(b->req_str("id", p.backend.id, 32));
    CLM_RETURN_IF_ERROR(b->opt_u32("min_contract_version", p.backend.min_contract_version, 1, 1000));
    CLM_ASSIGN_OR_RETURN(const json* opts, b->opt_raw_object("options", 16));
    if (opts != nullptr) {
      for (auto it = opts->begin(); it != opts->end(); ++it) {  // std::map order = sorted by key
        BackendRef::Option opt;
        opt.key = it.key();
        const json& v = it.value();
        if (v.is_boolean()) opt.value = v.get<bool>();
        else if (v.is_number_integer()) opt.value = v.get<std::int64_t>();
        else if (v.is_string() && v.get<std::string>().size() <= 64) opt.value = v.get<std::string>();
        else return bad("profile.backend.options." + it.key().substr(0, 32) + " must be an integer, boolean or short string");
        p.backend.options.push_back(std::move(opt));
      }
    }
    CLM_RETURN_IF_ERROR(b->finish());
  }
  {  // context
    CLM_ASSIGN_OR_RETURN(auto c, o.opt_obj("context"));
    if (!c) return bad("profile.context is required");
    std::uint64_t d = 0, m = 0;
    CLM_RETURN_IF_ERROR(c->req_u64("default_tokens", d, 256, 4194304));
    CLM_RETURN_IF_ERROR(c->req_u64("max_tokens", m, 256, 4194304));
    p.context.default_tokens = static_cast<std::uint32_t>(d);
    p.context.max_tokens = static_cast<std::uint32_t>(m);
    CLM_ASSIGN_OR_RETURN(const json* off, c->opt_array("offered_profiles", 16));
    if (off != nullptr)
      for (const auto& e : *off) {
        if (!e.is_number_unsigned() || e.get<std::uint64_t>() < 256 || e.get<std::uint64_t>() > 4194304)
          return bad("profile.context.offered_profiles items must be integers in 256..4194304");
        p.context.offered_profiles.push_back(e.get<std::uint32_t>());
      }
    CLM_RETURN_IF_ERROR(c->finish());
  }
  {  // topology
    CLM_ASSIGN_OR_RETURN(auto t, o.opt_obj("topology"));
    if (!t) return bad("profile.topology is required");
    CLM_ASSIGN_OR_RETURN(const json* slots, t->req_array("slots", 1, 1 + kMaxWorkerSlots));
    for (const auto& sj : *slots) {
      CLM_ASSIGN_OR_RETURN(Obj so, Obj::of(sj, "profile.topology.slots[]"));
      Slot s;
      CLM_RETURN_IF_ERROR(so.req_str("slot", s.slot, 32));
      std::string kind;
      CLM_RETURN_IF_ERROR(so.req_str("kind", kind, 16));
      if (kind == "host") s.kind = Slot::Kind::kHost;
      else if (kind == "worker") s.kind = Slot::Kind::kWorker;
      else return bad("profile.topology.slots[].kind must be host or worker");
      CLM_RETURN_IF_ERROR(so.opt_str("label", s.label, 80));
      CLM_RETURN_IF_ERROR(so.opt_bool("optional", s.optional));
      CLM_ASSIGN_OR_RETURN(auto sel, so.opt_obj("select"));
      if (sel) {
        CLM_ASSIGN_OR_RETURN(auto parsed, read_selector(*sel));
        s.select = std::move(parsed);
      }
      CLM_RETURN_IF_ERROR(so.finish());
      p.topology.slots.push_back(std::move(s));
    }
    CLM_RETURN_IF_ERROR(t->opt_opt_u32("min_workers", p.topology.min_workers, 0, 8));
    CLM_RETURN_IF_ERROR(t->opt_opt_u32("max_workers", p.topology.max_workers, 0, 8));
    CLM_RETURN_IF_ERROR(t->finish());
  }
  CLM_ASSIGN_OR_RETURN(auto res, o.opt_obj("resources"));
  if (res) {
    CLM_RETURN_IF_ERROR(read_enum(*res, "device_preference", kDevice, p.resources.device_preference));
    CLM_RETURN_IF_ERROR(res->opt_opt_u32("host_vram_margin_mib", p.resources.host_vram_margin_mib, 0, kMaxMib));
    CLM_RETURN_IF_ERROR(res->opt_opt_u32("host_ram_margin_mib", p.resources.host_ram_margin_mib, 0, kMaxMib));
    CLM_RETURN_IF_ERROR(res->opt_opt_u32("worker_vram_margin_mib", p.resources.worker_vram_margin_mib, 0, kMaxMib));
    CLM_RETURN_IF_ERROR(res->opt_opt_u32("worker_ram_margin_mib", p.resources.worker_ram_margin_mib, 0, kMaxMib));
    CLM_RETURN_IF_ERROR(res->opt_opt_u32("max_worker_temp_storage_mib", p.resources.max_worker_temp_storage_mib, 0, kMaxMib));
    CLM_RETURN_IF_ERROR(res->finish());
  }
  CLM_ASSIGN_OR_RETURN(auto pl, o.opt_obj("placement"));
  if (pl) {
    CLM_RETURN_IF_ERROR(read_enum(*pl, "mode", kPlaceMode, p.placement.mode));
    CLM_RETURN_IF_ERROR(read_enum(*pl, "objective", kObjective, p.placement.objective));
    CLM_ASSIGN_OR_RETURN(auto man, pl->opt_obj("manual"));
    if (man) {
      CLM_ASSIGN_OR_RETURN(const json* stages, man->req_array("stages", 1, 1 + kMaxWorkerSlots));
      for (const auto& sj : *stages) {
        CLM_ASSIGN_OR_RETURN(Obj so, Obj::of(sj, "profile.placement.manual.stages[]"));
        ManualStage st;
        CLM_RETURN_IF_ERROR(so.req_str("slot", st.slot, 32));
        std::uint64_t f = 0, e = 0;
        CLM_RETURN_IF_ERROR(so.req_u64("first_layer", f, 0, 65535));
        CLM_RETURN_IF_ERROR(so.req_u64("end_layer", e, 1, 65536));
        st.first_layer = static_cast<std::uint32_t>(f);
        st.end_layer = static_cast<std::uint32_t>(e);
        CLM_RETURN_IF_ERROR(so.finish());
        p.placement.manual.push_back(std::move(st));
      }
      CLM_RETURN_IF_ERROR(man->finish());
    }
    CLM_RETURN_IF_ERROR(pl->finish());
  }
  CLM_ASSIGN_OR_RETURN(auto sp, o.opt_obj("speculation"));
  if (sp) {
    CLM_RETURN_IF_ERROR(sp->opt_bool("enabled", p.speculation.enabled));
    CLM_RETURN_IF_ERROR(sp->opt_u32("max_q", p.speculation.max_q, 1, 16));
    CLM_RETURN_IF_ERROR(sp->finish());
  }
  CLM_ASSIGN_OR_RETURN(auto ct, o.opt_obj("chat_template"));
  if (ct) {
    CLM_RETURN_IF_ERROR(ct->opt_str("source", p.chat_template_source, 64));
    CLM_RETURN_IF_ERROR(ct->finish());
  }
  CLM_ASSIGN_OR_RETURN(auto lc, o.opt_obj("lifecycle"));
  if (lc) {
    CLM_RETURN_IF_ERROR(read_enum(*lc, "preparation", kPrep, p.lifecycle.preparation));
    CLM_RETURN_IF_ERROR(lc->opt_u32("release_after_idle_seconds", p.lifecycle.release_after_idle_seconds, 0, 604800));
    CLM_RETURN_IF_ERROR(lc->opt_bool("prepare_when_available", p.lifecycle.prepare_when_available));
    CLM_RETURN_IF_ERROR(lc->finish());
  }
  CLM_ASSIGN_OR_RETURN(auto wl, o.opt_obj("on_worker_loss"));
  if (wl) {
    CLM_RETURN_IF_ERROR(wl->opt_u32("max_retries", p.on_worker_loss.max_retries, 0, 5));
    CLM_RETURN_IF_ERROR(read_enum(*wl, "then", kThen, p.on_worker_loss.then));
    CLM_RETURN_IF_ERROR(wl->opt_nullable_str("fallback_profile_id", p.on_worker_loss.fallback_profile_id, 64));
    CLM_RETURN_IF_ERROR(wl->opt_bool("preserve_conversation", p.on_worker_loss.preserve_conversation));
    CLM_RETURN_IF_ERROR(wl->finish());
  }
  CLM_ASSIGN_OR_RETURN(auto ex, o.opt_obj("exposure"));
  if (ex) {
    CLM_RETURN_IF_ERROR(ex->opt_bool("api", p.exposure.api));
    CLM_RETURN_IF_ERROR(ex->opt_nullable_str("api_model_id", p.exposure.api_model_id, 64));
    CLM_RETURN_IF_ERROR(ex->opt_bool("allow_lan", p.exposure.allow_lan));
    CLM_RETURN_IF_ERROR(ex->finish());
  }
  CLM_ASSIGN_OR_RETURN(const json* goals, o.opt_array("goals", 8));
  if (goals != nullptr)
    for (const auto& gj : *goals) {
      CLM_ASSIGN_OR_RETURN(Obj go, Obj::of(gj, "profile.goals[]"));
      Goal g;
      CLM_RETURN_IF_ERROR(go.req_str("metric", g.metric, 48));
      CLM_RETURN_IF_ERROR(go.req_number("value", g.value));
      std::string status;
      CLM_RETURN_IF_ERROR(go.req_str("status", status, 32));
      if (status != "pending_qualification") return bad("profile.goals[].status must be pending_qualification");
      CLM_RETURN_IF_ERROR(go.opt_nullable_str("experiment", g.experiment, 32));
      CLM_RETURN_IF_ERROR(go.finish());
      p.goals.push_back(std::move(g));
    }
  CLM_RETURN_IF_ERROR(o.finish());
  CLM_RETURN_IF_ERROR(validate(p));
  return p;
}

// ---- writers (omit optional fields equal to their default) -----------------------------------------------------

json requirements_to(const Requirements& r) {
  json j = json::object();
  if (!r.gpu_vendor.empty()) j["gpu_vendor"] = r.gpu_vendor;
  if (r.min_vram_mib) j["min_vram_mib"] = *r.min_vram_mib;
  if (r.min_ram_mib) j["min_ram_mib"] = *r.min_ram_mib;
  if (r.min_compute_capability) j["min_compute_capability"] = *r.min_compute_capability;
  if (r.min_link_mbit) j["min_link_mbit"] = *r.min_link_mbit;
  if (r.cpu_only_ok) j["cpu_only_ok"] = true;
  return j;
}

json profile_to(const Profile& p) {
  json j = {{"schema", "clusterlm.profile"}, {"schema_version", 1}, {"id", p.id}, {"name", p.name}};
  if (!p.description.empty()) j["description"] = p.description;
  j["revision"] = p.revision;
  if (p.example_of) j["example_of"] = *p.example_of;
  if (!p.qualification_experiments.empty()) j["qualification_experiments"] = p.qualification_experiments;

  const auto& i = p.model.identity;
  json id = {{"family", i.family}, {"quant", i.quant}};
  if (!i.display_name.empty()) id["display_name"] = i.display_name;
  id["artifact_id"] = i.artifact_id ? json(*i.artifact_id) : json(nullptr);
  id["expected_root_hash"] = i.expected_root_hash ? json(*i.expected_root_hash) : json(nullptr);
  if (!i.expected_files.empty()) {
    json files = json::array();
    for (const auto& f : i.expected_files)
      files.push_back({{"role", f.role},
                       {"name", f.name ? json(*f.name) : json(nullptr)},
                       {"approx_bytes", f.approx_bytes ? json(*f.approx_bytes) : json(nullptr)}});
    id["expected_files"] = std::move(files);
  }
  json model = {{"identity", std::move(id)}};
  if (p.model.library_id) model["library_id"] = *p.model.library_id;
  j["model"] = std::move(model);

  json backend = {{"id", p.backend.id}};
  if (p.backend.min_contract_version != 1) backend["min_contract_version"] = p.backend.min_contract_version;
  if (!p.backend.options.empty()) {
    json opts = json::object();
    for (const auto& o : p.backend.options) std::visit([&](const auto& v) { opts[o.key] = v; }, o.value);
    backend["options"] = std::move(opts);
  }
  j["backend"] = std::move(backend);

  json ctx = {{"default_tokens", p.context.default_tokens}, {"max_tokens", p.context.max_tokens}};
  if (!p.context.offered_profiles.empty()) ctx["offered_profiles"] = p.context.offered_profiles;
  j["context"] = std::move(ctx);

  json slots = json::array();
  for (const auto& s : p.topology.slots) {
    json sj = {{"slot", s.slot}, {"kind", s.kind == Slot::Kind::kHost ? "host" : "worker"}};
    if (!s.label.empty()) sj["label"] = s.label;
    if (s.optional) sj["optional"] = true;
    if (s.select) {
      json sel = {{"mode", kSelMode[static_cast<std::size_t>(s.select->mode)]}};
      if (s.select->mode == Selector::Mode::kBinding) sel["binding"] = s.select->binding;
      if (s.select->mode == Selector::Mode::kMachine) sel["machine"] = s.select->machine;
      if (s.select->requirements) sel["requirements"] = requirements_to(*s.select->requirements);
      sj["select"] = std::move(sel);
    }
    slots.push_back(std::move(sj));
  }
  json topo = {{"slots", std::move(slots)}};
  if (p.topology.min_workers) topo["min_workers"] = *p.topology.min_workers;
  if (p.topology.max_workers) topo["max_workers"] = *p.topology.max_workers;
  j["topology"] = std::move(topo);

  const Resources defres;
  if (!(p.resources == defres)) {
    json r = json::object();
    if (p.resources.device_preference != Resources::Device::kAuto)
      r["device_preference"] = kDevice[static_cast<std::size_t>(p.resources.device_preference)];
    if (p.resources.host_vram_margin_mib) r["host_vram_margin_mib"] = *p.resources.host_vram_margin_mib;
    if (p.resources.host_ram_margin_mib) r["host_ram_margin_mib"] = *p.resources.host_ram_margin_mib;
    if (p.resources.worker_vram_margin_mib) r["worker_vram_margin_mib"] = *p.resources.worker_vram_margin_mib;
    if (p.resources.worker_ram_margin_mib) r["worker_ram_margin_mib"] = *p.resources.worker_ram_margin_mib;
    if (p.resources.max_worker_temp_storage_mib) r["max_worker_temp_storage_mib"] = *p.resources.max_worker_temp_storage_mib;
    j["resources"] = std::move(r);
  }
  if (!(p.placement == Placement{})) {
    json pl = json::object();
    if (p.placement.mode != Placement::Mode::kAuto) pl["mode"] = "manual";
    if (p.placement.objective != Placement::Objective::kBalanced) pl["objective"] = kObjective[static_cast<std::size_t>(p.placement.objective)];
    if (!p.placement.manual.empty()) {
      json st = json::array();
      for (const auto& s : p.placement.manual)
        st.push_back({{"slot", s.slot}, {"first_layer", s.first_layer}, {"end_layer", s.end_layer}});
      pl["manual"] = {{"stages", std::move(st)}};
    }
    j["placement"] = std::move(pl);
  }
  if (!(p.speculation == Speculation{})) {
    json sp = json::object();
    if (!p.speculation.enabled) sp["enabled"] = false;
    if (p.speculation.max_q != 1) sp["max_q"] = p.speculation.max_q;
    j["speculation"] = std::move(sp);
  }
  if (p.chat_template_source != "model") j["chat_template"] = {{"source", p.chat_template_source}};
  if (!(p.lifecycle == Lifecycle{})) {
    json lc = json::object();
    if (p.lifecycle.preparation != Lifecycle::Preparation::kOnDemand) lc["preparation"] = kPrep[static_cast<std::size_t>(p.lifecycle.preparation)];
    if (p.lifecycle.release_after_idle_seconds != 60) lc["release_after_idle_seconds"] = p.lifecycle.release_after_idle_seconds;
    if (p.lifecycle.prepare_when_available) lc["prepare_when_available"] = true;
    j["lifecycle"] = std::move(lc);
  }
  if (!(p.on_worker_loss == OnWorkerLoss{})) {
    json wl = json::object();
    if (p.on_worker_loss.max_retries != 1) wl["max_retries"] = p.on_worker_loss.max_retries;
    if (p.on_worker_loss.then != OnWorkerLoss::Then::kStop) wl["then"] = kThen[static_cast<std::size_t>(p.on_worker_loss.then)];
    if (p.on_worker_loss.fallback_profile_id) wl["fallback_profile_id"] = *p.on_worker_loss.fallback_profile_id;
    if (!p.on_worker_loss.preserve_conversation) wl["preserve_conversation"] = false;
    j["on_worker_loss"] = std::move(wl);
  }
  if (!(p.exposure == Exposure{})) {
    json ex = json::object();
    if (p.exposure.api) ex["api"] = true;
    if (p.exposure.api_model_id) ex["api_model_id"] = *p.exposure.api_model_id;
    if (p.exposure.allow_lan) ex["allow_lan"] = true;
    j["exposure"] = std::move(ex);
  }
  if (!p.goals.empty()) {
    json gs = json::array();
    for (const auto& g : p.goals) {
      json gj = {{"metric", g.metric}, {"value", g.value}, {"status", "pending_qualification"}};
      if (g.experiment) gj["experiment"] = *g.experiment;
      gs.push_back(std::move(gj));
    }
    j["goals"] = std::move(gs);
  }
  return j;
}

}  // namespace

WorkerBounds worker_count_bounds(const Topology& t) {
  std::uint32_t workers = 0, required = 0;
  for (const auto& s : t.slots)
    if (s.kind == Slot::Kind::kWorker) {
      ++workers;
      if (!s.optional) ++required;
    }
  return {t.min_workers.value_or(required), t.max_workers.value_or(workers)};
}

std::size_t worker_slot_count(const Topology& t) {
  return static_cast<std::size_t>(std::count_if(t.slots.begin(), t.slots.end(), [](const Slot& s) { return s.kind == Slot::Kind::kWorker; }));
}

Result<Profile> profile_from_json(std::string_view text) {
  CLM_ASSIGN_OR_RETURN(json doc, detail::parse_object_document(text));
  return read_profile(doc);
}

std::string to_json(const Profile& p) { return profile_to(p).dump(2) + "\n"; }

Result<RoutingAlias> alias_from_json(std::string_view text) {
  CLM_ASSIGN_OR_RETURN(json doc, detail::parse_object_document(text));
  CLM_ASSIGN_OR_RETURN(Obj o, Obj::of(doc, "alias"));
  RoutingAlias a;
  std::string schema;
  CLM_RETURN_IF_ERROR(o.req_str("schema", schema, 32));
  if (schema != "clusterlm.routing-alias") return bad("alias.schema must be 'clusterlm.routing-alias'");
  std::uint64_t ver = 0;
  CLM_RETURN_IF_ERROR(o.req_u64("schema_version", ver, 0, 1u << 30));
  if (ver != 1) return make_error(ErrorCode::kVersionMismatch, "alias schema_version " + std::to_string(ver) + " needs a newer ClusterLM");
  CLM_RETURN_IF_ERROR(o.req_str("id", a.id, 64));
  CLM_RETURN_IF_ERROR(o.opt_str("name", a.name, 80));
  CLM_RETURN_IF_ERROR(o.req_str("api_model_id", a.api_model_id, 64));
  CLM_RETURN_IF_ERROR(o.opt_bool("allow_lan", a.allow_lan));
  CLM_RETURN_IF_ERROR(read_enum(o, "policy", kPolicy, a.policy));
  CLM_RETURN_IF_ERROR(o.opt_bool("allow_prepare", a.allow_prepare));
  CLM_ASSIGN_OR_RETURN(const json* cands, o.req_array("candidates", 1, 8));
  for (const auto& cj : *cands) {
    CLM_ASSIGN_OR_RETURN(Obj co, Obj::of(cj, "alias.candidates[]"));
    AliasCandidate c;
    CLM_RETURN_IF_ERROR(co.req_str("profile_id", c.profile_id, 64));
    CLM_ASSIGN_OR_RETURN(auto when, co.opt_obj("when"));
    if (when) {
      c.has_when = true;
      CLM_RETURN_IF_ERROR(when->opt_string_array("workers_available", c.workers_available, 8, 32));
      CLM_RETURN_IF_ERROR(when->opt_opt_u32("min_workers_available", c.min_workers_available, 0, 8));
      CLM_RETURN_IF_ERROR(when->opt_bool("require_ready", c.require_ready));
      CLM_RETURN_IF_ERROR(when->opt_bool("avoid_on_battery", c.avoid_on_battery));
      CLM_RETURN_IF_ERROR(when->finish());
    }
    CLM_RETURN_IF_ERROR(co.finish());
    a.candidates.push_back(std::move(c));
  }
  CLM_RETURN_IF_ERROR(o.finish());
  CLM_RETURN_IF_ERROR(validate(a));
  return a;
}

std::string to_json(const RoutingAlias& a) {
  json j = {{"schema", "clusterlm.routing-alias"}, {"schema_version", 1}, {"id", a.id}};
  if (!a.name.empty()) j["name"] = a.name;
  j["api_model_id"] = a.api_model_id;
  if (a.allow_lan) j["allow_lan"] = true;
  if (a.policy != RoutingAlias::Policy::kFirstViable) j["policy"] = kPolicy[static_cast<std::size_t>(a.policy)];
  json cs = json::array();
  for (const auto& c : a.candidates) {
    json cj = {{"profile_id", c.profile_id}};
    if (c.has_when) {
      json w = json::object();
      if (!c.workers_available.empty()) w["workers_available"] = c.workers_available;
      if (c.min_workers_available) w["min_workers_available"] = *c.min_workers_available;
      if (c.require_ready) w["require_ready"] = true;
      if (!c.avoid_on_battery) w["avoid_on_battery"] = false;
      cj["when"] = std::move(w);
    }
    cs.push_back(std::move(cj));
  }
  j["candidates"] = std::move(cs);
  if (a.allow_prepare) j["allow_prepare"] = true;
  return j.dump(2) + "\n";
}

}  // namespace clusterlm::profiles
