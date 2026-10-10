#include "clusterlm/domain/backend_descriptor.hpp"

#include <algorithm>
#include <cctype>
#include <set>

#include "builtin_descriptors.hpp"
#include "clusterlm/common/strict_json.hpp"
#include "clusterlm/objects/ggml_types.hpp"

namespace clusterlm::domain {

using nlohmann::json;
using strict::bad;
using strict::Obj;

namespace {

constexpr std::size_t kMaxDocument = 1u << 20;
const std::vector<std::string_view> kLabels = {"Supported and qualified", "Supported, awaiting hardware qualification", "Experimental", "Unsupported"};

std::string upper(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}
std::string lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool in_set(const std::vector<std::string>& v, std::string_view x) { return std::find(v.begin(), v.end(), x) != v.end(); }

bool one_of(std::string_view s, std::initializer_list<std::string_view> allowed) {
  return std::find(allowed.begin(), allowed.end(), s) != allowed.end();
}

bool is_hq(std::string_view s) {
  // HQ-<A-Z0-9>+-<digits>+
  if (s.size() < 6 || s.size() > 32 || s.substr(0, 3) != "HQ-") return false;
  const auto dash = s.rfind('-');
  if (dash <= 3 || dash + 1 >= s.size()) return false;
  auto name = s.substr(3, dash - 3);
  auto num = s.substr(dash + 1);
  return std::all_of(name.begin(), name.end(), [](char c) { return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); }) &&
         std::all_of(num.begin(), num.end(), [](char c) { return c >= '0' && c <= '9'; });
}

Status read_label(Obj& o, const char* key, CompatLabel& out) {
  std::size_t idx = static_cast<std::size_t>(out);
  CLM_RETURN_IF_ERROR(o.opt_enum(key, kLabels, idx));
  out = static_cast<CompatLabel>(idx);
  return Status::ok();
}

Status req_enum_str(Obj& o, const char* key, std::string& out, std::initializer_list<std::string_view> allowed) {
  CLM_RETURN_IF_ERROR(o.req_str(key, out, 32));
  if (!one_of(out, allowed)) return bad(o.path() + "." + key + " has an unknown value");
  return Status::ok();
}

Status req_bool(Obj& o, const char* key, bool& out) {
  if (!o.has(key)) return bad(o.path() + "." + key + " is required (capabilities are explicit)");
  return o.opt_bool(key, out);
}

Status req_u32(Obj& o, const char* key, std::uint32_t& out, std::uint32_t lo, std::uint32_t hi) {
  if (!o.has(key)) return bad(o.path() + "." + key + " is required (capabilities are explicit)");
  return o.opt_u32(key, out, lo, hi);
}

}  // namespace

std::string_view to_string(CompatLabel l) noexcept { return kLabels[static_cast<std::size_t>(l)]; }

std::optional<CompatLabel> compat_label_from_string(std::string_view s) {
  for (std::size_t i = 0; i < kLabels.size(); ++i)
    if (kLabels[i] == s) return static_cast<CompatLabel>(i);
  return std::nullopt;
}

bool family_glob_match(std::string_view pattern, std::string_view text) {
  if (pattern.size() > 120 || text.size() > 120) return false;
  const std::string p = lower(std::string(pattern)), t = lower(std::string(text));
  std::size_t pi = 0, ti = 0, star = std::string::npos, mark = 0;
  while (ti < t.size()) {
    if (pi < p.size() && p[pi] == '*') {
      star = pi++;
      mark = ti;
    } else if (pi < p.size() && p[pi] == t[ti]) {
      ++pi;
      ++ti;
    } else if (star != std::string::npos) {
      pi = star + 1;
      ti = ++mark;
    } else {
      return false;
    }
  }
  while (pi < p.size() && p[pi] == '*') ++pi;
  return pi == p.size();
}

Status validate(const BackendDescriptor& d) {
  const bool pair_ok = (d.id == "llama-local" && d.factory_name == "llama") || (d.id == "strata-hybrid" && d.factory_name == "strata") ||
                       (d.id == "reference" && d.factory_name == "reference");
  if (!pair_ok) return bad("descriptor id and factory_name must be a known pair (llama-local/llama, strata-hybrid/strata, reference/reference)");
  if (d.display_name.empty() || d.display_name.size() > 80) return bad("display_name must be 1..80 characters");

  const auto& ms = d.model_support;
  if (ms.containers.empty() || ms.containers.size() > 8) return bad("containers needs 1..8 entries");
  for (const auto& c : ms.containers)
    if (!one_of(c, {"gguf", "gguf-split", "manifest-native"})) return bad("unknown container");
  if (ms.tensor_formats.empty() || ms.tensor_formats.size() > 64) return bad("tensor_formats needs 1..64 entries");
  {
    std::set<std::string> seen;
    for (const auto& f : ms.tensor_formats) {
      if (f != upper(f)) return bad("tensor_formats are canonical upper-case ggml names");
      if (objects::ggml_type_by_name(f) == nullptr) return bad("tensor format '" + f.substr(0, 32) + "' is not a ggml type this build knows");
      if (!seen.insert(f).second) return bad("tensor_formats has duplicates");
    }
  }
  if (ms.families.size() > 64) return bad("too many family entries");
  for (const auto& f : ms.families) {
    if (f.family_match.empty() || f.family_match.size() > 120) return bad("family_match must be 1..120 characters");
    if (f.evidence.size() > 32) return bad("too many evidence entries");
  }
  if (ms.unlisted_family_status != CompatLabel::kExperimental && ms.unlisted_family_status != CompatLabel::kUnsupported)
    return bad("unlisted_family_status may only be Experimental or Unsupported");

  const auto& dev = d.devices;
  for (const auto& v : dev.gpu_vendors)
    if (!one_of(v, {"nvidia", "amd", "intel"})) return bad("unknown GPU vendor");
  if (in_set(dev.gpu_vendors, "nvidia") && !dev.min_compute_capability) return bad("an NVIDIA backend must declare min_compute_capability");
  if (dev.requires_runtime.size() > 8) return bad("too many runtime requirements");

  const auto& ex = d.execution;
  if (!ex.single_host && !ex.cross_machine) return bad("a backend must run single-host, cross-machine or both");
  if (ex.validated_max_workers > 8) return bad("validated_max_workers must be 0..8");
  if (!ex.cross_machine && (ex.validated_max_workers != 0 || !ex.worker_roles.empty()))
    return bad("a backend that is not cross-machine has validated_max_workers 0 and no worker roles");
  if (ex.cross_machine && ex.validated_max_workers == 0) return bad("a cross-machine backend needs validated_max_workers >= 1 backed by evidence");
  if (ex.validated_topologies.empty() || ex.validated_topologies.size() > 16) return bad("validated_topologies needs 1..16 entries");
  if (!one_of(ex.layer_partitioning, {"none", "contiguous-layers"})) return bad("unknown layer_partitioning");
  for (const auto& r : ex.worker_roles)
    if (r != "middle") return bad("a Worker can only play the middle role");
  if (ex.worker_roles.size() > 1) return bad("worker_roles has duplicates");

  const auto& st = d.state;
  if (st.kinds.empty() || st.kinds.size() > 4) return bad("state.kinds needs 1..4 entries");
  for (const auto& k : st.kinds)
    if (!one_of(k, {"kv", "recurrent", "indexer", "ple-history"})) return bad("unknown state kind");
  if (!one_of(st.supports_rollback, {"native", "recompute", "none"})) return bad("unknown supports_rollback");
  if (st.max_sessions_per_domain < 1 || st.max_sessions_per_domain > 64) return bad("max_sessions_per_domain must be 1..64");

  const auto& sp = d.speculation;
  if (!one_of(sp.mtp, {"none", "one-hot-proposals", "full-distribution"})) return bad("unknown mtp form");
  if (sp.max_q < 1 || sp.max_q > 64) return bad("speculation.max_q must be 1..64");
  if (!sp.window_commit_abort && sp.max_q != 1) return bad("max_q must be 1 when window_commit_abort is false");

  if (d.limitations.size() > 32) return bad("too many limitations");
  if (d.options.size() > 16) return bad("too many backend options");
  for (std::size_t i = 0; i < d.options.size(); ++i) {
    if (i > 0 && !(d.options[i - 1].first < d.options[i].first)) return bad("options must be sorted by unique name");
    const auto& o = d.options[i].second;
    if (o.minimum && o.maximum && *o.minimum > *o.maximum) return bad("option minimum exceeds maximum");
    if (o.type != OptionSpec::Type::kString && !o.allowed.empty()) return bad("enum is only allowed on string options");
    if (o.allowed.size() > 16) return bad("too many option enum values");
  }

  const auto& q = d.qualification;
  if (q.hardware_experiments.size() > 32) return bad("too many hardware experiments");
  std::set<std::string> seen;
  for (const auto& h : q.hardware_experiments) {
    if (!is_hq(h)) return bad("hardware experiment ids look like HQ-NAME-01");
    if (!seen.insert(h).second) return bad("hardware_experiments has duplicates");
  }
  if (q.hardware_experiments.empty() && !d.development_only) return bad("a production backend must name its hardware experiments");
  if (q.notes.size() > 500) return bad("qualification notes too long");
  return Status::ok();
}

Result<BackendDescriptor> backend_descriptor_from_json(std::string_view text) {
  if (text.size() > kMaxDocument) return bad("descriptor larger than 1 MiB");
  if (!strict::depth_within(text, 16)) return bad("JSON nesting deeper than 16");
  json doc = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) return bad("not valid JSON");
  CLM_ASSIGN_OR_RETURN(Obj o, Obj::of(doc, "descriptor"));
  BackendDescriptor d;
  std::string s;
  CLM_RETURN_IF_ERROR(o.req_str("schema", s, 40));
  if (s != "clusterlm.backend-descriptor") return bad("descriptor.schema must be 'clusterlm.backend-descriptor'");
  std::uint64_t ver = 0;
  CLM_RETURN_IF_ERROR(o.req_u64("contract_version", ver, 0, 1u << 30));
  if (ver != 1) return make_error(ErrorCode::kVersionMismatch, "descriptor contract_version " + std::to_string(ver) + " needs a newer ClusterLM");
  CLM_RETURN_IF_ERROR(o.req_str("id", d.id, 32));
  CLM_RETURN_IF_ERROR(o.req_str("display_name", d.display_name, 80));
  CLM_RETURN_IF_ERROR(o.req_str("factory_name", d.factory_name, 32));
  std::string role;
  CLM_RETURN_IF_ERROR(req_enum_str(o, "role_in_product", role, {"production", "development-only"}));
  d.development_only = role == "development-only";

  {
    CLM_ASSIGN_OR_RETURN(auto m, o.opt_obj("model_support"));
    if (!m) return bad("descriptor.model_support is required");
    CLM_ASSIGN_OR_RETURN(const json* cs, m->req_array("containers", 1, 8));
    for (const auto& c : *cs) {
      if (!c.is_string()) return bad("containers must be strings");
      d.model_support.containers.push_back(c.get<std::string>());
    }
    CLM_ASSIGN_OR_RETURN(const json* tf, m->req_array("tensor_formats", 1, 64));
    for (const auto& t : *tf) {
      if (!t.is_string() || t.get<std::string>().size() > 32) return bad("tensor_formats must be short strings");
      d.model_support.tensor_formats.push_back(upper(t.get<std::string>()));
    }
    CLM_ASSIGN_OR_RETURN(const json* fs, m->req_array("families", 0, 64));
    for (const auto& fj : *fs) {
      CLM_ASSIGN_OR_RETURN(Obj fo, Obj::of(fj, "descriptor.model_support.families[]"));
      FamilyEntry f;
      CLM_RETURN_IF_ERROR(fo.req_str("family_match", f.family_match, 120));
      CLM_RETURN_IF_ERROR(fo.opt_nullable_str("architecture", f.architecture, 64));
      if (!fo.has("status")) return bad("families[].status is required");
      CLM_RETURN_IF_ERROR(read_label(fo, "status", f.status));
      CLM_RETURN_IF_ERROR(fo.opt_string_array("evidence", f.evidence, 32, 120));
      CLM_RETURN_IF_ERROR(fo.finish());
      d.model_support.families.push_back(std::move(f));
    }
    if (!m->has("unlisted_family_status")) return bad("model_support.unlisted_family_status is required");
    CLM_RETURN_IF_ERROR(read_label(*m, "unlisted_family_status", d.model_support.unlisted_family_status));
    CLM_RETURN_IF_ERROR(m->finish());
  }
  {
    CLM_ASSIGN_OR_RETURN(auto dv, o.opt_obj("devices"));
    if (!dv) return bad("descriptor.devices is required");
    CLM_RETURN_IF_ERROR(req_bool(*dv, "cpu", d.devices.cpu));
    CLM_ASSIGN_OR_RETURN(const json* gv, dv->req_array("gpu_vendors", 0, 3));
    for (const auto& v : *gv) {
      if (!v.is_string()) return bad("gpu_vendors must be strings");
      d.devices.gpu_vendors.push_back(v.get<std::string>());
    }
    CLM_RETURN_IF_ERROR(dv->opt_nullable_str("min_compute_capability", d.devices.min_compute_capability, 8));
    if (!dv->has("requires_runtime")) return bad("devices.requires_runtime is required (empty list = nothing)");
    CLM_RETURN_IF_ERROR(dv->opt_string_array("requires_runtime", d.devices.requires_runtime, 8, 64));
    CLM_RETURN_IF_ERROR(dv->finish());
  }
  {
    CLM_ASSIGN_OR_RETURN(auto e, o.opt_obj("execution"));
    if (!e) return bad("descriptor.execution is required");
    auto& x = d.execution;
    CLM_RETURN_IF_ERROR(req_bool(*e, "single_host", x.single_host));
    CLM_RETURN_IF_ERROR(req_bool(*e, "cross_machine", x.cross_machine));
    CLM_RETURN_IF_ERROR(req_u32(*e, "validated_max_workers", x.validated_max_workers, 0, 8));
    if (!e->has("validated_topologies")) return bad("execution.validated_topologies is required");
    CLM_RETURN_IF_ERROR(e->opt_string_array("validated_topologies", x.validated_topologies, 16, 160));
    CLM_RETURN_IF_ERROR(req_enum_str(*e, "layer_partitioning", x.layer_partitioning, {"none", "contiguous-layers"}));
    CLM_RETURN_IF_ERROR(req_bool(*e, "cpu_gpu_hybrid", x.cpu_gpu_hybrid));
    CLM_RETURN_IF_ERROR(req_bool(*e, "token_free_middle_stages", x.token_free_middle_stages));
    CLM_RETURN_IF_ERROR(req_bool(*e, "manual_layer_ranges", x.manual_layer_ranges));
    if (!e->has("worker_roles")) return bad("execution.worker_roles is required");
    CLM_RETURN_IF_ERROR(e->opt_string_array("worker_roles", x.worker_roles, 1, 16));
    CLM_RETURN_IF_ERROR(e->opt_string_array("evidence", x.evidence, 32, 120));
    CLM_RETURN_IF_ERROR(e->finish());
  }
  {
    CLM_ASSIGN_OR_RETURN(auto st, o.opt_obj("state"));
    if (!st) return bad("descriptor.state is required");
    if (!st->has("kinds")) return bad("state.kinds is required");
    CLM_RETURN_IF_ERROR(st->opt_string_array("kinds", d.state.kinds, 4, 16));
    CLM_RETURN_IF_ERROR(req_enum_str(*st, "supports_rollback", d.state.supports_rollback, {"native", "recompute", "none"}));
    CLM_RETURN_IF_ERROR(req_u32(*st, "max_sessions_per_domain", d.state.max_sessions_per_domain, 1, 64));
    CLM_RETURN_IF_ERROR(req_bool(*st, "per_client_isolation", d.state.per_client_isolation));
    CLM_RETURN_IF_ERROR(st->finish());
  }
  {
    CLM_ASSIGN_OR_RETURN(auto sp, o.opt_obj("speculation"));
    if (!sp) return bad("descriptor.speculation is required");
    CLM_RETURN_IF_ERROR(req_bool(*sp, "window_commit_abort", d.speculation.window_commit_abort));
    CLM_RETURN_IF_ERROR(req_enum_str(*sp, "mtp", d.speculation.mtp, {"none", "one-hot-proposals", "full-distribution"}));
    CLM_RETURN_IF_ERROR(req_u32(*sp, "max_q", d.speculation.max_q, 1, 64));
    CLM_RETURN_IF_ERROR(sp->finish());
  }
  {
    CLM_ASSIGN_OR_RETURN(auto sv, o.opt_obj("serving"));
    if (!sv) return bad("descriptor.serving is required");
    CLM_RETURN_IF_ERROR(req_bool(*sv, "full_logits_for_sampling", d.serving.full_logits_for_sampling));
    CLM_RETURN_IF_ERROR(req_bool(*sv, "constrained_decoding", d.serving.constrained_decoding));
    CLM_RETURN_IF_ERROR(req_bool(*sv, "logprobs", d.serving.logprobs));
    CLM_RETURN_IF_ERROR(sv->finish());
  }
  if (!o.has("limitations")) return bad("descriptor.limitations is required");
  CLM_RETURN_IF_ERROR(o.opt_string_array("limitations", d.limitations, 32, 300));
  CLM_ASSIGN_OR_RETURN(const json* opts, o.opt_raw_object("options_schema", 16));
  if (opts != nullptr)
    for (auto it = opts->begin(); it != opts->end(); ++it) {
      CLM_ASSIGN_OR_RETURN(Obj oo, Obj::of(it.value(), "descriptor.options_schema." + it.key().substr(0, 32)));
      OptionSpec spec;
      std::string type;
      CLM_RETURN_IF_ERROR(req_enum_str(oo, "type", type, {"integer", "boolean", "string"}));
      spec.type = type == "integer" ? OptionSpec::Type::kInteger : type == "boolean" ? OptionSpec::Type::kBoolean : OptionSpec::Type::kString;
      CLM_RETURN_IF_ERROR(oo.opt_nullable_i64("minimum", spec.minimum));
      CLM_RETURN_IF_ERROR(oo.opt_nullable_i64("maximum", spec.maximum));
      CLM_RETURN_IF_ERROR(oo.opt_string_array("enum", spec.allowed, 16, 64));
      CLM_RETURN_IF_ERROR(oo.opt_str("description", spec.description, 200));
      CLM_RETURN_IF_ERROR(oo.finish());
      d.options.emplace_back(it.key(), std::move(spec));
    }
  {
    CLM_ASSIGN_OR_RETURN(auto q, o.opt_obj("qualification"));
    if (!q) return bad("descriptor.qualification is required");
    if (!q->has("label")) return bad("qualification.label is required");
    CLM_RETURN_IF_ERROR(read_label(*q, "label", d.qualification.label));
    if (!q->has("hardware_experiments")) return bad("qualification.hardware_experiments is required");
    CLM_RETURN_IF_ERROR(q->opt_string_array("hardware_experiments", d.qualification.hardware_experiments, 32, 32));
    CLM_RETURN_IF_ERROR(q->opt_str("notes", d.qualification.notes, 500));
    CLM_RETURN_IF_ERROR(q->finish());
  }
  CLM_RETURN_IF_ERROR(o.finish());
  CLM_RETURN_IF_ERROR(validate(d));
  return d;
}

std::string to_json(const BackendDescriptor& d) {
  json fams = json::array();
  for (const auto& f : d.model_support.families) {
    json fj = {{"family_match", f.family_match}, {"status", std::string(to_string(f.status))}};
    if (f.architecture) fj["architecture"] = *f.architecture;
    if (!f.evidence.empty()) fj["evidence"] = f.evidence;
    fams.push_back(std::move(fj));
  }
  json j = {{"schema", "clusterlm.backend-descriptor"},
            {"contract_version", 1},
            {"id", d.id},
            {"display_name", d.display_name},
            {"factory_name", d.factory_name},
            {"role_in_product", d.development_only ? "development-only" : "production"},
            {"model_support",
             {{"containers", d.model_support.containers},
              {"tensor_formats", d.model_support.tensor_formats},
              {"families", std::move(fams)},
              {"unlisted_family_status", std::string(to_string(d.model_support.unlisted_family_status))}}}};
  json dev = {{"cpu", d.devices.cpu}, {"gpu_vendors", d.devices.gpu_vendors}, {"requires_runtime", d.devices.requires_runtime}};
  if (d.devices.min_compute_capability) dev["min_compute_capability"] = *d.devices.min_compute_capability;
  j["devices"] = std::move(dev);
  const auto& x = d.execution;
  json ex = {{"single_host", x.single_host},
             {"cross_machine", x.cross_machine},
             {"validated_max_workers", x.validated_max_workers},
             {"validated_topologies", x.validated_topologies},
             {"layer_partitioning", x.layer_partitioning},
             {"cpu_gpu_hybrid", x.cpu_gpu_hybrid},
             {"token_free_middle_stages", x.token_free_middle_stages},
             {"manual_layer_ranges", x.manual_layer_ranges},
             {"worker_roles", x.worker_roles}};
  if (!x.evidence.empty()) ex["evidence"] = x.evidence;
  j["execution"] = std::move(ex);
  j["state"] = {{"kinds", d.state.kinds},
                {"supports_rollback", d.state.supports_rollback},
                {"max_sessions_per_domain", d.state.max_sessions_per_domain},
                {"per_client_isolation", d.state.per_client_isolation}};
  j["speculation"] = {{"window_commit_abort", d.speculation.window_commit_abort}, {"mtp", d.speculation.mtp}, {"max_q", d.speculation.max_q}};
  j["serving"] = {{"full_logits_for_sampling", d.serving.full_logits_for_sampling},
                  {"constrained_decoding", d.serving.constrained_decoding},
                  {"logprobs", d.serving.logprobs}};
  j["limitations"] = d.limitations;
  if (!d.options.empty()) {
    json opts = json::object();
    for (const auto& [name, spec] : d.options) {
      json oj = {{"type", spec.type == OptionSpec::Type::kInteger ? "integer" : spec.type == OptionSpec::Type::kBoolean ? "boolean" : "string"}};
      if (spec.minimum) oj["minimum"] = *spec.minimum;
      if (spec.maximum) oj["maximum"] = *spec.maximum;
      if (!spec.allowed.empty()) oj["enum"] = spec.allowed;
      if (!spec.description.empty()) oj["description"] = spec.description;
      opts[name] = std::move(oj);
    }
    j["options_schema"] = std::move(opts);
  }
  json q = {{"label", std::string(to_string(d.qualification.label))}, {"hardware_experiments", d.qualification.hardware_experiments}};
  if (!d.qualification.notes.empty()) q["notes"] = d.qualification.notes;
  j["qualification"] = std::move(q);
  return j.dump(2) + "\n";
}

bool CompatReport::has_blocker() const {
  return std::any_of(findings.begin(), findings.end(), [](const Finding& f) { return f.severity == Finding::Severity::kBlocker; });
}

CompatReport check_model(const BackendDescriptor& d, const ModelFacts& m, const BackendRuntimeStatus& rt) {
  CompatReport r;
  auto add = [&](Finding::Severity s, std::string code, std::string msg) { r.findings.push_back({s, std::move(code), std::move(msg)}); };
  using Sev = Finding::Severity;

  if (d.development_only)
    add(Sev::kNote, "development_only_backend",
        "This backend exists to test ClusterLM itself; its label applies to its own fixtures, never to a real model.");

  // Runtime availability is reported separately from the model label; it is never hidden.
  if (!rt.built) add(Sev::kBlocker, "backend_not_built", "This backend is not built into this ClusterLM binary.");
  else if (!rt.runtime_present) add(Sev::kBlocker, "backend_runtime_missing", "The runtime this backend needs is not present on this machine.");
  else if (!rt.hardware_available) add(Sev::kWarning, "hardware_unavailable", "The hardware this backend needs is not available on this machine.");
  for (const auto& why : rt.reasons) add(Sev::kNote, "runtime_note", why);

  CompatLabel label = CompatLabel::kUnsupported;
  bool formats_ok = true;
  if (!in_set(d.model_support.containers, m.container)) {
    add(Sev::kBlocker, "container_unsupported", "This backend cannot read the model's file layout (" + m.container + ").");
    formats_ok = false;
  }
  {
    std::set<std::string> missing;
    for (const auto& t : m.tensor_types) {
      const auto u = upper(t);
      if (!in_set(d.model_support.tensor_formats, u)) missing.insert(u);
    }
    if (!missing.empty()) {
      std::string list;
      std::size_t n = 0;
      for (const auto& t : missing) {
        if (n++ == 3) {
          list += ", ...";
          break;
        }
        list += (list.empty() ? "" : ", ") + t;
      }
      add(Sev::kBlocker, "tensor_type_unsupported", "This backend cannot execute tensor type(s) used by the model: " + list + ".");
      formats_ok = false;
    }
  }
  if (formats_ok) {
    bool listed = false;
    for (const auto& f : d.model_support.families) {
      if (!family_glob_match(f.family_match, m.family)) continue;
      if (f.architecture && *f.architecture != m.architecture) continue;
      label = f.status;
      listed = true;
      break;
    }
    if (!listed) {
      label = d.model_support.unlisted_family_status;
      if (label == CompatLabel::kUnsupported) add(Sev::kBlocker, "family_unsupported", "This backend has no support for this model family.");
    }
    if (label == CompatLabel::kExperimental) {
      add(Sev::kWarning, "experimental", "Experimental: it may fail or be slow, it is never distributed across machines, and it is not covered by any test.");
      if (!m.experimental_opt_in)
        add(Sev::kBlocker, "experimental_opt_in_required", "Running an experimental model needs your explicit confirmation for this model.");
    }
  }
  // The backend-level label is a ceiling.
  if (rank(d.qualification.label) > rank(label)) label = d.qualification.label;
  r.label = label;
  r.hosts_alone = d.execution.single_host && label != CompatLabel::kUnsupported;
  r.distributable = d.execution.cross_machine && is_supported(label);
  r.max_workers = r.distributable ? d.execution.validated_max_workers : 0;
  if (!r.distributable && d.execution.cross_machine && label == CompatLabel::kExperimental)
    add(Sev::kNote, "not_distributable", "Only supported models are distributed across machines.");
  return r;
}

Result<BackendRegistry> BackendRegistry::of(std::vector<BackendDescriptor> descriptors) {
  BackendRegistry r;
  for (auto& d : descriptors) {
    CLM_RETURN_IF_ERROR(validate(d));
    if (std::any_of(r.descriptors_.begin(), r.descriptors_.end(), [&](const BackendDescriptor& x) { return x.id == d.id; }))
      return bad("duplicate backend descriptor id");
    r.descriptors_.push_back(std::move(d));
  }
  return r;
}

const BackendRegistry& BackendRegistry::builtin() {
  static const BackendRegistry instance = [] {
    std::vector<BackendDescriptor> v;
    for (const char* text : {detail::kReferenceDescriptorJson, detail::kLlamaDescriptorJson, detail::kStrataDescriptorJson}) {
      auto d = backend_descriptor_from_json(text);
      // An invalid embedded descriptor is a programming error caught by tests; never advertise a broken one.
      if (d.is_ok()) v.push_back(std::move(d).value());
    }
    auto r = BackendRegistry::of(std::move(v));
    return r.is_ok() ? std::move(r).value() : BackendRegistry{};
  }();
  return instance;
}

std::vector<const BackendDescriptor*> BackendRegistry::list() const {
  std::vector<const BackendDescriptor*> out;
  for (const auto& d : descriptors_) out.push_back(&d);
  return out;
}

const BackendDescriptor* BackendRegistry::find(std::string_view id) const {
  for (const auto& d : descriptors_)
    if (d.id == id) return &d;
  return nullptr;
}

}  // namespace clusterlm::domain
