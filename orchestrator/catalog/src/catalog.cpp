#include "clusterlm/catalog/catalog.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

namespace clusterlm::catalog {

using nlohmann::json;

std::string_view to_string(BackendKind b) noexcept {
  switch (b) {
    case BackendKind::kLlamaLocal: return "llama-local";
    case BackendKind::kStrataHybrid: return "strata-hybrid";
  }
  return "strata-hybrid";
}

std::string_view to_string(TargetStatus s) noexcept {
  switch (s) {
    case TargetStatus::kPendingQualification: return "pending_qualification";
    case TargetStatus::kMeasured: return "measured";
    case TargetStatus::kQualified: return "qualified";
  }
  return "pending_qualification";
}

bool ContextProfile::has_requirement(std::string_view key) const {
  return std::find(requirements.begin(), requirements.end(), key) != requirements.end();
}

const ContextProfile* TierEntry::find_context(std::uint32_t tokens) const {
  for (const auto& c : contexts)
    if (c.tokens == tokens) return &c;
  return nullptr;
}

std::size_t TierEntry::node_role_count() const {
  return static_cast<std::size_t>(std::count_if(roles.begin(), roles.end(), [](const std::string& r) {
    return r.rfind("node:", 0) == 0;
  }));
}

const TierEntry* Catalog::find(std::string_view tier_id) const {
  for (const auto& t : tiers_)
    if (t.id == tier_id) return &t;
  return nullptr;
}

std::vector<const TierEntry*> Catalog::fallbacks_after(std::string_view tier_id) const {
  std::vector<const TierEntry*> out;
  bool seen = false;
  for (const auto& id : fallback_) {
    if (seen) {
      if (const auto* t = find(id)) out.push_back(t);
    } else if (id == tier_id) {
      seen = true;
    }
  }
  return out;
}

namespace {

Status err(const std::string& msg) { return make_error(ErrorCode::kInvalidArgument, "catalog: " + msg); }

bool known_role(std::string_view r) { return r == kRoleFather || r == kRoleLaptop || r == kRole3060; }

bool is_hex64(const std::string& s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; });
}

std::size_t max_nodes_for(std::string_view tier_id) { return tier_id == "fast" ? 0 : tier_id == "strong" ? 1 : 2; }

// Nesting depth of a JSON text, ignoring string contents. Rejects pathological input before parsing.
bool depth_within(std::string_view text, int limit) {
  int depth = 0;
  bool in_str = false, esc = false;
  for (char c : text) {
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    else if (c == '{' || c == '[') {
      if (++depth > limit) return false;
    } else if (c == '}' || c == ']') {
      // Never below zero: stray closers must not "bank" depth that a later run of openers could spend.
      if (depth > 0) --depth;
    }
  }
  return true;
}

// Field readers: every failure names the JSON path.
Status get_string(const json& o, const char* key, const std::string& path, std::string& out) {
  if (!o.contains(key) || !o[key].is_string() || o[key].get<std::string>().empty())
    return err(path + "." + key + " is required (non-empty string)");
  out = o[key].get<std::string>();
  return Status::ok();
}

Status get_opt_string(const json& o, const char* key, const std::string& path, std::optional<std::string>& out) {
  out.reset();
  if (!o.contains(key) || o[key].is_null()) return Status::ok();
  if (!o[key].is_string() || o[key].get<std::string>().empty()) return err(path + "." + key + " must be null or a non-empty string");
  out = o[key].get<std::string>();
  return Status::ok();
}

Status get_bool(const json& o, const char* key, const std::string& path, bool& out) {
  if (!o.contains(key) || !o[key].is_boolean()) return err(path + "." + key + " is required (boolean)");
  out = o[key].get<bool>();
  return Status::ok();
}

Status get_string_array(const json& o, const char* key, const std::string& path, std::vector<std::string>& out, bool required) {
  out.clear();
  if (!o.contains(key)) return required ? err(path + "." + key + " is required (array)") : Status::ok();
  if (!o[key].is_array()) return err(path + "." + key + " must be an array");
  for (const auto& v : o[key]) {
    if (!v.is_string()) return err(path + "." + key + " must contain strings");
    out.push_back(v.get<std::string>());
  }
  return Status::ok();
}

Status parse_tier(const json& t, std::size_t index, TierEntry& out) {
  const std::string path = "tiers[" + std::to_string(index) + "]";
  if (!t.is_object()) return err(path + " must be an object");
  CLM_RETURN_IF_ERROR(get_string(t, "id", path, out.id));
  CLM_RETURN_IF_ERROR(get_string(t, "display_name", path, out.display_name));

  // model
  if (!t.contains("model") || !t["model"].is_object()) return err(path + ".model is required (object)");
  const auto& m = t["model"];
  const std::string mp = path + ".model";
  CLM_RETURN_IF_ERROR(get_string(m, "family", mp, out.model.family));
  CLM_RETURN_IF_ERROR(get_string(m, "display_name", mp, out.model.display_name));
  CLM_RETURN_IF_ERROR(get_string(m, "quant", mp, out.model.quant));
  CLM_RETURN_IF_ERROR(get_opt_string(m, "artifact_id", mp, out.model.artifact_id));
  CLM_RETURN_IF_ERROR(get_opt_string(m, "expected_root_hash", mp, out.model.expected_root_hash));
  if (out.model.expected_root_hash && !is_hex64(*out.model.expected_root_hash))
    return err(mp + ".expected_root_hash must be 64 hex characters or null");
  if (!m.contains("pin_status") || !m["pin_status"].is_string()) return err(mp + ".pin_status is required (\"unpinned\" | \"pinned\")");
  const auto pin = m["pin_status"].get<std::string>();
  if (pin != "unpinned" && pin != "pinned") return err(mp + ".pin_status must be \"unpinned\" or \"pinned\"");
  if ((pin == "pinned") != out.model.expected_root_hash.has_value())
    return err(mp + ": pin_status must say \"pinned\" exactly when expected_root_hash is set");
  if (!m.contains("expected_files") || !m["expected_files"].is_array() || m["expected_files"].empty())
    return err(mp + ".expected_files is required (non-empty array)");
  for (std::size_t i = 0; i < m["expected_files"].size(); ++i) {
    const auto& f = m["expected_files"][i];
    const std::string fp = mp + ".expected_files[" + std::to_string(i) + "]";
    if (!f.is_object()) return err(fp + " must be an object");
    ExpectedFile ef;
    CLM_RETURN_IF_ERROR(get_string(f, "role", fp, ef.role));
    CLM_RETURN_IF_ERROR(get_opt_string(f, "name", fp, ef.name));
    if (f.contains("approx_bytes") && !f["approx_bytes"].is_null()) {
      if (!f["approx_bytes"].is_number_unsigned()) return err(fp + ".approx_bytes must be null or an unsigned integer");
      ef.approx_bytes = f["approx_bytes"].get<std::uint64_t>();
    }
    out.model.expected_files.push_back(std::move(ef));
  }

  // backend
  std::string backend;
  CLM_RETURN_IF_ERROR(get_string(t, "backend", path, backend));
  if (backend == "llama-local") out.backend = BackendKind::kLlamaLocal;
  else if (backend == "strata-hybrid") out.backend = BackendKind::kStrataHybrid;
  else return err(path + ".backend must be \"llama-local\" or \"strata-hybrid\", got \"" + backend + "\"");

  CLM_RETURN_IF_ERROR(get_string_array(t, "roles", path, out.roles, true));

  // contexts
  if (!t.contains("contexts") || !t["contexts"].is_array() || t["contexts"].empty())
    return err(path + ".contexts is required (non-empty array)");
  for (std::size_t i = 0; i < t["contexts"].size(); ++i) {
    const auto& c = t["contexts"][i];
    const std::string cp = path + ".contexts[" + std::to_string(i) + "]";
    if (!c.is_object()) return err(cp + " must be an object");
    ContextProfile cx;
    if (!c.contains("tokens") || !c["tokens"].is_number_unsigned() || c["tokens"].get<std::uint64_t>() == 0 ||
        c["tokens"].get<std::uint64_t>() > (1u << 24))
      return err(cp + ".tokens is required (1..16777216)");
    cx.tokens = c["tokens"].get<std::uint32_t>();
    CLM_RETURN_IF_ERROR(get_bool(c, "offered", cp, cx.offered));
    CLM_RETURN_IF_ERROR(get_bool(c, "qualified", cp, cx.qualified));
    CLM_RETURN_IF_ERROR(get_string_array(c, "requirements", cp, cx.requirements, true));
    for (const auto& r : cx.requirements)
      if (r != kReqFeasiblePlacement && r != kReqQualificationRequired)
        return err(cp + ".requirements: unknown requirement \"" + r + "\"");
    out.contexts.push_back(std::move(cx));
  }

  // performance targets (optional)
  if (t.contains("performance_targets")) {
    if (!t["performance_targets"].is_array()) return err(path + ".performance_targets must be an array");
    for (std::size_t i = 0; i < t["performance_targets"].size(); ++i) {
      const auto& p = t["performance_targets"][i];
      const std::string pp = path + ".performance_targets[" + std::to_string(i) + "]";
      if (!p.is_object()) return err(pp + " must be an object");
      PerformanceTarget pt;
      CLM_RETURN_IF_ERROR(get_string(p, "metric", pp, pt.metric));
      CLM_RETURN_IF_ERROR(get_string(p, "experiment", pp, pt.experiment));
      if (!p.contains("value") || !p["value"].is_number() || !(p["value"].get<double>() > 0) ||
          !std::isfinite(p["value"].get<double>()))
        return err(pp + ".value is required (positive number)");
      pt.value = p["value"].get<double>();
      std::string st;
      CLM_RETURN_IF_ERROR(get_string(p, "status", pp, st));
      if (st == "pending_qualification") pt.status = TargetStatus::kPendingQualification;
      else if (st == "measured") pt.status = TargetStatus::kMeasured;
      else if (st == "qualified") pt.status = TargetStatus::kQualified;
      else return err(pp + ".status must be pending_qualification|measured|qualified");
      out.targets.push_back(std::move(pt));
    }
  }

  // on_session_loss
  if (!t.contains("on_session_loss") || !t["on_session_loss"].is_object()) return err(path + ".on_session_loss is required (object)");
  const auto& pl = t["on_session_loss"];
  if (!pl.contains("max_retries") || !pl["max_retries"].is_number_unsigned() || pl["max_retries"].get<std::uint64_t>() > 8)
    return err(path + ".on_session_loss.max_retries is required (0..8)");
  out.on_session_loss.max_retries = pl["max_retries"].get<std::uint32_t>();
  std::string then;
  CLM_RETURN_IF_ERROR(get_string(pl, "then", path + ".on_session_loss", then));
  if (then == "stop") out.on_session_loss.then = LossAction::kStop;
  else if (then == "downgrade") out.on_session_loss.then = LossAction::kDowngrade;
  else return err(path + ".on_session_loss.then must be \"stop\" or \"downgrade\"");

  CLM_RETURN_IF_ERROR(get_string_array(t, "qualification_experiments", path, out.qualification_experiments, false));
  return Status::ok();
}

}  // namespace

Status validate_tier(const TierEntry& t) {
  const std::string p = "tier \"" + t.id + "\": ";
  if (t.id != "fast" && t.id != "strong" && t.id != "ultra") return err(p + "id must be fast, strong or ultra");
  if (t.roles.empty() || t.roles.front() != kRoleFather) return err(p + "first role must be \"father\"");
  const auto nodes = t.node_role_count();
  if (t.id == "ultra" && nodes > 2) return err(p + "Ultra uses Father + laptop-class + designated 3060 only; a third node (Node 3) is excluded");
  std::set<std::string> seen;
  for (const auto& r : t.roles) {
    if (!known_role(r)) return err(p + "unknown role \"" + r + "\" (known: father, node:laptop-class, node:designated-3060)");
    if (!seen.insert(r).second) return err(p + "duplicate role \"" + r + "\"");
  }
  if (std::count(t.roles.begin(), t.roles.end(), std::string(kRoleFather)) != 1) return err(p + "exactly one father role");
  if (nodes != max_nodes_for(t.id))
    return err(p + "expects " + std::to_string(max_nodes_for(t.id)) + " node role(s), has " + std::to_string(nodes));
  if (t.id == "strong" && t.roles[1] != kRoleLaptop) return err(p + "Strong pairs Father with the laptop-class node");
  if (t.id == "ultra" && (t.roles[1] != kRoleLaptop || t.roles[2] != kRole3060))
    return err(p + "Ultra roles must be father, node:laptop-class, node:designated-3060 in pipeline order");
  if (t.backend == BackendKind::kLlamaLocal && nodes != 0) return err(p + "backend llama-local cannot span nodes");
  if (t.backend == BackendKind::kStrataHybrid && t.id == "fast") return err(p + "Fast uses the local llama backend");
  if (t.id != "fast" && t.backend != BackendKind::kStrataHybrid) return err(p + "Strong and Ultra use the strata-hybrid backend");
  if (t.model.display_name.empty() || t.model.family.empty()) return err(p + "model identity incomplete");
  std::set<std::uint32_t> ctx;
  for (const auto& c : t.contexts) {
    if (!ctx.insert(c.tokens).second) return err(p + "duplicate context profile " + std::to_string(c.tokens));
    if (c.qualified && !c.has_requirement(kReqFeasiblePlacement)) return err(p + "a qualified context must still require a feasible placement");
  }
  return Status::ok();
}

Result<Catalog> Catalog::parse(std::string_view text) {
  if (text.size() > kMaxCatalogBytes) return err("input exceeds " + std::to_string(kMaxCatalogBytes) + " bytes");
  if (!depth_within(text, kMaxJsonDepth)) return err("JSON nesting deeper than " + std::to_string(kMaxJsonDepth));
  json j = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) return err("not a valid JSON object");

  if (!j.contains("schema_version") || !j["schema_version"].is_number_unsigned() || j["schema_version"].get<unsigned>() != 1)
    return err("schema_version must be 1");
  Catalog c;
  CLM_RETURN_IF_ERROR(get_string(j, "catalog_id", "$", c.id_));
  if (!j.contains("tiers") || !j["tiers"].is_array()) return err("$.tiers is required (array)");
  if (j["tiers"].size() != 3) return err("exactly three tiers (fast, strong, ultra) are required");
  for (std::size_t i = 0; i < j["tiers"].size(); ++i) {
    TierEntry t;
    CLM_RETURN_IF_ERROR(parse_tier(j["tiers"][i], i, t));
    CLM_RETURN_IF_ERROR(validate_tier(t));
    if (c.find(t.id)) return err("duplicate tier \"" + t.id + "\"");
    c.tiers_.push_back(std::move(t));
  }
  CLM_RETURN_IF_ERROR(get_string_array(j, "fallback_order", "$", c.fallback_, true));
  if (c.fallback_ != std::vector<std::string>{"ultra", "strong", "fast"})
    return err("fallback_order must be [ultra, strong, fast]");
  return c;
}

Result<Catalog> Catalog::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return make_error(ErrorCode::kNotFound, "catalog: cannot open " + path);
  std::string text(kMaxCatalogBytes + 1, '\0');
  f.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(f.gcount()));
  return parse(text);
}

std::optional<std::string> TierAssignment::machine_for(std::string_view role) const {
  auto it = map_.find(role);
  if (it == map_.end()) return std::nullopt;
  return it->second;
}

Status TierAssignment::validate_for(const TierEntry& tier) const {
  std::set<std::string> machines;
  for (const auto& r : tier.roles) {
    auto m = machine_for(r);
    if (!m || m->empty())
      return make_error(ErrorCode::kFailedPrecondition, "no machine assigned to role \"" + r + "\" for tier " + tier.id);
    if (!machines.insert(*m).second)
      return make_error(ErrorCode::kInvalidArgument, "machine \"" + *m + "\" is assigned to two roles of tier " + tier.id);
  }
  return Status::ok();
}

}  // namespace clusterlm::catalog
