#include "clusterlm/placement/profile.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace clusterlm::placement {

using nlohmann::json;

std::string_view to_string(Provenance p) noexcept {
  switch (p) {
    case Provenance::kSynthetic: return "synthetic";
    case Provenance::kMeasured: return "measured";
    case Provenance::kQualified: return "qualified";
  }
  return "synthetic";
}

bool provenance_from_string(std::string_view s, Provenance& out) noexcept {
  if (s == "synthetic") out = Provenance::kSynthetic;
  else if (s == "measured") out = Provenance::kMeasured;
  else if (s == "qualified") out = Provenance::kQualified;
  else return false;
  return true;
}

const LinkProfile* NetworkProfile::find_link(std::string_view from, std::string_view to) const {
  for (const auto& l : links)
    if (l.from == from && l.to == to) return &l;
  for (const auto& l : links)
    if (l.from == to && l.to == from) return &l;
  return nullptr;
}

// ---- traversal -----------------------------------------------------------------------------------------

void for_each_quantity(HardwareProfile& p, const QuantityVisitor& fn) {
  fn("cpu.usable_threads", p.cpu.usable_threads);
  for (auto& [k, v] : p.cpu.expert_bytes_per_s) fn("cpu.expert_bytes_per_s." + k, v);
  fn("cpu.q_scaling", p.cpu.q_scaling);
  fn("cpu.sustained_factor", p.cpu.sustained_factor);
  fn("memory.ram_total", p.memory.ram_total);
  fn("memory.ram_safe_allowance", p.memory.ram_safe_allowance);
  fn("memory.ram_bandwidth", p.memory.ram_bandwidth);
  fn("memory.pinned_limit", p.memory.pinned_limit);
  fn("gpu.vram_total", p.gpu.vram_total);
  fn("gpu.vram_budget", p.gpu.vram_budget);
  fn("gpu.gpu_expert_bytes_per_s", p.gpu.gpu_expert_bytes_per_s);
  for (auto& [k, v] : p.gpu.dense_layer_ms) fn("gpu.dense_layer_ms." + k, v);
  fn("gpu.dense_q_scaling", p.gpu.dense_q_scaling);
  fn("gpu.pcie_h2d_bytes_per_s", p.gpu.pcie_h2d_bytes_per_s);
  fn("gpu.prefill_tokens_per_s", p.gpu.prefill_tokens_per_s);
  fn("overheads.os_reserve_ram", p.overheads.os_reserve_ram);
  fn("overheads.scratch_vram", p.overheads.scratch_vram);
  fn("overheads.staging_ram", p.overheads.staging_ram);
}

void for_each_quantity(NetworkProfile& n, const QuantityVisitor& fn) {
  fn("father_egress_bytes_per_s", n.father_egress_bytes_per_s);
  for (std::size_t i = 0; i < n.links.size(); ++i) {
    const std::string base = "links[" + n.links[i].from + "->" + n.links[i].to + "].";
    fn(base + "bandwidth_bytes_per_s", n.links[i].bandwidth_bytes_per_s);
    fn(base + "rtt_ms", n.links[i].rtt_ms);
    fn(base + "jitter_ms", n.links[i].jitter_ms);
  }
}

Provenance weakest_provenance(const HardwareProfile& p) {
  Provenance w = Provenance::kQualified;
  for_each_quantity(const_cast<HardwareProfile&>(p), [&](const std::string&, Quantity& q) { w = weakest(w, q.provenance); });
  return w;
}

Provenance weakest_provenance(const NetworkProfile& n) {
  Provenance w = Provenance::kQualified;
  for_each_quantity(const_cast<NetworkProfile&>(n), [&](const std::string&, Quantity& q) { w = weakest(w, q.provenance); });
  return w;
}

Provenance weakest_link_provenance(const LinkProfile& l) {
  return weakest(weakest(l.bandwidth_bytes_per_s.provenance, l.rtt_ms.provenance), l.jitter_ms.provenance);
}

// ---- validation ----------------------------------------------------------------------------------------

namespace {
Status bad(const std::string& id, const std::string& what) {
  return make_error(ErrorCode::kInvalidArgument, "profile '" + id + "': " + what);
}
}  // namespace

Status validate(const HardwareProfile& p) {
  if (p.id.empty()) return bad(p.id, "empty id");
  Status st;
  auto check = [&](const std::string& path, const Quantity& q) {
    if (!st.is_ok()) return;
    if (!std::isfinite(q.value) || q.value < 0) st = bad(p.id, path + " must be finite and >= 0");
  };
  for_each_quantity(const_cast<HardwareProfile&>(p), [&](const std::string& path, Quantity& q) { check(path, q); });
  CLM_RETURN_IF_ERROR(st);
  const double sf = p.cpu.sustained_factor.value;
  if (!(sf > 0 && sf <= 1)) return bad(p.id, "cpu.sustained_factor must be in (0,1]");
  if (p.memory.ram_safe_allowance.value > p.memory.ram_total.value)
    return bad(p.id, "ram_safe_allowance exceeds ram_total");
  if (p.gpu.vram_budget.value > p.gpu.vram_total.value) return bad(p.id, "vram_budget exceeds vram_total");
  return Status::ok();
}

Status validate(const NetworkProfile& n) {
  Status st;
  for_each_quantity(const_cast<NetworkProfile&>(n), [&](const std::string& path, Quantity& q) {
    if (st.is_ok() && (!std::isfinite(q.value) || q.value < 0))
      st = make_error(ErrorCode::kInvalidArgument, "network: " + path + " must be finite and >= 0");
  });
  CLM_RETURN_IF_ERROR(st);
  for (const auto& l : n.links)
    if (l.bandwidth_bytes_per_s.value <= 0)
      return make_error(ErrorCode::kInvalidArgument, "network: link " + l.from + "->" + l.to + " has zero bandwidth");
  return Status::ok();
}

// ---- qualification -------------------------------------------------------------------------------------

namespace {
template <typename T>
Status qualify_impl(T& obj, std::string_view id, std::string_view qualification_source) {
  std::vector<std::string> synthetic;
  for_each_quantity(obj, [&](const std::string& path, Quantity& q) {
    if (q.provenance == Provenance::kSynthetic) synthetic.push_back(path);
  });
  if (!synthetic.empty()) {
    std::string msg = "cannot mark '" + std::string(id) + "' Qualified: " + std::to_string(synthetic.size()) +
                      " synthetic quantit" + (synthetic.size() == 1 ? "y" : "ies") + " (first: " + synthetic.front() + ")";
    return make_error(ErrorCode::kFailedPrecondition, std::move(msg));
  }
  for_each_quantity(obj, [&](const std::string&, Quantity& q) {
    q.provenance = Provenance::kQualified;
    q.source += "; qualified: " + std::string(qualification_source);
  });
  return Status::ok();
}
}  // namespace

Status mark_qualified(HardwareProfile& p, std::string_view src) {
  CLM_RETURN_IF_ERROR(validate(p));
  return qualify_impl(p, p.id, src);
}
Status mark_qualified(NetworkProfile& n, std::string_view src) {
  CLM_RETURN_IF_ERROR(validate(n));
  return qualify_impl(n, "network", src);
}

// ---- JSON ----------------------------------------------------------------------------------------------

namespace {

struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

json q_to_json(const Quantity& q) {
  return json{{"value", q.value}, {"provenance", std::string(to_string(q.provenance))}, {"source", q.source}};
}

Quantity q_from_json(const json& j, const std::string& path) {
  if (!j.is_object()) throw ParseError(path + ": expected a quantity object {value, provenance, source}");
  Quantity q;
  if (!j.contains("value") || !j["value"].is_number()) throw ParseError(path + ".value missing or not a number");
  q.value = j["value"].get<double>();
  if (!j.contains("provenance") || !j["provenance"].is_string() ||
      !provenance_from_string(j["provenance"].get<std::string>(), q.provenance))
    throw ParseError(path + ".provenance missing or unknown");
  if (j.contains("source")) q.source = j["source"].get<std::string>();
  return q;
}

const json& need(const json& j, const char* key, const char* path) {
  if (!j.is_object() || !j.contains(key)) throw ParseError(std::string(path) + "." + key + " missing");
  return j[key];
}

Quantity q_field(const json& j, const char* key, const char* path) {
  return q_from_json(need(j, key, path), std::string(path) + "." + key);
}

json qmap_to_json(const std::map<std::string, Quantity>& m) {
  json o = json::object();
  for (const auto& [k, v] : m) o[k] = q_to_json(v);
  return o;
}

std::map<std::string, Quantity> qmap_from_json(const json& j, const std::string& path) {
  if (!j.is_object()) throw ParseError(path + ": expected an object");
  std::map<std::string, Quantity> m;
  for (auto it = j.begin(); it != j.end(); ++it) m[it.key()] = q_from_json(it.value(), path + "." + it.key());
  return m;
}

Status wrap(const std::exception& e) { return make_error(ErrorCode::kInvalidArgument, e.what()); }

Result<std::string> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return make_error(ErrorCode::kNotFound, "cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

Status write_file(const std::string& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return make_error(ErrorCode::kUnavailable, "cannot write " + path);
  out << data << '\n';
  return out ? Status::ok() : make_error(ErrorCode::kUnavailable, "write failed: " + path);
}

}  // namespace

std::string to_json(const HardwareProfile& p, int indent) {
  json j;
  j["schema"] = "clusterlm.hardware_profile.v1";
  j["id"] = p.id;
  j["role"] = p.role == DomainRole::kFather ? "father" : "node";
  j["synthetic"] = p.synthetic_fixture;
  j["note"] = p.note;
  j["cpu"] = {{"arch", p.cpu.arch},
              {"features", p.cpu.features},
              {"usable_threads", q_to_json(p.cpu.usable_threads)},
              {"expert_bytes_per_s", qmap_to_json(p.cpu.expert_bytes_per_s)},
              {"q_scaling", q_to_json(p.cpu.q_scaling)},
              {"sustained_factor", q_to_json(p.cpu.sustained_factor)}};
  j["memory"] = {{"ram_total", q_to_json(p.memory.ram_total)},
                 {"ram_safe_allowance", q_to_json(p.memory.ram_safe_allowance)},
                 {"ram_bandwidth", q_to_json(p.memory.ram_bandwidth)},
                 {"pinned_limit", q_to_json(p.memory.pinned_limit)}};
  j["gpu"] = {{"name", p.gpu.name},
              {"vram_total", q_to_json(p.gpu.vram_total)},
              {"vram_budget", q_to_json(p.gpu.vram_budget)},
              {"gpu_expert_bytes_per_s", q_to_json(p.gpu.gpu_expert_bytes_per_s)},
              {"dense_layer_ms", qmap_to_json(p.gpu.dense_layer_ms)},
              {"dense_q_scaling", q_to_json(p.gpu.dense_q_scaling)},
              {"pcie_h2d_bytes_per_s", q_to_json(p.gpu.pcie_h2d_bytes_per_s)},
              {"prefill_tokens_per_s", q_to_json(p.gpu.prefill_tokens_per_s)}};
  j["overheads"] = {{"os_reserve_ram", q_to_json(p.overheads.os_reserve_ram)},
                    {"scratch_vram", q_to_json(p.overheads.scratch_vram)},
                    {"staging_ram", q_to_json(p.overheads.staging_ram)}};
  return j.dump(indent);
}

Result<HardwareProfile> hardware_profile_from_json(std::string_view text) {
  try {
    const json j = json::parse(text);
    HardwareProfile p;
    p.id = need(j, "id", "profile").get<std::string>();
    const std::string role = need(j, "role", "profile").get<std::string>();
    if (role == "father") p.role = DomainRole::kFather;
    else if (role == "node") p.role = DomainRole::kNode;
    else throw ParseError("profile.role must be 'father' or 'node'");
    p.synthetic_fixture = j.value("synthetic", false);
    p.note = j.value("note", std::string());
    const json& c = need(j, "cpu", "profile");
    p.cpu.arch = need(c, "arch", "cpu").get<std::string>();
    p.cpu.features = need(c, "features", "cpu").get<std::vector<std::string>>();
    p.cpu.usable_threads = q_field(c, "usable_threads", "cpu");
    p.cpu.expert_bytes_per_s = qmap_from_json(need(c, "expert_bytes_per_s", "cpu"), "cpu.expert_bytes_per_s");
    p.cpu.q_scaling = q_field(c, "q_scaling", "cpu");
    p.cpu.sustained_factor = q_field(c, "sustained_factor", "cpu");
    const json& m = need(j, "memory", "profile");
    p.memory.ram_total = q_field(m, "ram_total", "memory");
    p.memory.ram_safe_allowance = q_field(m, "ram_safe_allowance", "memory");
    p.memory.ram_bandwidth = q_field(m, "ram_bandwidth", "memory");
    p.memory.pinned_limit = q_field(m, "pinned_limit", "memory");
    const json& g = need(j, "gpu", "profile");
    p.gpu.name = g.value("name", std::string());
    p.gpu.vram_total = q_field(g, "vram_total", "gpu");
    p.gpu.vram_budget = q_field(g, "vram_budget", "gpu");
    p.gpu.gpu_expert_bytes_per_s = q_field(g, "gpu_expert_bytes_per_s", "gpu");
    p.gpu.dense_layer_ms = qmap_from_json(need(g, "dense_layer_ms", "gpu"), "gpu.dense_layer_ms");
    p.gpu.dense_q_scaling = q_field(g, "dense_q_scaling", "gpu");
    p.gpu.pcie_h2d_bytes_per_s = q_field(g, "pcie_h2d_bytes_per_s", "gpu");
    p.gpu.prefill_tokens_per_s = q_field(g, "prefill_tokens_per_s", "gpu");
    const json& o = need(j, "overheads", "profile");
    p.overheads.os_reserve_ram = q_field(o, "os_reserve_ram", "overheads");
    p.overheads.scratch_vram = q_field(o, "scratch_vram", "overheads");
    p.overheads.staging_ram = q_field(o, "staging_ram", "overheads");
    CLM_RETURN_IF_ERROR(validate(p));
    return p;
  } catch (const std::exception& e) {
    return wrap(e);
  }
}

std::string to_json(const NetworkProfile& n, int indent) {
  json j;
  j["schema"] = "clusterlm.network_profile.v1";
  j["synthetic"] = n.synthetic_fixture;
  j["note"] = n.note;
  j["father_egress_bytes_per_s"] = q_to_json(n.father_egress_bytes_per_s);
  j["links"] = json::array();
  for (const auto& l : n.links)
    j["links"].push_back({{"from", l.from},
                          {"to", l.to},
                          {"bandwidth_bytes_per_s", q_to_json(l.bandwidth_bytes_per_s)},
                          {"rtt_ms", q_to_json(l.rtt_ms)},
                          {"jitter_ms", q_to_json(l.jitter_ms)}});
  return j.dump(indent);
}

Result<NetworkProfile> network_profile_from_json(std::string_view text) {
  try {
    const json j = json::parse(text);
    NetworkProfile n;
    n.synthetic_fixture = j.value("synthetic", false);
    n.note = j.value("note", std::string());
    n.father_egress_bytes_per_s = q_field(j, "father_egress_bytes_per_s", "network");
    const json& links = need(j, "links", "network");
    if (!links.is_array()) throw ParseError("network.links must be an array");
    for (const auto& lj : links) {
      LinkProfile l;
      l.from = need(lj, "from", "link").get<std::string>();
      l.to = need(lj, "to", "link").get<std::string>();
      l.bandwidth_bytes_per_s = q_field(lj, "bandwidth_bytes_per_s", "link");
      l.rtt_ms = q_field(lj, "rtt_ms", "link");
      l.jitter_ms = q_field(lj, "jitter_ms", "link");
      n.links.push_back(std::move(l));
    }
    CLM_RETURN_IF_ERROR(validate(n));
    return n;
  } catch (const std::exception& e) {
    return wrap(e);
  }
}

Result<HardwareProfile> load_hardware_profile(const std::string& path) {
  CLM_ASSIGN_OR_RETURN(std::string text, read_file(path));
  return hardware_profile_from_json(text);
}
Result<NetworkProfile> load_network_profile(const std::string& path) {
  CLM_ASSIGN_OR_RETURN(std::string text, read_file(path));
  return network_profile_from_json(text);
}
Status save_hardware_profile(const HardwareProfile& p, const std::string& path) { return write_file(path, to_json(p)); }
Status save_network_profile(const NetworkProfile& n, const std::string& path) { return write_file(path, to_json(n)); }

}  // namespace clusterlm::placement
