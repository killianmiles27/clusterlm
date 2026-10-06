#include "clusterlm/config/settings.hpp"

#include <algorithm>
#include <cctype>

#include <nlohmann/json.hpp>

#include "clusterlm/transport/transport.hpp"  // Endpoint::parse for address validation

namespace clusterlm::config {

using nlohmann::json;

namespace {

constexpr std::size_t kMaxName = 64;
constexpr std::size_t kMaxPath = 1024;
constexpr std::size_t kMaxDevices = 64;
constexpr std::uint32_t kMaxGib = 4096;

bool printable_label(std::string_view s, std::size_t max) {
  if (s.empty() || s.size() > max) return false;
  return std::all_of(s.begin(), s.end(), [](unsigned char c) { return c >= 0x20 && c != 0x7F; });
}

Status bad(std::string what) { return make_error(ErrorCode::kInvalidArgument, std::move(what)); }

// ---- typed field readers: absent -> keep the default, present with the wrong type -> error ----------------

template <typename T>
Status read(const json& obj, const char* key, T& out) {
  auto it = obj.find(key);
  if (it == obj.end()) return Status::ok();
  if constexpr (std::is_same_v<T, bool>) {
    if (!it->is_boolean()) return bad(std::string("field '") + key + "' must be a boolean");
    out = it->get<bool>();
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (!it->is_string()) return bad(std::string("field '") + key + "' must be a string");
    out = it->get<std::string>();
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    if (!it->is_number_integer()) return bad(std::string("field '") + key + "' must be an integer");
    out = it->get<std::int64_t>();
  } else {
    static_assert(std::is_same_v<T, std::uint32_t>);
    if (!it->is_number_unsigned() || it->get<std::uint64_t>() > 0xFFFFFFFFull)
      return bad(std::string("field '") + key + "' must be a non-negative 32-bit integer");
    out = it->get<std::uint32_t>();
  }
  return Status::ok();
}

Result<const json*> read_object(const json& obj, const char* key) {
  auto it = obj.find(key);
  if (it == obj.end()) return static_cast<const json*>(nullptr);
  if (!it->is_object()) return bad(std::string("field '") + key + "' must be an object");
  return &*it;
}

Result<json> parse_document(std::string_view text) {
  json doc = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object()) return bad("not a JSON object");
  return doc;
}

Result<PairedDevice> device_from(const json& j) {
  if (!j.is_object()) return bad("paired device must be an object");
  PairedDevice d;
  CLM_RETURN_IF_ERROR(read(j, "fingerprint", d.fingerprint));
  CLM_RETURN_IF_ERROR(read(j, "name", d.name));
  CLM_RETURN_IF_ERROR(read(j, "address", d.address));
  CLM_RETURN_IF_ERROR(read(j, "role", d.role));
  CLM_RETURN_IF_ERROR(read(j, "paired_at_unix", d.paired_at_unix));
  CLM_RETURN_IF_ERROR(validate(d));
  return d;
}

json device_to(const PairedDevice& d) {
  return json{{"fingerprint", d.fingerprint}, {"name", d.name}, {"address", d.address}, {"role", d.role},
              {"paired_at_unix", d.paired_at_unix}};
}

Result<std::map<std::string, std::string>> string_map(const json& obj, const char* key) {
  std::map<std::string, std::string> out;
  CLM_ASSIGN_OR_RETURN(const json* m, read_object(obj, key));
  if (m == nullptr) return out;
  for (auto it = m->begin(); it != m->end(); ++it) {
    if (!it.value().is_string()) return bad(std::string("values of '") + key + "' must be strings");
    out[it.key()] = it.value().get<std::string>();
  }
  return out;
}

}  // namespace

bool is_fingerprint(std::string_view t) {
  return t.size() == 64 &&
         std::all_of(t.begin(), t.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

Status validate(const PairedDevice& d) {
  if (!is_fingerprint(d.fingerprint)) return bad("device fingerprint must be 64 lowercase hex characters");
  if (!printable_label(d.name, kMaxName)) return bad("device name must be 1.." + std::to_string(kMaxName) + " printable characters");
  if (d.role != "node" && d.role != "father") return bad("device role must be 'node' or 'father'");
  if (!d.address.empty()) {
    auto ep = transport::Endpoint::parse(d.address);
    if (!ep.is_ok() || ep->port == 0) return bad("device address must be host:port");
  }
  if (d.role == "node" && d.address.empty()) return bad("a paired Node needs an address");
  if (d.paired_at_unix < 0) return bad("paired_at_unix must be >= 0");
  return Status::ok();
}

Status validate(const FatherSettings& s) {
  if (s.paired_nodes.size() > kMaxDevices) return bad("too many paired nodes");
  for (std::size_t i = 0; i < s.paired_nodes.size(); ++i) {
    CLM_RETURN_IF_ERROR(validate(s.paired_nodes[i]));
    if (s.paired_nodes[i].role != "node") return bad("paired_nodes may only hold role 'node'");
    for (std::size_t k = 0; k < i; ++k)
      if (s.paired_nodes[k].fingerprint == s.paired_nodes[i].fingerprint) return bad("duplicate paired node");
  }
  for (const auto& [role, fp] : s.assignments) {
    if (!printable_label(role, kMaxName)) return bad("assignment role is malformed");
    if (s.find_node(fp) == nullptr) return bad("assignment names a device that is not paired");
  }
  if (!printable_label(s.selected_tier, kMaxName)) return bad("selected_tier is malformed");
  if (s.context_tokens < 256 || s.context_tokens > (1u << 22)) return bad("context_tokens out of range");
  if (s.keep_ready.release_after_idle_minutes > 7 * 24 * 60) return bad("keep_ready.release_after_idle_minutes out of range");
  for (const auto& [tier, dir] : s.model_dirs) {
    if (!printable_label(tier, kMaxName)) return bad("model_dirs tier id is malformed");
    if (dir.empty() || dir.size() > kMaxPath) return bad("model_dirs path is malformed");
  }
  const auto& a = s.advanced;
  if (a.bench_results_dir.size() > kMaxPath) return bad("advanced.bench_results_dir too long");
  if (a.log_level != "debug" && a.log_level != "info" && a.log_level != "warn" && a.log_level != "error")
    return bad("advanced.log_level must be debug|info|warn|error");
  if (a.default_q < 1 || a.default_q > 16) return bad("advanced.default_q out of range");
  if (a.prefill_chunk < 1 || a.prefill_chunk > 65536) return bad("advanced.prefill_chunk out of range");
  return Status::ok();
}

Status validate(const NodeSettings& s) {
  if (!printable_label(s.name, kMaxName)) return bad("node name is malformed");
  if (s.paired_father) {
    CLM_RETURN_IF_ERROR(validate(*s.paired_father));
    if (s.paired_father->role != "father") return bad("paired_father must have role 'father'");
  }
  if (s.idle_seconds < 1 || s.idle_seconds > 7 * 24 * 3600) return bad("idle_seconds out of range");
  if (s.temp_storage_limit_gib > kMaxGib) return bad("temp_storage_limit_gib out of range");
  if (s.caps.ram_gib > kMaxGib || s.caps.vram_gib > kMaxGib) return bad("resource caps out of range");
  if (s.caps.threads > 4096) return bad("caps.threads out of range");
  return Status::ok();
}

const PairedDevice* FatherSettings::find_node(std::string_view fingerprint) const {
  for (const auto& n : paired_nodes)
    if (n.fingerprint == fingerprint) return &n;
  return nullptr;
}

void FatherSettings::upsert_node(PairedDevice device) {
  for (auto& n : paired_nodes)
    if (n.fingerprint == device.fingerprint) {
      n = std::move(device);
      return;
    }
  paired_nodes.push_back(std::move(device));
}

bool FatherSettings::remove_node(std::string_view fingerprint) {
  const auto before = paired_nodes.size();
  paired_nodes.erase(std::remove_if(paired_nodes.begin(), paired_nodes.end(),
                                    [&](const PairedDevice& n) { return n.fingerprint == fingerprint; }),
                     paired_nodes.end());
  for (auto it = assignments.begin(); it != assignments.end();) {
    if (it->second == fingerprint) it = assignments.erase(it);
    else ++it;
  }
  return paired_nodes.size() != before;
}

// ---------------------------------------------------------------------------------------------- Father JSON

std::string to_json(const FatherSettings& s) {
  json doc;
  doc["version"] = kFatherSettingsVersion;
  doc["kind"] = "clusterlm-father-settings";
  json nodes = json::array();
  for (const auto& n : s.paired_nodes) nodes.push_back(device_to(n));
  doc["paired_nodes"] = std::move(nodes);
  doc["assignments"] = s.assignments;
  doc["selected_tier"] = s.selected_tier;
  doc["context_tokens"] = s.context_tokens;
  doc["keep_ready"] = {{"enabled", s.keep_ready.enabled}, {"release_after_idle_minutes", s.keep_ready.release_after_idle_minutes}};
  doc["model_dirs"] = s.model_dirs;
  doc["advanced"] = {{"bench_results_dir", s.advanced.bench_results_dir}, {"log_level", s.advanced.log_level},
                     {"direct_peer", s.advanced.direct_peer},           {"default_q", s.advanced.default_q},
                     {"prefill_chunk", s.advanced.prefill_chunk}};
  return doc.dump(2) + "\n";
}

Result<FatherSettings> father_settings_from_json(std::string_view text) {
  CLM_ASSIGN_OR_RETURN(json doc, parse_document(text));
  FatherSettings s;
  if (auto it = doc.find("paired_nodes"); it != doc.end()) {
    if (!it->is_array()) return bad("field 'paired_nodes' must be an array");
    for (const auto& j : *it) {
      CLM_ASSIGN_OR_RETURN(auto d, device_from(j));
      s.paired_nodes.push_back(std::move(d));
    }
  }
  CLM_ASSIGN_OR_RETURN(s.assignments, string_map(doc, "assignments"));
  CLM_RETURN_IF_ERROR(read(doc, "selected_tier", s.selected_tier));
  CLM_RETURN_IF_ERROR(read(doc, "context_tokens", s.context_tokens));
  CLM_ASSIGN_OR_RETURN(const json* kr, read_object(doc, "keep_ready"));
  if (kr != nullptr) {
    CLM_RETURN_IF_ERROR(read(*kr, "enabled", s.keep_ready.enabled));
    CLM_RETURN_IF_ERROR(read(*kr, "release_after_idle_minutes", s.keep_ready.release_after_idle_minutes));
  }
  CLM_ASSIGN_OR_RETURN(s.model_dirs, string_map(doc, "model_dirs"));
  CLM_ASSIGN_OR_RETURN(const json* adv, read_object(doc, "advanced"));
  if (adv != nullptr) {
    CLM_RETURN_IF_ERROR(read(*adv, "bench_results_dir", s.advanced.bench_results_dir));
    CLM_RETURN_IF_ERROR(read(*adv, "log_level", s.advanced.log_level));
    CLM_RETURN_IF_ERROR(read(*adv, "direct_peer", s.advanced.direct_peer));
    CLM_RETURN_IF_ERROR(read(*adv, "default_q", s.advanced.default_q));
    CLM_RETURN_IF_ERROR(read(*adv, "prefill_chunk", s.advanced.prefill_chunk));
  }
  CLM_RETURN_IF_ERROR(validate(s));
  return s;
}

// ---------------------------------------------------------------------------------------------- Node JSON

std::string to_json(const NodeSettings& s) {
  json doc;
  doc["version"] = kNodeSettingsVersion;
  doc["kind"] = "clusterlm-node-settings";
  doc["name"] = s.name;
  doc["paired_father"] = s.paired_father ? device_to(*s.paired_father) : json(nullptr);
  doc["allow_when_idle"] = s.allow_when_idle;
  doc["idle_seconds"] = s.idle_seconds;
  doc["ac_only"] = s.ac_only;
  doc["temp_storage_limit_gib"] = s.temp_storage_limit_gib;
  doc["paused"] = s.paused;
  doc["start_with_system"] = s.start_with_system;
  doc["caps"] = {{"ram_gib", s.caps.ram_gib}, {"vram_gib", s.caps.vram_gib}, {"threads", s.caps.threads}};
  return doc.dump(2) + "\n";
}

Result<NodeSettings> node_settings_from_json(std::string_view text) {
  CLM_ASSIGN_OR_RETURN(json doc, parse_document(text));
  NodeSettings s;
  CLM_RETURN_IF_ERROR(read(doc, "name", s.name));
  if (auto it = doc.find("paired_father"); it != doc.end() && !it->is_null()) {
    CLM_ASSIGN_OR_RETURN(auto d, device_from(*it));
    s.paired_father = std::move(d);
  }
  CLM_RETURN_IF_ERROR(read(doc, "allow_when_idle", s.allow_when_idle));
  CLM_RETURN_IF_ERROR(read(doc, "idle_seconds", s.idle_seconds));
  CLM_RETURN_IF_ERROR(read(doc, "ac_only", s.ac_only));
  CLM_RETURN_IF_ERROR(read(doc, "temp_storage_limit_gib", s.temp_storage_limit_gib));
  CLM_RETURN_IF_ERROR(read(doc, "paused", s.paused));
  CLM_RETURN_IF_ERROR(read(doc, "start_with_system", s.start_with_system));
  CLM_ASSIGN_OR_RETURN(const json* caps, read_object(doc, "caps"));
  if (caps != nullptr) {
    CLM_RETURN_IF_ERROR(read(*caps, "ram_gib", s.caps.ram_gib));
    CLM_RETURN_IF_ERROR(read(*caps, "vram_gib", s.caps.vram_gib));
    CLM_RETURN_IF_ERROR(read(*caps, "threads", s.caps.threads));
  }
  CLM_RETURN_IF_ERROR(validate(s));
  return s;
}

}  // namespace clusterlm::config
