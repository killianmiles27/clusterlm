#include "clusterlm/profiles/topology.hpp"

#include <algorithm>
#include <set>

#include "strict_json.hpp"

namespace clusterlm::profiles {

using detail::bad;
using detail::Obj;
using nlohmann::json;

std::string_view to_string(WorkerAvailability a) noexcept {
  switch (a) {
    case WorkerAvailability::kAvailable: return "available";
    case WorkerAvailability::kBusy: return "busy";
    case WorkerAvailability::kOffline: return "offline";
    case WorkerAvailability::kPaused: return "paused";
    case WorkerAvailability::kReleasing: return "releasing";
  }
  return "offline";
}

namespace {

bool parse_cc(std::string_view s, int& major, int& minor) {
  const auto dot = s.find('.');
  if (dot == std::string_view::npos || dot == 0 || dot + 1 >= s.size() || s.size() > 5) return false;
  major = 0;
  minor = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i == dot) continue;
    if (s[i] < '0' || s[i] > '9') return false;
    int& target = i < dot ? major : minor;
    target = target * 10 + (s[i] - '0');
  }
  return true;
}

std::uint32_t max_vram(const WorkerCapability& c) {
  std::uint32_t v = 0;
  for (const auto& g : c.gpus) v = std::max(v, g.vram_mib);
  return v;
}

bool is_hex64(std::string_view s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

// Reason a Worker cannot be used at all right now ("" = usable).
std::string unusable_reason(const WorkerCapability& w, const ResolveOptions& opt) {
  switch (w.availability) {
    case WorkerAvailability::kBusy: return w.name + " is busy (a local user is active)";
    case WorkerAvailability::kOffline: return w.name + " is offline";
    case WorkerAvailability::kPaused: return w.name + " is paused";
    case WorkerAvailability::kReleasing: return w.name + " is releasing its resources";
    case WorkerAvailability::kAvailable: break;
  }
  if (w.on_battery && !opt.allow_on_battery) return w.name + " is on battery";
  if (!opt.backend_id.empty() && std::find(w.backends.begin(), w.backends.end(), opt.backend_id) == w.backends.end())
    return w.name + " does not have the " + opt.backend_id + " backend";
  return {};
}

}  // namespace

bool compute_capability_at_least(std::string_view have, std::string_view need) {
  int hm = 0, hn = 0, nm = 0, nn = 0;
  if (!parse_cc(have, hm, hn) || !parse_cc(need, nm, nn)) return false;
  return hm > nm || (hm == nm && hn >= nn);
}

bool meets_requirements(const WorkerCapability& cap, const Requirements& req, std::string* why) {
  auto fail = [&](std::string w) {
    if (why != nullptr) *why = std::move(w);
    return false;
  };
  if (req.min_ram_mib && cap.ram_mib < *req.min_ram_mib) return fail(cap.name + " has less RAM than required");
  if (req.min_link_mbit) {
    if (cap.link_mbit == 0) return fail("link speed to " + cap.name + " is unknown");
    if (cap.link_mbit < *req.min_link_mbit) return fail("link to " + cap.name + " is slower than required");
  }
  if (cap.gpus.empty()) {
    if (req.cpu_only_ok) return true;
    return fail(cap.name + " has no usable GPU");
  }
  // At least one GPU must satisfy vendor + VRAM + compute capability together.
  bool vendor_seen = req.gpu_vendor.empty(), vram_seen = !req.min_vram_mib, cc_seen = !req.min_compute_capability;
  for (const auto& g : cap.gpus) {
    const bool v = req.gpu_vendor.empty() || std::find(req.gpu_vendor.begin(), req.gpu_vendor.end(), g.vendor) != req.gpu_vendor.end();
    const bool m = !req.min_vram_mib || g.vram_mib >= *req.min_vram_mib;
    const bool c = !req.min_compute_capability || compute_capability_at_least(g.compute_capability, *req.min_compute_capability);
    vendor_seen |= v;
    vram_seen |= m;
    cc_seen |= c;
    if (v && m && c) return true;
  }
  if (!vendor_seen) return fail(cap.name + " has no GPU from the required vendor");
  if (!vram_seen) return fail(cap.name + " has less VRAM than required");
  if (!cc_seen) return fail(cap.name + " has a GPU below the required compute capability");
  return fail(cap.name + " has no single GPU meeting all GPU requirements");
}

std::vector<std::string> TopologyResolution::fingerprints() const {
  std::vector<std::string> out;
  for (const auto& s : slots)
    if (s.state == SlotResolution::State::kFilled && s.fingerprint) out.push_back(*s.fingerprint);
  return out;
}

TopologyResolution resolve_topology(const Topology& topology, const BindingMap& bindings, const std::vector<WorkerCapability>& workers,
                                    const ResolveOptions& options) {
  TopologyResolution out;
  std::vector<const Slot*> worker_slots;
  for (const auto& s : topology.slots)
    if (s.kind == Slot::Kind::kWorker) worker_slots.push_back(&s);
  out.slots.resize(worker_slots.size());

  auto find_worker = [&](const std::string& fp) -> const WorkerCapability* {
    for (const auto& w : workers)
      if (w.fingerprint == fp) return &w;
    return nullptr;
  };
  std::set<std::string> chosen;
  std::map<std::string, std::string> chosen_by;  // fingerprint -> slot label

  auto label_of = [](const Slot& s) { return s.label.empty() ? s.slot : s.label; };
  auto fill = [&](std::size_t i, const WorkerCapability& w) {
    out.slots[i].state = SlotResolution::State::kFilled;
    out.slots[i].fingerprint = w.fingerprint;
    out.slots[i].reason.clear();
    chosen.insert(w.fingerprint);
    chosen_by[w.fingerprint] = label_of(*worker_slots[i]);
  };
  auto reject = [&](std::size_t i, std::string reason) {
    out.slots[i].state = worker_slots[i]->optional ? SlotResolution::State::kEmptyOptional : SlotResolution::State::kUnsatisfied;
    out.slots[i].reason = std::move(reason);
  };
  for (std::size_t i = 0; i < worker_slots.size(); ++i) {
    out.slots[i].slot = worker_slots[i]->slot;
    out.slots[i].label = label_of(*worker_slots[i]);
  }

  // Pass 1: binding / machine selectors, in slot order.
  for (std::size_t i = 0; i < worker_slots.size(); ++i) {
    const Slot& s = *worker_slots[i];
    if (!s.select || s.select->mode == Selector::Mode::kRequirements) continue;
    std::string fp;
    if (s.select->mode == Selector::Mode::kMachine) {
      fp = s.select->machine;
    } else {
      auto it = bindings.find(s.select->binding);
      if (it == bindings.end()) {
        reject(i, "Worker slot " + label_of(s) + " is not assigned");
        continue;
      }
      fp = it->second;
    }
    const WorkerCapability* w = find_worker(fp);
    if (w == nullptr) {
      reject(i, "the machine for " + label_of(s) + " is not paired");
      continue;
    }
    if (chosen.count(fp) != 0) {
      reject(i, w->name + " already fills " + chosen_by[fp]);
      continue;
    }
    if (auto why = unusable_reason(*w, options); !why.empty()) {
      reject(i, std::move(why));
      continue;
    }
    if (s.select->requirements) {
      std::string why;
      if (!meets_requirements(*w, *s.select->requirements, &why)) {
        reject(i, std::move(why));
        continue;
      }
    }
    fill(i, *w);
  }

  // Pass 2: requirements selectors over the machines not yet chosen.
  for (std::size_t i = 0; i < worker_slots.size(); ++i) {
    const Slot& s = *worker_slots[i];
    if (!s.select || s.select->mode != Selector::Mode::kRequirements) continue;
    const WorkerCapability* best = nullptr;
    std::string first_why;
    for (const auto& w : workers) {
      if (chosen.count(w.fingerprint) != 0) continue;
      if (auto why = unusable_reason(w, options); !why.empty()) {
        if (first_why.empty()) first_why = std::move(why);
        continue;
      }
      std::string why;
      if (!meets_requirements(w, *s.select->requirements, &why)) {
        if (first_why.empty()) first_why = std::move(why);
        continue;
      }
      if (best == nullptr || max_vram(w) > max_vram(*best) || (max_vram(w) == max_vram(*best) && w.fingerprint < best->fingerprint)) best = &w;
    }
    if (best != nullptr) fill(i, *best);
    else reject(i, first_why.empty() ? "no paired Worker meets the requirements of " + label_of(s) : first_why);
  }

  // Count rules: shed optional Workers beyond max_workers (last slots first), then check min and required slots.
  const auto bounds = worker_count_bounds(topology);
  for (const auto& s : out.slots) out.filled += s.state == SlotResolution::State::kFilled ? 1u : 0u;
  for (std::size_t i = worker_slots.size(); i-- > 0 && out.filled > bounds.max;) {
    if (worker_slots[i]->optional && out.slots[i].state == SlotResolution::State::kFilled) {
      out.slots[i].state = SlotResolution::State::kEmptyOptional;
      out.slots[i].fingerprint.reset();
      out.slots[i].reason = "not used: the profile allows at most " + std::to_string(bounds.max) + " Workers";
      --out.filled;
    }
  }
  bool required_ok = true;
  for (std::size_t i = 0; i < out.slots.size(); ++i) {
    if (!worker_slots[i]->optional && out.slots[i].state != SlotResolution::State::kFilled) {
      required_ok = false;
      if (out.headline.empty()) out.headline = out.slots[i].reason;
    }
  }
  if (required_ok && out.filled < bounds.min) {
    required_ok = false;
    for (const auto& s : out.slots)
      if (s.state != SlotResolution::State::kFilled && !s.reason.empty()) {
        out.headline = s.reason;
        break;
      }
    if (out.headline.empty()) out.headline = "fewer than " + std::to_string(bounds.min) + " Workers are available";
  }
  out.satisfiable = required_ok && out.filled >= bounds.min && out.filled <= bounds.max;
  if (out.satisfiable) out.headline.clear();
  return out;
}

// ---- WorkerCapability JSON -------------------------------------------------------------------------------------------

Status validate(const WorkerCapability& c) {
  if (!is_hex64(c.fingerprint)) return bad("worker fingerprint must be 64 lowercase hex characters");
  if (c.name.empty() || c.name.size() > 64) return bad("worker name must be 1..64 characters");
  if (c.ram_mib > 16777216 || c.cpu_threads > 4096 || c.link_mbit > 1000000) return bad("worker capability value out of range");
  if (c.gpus.size() > 16) return bad("too many GPUs");
  for (const auto& g : c.gpus) {
    if (g.vendor != "nvidia" && g.vendor != "amd" && g.vendor != "intel" && g.vendor != "other") return bad("unknown GPU vendor");
    if (g.name.size() > 120 || g.vram_mib > 16777216) return bad("GPU field out of range");
    int a = 0, b = 0;
    if (!g.compute_capability.empty() && !parse_cc(g.compute_capability, a, b)) return bad("compute capability is malformed");
  }
  if (c.backends.size() > 16) return bad("too many backends");
  for (const auto& b : c.backends)
    if (b.empty() || b.size() > 32) return bad("backend id is malformed");
  if (c.build_hash.size() > 128) return bad("build_hash too long");
  if (c.provenance != "declared" && c.provenance != "measured") return bad("provenance must be declared or measured");
  if (c.source.size() > 200) return bad("source too long");
  return Status::ok();
}

std::string to_json(const WorkerCapability& c) {
  json gpus = json::array();
  for (const auto& g : c.gpus)
    gpus.push_back({{"vendor", g.vendor}, {"name", g.name}, {"vram_mib", g.vram_mib}, {"compute_capability", g.compute_capability}});
  json j = {{"fingerprint", c.fingerprint}, {"name", c.name},           {"ram_mib", c.ram_mib},   {"cpu_threads", c.cpu_threads},
            {"gpus", std::move(gpus)},      {"link_mbit", c.link_mbit}, {"backends", c.backends}, {"build_hash", c.build_hash},
            {"provenance", c.provenance},   {"source", c.source},       {"availability", std::string(to_string(c.availability))},
            {"on_battery", c.on_battery}};
  return j.dump(2) + "\n";
}

Result<WorkerCapability> worker_capability_from_json(std::string_view text) {
  CLM_ASSIGN_OR_RETURN(json doc, detail::parse_object_document(text));
  CLM_ASSIGN_OR_RETURN(Obj o, Obj::of(doc, "worker"));
  WorkerCapability c;
  CLM_RETURN_IF_ERROR(o.req_str("fingerprint", c.fingerprint, 64, 64));
  CLM_RETURN_IF_ERROR(o.req_str("name", c.name, 64));
  CLM_RETURN_IF_ERROR(o.opt_u32("ram_mib", c.ram_mib, 0, 16777216));
  CLM_RETURN_IF_ERROR(o.opt_u32("cpu_threads", c.cpu_threads, 0, 4096));
  CLM_RETURN_IF_ERROR(o.opt_u32("link_mbit", c.link_mbit, 0, 1000000));
  CLM_ASSIGN_OR_RETURN(const json* gpus, o.opt_array("gpus", 16));
  if (gpus != nullptr)
    for (const auto& gj : *gpus) {
      CLM_ASSIGN_OR_RETURN(Obj go, Obj::of(gj, "worker.gpus[]"));
      GpuInfo g;
      CLM_RETURN_IF_ERROR(go.req_str("vendor", g.vendor, 16));
      CLM_RETURN_IF_ERROR(go.opt_str("name", g.name, 120));
      CLM_RETURN_IF_ERROR(go.opt_u32("vram_mib", g.vram_mib, 0, 16777216));
      CLM_RETURN_IF_ERROR(go.opt_str("compute_capability", g.compute_capability, 8));
      CLM_RETURN_IF_ERROR(go.finish());
      c.gpus.push_back(std::move(g));
    }
  CLM_RETURN_IF_ERROR(o.opt_string_array("backends", c.backends, 16, 32));
  CLM_RETURN_IF_ERROR(o.opt_str("build_hash", c.build_hash, 128));
  CLM_RETURN_IF_ERROR(o.opt_str("provenance", c.provenance, 16));
  CLM_RETURN_IF_ERROR(o.opt_str("source", c.source, 200));
  std::size_t av = 0;
  CLM_RETURN_IF_ERROR(o.opt_enum("availability", {"available", "busy", "offline", "paused", "releasing"}, av));
  c.availability = static_cast<WorkerAvailability>(av);
  CLM_RETURN_IF_ERROR(o.opt_bool("on_battery", c.on_battery));
  CLM_RETURN_IF_ERROR(o.finish());
  CLM_RETURN_IF_ERROR(validate(c));
  return c;
}

}  // namespace clusterlm::profiles
