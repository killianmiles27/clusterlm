#include "clusterlm/profiles/validate.hpp"

#include <algorithm>
#include <regex>
#include <map>
#include <set>

#include "strict_json.hpp"

namespace clusterlm::profiles {

using detail::bad;

namespace {

bool matches(const char* pattern, std::string_view s) {
  // Patterns are tiny and anchored; std::regex is acceptable here (never on untrusted-size input: callers bound lengths first).
  if (s.size() > 256) return false;
  return std::regex_match(s.begin(), s.end(), std::regex(pattern));
}

bool is_hex(std::string_view s, std::size_t n) {
  return s.size() == n && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

bool text_ok(std::string_view s, std::size_t max, bool allow_newline = false) {
  if (s.size() > max) return false;
  return std::all_of(s.begin(), s.end(), [&](unsigned char c) { return (c >= 0x20 && c != 0x7F) || (allow_newline && (c == '\n' || c == '\t')); });
}

bool is_slot_name(std::string_view s) { return matches("[a-z][a-z0-9_-]{0,31}", s); }
bool is_hq_id(std::string_view s) { return s.size() <= 32 && matches("HQ-[A-Z0-9]+-[0-9]+", s); }

Status check_requirements(const Requirements& r, const std::string& where) {
  std::set<std::string> seen;
  for (const auto& v : r.gpu_vendor) {
    if (v != "nvidia" && v != "amd" && v != "intel") return bad(where + ".gpu_vendor has an unknown vendor");
    if (!seen.insert(v).second) return bad(where + ".gpu_vendor has duplicates");
  }
  if (r.min_compute_capability && !matches("[0-9]{1,2}\\.[0-9]{1,2}", *r.min_compute_capability))
    return bad(where + ".min_compute_capability must look like 8.6");
  return Status::ok();
}

}  // namespace

bool is_profile_id(std::string_view s) { return s.size() <= 53 && matches("prof_[a-z0-9][a-z0-9_]{2,47}", s); }
bool is_alias_id(std::string_view s) { return s.size() <= 53 && matches("alias_[a-z0-9][a-z0-9_]{2,47}", s); }
bool is_api_model_id(std::string_view s) { return s.size() <= 64 && matches("[a-z0-9][a-z0-9._-]{0,63}", s); }

Status validate(const Profile& p) {
  if (!is_profile_id(p.id)) return bad("profile id must match prof_[a-z0-9][a-z0-9_]{2,47}");
  if (p.name.empty() || !text_ok(p.name, 80)) return bad("profile name must be 1..80 printable characters");
  if (!text_ok(p.description, 1000, true)) return bad("profile description is malformed");
  if (p.revision < 1) return bad("profile revision must be >= 1");
  if (p.example_of && !matches("[a-z0-9][a-z0-9:_-]{0,63}", *p.example_of)) return bad("example_of is malformed");
  {
    std::set<std::string> seen;
    for (const auto& e : p.qualification_experiments) {
      if (!is_hq_id(e)) return bad("qualification experiment ids look like HQ-NAME-01");
      if (!seen.insert(e).second) return bad("qualification_experiments has duplicates");
    }
  }

  const auto& id = p.model.identity;
  if (id.family.empty() || !text_ok(id.family, 120)) return bad("model family must be 1..120 printable characters");
  if (!text_ok(id.display_name, 200)) return bad("model display_name is malformed");
  if (!matches("[A-Za-z0-9_+.-]{1,32}", id.quant)) return bad("model quant must be a ggml type name such as IQ3_S");
  if (id.artifact_id && !text_ok(*id.artifact_id, 200)) return bad("artifact_id is malformed");
  if (id.expected_root_hash && !is_hex(*id.expected_root_hash, 64)) return bad("expected_root_hash must be 64 lowercase hex characters or null");
  for (const auto& f : id.expected_files) {
    if (!matches("[a-z][a-z0-9-]{0,31}", f.role)) return bad("expected file role is malformed");
    if (f.name && (f.name->empty() || f.name->size() > 255 || f.name->find_first_of("/\\:*?\"<>|") != std::string::npos || !text_ok(*f.name, 255)))
      return bad("expected file names are bare file names, never paths");
  }
  if (p.model.library_id && !matches("mdl_[0-9a-f]{24}", *p.model.library_id)) return bad("library_id must be mdl_ + 24 hex characters");

  if (!matches("[a-z][a-z0-9-]{1,31}", p.backend.id)) return bad("backend id is malformed");
  if (p.backend.options.size() > 16) return bad("too many backend options");
  for (std::size_t i = 0; i < p.backend.options.size(); ++i) {
    if (!matches("[a-z][a-z0-9_]{0,31}", p.backend.options[i].key)) return bad("backend option key is malformed");
    if (i > 0 && !(p.backend.options[i - 1].key < p.backend.options[i].key)) return bad("backend options must be sorted by unique key");
  }

  const auto& c = p.context;
  if (c.default_tokens < 256 || c.max_tokens > 4194304 || c.default_tokens > c.max_tokens) return bad("context needs 256 <= default_tokens <= max_tokens <= 4194304");
  if (c.offered_profiles.size() > 16) return bad("too many offered context sizes");
  for (std::size_t i = 0; i < c.offered_profiles.size(); ++i) {
    if (c.offered_profiles[i] < 256 || c.offered_profiles[i] > 4194304) return bad("offered context size out of range");
    if (i > 0 && c.offered_profiles[i] <= c.offered_profiles[i - 1]) return bad("offered_profiles must be strictly ascending");
  }
  if (!c.offered_profiles.empty() && c.offered_profiles.back() != c.max_tokens) return bad("max_tokens must be the largest offered context size");

  // topology
  const auto& slots = p.topology.slots;
  if (slots.empty() || slots.size() > 1 + kMaxWorkerSlots) return bad("topology needs a host slot and at most 8 worker slots");
  if (slots[0].kind != Slot::Kind::kHost) return bad("slot 0 must be the host slot");
  std::set<std::string> names, bindings, machines;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const Slot& s = slots[i];
    if (!is_slot_name(s.slot)) return bad("slot name is malformed");
    if (!names.insert(s.slot).second) return bad("slot names must be unique");
    if (!text_ok(s.label, 80)) return bad("slot label is malformed");
    if (s.kind == Slot::Kind::kHost) {
      if (i != 0) return bad("only slot 0 may be a host slot");
      if (s.select || s.optional) return bad("the host slot has no selector and cannot be optional");
      continue;
    }
    if (!s.select) return bad("worker slot '" + s.slot + "' needs a selector");
    const Selector& sel = *s.select;
    switch (sel.mode) {
      case Selector::Mode::kBinding:
        if (!matches("[a-z][a-z0-9:_-]{0,47}", sel.binding)) return bad("binding name is malformed");
        if (!sel.machine.empty()) return bad("a binding selector cannot carry a machine");
        if (!bindings.insert(sel.binding).second) return bad("two slots name the same binding");
        break;
      case Selector::Mode::kMachine:
        if (!is_hex(sel.machine, 64)) return bad("machine selector needs a 64-hex device fingerprint");
        if (!sel.binding.empty() || sel.requirements) return bad("a machine selector carries only the machine");
        if (!machines.insert(sel.machine).second) return bad("two slots name the same machine");
        break;
      case Selector::Mode::kRequirements:
        if (!sel.requirements) return bad("a requirements selector needs requirements");
        if (!sel.binding.empty() || !sel.machine.empty()) return bad("a requirements selector carries only requirements");
        break;
    }
    if (sel.requirements) CLM_RETURN_IF_ERROR(check_requirements(*sel.requirements, "slot '" + s.slot + "' requirements"));
  }
  {
    std::uint32_t workers = 0, required = 0;
    for (const auto& s : slots)
      if (s.kind == Slot::Kind::kWorker) {
        ++workers;
        if (!s.optional) ++required;
      }
    if (workers > kMaxWorkerSlots) return bad("at most 8 worker slots");
    const auto b = worker_count_bounds(p.topology);
    if (!(required <= b.min && b.min <= b.max && b.max <= workers))
      return bad("worker counts need non-optional slots <= min_workers <= max_workers <= worker slots");
  }

  if (p.placement.mode == Placement::Mode::kManual) {
    if (p.placement.manual.empty()) return bad("manual placement needs stages");
    std::uint32_t next = 0;
    for (const auto& st : p.placement.manual) {
      if (names.count(st.slot) == 0) return bad("manual stage names a slot that does not exist");
      if (st.first_layer != next || st.end_layer <= st.first_layer) return bad("manual stages must be non-empty and contiguous from layer 0");
      next = st.end_layer;
    }
  } else if (!p.placement.manual.empty()) {
    return bad("manual stages are only allowed with placement.mode = manual");
  }

  if (p.speculation.max_q < 1 || p.speculation.max_q > 16) return bad("speculation.max_q must be 1..16");
  if (!matches("(model|builtin:[a-z0-9._-]{1,48})", p.chat_template_source)) return bad("chat_template.source is malformed");
  if (p.lifecycle.prepare_when_available && p.lifecycle.preparation == Lifecycle::Preparation::kManual)
    return bad("prepare_when_available cannot be combined with preparation = manual");

  const auto& wl = p.on_worker_loss;
  if (wl.then == OnWorkerLoss::Then::kFallbackProfile && !wl.fallback_profile_id) return bad("then = fallback-profile needs fallback_profile_id");
  if (wl.fallback_profile_id) {
    if (!is_profile_id(*wl.fallback_profile_id)) return bad("fallback_profile_id is malformed");
    if (*wl.fallback_profile_id == p.id) return bad("a profile cannot fall back to itself");
  }

  if (p.exposure.api && !p.exposure.api_model_id) return bad("exposure.api = true needs api_model_id");
  if (p.exposure.api_model_id) {
    const auto& m = *p.exposure.api_model_id;
    if (!is_api_model_id(m)) return bad("api_model_id is malformed");
    if (m.rfind("clusterlm-", 0) == 0) return bad("api_model_id may not start with the reserved prefix 'clusterlm-'");
    if (m == "auto" || m == "default") return bad("'auto' and 'default' are reserved for routing aliases");
  }

  if (p.goals.size() > 8) return bad("too many goals");
  for (const auto& g : p.goals) {
    if (!matches("[a-z0-9_.<>-]{1,48}", g.metric)) return bad("goal metric is malformed");
    if (g.experiment && !is_hq_id(*g.experiment)) return bad("goal experiment id is malformed");
  }
  return Status::ok();
}

Status validate(const RoutingAlias& a) {
  if (!is_alias_id(a.id)) return bad("alias id must match alias_[a-z0-9][a-z0-9_]{2,47}");
  if (!text_ok(a.name, 80)) return bad("alias name is malformed");
  if (!is_api_model_id(a.api_model_id)) return bad("alias api_model_id is malformed");
  if (a.api_model_id.rfind("clusterlm-", 0) == 0) return bad("api_model_id may not start with the reserved prefix 'clusterlm-'");
  if (a.candidates.empty() || a.candidates.size() > 8) return bad("an alias needs 1..8 candidates");
  std::set<std::string> seen;
  for (const auto& c : a.candidates) {
    if (!is_profile_id(c.profile_id)) return bad("candidate profile_id is malformed");
    if (!seen.insert(c.profile_id).second) return bad("candidate profile ids must be unique");
    std::set<std::string> w;
    for (const auto& n : c.workers_available) {
      if (!is_slot_name(n)) return bad("workers_available names are slot names");
      if (!w.insert(n).second) return bad("workers_available has duplicates");
    }
  }
  return Status::ok();
}

Status validate_set(const std::vector<Profile>& profiles, const std::vector<RoutingAlias>& aliases) {
  if (profiles.size() > kMaxProfiles || aliases.size() > kMaxAliases) return bad("too many profiles or aliases");
  std::map<std::string, const Profile*> by_id;
  std::set<std::string> api_ids;
  for (const auto& p : profiles) {
    CLM_RETURN_IF_ERROR(validate(p));
    if (!by_id.emplace(p.id, &p).second) return bad("duplicate profile id");
    if (p.exposure.api_model_id && !api_ids.insert(*p.exposure.api_model_id).second) return bad("api_model_id is used twice");
  }
  std::set<std::string> alias_ids;
  for (const auto& a : aliases) {
    CLM_RETURN_IF_ERROR(validate(a));
    if (!alias_ids.insert(a.id).second) return bad("duplicate alias id");
    if (!api_ids.insert(a.api_model_id).second) return bad("api_model_id is used twice");
    for (const auto& c : a.candidates) {
      auto it = by_id.find(c.profile_id);
      if (it == by_id.end()) return bad("alias candidate names a profile that does not exist");
      for (const auto& w : c.workers_available) {
        const bool ok = std::any_of(it->second->topology.slots.begin(), it->second->topology.slots.end(),
                                    [&](const Slot& s) { return s.kind == Slot::Kind::kWorker && s.slot == w; });
        if (!ok) return bad("workers_available names a slot the candidate profile does not have");
      }
    }
  }
  for (const auto& p : profiles) {
    if (!p.on_worker_loss.fallback_profile_id) continue;
    std::set<std::string> visited{p.id};
    const Profile* cur = &p;
    while (cur->on_worker_loss.fallback_profile_id) {
      auto it = by_id.find(*cur->on_worker_loss.fallback_profile_id);
      if (it == by_id.end()) return bad("fallback_profile_id names a profile that does not exist");
      if (!visited.insert(it->first).second) return bad("fallback profiles form a cycle");
      cur = it->second;
    }
  }
  return Status::ok();
}

std::vector<std::string> fallback_chain(const std::vector<Profile>& profiles, const std::string& id) {
  std::vector<std::string> chain;
  std::set<std::string> visited;
  std::string cur = id;
  while (visited.insert(cur).second) {
    auto it = std::find_if(profiles.begin(), profiles.end(), [&](const Profile& p) { return p.id == cur; });
    if (it == profiles.end()) break;
    chain.push_back(cur);
    if (it->on_worker_loss.then != OnWorkerLoss::Then::kFallbackProfile || !it->on_worker_loss.fallback_profile_id) break;
    cur = *it->on_worker_loss.fallback_profile_id;
  }
  return chain;
}

}  // namespace clusterlm::profiles
