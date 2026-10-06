#include "clusterlm/father/production.hpp"

#include <algorithm>
#include <set>

#include "clusterlm/common/log.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/placement/placement.hpp"
#include "clusterlm/placement/profile.hpp"
#include "clusterlm/planning/planning.hpp"
#include "clusterlm/protocol/wire.hpp"

namespace clusterlm::father {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------- progress

void ProvisioningBoard::update(const std::string& tier_id, const coordinator::PrepareProgress& p) {
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard lk(mu_);
  Entry& e = entries_[tier_id];
  if (!e.started_set && p.bytes_sent > 0) {
    e.started_set = true;
    e.started = now;
    e.bytes_at_start = p.bytes_sent;
  }
  e.progress.bytes_done = p.bytes_sent;
  e.progress.bytes_total = p.bytes_total;
  if (e.started_set && p.bytes_sent > e.bytes_at_start) {
    const double secs = std::chrono::duration<double>(now - e.started).count();
    if (secs >= 0.5) {
      e.progress.rate_bytes_per_s = static_cast<double>(p.bytes_sent - e.bytes_at_start) / secs;
      e.progress.rate_is_measured = true;
    }
  }
}

void ProvisioningBoard::clear(const std::string& tier_id) {
  std::lock_guard lk(mu_);
  entries_.erase(tier_id);
}

std::optional<catalog::ProvisioningProgress> ProvisioningBoard::get(const std::string& tier_id) const {
  std::lock_guard lk(mu_);
  auto it = entries_.find(tier_id);
  if (it == entries_.end()) return std::nullopt;
  return it->second.progress;
}

PrepareObserver ProvisioningBoard::observer() {
  PrepareObserver o;
  o.on_progress = [this](const std::string& tier, const coordinator::PrepareProgress& p) { update(tier, p); };
  o.on_finished = [this](const std::string& tier) { clear(tier); };
  return o;
}

std::function<std::optional<catalog::ProvisioningProgress>(const std::string&)> ProvisioningBoard::provider() {
  return [this](const std::string& tier) { return get(tier); };
}

// ---------------------------------------------------------------------------------------------- details

void DetailsBoard::set(const std::string& tier_id, std::string key, std::string text) {
  std::lock_guard lk(mu_);
  notes_[tier_id][std::move(key)] = std::move(text);
}

std::vector<std::string> DetailsBoard::get(const std::string& tier_id) const {
  std::lock_guard lk(mu_);
  std::vector<std::string> out;
  auto it = notes_.find(tier_id);
  if (it != notes_.end())
    for (const auto& [k, v] : it->second) out.push_back(v);
  return out;
}

namespace {

constexpr const char* kDevLabel =
    "DEVELOPMENT FIXTURE MODEL: a small test model on the reference backend stands in for this tier's real model";

Status precondition(std::string why) { return make_error(ErrorCode::kFailedPrecondition, std::move(why)); }

// Profile for one machine: a bench-produced file for this exact machine wins, else the Synthetic fixture for the
// role. `source` says which, for the details board.
struct LoadedProfile {
  placement::HardwareProfile profile;
  std::string source;
};

Result<placement::HardwareProfile> load_if_exists(const fs::path& p) {
  std::error_code ec;
  if (p.empty() || !fs::exists(p, ec)) return make_error(ErrorCode::kNotFound, "no profile");
  return placement::load_hardware_profile(p.string());
}

Result<LoadedProfile> profile_for(const ProductionOptions& opt, const config::FatherSettings& s,
                                  const config::PairedDevice* node, std::string_view role) {
  const fs::path bench = s.advanced.bench_results_dir;
  if (!bench.empty() && node != nullptr) {
    for (const auto& stem : {node->fingerprint, node->name}) {
      auto p = load_if_exists(bench / (stem + ".hardware.json"));
      if (p.is_ok()) return LoadedProfile{std::move(p).value(), "bench results (" + stem + ")"};
    }
  }
  if (!bench.empty() && node == nullptr) {
    auto p = load_if_exists(bench / "father.hardware.json");
    if (p.is_ok()) return LoadedProfile{std::move(p).value(), "bench results (father)"};
  }
  const char* fixture = "Father-4060Ti-7600.json";
  if (role == catalog::kRoleLaptop) fixture = "Node-G14-4070-8945HS.json";
  else if (role == catalog::kRole3060) fixture = "Node-3060-5600.json";
  else if (role != catalog::kRoleFather) return precondition("no hardware profile for role " + std::string(role));
  auto p = load_if_exists(opt.profiles_dir / fixture);
  if (!p.is_ok()) return precondition("no hardware profile for " + std::string(role) + " (no bench result, no fixture)");
  return LoadedProfile{std::move(p).value(), std::string("development fixture ") + fixture};
}

Result<placement::NetworkProfile> network_profile(const ProductionOptions& opt, const config::FatherSettings& s,
                                                  std::string& source) {
  const fs::path bench = s.advanced.bench_results_dir;
  std::error_code ec;
  if (!bench.empty() && fs::exists(bench / "network.json", ec)) {
    auto n = placement::load_network_profile((bench / "network.json").string());
    if (n.is_ok()) {
      source = "bench results";
      return n;
    }
  }
  const fs::path fixture = opt.profiles_dir / "network-gige-simulated.json";
  if (opt.profiles_dir.empty() || !fs::exists(fixture, ec)) return precondition("no network profile available");
  source = "development fixture network-gige-simulated.json";
  return placement::load_network_profile(fixture.string());
}

std::string provenance_note(placement::Provenance p, const std::string& sources) {
  switch (p) {
    case placement::Provenance::kSynthetic:
      return "Placement used SYNTHETIC profiles (" + sources + "): a development estimate, not a measurement of your machines";
    case placement::Provenance::kMeasured:
      return "Placement used MEASURED profiles (" + sources + "); not yet qualified";
    case placement::Provenance::kQualified: return "Placement used qualified profiles (" + sources + ")";
  }
  return {};
}

Result<Deployment> plan_tier(const ProductionOptions& opt, const catalog::TierEntry& tier, std::uint32_t context_tokens) {
  const config::FatherSettings s = opt.settings->get();
  auto dir_it = s.model_dirs.find(tier.id);
  if (dir_it == s.model_dirs.end()) return precondition("no model directory is configured for " + tier.display_name);
  CLM_ASSIGN_OR_RETURN(auto store, objects::CanonicalModelStore::open(dir_it->second));
  const objects::ModelManifest& manifest = store->manifest();

  Deployment d;
  d.config.model_dir = dir_it->second;
  d.config.security.mode = transport::SecurityConfig::Mode::kMutualTls;
  d.config.security.identity = opt.identity;
  d.config.direct_peer = s.advanced.direct_peer;
  d.config.request_timeout = std::chrono::milliseconds(10'000);
  d.config.window_timeout = std::chrono::milliseconds(30'000);

  std::vector<const config::PairedDevice*> nodes;
  std::vector<std::string> node_roles;
  for (const auto& role : tier.roles) {
    if (role == catalog::kRoleFather) continue;
    auto a = s.assignments.find(role);
    if (a == s.assignments.end()) return precondition("no machine is assigned to " + role);
    const auto* node = s.find_node(a->second);
    if (node == nullptr) return precondition("the machine assigned to " + role + " is not paired");
    CLM_ASSIGN_OR_RETURN(auto ep, transport::Endpoint::parse(node->address));
    d.config.nodes.push_back({node->name, ep, node->fingerprint});
    d.config.security.trusted_peers.push_back(node->fingerprint);  // pinned: only the paired identities
    nodes.push_back(node);
    node_roles.push_back(role);
  }

  const std::uint32_t n_layers = manifest.geometry.n_layers;
  const std::uint32_t q = std::max(1u, s.advanced.default_q);
  if (nodes.empty()) {
    CLM_ASSIGN_OR_RETURN(d.plan, coordinator::ClusterPlan::parse(
                                     "0-" + std::to_string(n_layers) + "@father," + std::to_string(n_layers) + "-" +
                                         std::to_string(n_layers) + "@father",
                                     n_layers));
    d.plan.max_context = context_tokens;
    d.plan.max_window = std::max(q, s.advanced.prefill_chunk);
    if (opt.details) opt.details->set(tier.id, "placement", "Father-only plan: no placement search needed");
    return d;
  }

  // ---- placement search over the best available profiles ----
  placement::PlacementRequest req;
  std::vector<std::string> sources;
  CLM_ASSIGN_OR_RETURN(auto father, profile_for(opt, s, nullptr, catalog::kRoleFather));
  sources.push_back("Father: " + father.source);
  req.father = std::move(father.profile);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    CLM_ASSIGN_OR_RETURN(auto lp, profile_for(opt, s, nodes[i], node_roles[i]));
    sources.push_back(nodes[i]->name + ": " + lp.source);
    req.nodes.push_back(std::move(lp.profile));
  }
  std::string net_source;
  CLM_ASSIGN_OR_RETURN(req.network, network_profile(opt, s, net_source));
  sources.push_back("network: " + net_source);
  if (opt.dev_fixture_model) {
    // The fixture model uses its own tensor representations; give the CPU a Synthetic throughput for them.
    auto scaffold = [](placement::HardwareProfile& p) {
      if (p.cpu.expert_bytes_per_s.empty()) return;
      const auto any = p.cpu.expert_bytes_per_s.begin()->second;
      for (const char* quant : {"f32", "q8_0-fixture"})
        p.cpu.expert_bytes_per_s[quant] = placement::Quantity::synthetic(any.value, "synthetic: dev fixture scaffolding");
    };
    scaffold(req.father);
    for (auto& n : req.nodes) scaffold(n);
  }
  CLM_ASSIGN_OR_RETURN(req.model, planning::cost_inputs_from_manifest(manifest));
  req.context_tokens = context_tokens;
  req.q = q;
  req.prefix_layers_options = {4};
  CLM_ASSIGN_OR_RETURN(auto result, placement::search_placements(req));

  std::set<std::string> want;
  std::map<std::string, int> index;
  for (std::size_t i = 0; i < req.nodes.size(); ++i) {
    want.insert(req.nodes[i].id);
    index[req.nodes[i].id] = static_cast<int>(i);
  }
  const placement::PlacementPlan* chosen = nullptr;
  for (const auto& c : result.candidates) {
    const auto ids = c.node_ids();
    if (std::set<std::string>(ids.begin(), ids.end()) == want) {
      chosen = &c;
      break;
    }
  }
  if (chosen == nullptr) return precondition("no feasible placement uses every assigned machine for this context");
  CLM_ASSIGN_OR_RETURN(d.plan, planning::to_cluster_plan(*chosen, manifest, req.father.id, index, context_tokens,
                                                         std::max(q, s.advanced.prefill_chunk)));
  std::string joined;
  for (const auto& src : sources) joined += (joined.empty() ? "" : "; ") + src;
  if (opt.details) opt.details->set(tier.id, "placement", provenance_note(chosen->provenance, joined));
  return d;
}

}  // namespace

Result<Deployment> ConfigDeploymentProvider::resolve(const catalog::TierEntry& tier, std::uint32_t context_tokens) {
  return plan_tier(opt_, tier, context_tokens);
}

// ---------------------------------------------------------------------------------------------- readiness

LiveReadinessSource::~LiveReadinessSource() {
  stopping_.store(true);
  for (auto& [k, t] : verifiers_)
    if (t.joinable()) t.join();
}

LiveReadinessSource::Verify LiveReadinessSource::verification(const fs::path& dir, const std::string& root_hex) {
  const std::string key = dir.string() + "|" + root_hex;
  std::lock_guard lk(mu_);
  auto it = verified_.find(key);
  if (it != verified_.end()) return it->second;
  verified_[key] = Verify{false, false, root_hex};
  verifiers_[key] = std::thread([this, dir, key] {
    bool ok = false;
    auto store = objects::CanonicalModelStore::open(dir);
    if (store.is_ok()) {
      ok = true;
      for (const auto& obj : store.value()->manifest().objects) {
        if (stopping_.load()) return;
        auto st = store.value()->stream_object(obj.name, 1u << 20, [](std::uint64_t, ByteSpan) { return Status::ok(); });
        if (!st.is_ok()) {
          ok = false;
          break;
        }
      }
    }
    std::lock_guard l2(mu_);
    auto& v = verified_[key];
    v.done = true;
    v.ok = ok;
  });
  return verified_[key];
}

LiveReadinessSource::Probe LiveReadinessSource::probe_node(const config::PairedDevice& node) {
  const auto now = std::chrono::steady_clock::now();
  {
    std::lock_guard lk(mu_);
    auto it = probes_.find(node.fingerprint);
    const SessionPhase phase = opt_.session_phase();
    if (phase != SessionPhase::kNone) {
      Probe p = it != probes_.end() ? it->second : Probe{};
      p.valid = true;
      if (p.state == catalog::MachineState::kOffline) return p;
      p.state = phase == SessionPhase::kPreparing   ? catalog::MachineState::kPreparing
                : phase == SessionPhase::kReadyIdle ? catalog::MachineState::kReady
                                                    : catalog::MachineState::kInferencing;
      return p;
    }
    if (it != probes_.end() && it->second.valid && now - it->second.at < opt_.probe_ttl) return it->second;
  }
  Probe p;
  p.valid = true;
  p.at = now;
  auto ep = transport::Endpoint::parse(node.address);
  if (!ep.is_ok()) {
    p.state = catalog::MachineState::kOffline;
  } else {
    transport::SecurityConfig sec;
    sec.mode = transport::SecurityConfig::Mode::kMutualTls;
    sec.identity = opt_.identity;
    sec.trusted_peers = {node.fingerprint};
    auto conn = transport::connect(ep.value(), sec, node.fingerprint, opt_.probe_timeout);
    if (!conn.is_ok()) {
      p.state = catalog::MachineState::kOffline;
    } else {
      protocol::MessageStream stream(std::move(conn).value(), protocol::Channel::kControl);
      protocol::Hello hello;
      hello.role = protocol::NodeRole::kFather;
      hello.channel = protocol::Channel::kControl;
      hello.device_id = opt_.identity->fingerprint();
      if (!stream.send(hello).is_ok() || !stream.expect<protocol::HelloAck>(opt_.probe_timeout).is_ok()) {
        p.state = catalog::MachineState::kOffline;
      } else {
        // An Available Node offers resources right after the handshake; a Busy one stays silent.
        auto offer = stream.expect<protocol::OfferResources>(opt_.probe_timeout);
        if (offer.is_ok()) {
          p.state = catalog::MachineState::kAvailable;
          p.on_ac = offer->power != protocol::PowerSource::kBattery;
        } else {
          p.state = catalog::MachineState::kBusy;
        }
      }
      stream.close();
    }
  }
  std::lock_guard lk(mu_);
  probes_[node.fingerprint] = p;
  return p;
}

catalog::ReadinessInputs LiveReadinessSource::observe(const catalog::TierEntry& tier,
                                                      const catalog::TierAssignment& assignment,
                                                      std::uint32_t context_tokens) {
  catalog::ReadinessInputs in;
  in.context_tokens = context_tokens;
  const config::FatherSettings s = opt_.settings->get();

  // ---- model ----
  auto dir_it = s.model_dirs.find(tier.id);
  if (dir_it != s.model_dirs.end()) {
    auto store = objects::CanonicalModelStore::open(dir_it->second);
    if (store.is_ok()) {
      in.model.manifest_present = true;
      in.model.manifest_root_hex = store.value()->manifest().root_hash().hex();
      in.model.hashes_verified = verification(dir_it->second, in.model.manifest_root_hex).ok;
      auto c = s.confirmed_model_roots.find(tier.id);
      if (opt_.dev_fixture_model) in.model.user_confirmed_root_hex = in.model.manifest_root_hex;
      else if (c != s.confirmed_model_roots.end()) in.model.user_confirmed_root_hex = c->second;
    }
  }
  if (opt_.details) {
    if (dir_it == s.model_dirs.end()) opt_.details->set(tier.id, "model", "No model directory is configured for this tier");
    if (opt_.dev_fixture_model) opt_.details->set(tier.id, "dev", kDevLabel);
  }

  // ---- backend: no Strata/llama backend is built into this binary ----
  if (opt_.dev_fixture_model) {
    in.backend = {"reference backend, dev fixture", true};
  } else {
    in.backend = {"backend not available in this build", false};
    if (opt_.details) opt_.details->set(tier.id, "backend", "Backend not available in this build: no Strata/llama inference backend is compiled in");
  }

  // ---- machines ----
  for (const auto& role : tier.roles) {
    catalog::MachineInputs m;
    m.role = role;
    if (role == catalog::kRoleFather) {
      m.machine_id = "Father";
      m.state = catalog::MachineState::kAvailable;
      m.power = opt_.father_power();
    } else {
      const auto fp = assignment.machine_for(role);
      const auto* node = fp ? s.find_node(*fp) : nullptr;
      if (node == nullptr) {
        m.machine_id = role;
        m.paired = false;
        m.state = catalog::MachineState::kOffline;
      } else {
        m.machine_id = node->name;
        const Probe p = probe_node(*node);
        m.state = p.state;
        m.power.on_ac = p.on_ac;
      }
    }
    in.machines.push_back(std::move(m));
  }

  // ---- placement feasibility (cached per configuration) ----
  if (in.model.manifest_present) {
    const std::string key = tier.id + "|" + std::to_string(context_tokens) + "|" + to_json(s);
    bool feasible = false;
    bool have = false;
    {
      std::lock_guard lk(mu_);
      auto it = feasible_.find(key);
      if (it != feasible_.end()) {
        feasible = it->second;
        have = true;
      }
    }
    if (!have) {
      auto d = plan_tier(opt_, tier, context_tokens);
      feasible = d.is_ok();
      if (!feasible && opt_.details) opt_.details->set(tier.id, "placement", "Placement: " + d.status().message());
      std::lock_guard lk(mu_);
      if (feasible_.size() > 64) feasible_.clear();
      feasible_[key] = feasible;
    }
    in.plan.feasible_for_context = feasible;
  }
  if (opt_.provisioning) in.provisioning = opt_.provisioning(tier.id);
  return in;
}

}  // namespace clusterlm::father
