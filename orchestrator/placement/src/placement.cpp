#include "clusterlm/placement/placement.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace clusterlm::placement {

using nlohmann::json;

std::string_view to_string(StageRole r) noexcept {
  switch (r) {
    case StageRole::kPrefix: return "prefix";
    case StageRole::kMiddle: return "middle";
    case StageRole::kTail: return "tail";
    case StageRole::kFull: return "full";
  }
  return "?";
}

std::string plan_key(const std::vector<PlanStage>& stages) {
  std::string k;
  for (const auto& s : stages) {
    if (!k.empty()) k += '>';
    k += s.domain_id + "[" + std::to_string(s.layers.begin) + "," + std::to_string(s.layers.end) + ")";
  }
  return k;
}

std::vector<std::string> PlacementPlan::node_ids() const {
  std::vector<std::string> ids;
  for (const auto& s : stages)
    if (s.role == StageRole::kMiddle) ids.push_back(s.domain_id);
  return ids;
}

const PlacementPlan* PlacementResult::find(const std::string& key) const {
  for (const auto& p : candidates)
    if (p.key == key) return &p;
  return nullptr;
}
const RejectedPlan* PlacementResult::find_rejected(const std::string& key) const {
  for (const auto& r : rejected)
    if (r.key == key) return &r;
  return nullptr;
}

Status require_qualified(const PlacementPlan& plan) {
  if (plan.provenance == Provenance::kQualified) return Status::ok();
  std::string msg = "plan '" + plan.key + "' is " + std::string(to_string(plan.provenance)) +
                    ", not qualified; non-qualified inputs:";
  for (const auto& s : plan.non_qualified_inputs) msg += " [" + s + "]";
  return make_error(ErrorCode::kFailedPrecondition, std::move(msg));
}

namespace {

constexpr double kMs = 1000.0;

// P(expert selected at least once among n independent positions) with per-position probability p.
double union_prob(double p, double n) { return 1.0 - std::pow(1.0 - p, n); }

std::string fmt_bytes(double b) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.2f GiB", b / (1024.0 * 1024.0 * 1024.0));
  return buf;
}

std::string ranges_key(const std::vector<LayerRange>& rs) {
  std::string k;
  for (const auto& r : rs) k += "[" + std::to_string(r.begin) + "," + std::to_string(r.end) + ")";
  return k;
}

// Everything computed about one domain hosting a set of layer ranges: admission, residency, per-layer times.
struct DomainEval {
  bool ok = false;
  std::vector<std::string> reasons;
  MemoryLedger ledger;
  ExpertResidency residency;
  bool has_gpu = false;
  // Per absolute layer (zero outside the domain's ranges).
  std::vector<double> dense_ms, cpu_ms, gpu_ms, cpu_miss_bytes, gpu_hit_bytes, prefill_cpu_chunk_ms;
  double prefill_rate = 0;  // tokens/s if this domain ran all layers
  double upload_s = 0;      // host->GPU upload of dense + GPU experts
};

class Evaluator {
 public:
  explicit Evaluator(const PlacementRequest& r) : req_(r), n_(r.model.n_layers()) {
    q_ = static_cast<double>(r.q);
    chunk_tokens_ = std::min(r.prefill_chunk, r.context_tokens);
    const auto& m = r.model;
    order_.resize(n_);
    pq_.resize(n_);
    union_total_.assign(n_, 0.0);
    chunk_total_.assign(n_, 0.0);
    pc_.resize(n_);
    for (std::uint32_t l = 0; l < n_; ++l) {
      const auto& row = m.routing_freq[l];
      order_[l].resize(m.n_experts);
      std::iota(order_[l].begin(), order_[l].end(), std::uint16_t{0});
      std::stable_sort(order_[l].begin(), order_[l].end(), [&](std::uint16_t a, std::uint16_t b) { return row[a] > row[b]; });
      pq_[l].resize(m.n_experts);
      pc_[l].resize(m.n_experts);
      const double bytes = static_cast<double>(m.layers[l].expert_bytes);
      for (std::uint32_t e = 0; e < m.n_experts; ++e) {
        pq_[l][e] = union_prob(row[e], q_);
        pc_[l][e] = union_prob(row[e], static_cast<double>(std::max(1u, chunk_tokens_)));
        union_total_[l] += pq_[l][e] * bytes;
        chunk_total_[l] += pc_[l][e] * bytes;
      }
    }
    domains_[r.father.id] = &r.father;
    for (const auto& n : r.nodes) domains_[n.id] = &n;
    for (const auto& [id, p] : domains_) profile_prov_[id] = weakest_provenance(*p);
    network_prov_ = weakest_provenance(r.network);
  }

  PlanEvaluation evaluate(const std::vector<PlanStage>& stages);

 private:
  const DomainEval& domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges);
  DomainEval compute_domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges) const;
  std::uint64_t layer_object_bytes(std::uint32_t l) const {
    return req_.model.layers[l].dense_bytes + std::uint64_t{req_.model.n_experts} * req_.model.layers[l].expert_bytes;
  }

  const PlacementRequest& req_;
  std::uint32_t n_;
  double q_ = 1;
  std::uint32_t chunk_tokens_ = 0;
  std::vector<std::vector<std::uint16_t>> order_;  // experts of each layer by routing frequency, descending
  std::vector<std::vector<double>> pq_, pc_;       // union probability over q positions / one prefill chunk
  std::vector<double> union_total_, chunk_total_;  // sum_e bytes * prob, per layer
  std::map<std::string, const HardwareProfile*> domains_;
  std::map<std::string, Provenance> profile_prov_;
  Provenance network_prov_ = Provenance::kSynthetic;
  std::map<std::string, DomainEval> cache_;
};

DomainEval Evaluator::compute_domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges) const {
  const auto& m = req_.model;
  DomainEval d;
  d.dense_ms.assign(n_, 0.0);
  d.cpu_ms = d.gpu_ms = d.cpu_miss_bytes = d.gpu_hit_bytes = d.prefill_cpu_chunk_ms = d.dense_ms;
  d.ledger.domain_id = p.id;
  d.residency.domain_id = p.id;
  d.residency.gpu_experts_by_layer.assign(n_, {});
  auto reject = [&](std::string why) { d.reasons.push_back(std::move(why)); };

  std::vector<std::uint32_t> layers;
  for (const auto& r : ranges)
    for (std::uint32_t l = r.begin; l < r.end; ++l) layers.push_back(l);

  const double gpu_bw = p.gpu.gpu_expert_bytes_per_s.value;
  d.has_gpu = p.gpu.vram_budget.value > 0 && gpu_bw > 0;
  d.prefill_rate = p.gpu.prefill_tokens_per_s.value;
  const double sustained = p.cpu.sustained_factor.value;
  const double cpu_scale = 1.0 + p.cpu.q_scaling.value * (q_ - 1.0);
  const double dense_scale = 1.0 + p.gpu.dense_q_scaling.value * (q_ - 1.0);

  // Per-layer effective CPU throughput (derated) and dense time; missing measurements are rejections, never
  // silently defaulted.
  std::vector<double> cpu_eff(n_, 0.0);
  std::set<std::string> missing;
  for (std::uint32_t l : layers) {
    const auto& lc = m.layers[l];
    auto it = p.cpu.expert_bytes_per_s.find(lc.quant);
    if (it == p.cpu.expert_bytes_per_s.end() || it->second.value <= 0)
      missing.insert("no CPU expert throughput for quant '" + lc.quant + "'");
    else
      cpu_eff[l] = it->second.value * sustained;
    auto dk = p.gpu.dense_layer_ms.find(std::string(to_string(lc.kind)));
    if (dk == p.gpu.dense_layer_ms.end())
      missing.insert("no dense_layer_ms for kind '" + std::string(to_string(lc.kind)) + "'");
    else
      d.dense_ms[l] = dk->second.value * dense_scale;
  }
  for (const auto& s : missing) reject(s);
  if (req_.context_tokens > 0 && d.prefill_rate <= 0) reject("prefill_tokens_per_s is zero");
  if (!d.reasons.empty()) return d;

  // ---- memory ledger ----
  auto& led = d.ledger;
  led.dense_on_gpu = d.has_gpu;
  led.vram_budget = static_cast<std::uint64_t>(p.gpu.vram_budget.value);
  led.ram_safe_allowance = static_cast<std::uint64_t>(p.memory.ram_safe_allowance.value);
  for (std::uint32_t l : layers) {
    const auto& lc = m.layers[l];
    led.dense += lc.dense_bytes;
    led.state += lc.fixed_state_bytes + m.state_bytes_per_context_token[static_cast<std::size_t>(lc.kind)] * req_.context_tokens;
  }
  led.scratch_vram = d.has_gpu ? static_cast<std::uint64_t>(p.overheads.scratch_vram.value) : 0;
  led.staging_ram = static_cast<std::uint64_t>(p.overheads.staging_ram.value);
  led.os_reserve_ram = static_cast<std::uint64_t>(p.overheads.os_reserve_ram.value);
  led.father_only = p.role == DomainRole::kFather ? m.father_only.total() : 0;

  // ---- GPU residency ----
  // Rank experts by critical-path time saved per byte of VRAM spent: P(selected in the q-position union) *
  // (1/cpu_bw - 1/gpu_bw). Within a layer all experts have equal size, so this is frequency order; across
  // layers (different quant / size) the per-byte coefficient differs. Greedy fill; an expert that does not fit
  // ends its layer's stream (same size in a layer) while smaller layers may still be filled.
  std::vector<std::vector<std::uint16_t>> chosen(n_);
  std::vector<double> gpu_union(n_, 0.0), gpu_chunk(n_, 0.0);
  std::uint64_t remaining = 0;
  if (d.has_gpu) {
    const std::uint64_t fixed = led.dense + led.state + led.scratch_vram;
    if (fixed > led.vram_budget)
      reject("VRAM: dense " + fmt_bytes(static_cast<double>(led.dense)) + " + state " +
             fmt_bytes(static_cast<double>(led.state)) + " + scratch " + fmt_bytes(static_cast<double>(led.scratch_vram)) +
             " exceed budget " + fmt_bytes(static_cast<double>(led.vram_budget)));
    else
      remaining = led.vram_budget - fixed;

    struct Item {
      double score;
      std::uint32_t layer, rank;
    };
    auto worse = [](const Item& a, const Item& b) {
      if (a.score != b.score) return a.score < b.score;
      if (a.layer != b.layer) return a.layer > b.layer;
      return a.rank > b.rank;
    };
    std::priority_queue<Item, std::vector<Item>, decltype(worse)> heap(worse);
    std::vector<double> coef(n_, 0.0);
    for (std::uint32_t l : layers) {
      coef[l] = 1.0 / cpu_eff[l] - 1.0 / gpu_bw;  // seconds saved per byte served from the GPU
      if (coef[l] > 0) heap.push({pq_[l][order_[l][0]] * coef[l], l, 0});
    }
    while (!heap.empty() && remaining > 0) {
      const Item it = heap.top();
      heap.pop();
      const std::uint64_t bytes = m.layers[it.layer].expert_bytes;
      if (bytes > remaining) continue;  // this layer's stream ends: all its experts are the same size
      remaining -= bytes;
      const std::uint16_t e = order_[it.layer][it.rank];
      chosen[it.layer].push_back(e);
      gpu_union[it.layer] += pq_[it.layer][e] * static_cast<double>(bytes);
      gpu_chunk[it.layer] += pc_[it.layer][e] * static_cast<double>(bytes);
      if (it.rank + 1 < m.n_experts)
        heap.push({pq_[it.layer][order_[it.layer][it.rank + 1]] * coef[it.layer], it.layer, it.rank + 1});
    }
  }

  // ---- per-layer timings and ledger totals ----
  for (std::uint32_t l : layers) {
    const double bytes = static_cast<double>(m.layers[l].expert_bytes);
    const std::uint64_t ng = chosen[l].size();
    std::sort(chosen[l].begin(), chosen[l].end());
    led.gpu_experts += ng * m.layers[l].expert_bytes;
    led.cpu_experts += (m.n_experts - ng) * m.layers[l].expert_bytes;
    d.residency.gpu_expert_count += ng;
    d.residency.cpu_expert_count += m.n_experts - ng;
    d.cpu_miss_bytes[l] = std::max(0.0, union_total_[l] - gpu_union[l]);
    d.gpu_hit_bytes[l] = gpu_union[l];
    d.cpu_ms[l] = d.cpu_miss_bytes[l] / cpu_eff[l] * kMs * cpu_scale;
    d.gpu_ms[l] = gpu_bw > 0 ? gpu_union[l] / gpu_bw * kMs : 0.0;
    d.prefill_cpu_chunk_ms[l] = std::max(0.0, chunk_total_[l] - gpu_chunk[l]) / cpu_eff[l] * kMs;
    (void)bytes;
    d.residency.gpu_experts_by_layer[l] = std::move(chosen[l]);
  }

  // ---- admission ----
  if (led.ram_used() > led.ram_safe_allowance)
    reject("RAM: need " + fmt_bytes(static_cast<double>(led.ram_used())) + " (cpu experts " +
           fmt_bytes(static_cast<double>(led.cpu_experts)) + " + staging + os reserve" +
           (led.father_only ? " + father-only" : "") + ") > safe allowance " +
           fmt_bytes(static_cast<double>(led.ram_safe_allowance)));
  if (p.memory.pinned_limit.value > 0 && static_cast<double>(led.staging_ram) > p.memory.pinned_limit.value)
    reject("RAM: staging exceeds pinned limit");
  if (led.vram_used() > led.vram_budget && d.has_gpu) reject("VRAM: over budget");
  const double upload_bytes = static_cast<double>((d.has_gpu ? led.dense : 0) + led.gpu_experts);
  if (upload_bytes > 0) {
    if (p.gpu.pcie_h2d_bytes_per_s.value <= 0)
      reject("pcie_h2d_bytes_per_s is zero but GPU upload is required");
    else
      d.upload_s = upload_bytes / p.gpu.pcie_h2d_bytes_per_s.value;
  }
  d.ok = d.reasons.empty();
  return d;
}

const DomainEval& Evaluator::domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges) {
  const std::string key = p.id + "|" + ranges_key(ranges);
  auto it = cache_.find(key);
  if (it == cache_.end()) it = cache_.emplace(key, compute_domain(p, ranges)).first;
  return it->second;
}

PlanEvaluation Evaluator::evaluate(const std::vector<PlanStage>& stages) {
  PlanEvaluation out;
  out.key = plan_key(stages);
  std::vector<std::string> reasons;
  const auto& m = req_.model;
  const auto& father = req_.father;

  // ---- structural checks ----
  if (stages.empty()) {
    out.reasons = {"no stages"};
    return out;
  }
  std::uint32_t expect = 0;
  for (const auto& s : stages) {
    if (s.layers.begin != expect || s.layers.end <= s.layers.begin) {
      out.reasons = {"stages are not a contiguous, non-empty cover of the layers"};
      return out;
    }
    expect = s.layers.end;
    if (!domains_.count(s.domain_id)) {
      out.reasons = {"unknown domain '" + s.domain_id + "'"};
      return out;
    }
  }
  if (expect != n_) {
    out.reasons = {"stages do not cover all layers"};
    return out;
  }
  std::vector<std::string> order;  // domains in first-appearance (pipeline) order
  std::map<std::string, std::vector<LayerRange>> ranges;
  for (const auto& s : stages) {
    if (!ranges.count(s.domain_id)) order.push_back(s.domain_id);
    ranges[s.domain_id].push_back(s.layers);
  }
  for (const auto& id : order)
    if (id != father.id && ranges[id].size() != 1) {
      out.reasons = {"node '" + id + "' appears in more than one stage"};
      return out;
    }
  if (m.ple_layer >= 0) {
    for (const auto& s : stages)
      if (s.layers.begin <= static_cast<std::uint32_t>(m.ple_layer) &&
          static_cast<std::uint32_t>(m.ple_layer) < s.layers.end && s.domain_id != father.id)
        reasons.push_back("per-layer-embedding layer " + std::to_string(m.ple_layer) + " must stay on Father");
  }

  // ---- per-domain admission ----
  std::vector<const DomainEval*> evals;
  for (const auto& id : order) {
    const DomainEval& d = domain(*domains_[id], ranges[id]);
    evals.push_back(&d);
    for (const auto& r : d.reasons) reasons.push_back(id + ": " + r);
  }

  // ---- links ----
  struct Hop {
    std::string from, to;
    const LinkProfile* link;
  };
  std::vector<Hop> hops;
  auto add_hop = [&](const std::string& a, const std::string& b) {
    const LinkProfile* l = req_.network.find_link(a, b);
    if (!l) reasons.push_back("no network link " + a + " -> " + b);
    hops.push_back({a, b, l});
  };
  for (std::size_t i = 0; i + 1 < stages.size(); ++i)
    if (stages[i].domain_id != stages[i + 1].domain_id) add_hop(stages[i].domain_id, stages[i + 1].domain_id);
  if (stages.back().domain_id != father.id) add_hop(stages.back().domain_id, father.id);
  std::map<std::string, const LinkProfile*> father_links;
  for (const auto& id : order) {
    if (id == father.id) continue;
    const LinkProfile* l = req_.network.find_link(father.id, id);
    if (!l) reasons.push_back("no network link " + father.id + " <-> " + id + " (provisioning/commit)");
    father_links[id] = l;
  }
  if (!reasons.empty()) {
    out.reasons = std::move(reasons);
    return out;
  }

  // ---- cost model ----
  PlacementPlan plan;
  plan.key = out.key;
  plan.stages = stages;
  PredictedMetrics& pm = plan.metrics;
  pm.draft_ms = m.draft_ms.value;

  // Stage verify time: dense + CPU-miss + GPU-hit, serial unless overlap_factor > 0.
  for (const auto& s : stages) {
    const DomainEval& d = *evals[static_cast<std::size_t>(std::find(order.begin(), order.end(), s.domain_id) - order.begin())];
    StageMetrics sm;
    sm.domain_id = s.domain_id;
    sm.role = s.role;
    sm.layers = s.layers;
    double chunk_cpu = 0;
    for (std::uint32_t l = s.layers.begin; l < s.layers.end; ++l) {
      sm.dense_ms += d.dense_ms[l];
      sm.cpu_ms += d.cpu_ms[l];
      sm.gpu_ms += d.gpu_ms[l];
      sm.cpu_miss_bytes_expected += d.cpu_miss_bytes[l];
      sm.gpu_hit_bytes_expected += d.gpu_hit_bytes[l];
      chunk_cpu += d.prefill_cpu_chunk_ms[l];
    }
    sm.stage_ms = sm.dense_ms + sm.cpu_ms + sm.gpu_ms - req_.overlap_factor * std::min(sm.cpu_ms, sm.dense_ms + sm.gpu_ms);
    // Prefill chunk: GPU-pipeline rate scaled by the layer fraction, plus streaming the CPU-resident experts
    // touched by the chunk (a chunk touches most experts, so this is a weight-streaming cost).
    if (chunk_tokens_ > 0)
      sm.prefill_chunk_ms = static_cast<double>(chunk_tokens_) /
                                (d.prefill_rate * static_cast<double>(n_) / static_cast<double>(s.layers.size())) * kMs +
                            chunk_cpu;
    pm.stage_ms_total += sm.stage_ms;
    pm.stages.push_back(std::move(sm));
  }

  // Transport per boundary: q positions of boundary ABI + half an RTT (one-way latency).
  const double boundary = static_cast<double>(m.boundary_bytes_per_position);
  for (const auto& h : hops) {
    TransportMetrics t;
    t.from = h.from;
    t.to = h.to;
    const double bw = h.link->bandwidth_bytes_per_s.value;
    t.bytes_per_round = q_ * boundary;
    t.ms = t.bytes_per_round / bw * kMs + h.link->rtt_ms.value / 2.0;
    t.prefill_chunk_ms = static_cast<double>(chunk_tokens_) * boundary / bw * kMs + h.link->rtt_ms.value / 2.0;
    pm.transport_ms_total += t.ms;
    pm.transport_bytes_per_round += t.bytes_per_round;
    pm.transports.push_back(std::move(t));
  }
  for (const auto& [id, l] : father_links) pm.commit_ms = std::max(pm.commit_ms, l->rtt_ms.value);

  pm.round_ms = pm.draft_ms + pm.stage_ms_total + pm.transport_ms_total + pm.commit_ms;
  pm.decode_tok_s = pm.round_ms > 0 ? kMs * req_.acceptance.value / pm.round_ms : 0.0;

  // Prefill: chunks flow through the stages and links like a pipeline. First chunk pays the whole path, each
  // further chunk is bounded by the slowest element.
  if (chunk_tokens_ > 0) {
    double first = 0, slowest = 0;
    for (const auto& s : pm.stages) {
      first += s.prefill_chunk_ms;
      slowest = std::max(slowest, s.prefill_chunk_ms);
    }
    for (const auto& t : pm.transports) {
      first += t.prefill_chunk_ms;
      slowest = std::max(slowest, t.prefill_chunk_ms);
    }
    const double n_chunks = std::ceil(static_cast<double>(req_.context_tokens) / static_cast<double>(chunk_tokens_));
    pm.prefill_s = (first + (n_chunks - 1.0) * slowest) / kMs;
  }

  // Preparation. Provisioning shares Father's NIC, so it is NOT the sum of per-node times: it is the larger of
  // (all bytes / egress) and (the slowest single node's bytes / its own link).
  double total_new_bytes = 0;
  std::map<std::string, double> node_bytes, node_link_s, node_upload_s;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const std::string& id = order[i];
    pm.upload_s = std::max(pm.upload_s, evals[i]->upload_s);
    if (id == father.id) continue;
    const LayerRange r = ranges[id].front();
    double bytes = 0;
    const auto prov = req_.already_provisioned.find(id);
    for (std::uint32_t l = r.begin; l < r.end; ++l) {
      bool have = false;
      if (prov != req_.already_provisioned.end())
        for (const auto& pr : prov->second)
          if (l >= pr.begin && l < pr.end) have = true;
      if (!have) bytes += static_cast<double>(layer_object_bytes(l));
    }
    node_bytes[id] = bytes;
    total_new_bytes += bytes;
    node_link_s[id] = bytes / father_links[id]->bandwidth_bytes_per_s.value;
    node_upload_s[id] = evals[i]->upload_s;
  }
  pm.provisioning_bytes = total_new_bytes;
  double slowest_link_s = 0;
  for (const auto& [id, s] : node_link_s) slowest_link_s = std::max(slowest_link_s, s);
  pm.provisioning_s = node_bytes.empty() ? 0.0 : std::max(total_new_bytes / req_.network.father_egress_bytes_per_s.value, slowest_link_s);
  pm.prepare_s = pm.provisioning_s + pm.upload_s;
  pm.t_request_s = pm.prepare_s + pm.prefill_s + static_cast<double>(req_.output_tokens) / pm.decode_tok_s;

  // Lease amortisation: one request bears output_tokens/expected_tokens of a node's preparation (capped at
  // 1 for short leases). A node's provisioning share is attributed pro rata to its bytes. Father's own upload
  // is not amortised (share 1). A node whose own preparation exceeds its expected lease is flagged and the
  // overrun is penalised in the objective.
  double eff = 0, upload_eff = 0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const std::string& id = order[i];
    if (id == father.id) {
      upload_eff = std::max(upload_eff, evals[i]->upload_s);
      continue;
    }
    LeaseExpectation lease = req_.default_lease;
    if (auto it = req_.node_lease.find(id); it != req_.node_lease.end()) lease = it->second;
    const double share = lease.expected_tokens > 0 ? std::min(1.0, static_cast<double>(req_.output_tokens) / lease.expected_tokens) : 1.0;
    const double prov_attr = total_new_bytes > 0 ? pm.provisioning_s * node_bytes[id] / total_new_bytes : 0.0;
    eff += share * prov_attr;
    upload_eff = std::max(upload_eff, share * node_upload_s[id]);
    const double own_prep = prov_attr + node_upload_s[id];
    if (own_prep > lease.expected_lease_s) {
      pm.lease_overrun_s += own_prep - lease.expected_lease_s;
      pm.flags.push_back("lease_overrun:" + id);
    }
  }
  pm.effective_prepare_s = eff + upload_eff;
  const auto& w = req_.weights;
  pm.objective = w.preparation * (pm.effective_prepare_s + w.lease_overrun_penalty * pm.lease_overrun_s) +
                 w.throughput * (pm.prefill_s + static_cast<double>(req_.output_tokens) / pm.decode_tok_s);

  // ---- ledgers, residency, provenance, hash ----
  for (std::size_t i = 0; i < order.size(); ++i) {
    plan.ledgers.push_back(evals[i]->ledger);
    plan.residency.push_back(evals[i]->residency);
  }
  plan.provenance = Provenance::kQualified;
  auto fold = [&](const std::string& name, Provenance pv) {
    plan.provenance = weakest(plan.provenance, pv);
    if (pv != Provenance::kQualified) plan.non_qualified_inputs.push_back(name + "=" + std::string(to_string(pv)));
  };
  for (const auto& id : order) fold("profile:" + id, profile_prov_[id]);
  {
    std::set<std::string> seen;
    for (const auto& h : hops)
      if (seen.insert(h.from + ">" + h.to).second) fold("link:" + h.from + ">" + h.to, weakest_link_provenance(*h.link));
    for (const auto& [id, l] : father_links)
      if (seen.insert(father.id + ">" + id).second) fold("link:" + father.id + ">" + id, weakest_link_provenance(*l));
  }
  if (!father_links.empty()) fold("network:father_egress", req_.network.father_egress_bytes_per_s.provenance);
  fold("model:structure", m.provenance);
  fold("model:draft_ms", m.draft_ms.provenance);
  fold("acceptance", req_.acceptance.provenance);

  ByteWriter w8;
  w8.str("clusterlm.placement.plan.v1");
  w8.str(m.name);
  w8.u32(req_.q);
  w8.u32(req_.context_tokens);
  w8.u32(static_cast<std::uint32_t>(stages.size()));
  for (const auto& s : stages) {
    w8.str(s.domain_id);
    w8.u8(static_cast<std::uint8_t>(s.role));
    w8.u32(s.layers.begin);
    w8.u32(s.layers.end);
  }
  for (std::size_t i = 0; i < plan.ledgers.size(); ++i) {
    const auto& L = plan.ledgers[i];
    w8.str(L.domain_id);
    for (std::uint64_t v : {L.dense, L.gpu_experts, L.cpu_experts, L.state, L.scratch_vram, L.staging_ram, L.os_reserve_ram, L.father_only})
      w8.u64(v);
    const auto& R = plan.residency[i];
    w8.u64(R.gpu_expert_count);
    w8.u64(R.cpu_expert_count);
    for (const auto& per_layer : R.gpu_experts_by_layer) {
      w8.u32(static_cast<std::uint32_t>(per_layer.size()));
      for (std::uint16_t e : per_layer) w8.u16(e);
    }
  }
  plan.plan_hash = Sha256::of(ByteSpan(w8.bytes()));
  out.plan = std::move(plan);
  return out;
}

Status validate_request(const PlacementRequest& r) {
  auto err = [](std::string m) { return make_error(ErrorCode::kInvalidArgument, "placement request: " + std::move(m)); };
  CLM_RETURN_IF_ERROR(validate(r.model));
  CLM_RETURN_IF_ERROR(validate(r.father));
  if (r.father.role != DomainRole::kFather) return err("father profile must have role father");
  std::set<std::string> ids{r.father.id};
  for (const auto& n : r.nodes) {
    CLM_RETURN_IF_ERROR(validate(n));
    if (n.role != DomainRole::kNode) return err("node profile '" + n.id + "' must have role node");
    if (!ids.insert(n.id).second) return err("duplicate domain id '" + n.id + "'");
  }
  CLM_RETURN_IF_ERROR(validate(r.network));
  if (!r.nodes.empty() && r.network.father_egress_bytes_per_s.value <= 0) return err("father_egress_bytes_per_s must be > 0");
  if (r.q < 1) return err("q must be >= 1");
  if (r.granularity < 1) return err("granularity must be >= 1");
  if (r.prefill_chunk < 1) return err("prefill_chunk must be >= 1");
  if (!(r.acceptance.value > 0)) return err("acceptance must be > 0");
  if (r.output_tokens < 1) return err("output_tokens must be >= 1");
  if (!(r.overlap_factor >= 0 && r.overlap_factor <= 1)) return err("overlap_factor must be in [0,1]");
  if (r.weights.throughput < 0 || r.weights.preparation < 0 || r.weights.lease_overrun_penalty < 0)
    return err("negative objective weight");
  for (std::uint32_t p : r.prefix_layers_options)
    if (p < 1 || p >= r.model.n_layers()) return err("prefix_layers option out of range");
  return Status::ok();
}

}  // namespace

PlanEvaluation evaluate_plan(const PlacementRequest& request, const std::vector<PlanStage>& stages) {
  if (Status st = validate_request(request); !st.is_ok()) {
    PlanEvaluation e;
    e.key = plan_key(stages);
    e.reasons = {st.message()};
    return e;
  }
  Evaluator ev(request);
  return ev.evaluate(stages);
}

Result<PlacementResult> search_placements(const PlacementRequest& r) {
  CLM_RETURN_IF_ERROR(validate_request(r));
  Evaluator ev(r);
  PlacementResult res;
  const std::uint32_t n = r.model.n_layers();

  auto consider = [&](std::vector<PlanStage> stages) {
    PlanEvaluation e = ev.evaluate(stages);
    if (e.plan)
      res.candidates.push_back(std::move(*e.plan));
    else
      res.rejected.push_back({std::move(e.key), std::move(stages), std::move(e.reasons)});
  };

  // Father-only.
  consider({{r.father.id, StageRole::kFull, {0, n}}});

  std::vector<std::uint32_t> prefixes = r.prefix_layers_options;
  std::sort(prefixes.begin(), prefixes.end());
  prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());

  for (std::uint32_t prefix : prefixes) {
    // Cut points: prefix + k*granularity while < n, and n itself (empty Father tail).
    std::vector<std::uint32_t> cuts;
    for (std::uint32_t c = prefix + r.granularity; c < n; c += r.granularity) cuts.push_back(c);
    cuts.push_back(n);

    auto build = [&](const std::vector<std::pair<const HardwareProfile*, std::uint32_t>>& nodes_and_ends) {
      std::vector<PlanStage> stages;
      stages.push_back({r.father.id, StageRole::kPrefix, {0, prefix}});
      std::uint32_t at = prefix;
      for (const auto& [node, end] : nodes_and_ends) {
        stages.push_back({node->id, StageRole::kMiddle, {at, end}});
        at = end;
      }
      if (at < n) stages.push_back({r.father.id, StageRole::kTail, {at, n}});
      return stages;
    };

    for (const auto& a : r.nodes)
      for (std::uint32_t c : cuts) consider(build({{&a, c}}));

    for (std::size_t i = 0; i < r.nodes.size(); ++i)
      for (std::size_t j = 0; j < r.nodes.size(); ++j) {
        if (i == j || (!r.allow_node_orders && i > j)) continue;
        for (std::size_t x = 0; x < cuts.size(); ++x)
          for (std::size_t y = x + 1; y < cuts.size(); ++y) consider(build({{&r.nodes[i], cuts[x]}, {&r.nodes[j], cuts[y]}}));
      }
  }

  std::sort(res.candidates.begin(), res.candidates.end(), [](const PlacementPlan& a, const PlacementPlan& b) {
    if (a.metrics.objective != b.metrics.objective) return a.metrics.objective < b.metrics.objective;
    if (a.plan_hash != b.plan_hash) return a.plan_hash < b.plan_hash;
    return a.key < b.key;
  });

  const std::size_t nc = res.candidates.size();
  if (nc > 0) {
    res.recommended = 0;
    res.best_throughput = 0;
    res.best_preparation = 0;
    for (std::size_t i = 1; i < nc; ++i) {
      if (res.candidates[i].metrics.decode_tok_s > res.candidates[*res.best_throughput].metrics.decode_tok_s) res.best_throughput = i;
      if (res.candidates[i].metrics.prepare_s < res.candidates[*res.best_preparation].metrics.prepare_s) res.best_preparation = i;
    }
    // Pareto frontier: minimise prepare_s, maximise decode_tok_s.
    std::vector<std::size_t> idx(nc);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
      const auto& ma = res.candidates[a].metrics;
      const auto& mb = res.candidates[b].metrics;
      if (ma.prepare_s != mb.prepare_s) return ma.prepare_s < mb.prepare_s;
      return ma.decode_tok_s > mb.decode_tok_s;
    });
    double best = -1;
    for (std::size_t i : idx)
      if (res.candidates[i].metrics.decode_tok_s > best) {
        best = res.candidates[i].metrics.decode_tok_s;
        res.pareto.push_back(i);
      }
  }

  res.provenance = std::min(weakest_provenance(r.father), weakest_provenance(r.network));
  for (const auto& nd : r.nodes) res.provenance = weakest(res.provenance, weakest_provenance(nd));
  res.provenance = weakest(res.provenance, weakest(r.model.provenance, weakest(r.model.draft_ms.provenance, r.acceptance.provenance)));
  return res;
}

// ---- JSON report ---------------------------------------------------------------------------------------

namespace {

json stage_json(const PlanStage& s) {
  return {{"domain", s.domain_id}, {"role", std::string(to_string(s.role))}, {"layer_begin", s.layers.begin}, {"layer_end", s.layers.end}};
}

std::string provenance_banner(Provenance p) {
  if (p == Provenance::kQualified) return "QUALIFIED: every input quantity is qualified";
  if (p == Provenance::kMeasured) return "MEASURED, NOT QUALIFIED: at least one input is an unqualified measurement";
  return "SYNTHETIC: development estimate, not a measurement; do not use as a qualified placement decision";
}

json plan_json_value(const PlacementPlan& p, bool lists) {
  json j;
  j["provenance"] = std::string(to_string(p.provenance));
  j["provenance_banner"] = provenance_banner(p.provenance);
  j["non_qualified_inputs"] = p.non_qualified_inputs;
  j["key"] = p.key;
  j["plan_hash"] = p.plan_hash.hex();
  j["stages"] = json::array();
  for (const auto& s : p.stages) j["stages"].push_back(stage_json(s));
  j["memory_ledgers"] = json::array();
  for (const auto& l : p.ledgers)
    j["memory_ledgers"].push_back({{"domain", l.domain_id},
                                   {"dense_bytes", l.dense},
                                   {"gpu_expert_bytes", l.gpu_experts},
                                   {"cpu_expert_bytes", l.cpu_experts},
                                   {"state_bytes", l.state},
                                   {"scratch_vram_bytes", l.scratch_vram},
                                   {"staging_ram_bytes", l.staging_ram},
                                   {"os_reserve_ram_bytes", l.os_reserve_ram},
                                   {"father_only_bytes", l.father_only},
                                   {"vram_used_bytes", l.vram_used()},
                                   {"vram_budget_bytes", l.vram_budget},
                                   {"ram_used_bytes", l.ram_used()},
                                   {"ram_safe_allowance_bytes", l.ram_safe_allowance}});
  j["expert_residency"] = json::array();
  for (const auto& r : p.residency) {
    json rj = {{"domain", r.domain_id}, {"gpu_experts", r.gpu_expert_count}, {"cpu_experts", r.cpu_expert_count}};
    if (lists) {
      rj["gpu_experts_by_layer"] = json::object();
      for (std::size_t l = 0; l < r.gpu_experts_by_layer.size(); ++l)
        if (!r.gpu_experts_by_layer[l].empty()) rj["gpu_experts_by_layer"][std::to_string(l)] = r.gpu_experts_by_layer[l];
    }
    j["expert_residency"].push_back(std::move(rj));
  }
  const auto& m = p.metrics;
  json mj = {{"decode_tok_s", m.decode_tok_s},
             {"round_ms", m.round_ms},
             {"draft_ms", m.draft_ms},
             {"stage_ms_total", m.stage_ms_total},
             {"transport_ms_total", m.transport_ms_total},
             {"commit_ms", m.commit_ms},
             {"transport_bytes_per_round", m.transport_bytes_per_round},
             {"prefill_s", m.prefill_s},
             {"provisioning_bytes", m.provisioning_bytes},
             {"provisioning_s", m.provisioning_s},
             {"upload_s", m.upload_s},
             {"prepare_s", m.prepare_s},
             {"effective_prepare_s", m.effective_prepare_s},
             {"lease_overrun_s", m.lease_overrun_s},
             {"t_request_s", m.t_request_s},
             {"objective", m.objective},
             {"flags", m.flags}};
  mj["stages"] = json::array();
  for (const auto& s : m.stages)
    mj["stages"].push_back({{"domain", s.domain_id},
                            {"role", std::string(to_string(s.role))},
                            {"layer_begin", s.layers.begin},
                            {"layer_end", s.layers.end},
                            {"dense_ms", s.dense_ms},
                            {"cpu_ms", s.cpu_ms},
                            {"gpu_ms", s.gpu_ms},
                            {"stage_ms", s.stage_ms},
                            {"cpu_miss_bytes_expected", s.cpu_miss_bytes_expected},
                            {"gpu_hit_bytes_expected", s.gpu_hit_bytes_expected},
                            {"prefill_chunk_ms", s.prefill_chunk_ms}});
  mj["transports"] = json::array();
  for (const auto& t : m.transports)
    mj["transports"].push_back({{"from", t.from}, {"to", t.to}, {"bytes_per_round", t.bytes_per_round}, {"ms", t.ms}, {"prefill_chunk_ms", t.prefill_chunk_ms}});
  j["predicted"] = std::move(mj);
  return j;
}

}  // namespace

std::string plan_to_json(const PlacementPlan& plan, bool include_expert_lists, int indent) {
  return plan_json_value(plan, include_expert_lists).dump(indent);
}

std::string report_to_json(const PlacementResult& r, bool include_expert_lists, int indent) {
  json j;
  j["schema"] = "clusterlm.placement_report.v1";
  j["provenance"] = std::string(to_string(r.provenance));
  j["provenance_banner"] = provenance_banner(r.provenance);
  j["feasible_count"] = r.candidates.size();
  j["rejected_count"] = r.rejected.size();
  auto ref = [&](std::optional<std::size_t> i) -> json {
    return i ? json(r.candidates[*i].key) : json(nullptr);
  };
  j["recommended"] = ref(r.recommended);
  j["best_throughput"] = ref(r.best_throughput);
  j["best_preparation"] = ref(r.best_preparation);
  if (r.recommended) j["recommended_plan"] = plan_json_value(r.candidates[*r.recommended], include_expert_lists);
  j["pareto"] = json::array();
  for (std::size_t i : r.pareto)
    j["pareto"].push_back({{"key", r.candidates[i].key},
                           {"prepare_s", r.candidates[i].metrics.prepare_s},
                           {"decode_tok_s", r.candidates[i].metrics.decode_tok_s}});
  j["candidates"] = json::array();
  for (const auto& p : r.candidates)
    j["candidates"].push_back({{"key", p.key},
                               {"plan_hash", p.plan_hash.hex()},
                               {"provenance", std::string(to_string(p.provenance))},
                               {"decode_tok_s", p.metrics.decode_tok_s},
                               {"prepare_s", p.metrics.prepare_s},
                               {"objective", p.metrics.objective},
                               {"flags", p.metrics.flags}});
  j["rejected"] = json::array();
  for (const auto& x : r.rejected) j["rejected"].push_back({{"key", x.key}, {"reasons", x.reasons}});
  return j.dump(indent);
}

}  // namespace clusterlm::placement
