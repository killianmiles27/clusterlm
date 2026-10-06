#include "clusterlm/placement/placement.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
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

std::uint32_t floor_pow2(std::uint32_t v) {
  if (v == 0) return 0;
  std::uint32_t p = 1;
  while (p <= v / 2) p *= 2;
  return p;
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
  Digest256 digest;         // over id, ranges, ledger and residency (feeds the plan hash)
};

// One costed stage assignment: metrics are final, ledgers/residency/provenance/hash are materialised later so
// that candidates dropped by pruning never pay for them.
struct Hop {
  std::string from, to;
  const LinkProfile* link = nullptr;
};
struct Costed {
  std::string key;
  std::vector<PlanStage> stages;
  PredictedMetrics metrics;
  std::vector<std::string> order;  // domains in first-appearance (pipeline) order
  std::vector<const DomainEval*> evals;
  std::vector<Hop> hops;
  std::map<std::string, const LinkProfile*> father_links;
};
struct CostOutcome {
  std::optional<Costed> costed;
  std::vector<std::string> reasons;
  std::string key;
};

class Evaluator {
 public:
  explicit Evaluator(const PlacementRequest& r) : req_(r), n_(r.model.n_layers()) {
    q_ = static_cast<double>(r.q);
    prompt_ = r.prompt_tokens > 0 ? r.prompt_tokens : r.context_tokens;
    chunk_tokens_ = std::min(r.prefill_chunk, prompt_);
    const auto& m = r.model;
    order_.resize(n_);
    pq_.resize(n_);
    union_total_.assign(n_, 0.0);
    for (std::uint32_t l = 0; l < n_; ++l) {
      const auto& row = m.routing_freq[l];
      order_[l].resize(m.n_experts);
      std::iota(order_[l].begin(), order_[l].end(), std::uint16_t{0});
      std::stable_sort(order_[l].begin(), order_[l].end(), [&](std::uint16_t a, std::uint16_t b) { return row[a] > row[b]; });
      pq_[l].resize(m.n_experts);
      const double bytes = static_cast<double>(m.layers[l].expert_bytes);
      for (std::uint32_t e = 0; e < m.n_experts; ++e) {
        pq_[l][e] = union_prob(row[e], q_);
        union_total_[l] += pq_[l][e] * bytes;
      }
    }
    domains_[r.father.id] = &r.father;
    for (const auto& n : r.nodes) domains_[n.id] = &n;
  }

  CostOutcome cost(const std::vector<PlanStage>& stages);
  PlacementPlan materialize(const Costed& c) const;
  PlanEvaluation evaluate(const std::vector<PlanStage>& stages) {
    PlanEvaluation out;
    CostOutcome co = cost(stages);
    out.key = std::move(co.key);
    if (co.costed)
      out.plan = materialize(*co.costed);
    else
      out.reasons = std::move(co.reasons);
    return out;
  }

 private:
  // Union probability over one prefill micro-batch of `batch` tokens, per layer/expert, and its byte total.
  struct ChunkTable {
    std::vector<std::vector<double>> pc;
    std::vector<double> total;
  };
  const ChunkTable& chunk_table(std::uint32_t batch);
  const DomainEval& domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges);
  DomainEval compute_domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges);
  std::uint64_t layer_object_bytes(std::uint32_t l) const {
    return req_.model.layers[l].dense_bytes + std::uint64_t{req_.model.n_experts} * req_.model.layers[l].expert_bytes;
  }

  const PlacementRequest& req_;
  std::uint32_t n_;
  double q_ = 1;
  std::uint32_t prompt_ = 0;
  std::uint32_t chunk_tokens_ = 0;
  std::vector<std::vector<std::uint16_t>> order_;  // experts of each layer by routing frequency, descending
  std::vector<std::vector<double>> pq_;            // union probability over q positions
  std::vector<double> union_total_;                // sum_e bytes * prob over q positions, per layer
  std::map<std::uint32_t, ChunkTable> chunk_tables_;
  std::map<std::string, const HardwareProfile*> domains_;
  std::map<std::string, DomainEval> cache_;  // node-based: references stay valid
};

const Evaluator::ChunkTable& Evaluator::chunk_table(std::uint32_t batch) {
  auto it = chunk_tables_.find(batch);
  if (it != chunk_tables_.end()) return it->second;
  const auto& m = req_.model;
  ChunkTable t;
  t.pc.resize(n_);
  t.total.assign(n_, 0.0);
  for (std::uint32_t l = 0; l < n_; ++l) {
    t.pc[l].resize(m.n_experts);
    const double bytes = static_cast<double>(m.layers[l].expert_bytes);
    for (std::uint32_t e = 0; e < m.n_experts; ++e) {
      t.pc[l][e] = union_prob(m.routing_freq[l][e], static_cast<double>(std::max(1u, batch)));
      t.total[l] += t.pc[l][e] * bytes;
    }
  }
  return chunk_tables_.emplace(batch, std::move(t)).first->second;
}

DomainEval Evaluator::compute_domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges) {
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
  if (chunk_tokens_ > 0 && d.prefill_rate <= 0) reject("prefill_tokens_per_s is zero");
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

  // ---- local micro-batch ----
  // The cluster-wide prefill chunk is a pipeline property; each domain runs it in micro-batches sized to its
  // OWN VRAM headroom, so the smallest GPU never forces a small chunk on everyone else.
  const std::uint32_t wanted = std::max(chunk_tokens_, req_.q);
  led.local_batch = wanted;
  std::uint64_t remaining = 0;
  bool admitted_vram = true;
  if (d.has_gpu) {
    const std::uint64_t fixed = led.dense + led.state + led.scratch_vram;
    if (fixed > led.vram_budget) {
      admitted_vram = false;
      reject("VRAM: dense " + fmt_bytes(static_cast<double>(led.dense)) + " + state " +
             fmt_bytes(static_cast<double>(led.state)) + " + scratch " + fmt_bytes(static_cast<double>(led.scratch_vram)) +
             " exceed budget " + fmt_bytes(static_cast<double>(led.vram_budget)));
    } else {
      remaining = led.vram_budget - fixed;
      const std::uint64_t per_token = m.batch_scratch_bytes_per_token;
      if (per_token > 0) {
        const auto cap_bytes = static_cast<std::uint64_t>(static_cast<double>(remaining) * req_.batch_headroom_fraction);
        const std::uint64_t fit = cap_bytes / per_token;
        std::uint32_t batch = wanted;
        if (fit < wanted) batch = floor_pow2(static_cast<std::uint32_t>(fit));
        if (batch < std::max(req_.min_local_batch, req_.q)) {
          admitted_vram = false;
          reject("VRAM: headroom " + fmt_bytes(static_cast<double>(remaining)) + " cannot hold a local batch of " +
                 std::to_string(std::max(req_.min_local_batch, req_.q)) + " tokens");
        } else {
          led.local_batch = batch;
          led.batch_workspace = std::uint64_t{batch} * per_token;
          remaining -= led.batch_workspace;
        }
      }
    }
  }

  // ---- GPU residency ----
  // Rank experts by critical-path time saved per byte of VRAM spent: P(selected in the q-position union) *
  // (1/cpu_bw - 1/gpu_bw). Within a layer all experts have equal size, so this is frequency order; across
  // layers (different quant / size) the per-byte coefficient differs. Greedy fill; an expert that does not fit
  // ends its layer's stream (same size in a layer) while smaller layers may still be filled.
  //
  // Deterministic tie-breaking: scores are snapped to 12 significant digits relative to the best initial score
  // (so last-bit platform differences cannot reorder experts), then ties resolve to the LOWER layer, then the
  // lower rank (rank order within a layer is stable in expert id). The fill is a pure function of its inputs:
  // it has no memory of any previous plan (no hysteresis).
  const ChunkTable& ct = chunk_table(led.local_batch);
  std::vector<std::vector<std::uint16_t>> chosen(n_);
  std::vector<double> gpu_union(n_, 0.0), gpu_chunk(n_, 0.0);
  if (d.has_gpu && admitted_vram) {
    struct Item {
      std::int64_t score;
      std::uint32_t layer, rank;
    };
    auto worse = [](const Item& a, const Item& b) {
      if (a.score != b.score) return a.score < b.score;
      if (a.layer != b.layer) return a.layer > b.layer;
      return a.rank > b.rank;
    };
    std::vector<double> coef(n_, 0.0);
    double scale = 0;
    for (std::uint32_t l : layers) {
      coef[l] = 1.0 / cpu_eff[l] - 1.0 / gpu_bw;  // seconds saved per byte served from the GPU
      if (coef[l] > 0) scale = std::max(scale, pq_[l][order_[l][0]] * coef[l]);
    }
    auto snap = [&](double v) { return static_cast<std::int64_t>(std::llround(v / scale * 1e12)); };
    std::priority_queue<Item, std::vector<Item>, decltype(worse)> heap(worse);
    if (scale > 0)
      for (std::uint32_t l : layers)
        if (coef[l] > 0) heap.push({snap(pq_[l][order_[l][0]] * coef[l]), l, 0});
    while (!heap.empty() && remaining > 0) {
      const Item it = heap.top();
      heap.pop();
      const std::uint64_t bytes = m.layers[it.layer].expert_bytes;
      if (bytes > remaining) continue;  // this layer's stream ends: all its experts are the same size
      remaining -= bytes;
      const std::uint16_t e = order_[it.layer][it.rank];
      chosen[it.layer].push_back(e);
      gpu_union[it.layer] += pq_[it.layer][e] * static_cast<double>(bytes);
      gpu_chunk[it.layer] += ct.pc[it.layer][e] * static_cast<double>(bytes);
      if (it.rank + 1 < m.n_experts)
        heap.push({snap(pq_[it.layer][order_[it.layer][it.rank + 1]] * coef[it.layer]), it.layer, it.rank + 1});
    }
  }

  // ---- per-layer timings and ledger totals ----
  const double micro_batches =
      chunk_tokens_ > 0 ? std::ceil(static_cast<double>(chunk_tokens_) / static_cast<double>(std::max(1u, led.local_batch))) : 0.0;
  for (std::uint32_t l : layers) {
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
    // Each micro-batch streams the CPU-resident experts its own tokens touch.
    d.prefill_cpu_chunk_ms[l] = micro_batches * std::max(0.0, ct.total[l] - gpu_chunk[l]) / cpu_eff[l] * kMs;
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
  if (led.vram_used() > led.vram_budget && d.has_gpu && admitted_vram) reject("VRAM: over budget");
  const double upload_bytes = static_cast<double>((d.has_gpu ? led.dense : 0) + led.gpu_experts);
  if (upload_bytes > 0) {
    if (p.gpu.pcie_h2d_bytes_per_s.value <= 0)
      reject("pcie_h2d_bytes_per_s is zero but GPU upload is required");
    else
      d.upload_s = upload_bytes / p.gpu.pcie_h2d_bytes_per_s.value;
  }
  d.ok = d.reasons.empty();

  ByteWriter w;
  w.str(p.id);
  for (const auto& r : ranges) {
    w.u32(r.begin);
    w.u32(r.end);
  }
  w.u32(led.local_batch);
  for (std::uint64_t v : {led.dense, led.gpu_experts, led.cpu_experts, led.state, led.scratch_vram, led.staging_ram,
                          led.os_reserve_ram, led.father_only, led.batch_workspace})
    w.u64(v);
  w.u64(d.residency.gpu_expert_count);
  w.u64(d.residency.cpu_expert_count);
  for (const auto& per_layer : d.residency.gpu_experts_by_layer) {
    w.u32(static_cast<std::uint32_t>(per_layer.size()));
    for (std::uint16_t e : per_layer) w.u16(e);
  }
  d.digest = Sha256::of(ByteSpan(w.bytes()));
  return d;
}

const DomainEval& Evaluator::domain(const HardwareProfile& p, const std::vector<LayerRange>& ranges) {
  const std::string key = p.id + "|" + ranges_key(ranges);
  auto it = cache_.find(key);
  if (it == cache_.end()) it = cache_.emplace(key, compute_domain(p, ranges)).first;
  return it->second;
}

CostOutcome Evaluator::cost(const std::vector<PlanStage>& stages) {
  CostOutcome out;
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
  Costed c;
  c.key = out.key;
  c.stages = stages;
  std::map<std::string, std::vector<LayerRange>> ranges;
  for (const auto& s : stages) {
    if (!ranges.count(s.domain_id)) c.order.push_back(s.domain_id);
    ranges[s.domain_id].push_back(s.layers);
  }
  for (const auto& id : c.order)
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
  for (const auto& id : c.order) {
    const DomainEval& d = domain(*domains_[id], ranges[id]);
    c.evals.push_back(&d);
    for (const auto& r : d.reasons) reasons.push_back(id + ": " + r);
  }

  // ---- links ----
  auto add_hop = [&](const std::string& a, const std::string& b) {
    const LinkProfile* l = req_.network.find_link(a, b);
    if (!l) reasons.push_back("no network link " + a + " -> " + b);
    c.hops.push_back({a, b, l});
  };
  for (std::size_t i = 0; i + 1 < stages.size(); ++i)
    if (stages[i].domain_id != stages[i + 1].domain_id) add_hop(stages[i].domain_id, stages[i + 1].domain_id);
  if (stages.back().domain_id != father.id) add_hop(stages.back().domain_id, father.id);
  for (const auto& id : c.order) {
    if (id == father.id) continue;
    const LinkProfile* l = req_.network.find_link(father.id, id);
    if (!l) reasons.push_back("no network link " + father.id + " <-> " + id + " (provisioning/commit)");
    c.father_links[id] = l;
  }
  if (!reasons.empty()) {
    out.reasons = std::move(reasons);
    return out;
  }

  // ---- cost model ----
  PredictedMetrics& pm = c.metrics;
  pm.draft_ms = m.draft_ms.value;

  // Stage verify time: dense + CPU-miss + GPU-hit, serial unless overlap_factor > 0.
  for (const auto& s : stages) {
    const DomainEval& d = *c.evals[static_cast<std::size_t>(std::find(c.order.begin(), c.order.end(), s.domain_id) - c.order.begin())];
    StageMetrics sm;
    sm.domain_id = s.domain_id;
    sm.role = s.role;
    sm.layers = s.layers;
    sm.local_batch = d.ledger.local_batch;
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
    // touched by each micro-batch of the chunk (a micro-batch touches most experts: a weight-streaming cost).
    if (chunk_tokens_ > 0)
      sm.prefill_chunk_ms = static_cast<double>(chunk_tokens_) /
                                (d.prefill_rate * static_cast<double>(n_) / static_cast<double>(s.layers.size())) * kMs +
                            chunk_cpu;
    pm.stage_ms_total += sm.stage_ms;
    pm.stages.push_back(std::move(sm));
  }

  // Transport per boundary: q positions of boundary ABI + half an RTT (one-way latency).
  const double boundary = static_cast<double>(m.boundary_bytes_per_position);
  for (const auto& h : c.hops) {
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
  for (const auto& [id, l] : c.father_links) pm.commit_ms = std::max(pm.commit_ms, l->rtt_ms.value);

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
    const double n_chunks = std::ceil(static_cast<double>(prompt_) / static_cast<double>(chunk_tokens_));
    pm.prefill_s = (first + (n_chunks - 1.0) * slowest) / kMs;
  }

  // Preparation. Provisioning shares Father's NIC, so it is NOT the sum of per-node times: it is the larger of
  // (all bytes / egress) and (the slowest single node's bytes / its own link).
  double total_new_bytes = 0;
  std::map<std::string, double> node_bytes, node_link_s, node_upload_s;
  for (std::size_t i = 0; i < c.order.size(); ++i) {
    const std::string& id = c.order[i];
    pm.upload_s = std::max(pm.upload_s, c.evals[i]->upload_s);
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
    node_link_s[id] = bytes / c.father_links[id]->bandwidth_bytes_per_s.value;
    node_upload_s[id] = c.evals[i]->upload_s;
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
  for (std::size_t i = 0; i < c.order.size(); ++i) {
    const std::string& id = c.order[i];
    if (id == father.id) {
      upload_eff = std::max(upload_eff, c.evals[i]->upload_s);
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
  out.costed = std::move(c);
  return out;
}

// Provenance of the inputs a domain actually uses: quant- and layer-kind-keyed fields only count for the quants
// and kinds of the layers it hosts. Returns the weakest and appends the field paths at that level.
Provenance used_profile_provenance(const HardwareProfile& p, const ModelCostInputs& m, const std::vector<LayerRange>& rs,
                                   std::vector<std::string>* weakest_paths) {
  std::set<std::string> quants, kinds;
  for (const auto& r : rs)
    for (std::uint32_t l = r.begin; l < r.end; ++l) {
      quants.insert(m.layers[l].quant);
      kinds.insert(std::string(to_string(m.layers[l].kind)));
    }
  Provenance w = Provenance::kQualified;
  std::vector<std::pair<std::string, Provenance>> all;
  for_each_quantity(const_cast<HardwareProfile&>(p), [&](const std::string& path, Quantity& q) {
    static const std::string kq = "cpu.expert_bytes_per_s.", kk = "gpu.dense_layer_ms.";
    if (path.rfind(kq, 0) == 0 && !quants.count(path.substr(kq.size()))) return;
    if (path.rfind(kk, 0) == 0 && !kinds.count(path.substr(kk.size()))) return;
    all.emplace_back(path, q.provenance);
    w = weakest(w, q.provenance);
  });
  if (weakest_paths && w != Provenance::kQualified)
    for (const auto& [path, pv] : all)
      if (pv == w) weakest_paths->push_back(path);
  return w;
}

PlacementPlan Evaluator::materialize(const Costed& c) const {
  const auto& m = req_.model;
  const auto& father = req_.father;
  PlacementPlan plan;
  plan.key = c.key;
  plan.q = req_.q;
  plan.context_tokens = req_.context_tokens;
  plan.prompt_tokens = prompt_;
  plan.output_tokens = req_.output_tokens;
  plan.stages = c.stages;
  plan.metrics = c.metrics;
  for (const DomainEval* d : c.evals) {
    plan.ledgers.push_back(d->ledger);
    plan.residency.push_back(d->residency);
  }

  plan.provenance = Provenance::kQualified;
  auto fold = [&](const std::string& name, Provenance pv) {
    plan.provenance = weakest(plan.provenance, pv);
    if (pv != Provenance::kQualified) plan.non_qualified_inputs.push_back(name + "=" + std::string(to_string(pv)));
  };
  std::map<std::string, std::vector<LayerRange>> ranges;
  for (const auto& s : c.stages) ranges[s.domain_id].push_back(s.layers);
  for (const auto& id : c.order) {
    std::vector<std::string> paths;
    const Provenance pv = used_profile_provenance(*domains_.at(id), m, ranges[id], &paths);
    std::string name = "profile:" + id;
    if (!paths.empty()) {
      name += " [";
      for (std::size_t i = 0; i < paths.size() && i < 4; ++i) name += (i ? "," : "") + paths[i];
      if (paths.size() > 4) name += ",+" + std::to_string(paths.size() - 4) + " more";
      name += "]";
    }
    fold(name, pv);
  }
  {
    std::set<std::string> seen;
    for (const auto& h : c.hops)
      if (seen.insert(h.from + ">" + h.to).second) fold("link:" + h.from + ">" + h.to, weakest_link_provenance(*h.link));
    for (const auto& [id, l] : c.father_links)
      if (seen.insert(father.id + ">" + id).second) fold("link:" + father.id + ">" + id, weakest_link_provenance(*l));
  }
  if (!c.father_links.empty()) fold("network:father_egress", req_.network.father_egress_bytes_per_s.provenance);
  fold("model:structure", m.provenance);
  fold("model:draft_ms", m.draft_ms.provenance);
  fold("acceptance", req_.acceptance.provenance);

  ByteWriter w8;
  w8.str("clusterlm.placement.plan.v2");
  w8.str(m.name);
  w8.u32(req_.q);
  w8.u32(req_.context_tokens);
  w8.u32(static_cast<std::uint32_t>(c.stages.size()));
  for (const auto& s : c.stages) {
    w8.str(s.domain_id);
    w8.u8(static_cast<std::uint8_t>(s.role));
    w8.u32(s.layers.begin);
    w8.u32(s.layers.end);
  }
  for (const DomainEval* d : c.evals)
    for (std::uint8_t b : d->digest.bytes) w8.u8(b);
  plan.plan_hash = Sha256::of(ByteSpan(w8.bytes()));
  return plan;
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
  if (!(r.batch_headroom_fraction > 0 && r.batch_headroom_fraction <= 1)) return err("batch_headroom_fraction must be in (0,1]");
  if (r.min_local_batch < 1) return err("min_local_batch must be >= 1");
  for (std::uint32_t p : r.prefix_layers_options)
    if (p < 1 || p >= r.model.n_layers()) return err("prefix_layers option out of range");
  return Status::ok();
}

// Strict 5-D dominance used by pruning: a is no worse than b on every quantity the objective or the Pareto
// frontier reads, and strictly better on at least one.
bool dominates5(const PredictedMetrics& a, const PredictedMetrics& b) {
  const bool no_worse = a.prepare_s <= b.prepare_s && a.effective_prepare_s <= b.effective_prepare_s &&
                        a.lease_overrun_s <= b.lease_overrun_s && a.prefill_s <= b.prefill_s && a.decode_tok_s >= b.decode_tok_s;
  const bool better = a.prepare_s < b.prepare_s || a.effective_prepare_s < b.effective_prepare_s ||
                      a.lease_overrun_s < b.lease_overrun_s || a.prefill_s < b.prefill_s || a.decode_tok_s > b.decode_tok_s;
  return no_worse && better;
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

std::vector<std::size_t> pareto_frontier(const std::vector<const PlacementPlan*>& plans) {
  std::vector<std::size_t> idx(plans.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    const auto& ma = plans[a]->metrics;
    const auto& mb = plans[b]->metrics;
    if (ma.prepare_s != mb.prepare_s) return ma.prepare_s < mb.prepare_s;
    if (ma.decode_tok_s != mb.decode_tok_s) return ma.decode_tok_s > mb.decode_tok_s;
    return ma.prefill_s < mb.prefill_s;
  });
  // Sorted by prepare_s, a later plan can only be dominated by an earlier frontier member (or tie it).
  std::vector<std::size_t> front;
  for (std::size_t i : idx) {
    const auto& mi = plans[i]->metrics;
    bool dominated = false;
    for (std::size_t j : front) {
      const auto& mj = plans[j]->metrics;
      if (mj.prepare_s <= mi.prepare_s && mj.decode_tok_s >= mi.decode_tok_s && mj.prefill_s <= mi.prefill_s) {
        dominated = true;  // dominates or ties (the earlier plan wins a tie)
        break;
      }
    }
    if (!dominated) front.push_back(i);
  }
  return front;
}

Result<PlacementResult> search_placements(const PlacementRequest& r) {
  CLM_RETURN_IF_ERROR(validate_request(r));
  Evaluator ev(r);
  PlacementResult res;
  const std::uint32_t n = r.model.n_layers();
  const std::uint32_t g = r.granularity;

  std::vector<Costed> kept;
  auto consider = [&](std::vector<PlanStage> stages) {
    ++res.enumerated;
    CostOutcome co = ev.cost(stages);
    if (co.costed)
      kept.push_back(std::move(*co.costed));
    else
      res.rejected.push_back({std::move(co.key), std::move(stages), std::move(co.reasons)});
  };

  // Father-only.
  consider({{r.father.id, StageRole::kFull, {0, n}}});

  // Prefix sizes: explicit options plus the optional sweep from the smallest legal prefix.
  std::vector<std::uint32_t> prefixes = r.prefix_layers_options;
  if (r.sweep_prefix) {
    const std::uint32_t pmin = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(r.model.ple_layer + 1));
    if (pmin < n) prefixes.push_back(pmin);
    for (std::uint32_t p = ((pmin + g - 1) / g) * g; p <= pmin + r.prefix_sweep_span && p < n; p += g) prefixes.push_back(p);
  }
  std::sort(prefixes.begin(), prefixes.end());
  prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());

  const std::size_t max_k = std::min<std::size_t>(r.max_remote_nodes, r.nodes.size());
  for (std::uint32_t prefix : prefixes) {
    std::vector<std::uint32_t> cuts;
    for (std::uint32_t c = prefix + 1; c < n; ++c)
      if ((c - prefix) % g == 0 || c % g == 0 || (n - c) % g == 0) cuts.push_back(c);
    cuts.push_back(n);  // n itself = empty Father tail

    // Depth-first over ordered node subsets and strictly increasing range ends.
    std::vector<std::pair<const HardwareProfile*, std::uint32_t>> chosen;
    std::vector<bool> used(r.nodes.size(), false);
    auto emit = [&]() {
      std::vector<PlanStage> stages;
      stages.push_back({r.father.id, StageRole::kPrefix, {0, prefix}});
      std::uint32_t at = prefix;
      for (const auto& [node, end] : chosen) {
        stages.push_back({node->id, StageRole::kMiddle, {at, end}});
        at = end;
      }
      if (at < n) stages.push_back({r.father.id, StageRole::kTail, {at, n}});
      consider(std::move(stages));
    };
    std::function<void(std::size_t, std::size_t, std::size_t)> rec = [&](std::size_t depth, std::size_t from_cut, std::size_t min_node) {
      if (depth > 0) emit();
      if (depth == max_k) return;
      for (std::size_t i = r.allow_node_orders ? 0 : min_node; i < r.nodes.size(); ++i) {
        if (used[i]) continue;
        used[i] = true;
        for (std::size_t x = from_cut; x < cuts.size(); ++x) {
          chosen.emplace_back(&r.nodes[i], cuts[x]);
          // An end of n closes the chain: every further node would need at least one layer.
          rec(depth + 1, cuts[x] < n ? x + 1 : cuts.size(), i + 1);
          chosen.pop_back();
        }
        used[i] = false;
      }
    };
    rec(0, 0, 0);
  }

  if (r.prune_dominated && kept.size() > 1) {
    // Sort by prepare_s (a dominator can never have larger prepare_s), then test against the survivors only.
    std::vector<std::size_t> idx(kept.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return kept[a].metrics.prepare_s < kept[b].metrics.prepare_s; });
    std::vector<std::size_t> front;
    std::vector<bool> keep(kept.size(), false);
    for (std::size_t i : idx) {
      bool dominated = false;
      for (std::size_t j : front)
        if (dominates5(kept[j].metrics, kept[i].metrics)) {
          dominated = true;
          break;
        }
      if (dominated) continue;
      // A newcomer with equal prepare_s can also dominate earlier survivors.
      std::erase_if(front, [&](std::size_t j) {
        const bool d = dominates5(kept[i].metrics, kept[j].metrics);
        if (d) keep[j] = false;
        return d;
      });
      front.push_back(i);
      keep[i] = true;
    }
    std::vector<Costed> survivors;
    for (std::size_t i = 0; i < kept.size(); ++i)
      if (keep[i]) survivors.push_back(std::move(kept[i]));
    res.pruned_dominated = kept.size() - survivors.size();
    kept = std::move(survivors);
  }

  res.candidates.reserve(kept.size());
  for (const auto& c : kept) res.candidates.push_back(ev.materialize(c));
  kept.clear();

  // Deterministic order: objective, then the layer boundaries (independent of domain names), then the hash.
  auto boundaries = [](const PlacementPlan& p) {
    std::vector<std::uint32_t> b;
    for (const auto& s : p.stages) b.push_back(s.layers.end);
    return b;
  };
  std::sort(res.candidates.begin(), res.candidates.end(), [&](const PlacementPlan& a, const PlacementPlan& b) {
    if (a.metrics.objective != b.metrics.objective) return a.metrics.objective < b.metrics.objective;
    const auto ba = boundaries(a), bb = boundaries(b);
    if (ba != bb) return ba < bb;
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
    std::vector<const PlacementPlan*> ptrs;
    ptrs.reserve(nc);
    for (const auto& p : res.candidates) ptrs.push_back(&p);
    res.pareto = pareto_frontier(ptrs);
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
  j["q"] = p.q;
  j["context_tokens"] = p.context_tokens;
  j["prompt_tokens"] = p.prompt_tokens;
  j["output_tokens"] = p.output_tokens;
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
                                   {"batch_workspace_bytes", l.batch_workspace},
                                   {"local_batch", l.local_batch},
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
                            {"local_batch", s.local_batch},
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
  j["enumerated"] = r.enumerated;
  j["pruned_dominated"] = r.pruned_dominated;
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
                           {"decode_tok_s", r.candidates[i].metrics.decode_tok_s},
                           {"prefill_s", r.candidates[i].metrics.prefill_s}});
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
