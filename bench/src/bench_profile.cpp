#include "bench_profile.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <random>
#include <set>

namespace clusterlm::bench {

using placement::HardwareProfile;
using placement::NetworkProfile;
using placement::Provenance;
using placement::Quantity;

std::string bench_source(const std::string& run_id, const std::string& machine_id) {
  return "bench:" + run_id + "@" + machine_id;
}

std::optional<ParsedSource> parse_bench_source(const std::string& source) {
  // The source may carry a suffix appended by other tools ("; qualified: ...").
  if (source.rfind("bench:", 0) != 0) return std::nullopt;
  const std::string rest = source.substr(6, source.find(';') == std::string::npos ? std::string::npos : source.find(';') - 6);
  const auto at = rest.find('@');
  if (at == std::string::npos || at == 0 || at + 1 >= rest.size()) return std::nullopt;
  return ParsedSource{rest.substr(0, at), rest.substr(at + 1)};
}

std::string make_run_id() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
  std::random_device rd;
  char tail[8];
  std::snprintf(tail, sizeof tail, "%04x", static_cast<unsigned>(rd() & 0xFFFFu));
  return std::string("run-") + buf + "-" + tail;
}

namespace {

constexpr const char* kUnmeasured = "synthetic: not measured by clusterlm-bench";

Quantity syn(double v, const char* why = kUnmeasured) { return Quantity::synthetic(v, why); }

double bandwidth_best(const MemoryMeasurement& m) {
  double best = 0;
  for (const auto& b : m.bandwidth) best = std::max(best, b.bytes_per_s.median());
  return best;
}

}  // namespace

HardwareProfile build_measured_profile(const HardwareMeasurements& m) {
  HardwareProfile p;
  const std::string src = bench_source(m.run_id, m.machine_id);
  auto meas = [&](double v) { return Quantity::measured(v, src); };
  p.id = m.machine_id;
  p.role = m.role;

  // ---- CPU ----
  p.cpu.arch = m.host.arch;
  p.cpu.features = m.host.cpu_features;
  p.cpu.usable_threads = syn(static_cast<double>(std::max(1u, m.host.logical_cpus > 1 ? m.host.logical_cpus - 1 : 1)),
                             "synthetic: logical CPUs minus one, no thread sweep run");
  p.cpu.q_scaling = syn(0.05);
  p.cpu.sustained_factor = syn(0.9, "synthetic: placeholder, no 30-minute sustained run");
  std::string provider_note;
  if (m.cpu) {
    unsigned usable = 1;
    for (const auto& rr : m.cpu->representations) {
      usable = std::max(usable, rr.usable_threads);
      for (const auto& pt : rr.q_points)
        if (pt.q == 1 && !pt.bytes_per_s.empty()) p.cpu.expert_bytes_per_s[rr.representation] = meas(pt.bytes_per_s.median());
    }
    p.cpu.usable_threads = meas(static_cast<double>(usable));
    double qs = 0;
    bool have = false;
    for (const auto& rr : m.cpu->representations)
      if (rr.q_scaling_valid) {
        qs = std::max(qs, rr.q_scaling);
        have = true;
      }
    if (have) p.cpu.q_scaling = meas(qs);
    provider_note = " expert kernels: provider '" + m.cpu->provider_id + "' (" + m.cpu->isa + ").";
  }
  if (m.sustained && m.sustained->minutes >= kMinSustainedMinutes && !m.sustained->samples.empty())
    p.cpu.sustained_factor = meas(std::clamp(m.sustained->sustained_factor, 0.01, 1.0));

  // ---- memory ----
  const std::uint64_t total = m.memory ? m.memory->info.total_physical : m.host.ram_bytes;
  p.memory.ram_total = total > 0 ? meas(static_cast<double>(total)) : syn(0);
  p.memory.ram_safe_allowance = syn(0.75 * static_cast<double>(total), "synthetic: policy placeholder, 75% of RAM total");
  p.memory.ram_bandwidth = syn(0);
  p.memory.pinned_limit = syn(0);
  if (m.memory) {
    const double bw = bandwidth_best(*m.memory);
    if (bw > 0) p.memory.ram_bandwidth = meas(bw);
  }

  // ---- GPU ----
  p.gpu.vram_total = syn(0);
  p.gpu.vram_budget = syn(0);
  p.gpu.gpu_expert_bytes_per_s = syn(0);
  p.gpu.dense_layer_ms["recurrent"] = syn(0);
  p.gpu.dense_layer_ms["attention"] = syn(0);
  p.gpu.dense_q_scaling = syn(0);
  p.gpu.pcie_h2d_bytes_per_s = syn(0);
  p.gpu.prefill_tokens_per_s = syn(0);
  if (m.gpu && m.gpu->available && !m.gpu->devices.empty()) {
    const GpuDeviceReport& d = m.gpu->devices.front();
    p.gpu.name = d.info.name;
    p.gpu.vram_total = meas(static_cast<double>(d.info.total_bytes));
    std::uint64_t budget = d.info.free_bytes;
    if (!m.dxgi_budget_bytes.empty() && m.dxgi_budget_bytes.front() > 0) budget = m.dxgi_budget_bytes.front();
    p.gpu.vram_budget = meas(static_cast<double>(std::min(budget, d.info.total_bytes)));
    // Pinned H2D at the 256 MiB ring (the ring size the staging path uses), else the largest pinned ring measured.
    const TransferPoint* pick = nullptr;
    for (const auto& t : d.transfers) {
      if (t.kind != HostMemoryKind::kPinnedAlloc || t.h2d_bytes_per_s.empty()) continue;
      if (t.bytes == (256ull << 20)) {
        pick = &t;
        break;
      }
      if (!pick || t.bytes > pick->bytes) pick = &t;
    }
    if (pick) p.gpu.pcie_h2d_bytes_per_s = meas(pick->h2d_bytes_per_s.median());
    if (d.pinned.largest_ok_bytes > 0) p.memory.pinned_limit = meas(static_cast<double>(d.pinned.largest_ok_bytes));
  }

  // ---- overheads ----
  p.overheads.os_reserve_ram = syn(0);
  p.overheads.scratch_vram = syn(0);
  p.overheads.staging_ram = syn(0);

  p.synthetic_fixture = false;
  p.note = "Calibrated by clusterlm-bench run " + m.run_id + " on " + m.machine_id +
           "; Measured fields carry bench provenance, every other field is a Synthetic placeholder. Never Qualified by the tool." +
           provider_note;
  return p;
}

std::vector<std::string> measured_paths(HardwareProfile& p) {
  std::vector<std::string> out;
  placement::for_each_quantity(p, [&](const std::string& path, Quantity& q) {
    if (q.provenance != Provenance::kSynthetic) out.push_back(path);
  });
  return out;
}

namespace {

enum class MergeOutcome { kReplaced, kKeptNewer, kKeptQualified, kKeptUnordered, kNoChange };

MergeOutcome merge_quantity(Quantity& dst, const Quantity& src) {
  if (src.provenance == Provenance::kSynthetic) return MergeOutcome::kNoChange;  // never replaces anything
  Quantity incoming = src;
  incoming.provenance = Provenance::kMeasured;  // the bench never promotes beyond Measured
  switch (dst.provenance) {
    case Provenance::kSynthetic:
      dst = incoming;
      return MergeOutcome::kReplaced;
    case Provenance::kQualified:
      return MergeOutcome::kKeptQualified;
    case Provenance::kMeasured: {
      const auto a = parse_bench_source(dst.source), b = parse_bench_source(incoming.source);
      if (!a || !b) return MergeOutcome::kKeptUnordered;
      if (b->run_id >= a->run_id) {
        dst = incoming;
        return MergeOutcome::kReplaced;
      }
      return MergeOutcome::kKeptNewer;
    }
  }
  return MergeOutcome::kNoChange;
}

void record(MergeReport& r, MergeOutcome o, const std::string& path) {
  switch (o) {
    case MergeOutcome::kReplaced: r.replaced.push_back(path); break;
    case MergeOutcome::kKeptNewer: r.kept_newer.push_back(path); break;
    case MergeOutcome::kKeptQualified: r.kept_qualified.push_back(path); break;
    case MergeOutcome::kKeptUnordered: r.kept_unordered.push_back(path); break;
    case MergeOutcome::kNoChange: break;
  }
}

std::size_t count_measured(HardwareProfile& p) { return measured_paths(p).size(); }

}  // namespace

MergeReport merge_hardware_profile(HardwareProfile& base, const HardwareProfile& incoming_const) {
  MergeReport report;
  HardwareProfile incoming = incoming_const;
  const std::size_t incoming_measured = count_measured(incoming);

  // Keys of the by-name maps that the base does not know yet are added as they are (incoming Synthetic
  // placeholders are never added: an unmeasured key is simply absent).
  std::set<std::string> added;
  for (const auto& [k, q] : incoming.cpu.expert_bytes_per_s)
    if (!base.cpu.expert_bytes_per_s.count(k) && q.provenance != Provenance::kSynthetic) {
      Quantity copy = q;
      copy.provenance = Provenance::kMeasured;
      base.cpu.expert_bytes_per_s[k] = copy;
      added.insert("cpu.expert_bytes_per_s." + k);
      report.added.push_back("cpu.expert_bytes_per_s." + k);
    }
  for (const auto& [k, q] : incoming.gpu.dense_layer_ms)
    if (!base.gpu.dense_layer_ms.count(k)) {
      Quantity copy = q;
      if (copy.provenance != Provenance::kSynthetic) copy.provenance = Provenance::kMeasured;
      base.gpu.dense_layer_ms[k] = copy;
      added.insert("gpu.dense_layer_ms." + k);
      if (copy.provenance != Provenance::kSynthetic) report.added.push_back("gpu.dense_layer_ms." + k);
    }

  std::map<std::string, Quantity*> dst;
  placement::for_each_quantity(base, [&](const std::string& path, Quantity& q) { dst[path] = &q; });
  placement::for_each_quantity(incoming, [&](const std::string& path, Quantity& q) {
    if (added.count(path)) return;
    auto it = dst.find(path);
    if (it == dst.end()) return;
    record(report, merge_quantity(*it->second, q), path);
  });

  if (incoming_measured > 0) {
    base.id = incoming.id;
    base.role = incoming.role;
    if (!incoming.cpu.arch.empty()) base.cpu.arch = incoming.cpu.arch;
    if (!incoming.cpu.features.empty()) base.cpu.features = incoming.cpu.features;
    if (!incoming.gpu.name.empty()) base.gpu.name = incoming.gpu.name;
    base.note = incoming.note;
  }

  // A Synthetic placeholder inherited from a base file must not contradict a freshly measured limit.
  if (base.memory.ram_safe_allowance.provenance == Provenance::kSynthetic &&
      base.memory.ram_safe_allowance.value > base.memory.ram_total.value)
    base.memory.ram_safe_allowance.value = base.memory.ram_total.value;
  if (base.gpu.vram_budget.provenance == Provenance::kSynthetic && base.gpu.vram_budget.value > base.gpu.vram_total.value)
    base.gpu.vram_budget.value = base.gpu.vram_total.value;
  // A placeholder sustained factor must stay inside (0,1].
  if (!(base.cpu.sustained_factor.value > 0 && base.cpu.sustained_factor.value <= 1)) base.cpu.sustained_factor.value = 0.9;

  base.synthetic_fixture = count_measured(base) == 0;
  return report;
}

MergeReport merge_network_profile(NetworkProfile& base, const NetworkProfile& incoming) {
  MergeReport report;
  record(report, merge_quantity(base.father_egress_bytes_per_s, incoming.father_egress_bytes_per_s), "father_egress_bytes_per_s");
  for (const auto& l : incoming.links) {
    placement::LinkProfile* target = nullptr;
    for (auto& b : base.links)
      if ((b.from == l.from && b.to == l.to) || (b.from == l.to && b.to == l.from)) target = &b;
    const bool measured = weakest_link_provenance(l) != Provenance::kSynthetic ||
                          l.bandwidth_bytes_per_s.provenance != Provenance::kSynthetic ||
                          l.rtt_ms.provenance != Provenance::kSynthetic || l.jitter_ms.provenance != Provenance::kSynthetic;
    if (!target) {
      base.links.push_back(l);
      if (measured) report.added.push_back("links[" + l.from + "->" + l.to + "]");
      continue;
    }
    const std::string base_path = "links[" + target->from + "->" + target->to + "].";
    record(report, merge_quantity(target->bandwidth_bytes_per_s, l.bandwidth_bytes_per_s), base_path + "bandwidth_bytes_per_s");
    record(report, merge_quantity(target->rtt_ms, l.rtt_ms), base_path + "rtt_ms");
    record(report, merge_quantity(target->jitter_ms, l.jitter_ms), base_path + "jitter_ms");
  }
  bool any_measured = false;
  placement::for_each_quantity(base, [&](const std::string&, Quantity& q) { any_measured |= q.provenance != Provenance::kSynthetic; });
  base.synthetic_fixture = !any_measured;
  if (!incoming.note.empty()) base.note = incoming.note;
  return report;
}

NetworkProfile build_network_profile(const std::string& machine_id, const std::string& run_id,
                                     const std::vector<PeerLink>& peers, const ConcurrentMeasure* concurrent) {
  NetworkProfile n;
  const std::string src = bench_source(run_id, machine_id);
  bool any_real = false;
  double first_tx = 0;
  for (const auto& peer : peers) {
    const SizeMeasure* small = peer.measure.smallest();
    const SizeMeasure* large = peer.measure.largest();
    if (!small || !large || small->rtt_ms.empty()) continue;
    placement::LinkProfile l;
    l.from = machine_id;
    l.to = peer.peer_id;
    const double bw = std::max(1.0, std::min(large->tx_bytes_per_s, large->rx_bytes_per_s));
    const char* why = "synthetic: bench loopback or simulated link";
    auto q = [&](double v) { return peer.real_link ? Quantity::measured(v, src) : Quantity::synthetic(v, why); };
    l.bandwidth_bytes_per_s = q(bw);
    l.rtt_ms = q(small->rtt_ms.median());
    l.jitter_ms = q(small->jitter_ms());
    any_real |= peer.real_link;
    if (first_tx == 0) first_tx = large->tx_bytes_per_s;
    n.links.push_back(std::move(l));
  }
  if (concurrent && concurrent->connections >= 2 && any_real) {
    n.father_egress_bytes_per_s = Quantity::measured(concurrent->egress_bytes_per_s, src);
  } else if (concurrent && concurrent->connections >= 2) {
    n.father_egress_bytes_per_s = Quantity::synthetic(concurrent->egress_bytes_per_s, "synthetic: bench loopback concurrent egress");
  } else {
    n.father_egress_bytes_per_s =
        Quantity::synthetic(first_tx, "synthetic: single-peer tx rate, concurrent egress not measured");
  }
  n.synthetic_fixture = !any_real;
  n.note = "Network measured by clusterlm-bench run " + run_id + " from " + machine_id +
           (any_real ? "" : " over loopback/simulated links (Synthetic)") + "; never Qualified by the tool.";
  return n;
}

}  // namespace clusterlm::bench
