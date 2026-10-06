#include <doctest/doctest.h>

#include "bench_profile.hpp"
#include "clusterlm/placement/profile.hpp"

using namespace clusterlm;
using namespace clusterlm::bench;
using namespace clusterlm::placement;

namespace {

HardwareProfile fixture(const char* name = "Node-3060-5600.json") {
  auto r = load_hardware_profile(std::string(CLUSTERLM_SOURCE_DIR) + "/fixtures/profiles/" + name);
  REQUIRE_MESSAGE(r.is_ok(), r.status().to_string());
  return r.value();
}

const std::string kRunOld = "run-20260101T000000Z-aaaa";
const std::string kRunMid = "run-20260601T000000Z-bbbb";
const std::string kRunNew = "run-20261001T000000Z-cccc";

// A one-field incoming profile: every quantity Synthetic except ram_total (and optionally more).
HardwareProfile incoming_with_ram(const std::string& run, double bytes, const std::string& machine = "g14") {
  HardwareProfile p = fixture();
  for_each_quantity(p, [](const std::string&, Quantity& q) {
    q.provenance = Provenance::kSynthetic;
    q.source = "synthetic: not measured by clusterlm-bench";
  });
  p.id = machine;
  p.memory.ram_total = Quantity::measured(bytes, bench_source(run, machine));
  return p;
}

CpuBenchReport fake_cpu_report() {
  CpuBenchReport r;
  r.provider_id = "reference";
  r.isa = "scalar-fp32";
  RepresentationReport rr;
  rr.representation = "f32";
  rr.usable_threads = 3;
  rr.q_scaling = 0.4;
  rr.q_scaling_valid = true;
  QPoint q1;
  q1.q = 1;
  q1.pattern = SelectionPattern::kSame;
  for (double v : {9.0, 10.0, 11.0}) q1.bytes_per_s.add(v * 1e9);
  rr.q_points.push_back(q1);
  r.representations.push_back(rr);
  RepresentationReport rq = rr;
  rq.representation = "q8_0-fixture";
  rq.usable_threads = 5;
  rq.q_scaling = 0.7;
  rq.q_points[0].bytes_per_s.samples = {2e9, 2.2e9, 2.4e9};
  r.representations.push_back(rq);
  return r;
}

HardwareMeasurements base_measurements() {
  HardwareMeasurements m;
  m.machine_id = "g14";
  m.role = DomainRole::kNode;
  m.run_id = kRunMid;
  m.host.arch = "x86-64";
  m.host.cpu_features = {"avx2", "fma"};
  m.host.logical_cpus = 16;
  m.host.ram_bytes = 32ull << 30;
  return m;
}

}  // namespace

TEST_CASE("run ids sort chronologically and sources round-trip") {
  const std::string id = make_run_id();
  CHECK(id.rfind("run-", 0) == 0);
  CHECK(id.size() == std::string("run-20261001T000000Z-cccc").size());
  const auto src = bench_source(kRunMid, "father");
  CHECK(src == "bench:run-20260601T000000Z-bbbb@father");
  const auto parsed = parse_bench_source(src + "; qualified: later review");
  REQUIRE(parsed.has_value());
  CHECK(parsed->run_id == kRunMid);
  CHECK(parsed->machine_id == "father");
  CHECK_FALSE(parse_bench_source("synthetic: development estimate").has_value());
  CHECK_FALSE(parse_bench_source("bench:norun").has_value());
  CHECK(kRunOld < kRunMid);
  CHECK(kRunMid < kRunNew);
}

TEST_CASE("merge: a Measured field replaces a Synthetic one; everything else is untouched") {
  HardwareProfile base = fixture();
  const double before_bw = base.memory.ram_bandwidth.value;
  const auto report = merge_hardware_profile(base, incoming_with_ram(kRunMid, 30e9));
  CHECK(base.memory.ram_total.provenance == Provenance::kMeasured);
  CHECK(base.memory.ram_total.value == 30e9);
  CHECK(base.memory.ram_total.source == bench_source(kRunMid, "g14"));
  CHECK(report.replaced == std::vector<std::string>{"memory.ram_total"});
  CHECK(base.memory.ram_bandwidth.provenance == Provenance::kSynthetic);
  CHECK(base.memory.ram_bandwidth.value == before_bw);
  CHECK(base.synthetic_fixture == false);
  CHECK(validate(base).is_ok());
}

TEST_CASE("merge: a Measured field is replaced only by a newer Measured run") {
  HardwareProfile base = fixture();
  merge_hardware_profile(base, incoming_with_ram(kRunMid, 30e9));
  SUBCASE("older run does not replace") {
    const auto report = merge_hardware_profile(base, incoming_with_ram(kRunOld, 31e9));
    CHECK(base.memory.ram_total.value == 30e9);
    CHECK(report.kept_newer == std::vector<std::string>{"memory.ram_total"});
    CHECK(report.replaced.empty());
  }
  SUBCASE("newer run replaces") {
    const auto report = merge_hardware_profile(base, incoming_with_ram(kRunNew, 29e9));
    CHECK(base.memory.ram_total.value == 29e9);
    CHECK(base.memory.ram_total.source == bench_source(kRunNew, "g14"));
    CHECK(report.replaced == std::vector<std::string>{"memory.ram_total"});
  }
  SUBCASE("the same run merged twice is idempotent") {
    merge_hardware_profile(base, incoming_with_ram(kRunMid, 30e9));
    CHECK(base.memory.ram_total.value == 30e9);
    CHECK(base.memory.ram_total.provenance == Provenance::kMeasured);
  }
}

TEST_CASE("merge: Qualified fields are never touched, and Synthetic input never replaces anything") {
  HardwareProfile base = fixture();
  base.memory.ram_total = Quantity{25e9, Provenance::kQualified, "bench:run-20250101T000000Z-zzzz@g14; qualified: review"};
  base.memory.ram_bandwidth = Quantity::measured(55e9, bench_source(kRunOld, "g14"));
  HardwareProfile in = incoming_with_ram(kRunNew, 30e9);  // ram_total measured, everything else Synthetic
  const auto report = merge_hardware_profile(base, in);
  CHECK(base.memory.ram_total.value == 25e9);
  CHECK(base.memory.ram_total.provenance == Provenance::kQualified);
  CHECK(report.kept_qualified == std::vector<std::string>{"memory.ram_total"});
  // incoming Synthetic ram_bandwidth (0) must not downgrade the existing Measured one
  CHECK(base.memory.ram_bandwidth.provenance == Provenance::kMeasured);
  CHECK(base.memory.ram_bandwidth.value == 55e9);
}

TEST_CASE("merge: provenance is never upgraded beyond Measured") {
  HardwareProfile base = fixture();
  HardwareProfile in = incoming_with_ram(kRunNew, 30e9);
  in.memory.ram_total.provenance = Provenance::kQualified;  // a hand-edited or misbehaving producer
  merge_hardware_profile(base, in);
  CHECK(base.memory.ram_total.provenance == Provenance::kMeasured);
  CHECK(weakest_provenance(base) == Provenance::kSynthetic);
}

TEST_CASE("merge: Measured values of unknown origin are kept rather than guessed about") {
  HardwareProfile base = fixture();
  base.memory.ram_total = Quantity::measured(26e9, "manual: tape measure");
  const auto report = merge_hardware_profile(base, incoming_with_ram(kRunNew, 30e9));
  CHECK(base.memory.ram_total.value == 26e9);
  CHECK(report.kept_unordered == std::vector<std::string>{"memory.ram_total"});
}

TEST_CASE("merge: new expert-throughput keys are added, existing ones follow the replacement rule") {
  HardwareProfile base = fixture();  // has q4_k (Synthetic)
  HardwareProfile in = incoming_with_ram(kRunMid, 30e9);
  in.cpu.expert_bytes_per_s.clear();
  in.cpu.expert_bytes_per_s["f32"] = Quantity::measured(5e9, bench_source(kRunMid, "g14"));
  in.cpu.expert_bytes_per_s["q4_k"] = Quantity::measured(8e9, bench_source(kRunMid, "g14"));
  in.cpu.expert_bytes_per_s["iq3_s"] = Quantity::synthetic(0, "synthetic: not measured by clusterlm-bench");
  const auto report = merge_hardware_profile(base, in);
  CHECK(base.cpu.expert_bytes_per_s.at("f32").provenance == Provenance::kMeasured);
  CHECK(base.cpu.expert_bytes_per_s.at("q4_k").value == 8e9);
  CHECK(base.cpu.expert_bytes_per_s.at("q4_k").provenance == Provenance::kMeasured);
  CHECK(base.cpu.expert_bytes_per_s.count("iq3_s") == 0);  // an unmeasured key is not invented
  CHECK(std::find(report.added.begin(), report.added.end(), "cpu.expert_bytes_per_s.f32") != report.added.end());
}

TEST_CASE("merge: inherited Synthetic placeholders are clamped so the merged profile stays valid") {
  HardwareProfile base = fixture();  // ram_safe_allowance 12 GiB synthetic, vram_budget synthetic
  HardwareProfile in = incoming_with_ram(kRunMid, 8e9);
  in.gpu.vram_total = Quantity::measured(6e9, bench_source(kRunMid, "g14"));
  merge_hardware_profile(base, in);
  CHECK(base.memory.ram_safe_allowance.value <= base.memory.ram_total.value);
  CHECK(base.gpu.vram_budget.value <= base.gpu.vram_total.value);
  CHECK(validate(base).is_ok());
}

TEST_CASE("build_measured_profile: Measured only where measured, Synthetic placeholders elsewhere, never Qualified") {
  HardwareMeasurements m = base_measurements();
  m.cpu = fake_cpu_report();
  MemoryMeasurement mem;
  mem.info.total_physical = 31ull << 30;
  ReadBandwidth bw1, bw2;
  bw1.threads = 1;
  bw1.bytes_per_s.samples = {20e9, 21e9, 22e9};
  bw2.threads = 4;
  bw2.bytes_per_s.samples = {60e9, 61e9, 62e9};
  mem.bandwidth = {bw1, bw2};
  m.memory = mem;

  HardwareProfile p = build_measured_profile(m);
  const std::string src = bench_source(kRunMid, "g14");
  CHECK(p.id == "g14");
  CHECK(p.role == DomainRole::kNode);
  CHECK(p.cpu.features == std::vector<std::string>{"avx2", "fma"});
  CHECK(p.cpu.expert_bytes_per_s.at("f32").value == doctest::Approx(10e9));
  CHECK(p.cpu.expert_bytes_per_s.at("f32").provenance == Provenance::kMeasured);
  CHECK(p.cpu.expert_bytes_per_s.at("f32").source == src);
  CHECK(p.cpu.expert_bytes_per_s.at("q8_0-fixture").value == doctest::Approx(2.2e9));
  CHECK(p.cpu.usable_threads.value == 5);  // largest over representations
  CHECK(p.cpu.q_scaling.value == doctest::Approx(0.7));
  CHECK(p.cpu.q_scaling.provenance == Provenance::kMeasured);
  CHECK(p.memory.ram_total.value == doctest::Approx(31.0 * (1ull << 30)));
  CHECK(p.memory.ram_bandwidth.value == doctest::Approx(61e9));
  CHECK(p.memory.ram_bandwidth.source == src);

  // Not measured: sustained factor (no run), safe allowance (policy), GPU fields (no GPU), overheads.
  CHECK(p.cpu.sustained_factor.provenance == Provenance::kSynthetic);
  CHECK(p.memory.ram_safe_allowance.provenance == Provenance::kSynthetic);
  CHECK(p.gpu.vram_total.provenance == Provenance::kSynthetic);
  CHECK(p.gpu.pcie_h2d_bytes_per_s.provenance == Provenance::kSynthetic);
  CHECK(p.overheads.os_reserve_ram.provenance == Provenance::kSynthetic);

  for_each_quantity(p, [&](const std::string& path, Quantity& q) {
    CAPTURE(path);
    CHECK(q.provenance != Provenance::kQualified);
    if (q.provenance == Provenance::kMeasured) CHECK(q.source == src);
    else CHECK(q.source.rfind("synthetic:", 0) == 0);
  });
  CHECK(validate(p).is_ok());
  // The profile survives a JSON round trip, which is how placement loads it.
  auto back = hardware_profile_from_json(to_json(p));
  REQUIRE_MESSAGE(back.is_ok(), back.status().to_string());
  CHECK(back->cpu.expert_bytes_per_s.at("f32").source == src);
  CHECK(weakest_provenance(back.value()) == Provenance::kSynthetic);
  CHECK(mark_qualified(back.value(), "x").is_ok() == false);  // still has Synthetic fields: cannot be qualified
}

TEST_CASE("build_measured_profile: the sustained factor needs a 30-minute run") {
  HardwareMeasurements m = base_measurements();
  SustainedReport s;
  s.minutes = 5;
  s.sustained_factor = 0.8;
  s.samples.push_back({1, 1, std::nullopt});
  m.sustained = s;
  CHECK(build_measured_profile(m).cpu.sustained_factor.provenance == Provenance::kSynthetic);
  m.sustained->minutes = 30;
  const auto p = build_measured_profile(m);
  CHECK(p.cpu.sustained_factor.provenance == Provenance::kMeasured);
  CHECK(p.cpu.sustained_factor.value == doctest::Approx(0.8));
  m.sustained->sustained_factor = 7.0;  // clamped into (0,1]
  CHECK(build_measured_profile(m).cpu.sustained_factor.value == 1.0);
}

TEST_CASE("build_measured_profile: GPU mapping uses DXGI budget when known and pinned PCIe bandwidth") {
  HardwareMeasurements m = base_measurements();
  GpuBenchReport g;
  g.available = true;
  GpuDeviceReport d;
  d.info.name = "RTX 4070 Laptop";
  d.info.total_bytes = 8ull << 30;
  d.info.free_bytes = 7ull << 30;
  TransferPoint pageable, pinned64, pinned256;
  pageable.kind = HostMemoryKind::kPageable;
  pageable.bytes = 256ull << 20;
  pageable.h2d_bytes_per_s.samples = {6e9};
  pinned64.kind = HostMemoryKind::kPinnedAlloc;
  pinned64.bytes = 64ull << 20;
  pinned64.h2d_bytes_per_s.samples = {20e9};
  pinned256.kind = HostMemoryKind::kPinnedAlloc;
  pinned256.bytes = 256ull << 20;
  pinned256.h2d_bytes_per_s.samples = {22e9, 23e9, 24e9};
  d.transfers = {pageable, pinned64, pinned256};
  d.pinned.largest_ok_bytes = 4ull << 30;
  g.devices.push_back(d);
  m.gpu = g;

  SUBCASE("without DXGI: free VRAM") {
    const auto p = build_measured_profile(m);
    CHECK(p.gpu.name == "RTX 4070 Laptop");
    CHECK(p.gpu.vram_total.value == doctest::Approx(8.0 * (1ull << 30)));
    CHECK(p.gpu.vram_budget.value == doctest::Approx(7.0 * (1ull << 30)));
    CHECK(p.gpu.pcie_h2d_bytes_per_s.value == doctest::Approx(23e9));
    CHECK(p.memory.pinned_limit.value == doctest::Approx(4.0 * (1ull << 30)));
    CHECK(p.gpu.gpu_expert_bytes_per_s.provenance == Provenance::kSynthetic);  // needs the Strata kernels
    CHECK(p.gpu.dense_layer_ms.at("recurrent").provenance == Provenance::kSynthetic);
    CHECK(validate(p).is_ok());
  }
  SUBCASE("with DXGI: the OS budget, capped at total") {
    m.dxgi_budget_bytes = {6ull << 30};
    CHECK(build_measured_profile(m).gpu.vram_budget.value == doctest::Approx(6.0 * (1ull << 30)));
    m.dxgi_budget_bytes = {64ull << 30};
    CHECK(build_measured_profile(m).gpu.vram_budget.value == doctest::Approx(8.0 * (1ull << 30)));
  }
  SUBCASE("an unavailable GPU measures nothing") {
    m.gpu->available = false;
    const auto p = build_measured_profile(m);
    CHECK(p.gpu.vram_total.provenance == Provenance::kSynthetic);
    CHECK(p.memory.pinned_limit.provenance == Provenance::kSynthetic);
  }
}

namespace {
LinkMeasure fake_link(double rtt_ms, double tx, double rx) {
  LinkMeasure l;
  for (std::uint32_t size : {64u, 1048576u}) {
    SizeMeasure s;
    s.size = size;
    for (double j : {-0.01, 0.0, 0.01, 0.02, -0.02}) s.rtt_ms.add(rtt_ms + j);
    s.tx_bytes_per_s = tx;
    s.rx_bytes_per_s = rx;
    l.sizes.push_back(s);
  }
  return l;
}
}  // namespace

TEST_CASE("network profile: Measured for real links (slower direction), Synthetic for loopback") {
  PeerLink a{"node-a", fake_link(0.4, 118e6, 112e6), true};
  PeerLink b{"node-b", fake_link(0.5, 117e6, 119e6), true};
  ConcurrentMeasure c;
  c.connections = 2;
  c.egress_bytes_per_s = 117e6;
  const auto net = build_network_profile("father", kRunMid, {a, b}, &c);
  REQUIRE(net.links.size() == 2);
  CHECK(net.links[0].from == "father");
  CHECK(net.links[0].to == "node-a");
  CHECK(net.links[0].bandwidth_bytes_per_s.value == doctest::Approx(112e6));  // min(tx, rx)
  CHECK(net.links[0].rtt_ms.value == doctest::Approx(0.4));
  CHECK(net.links[0].jitter_ms.value > 0);
  CHECK(net.links[0].rtt_ms.provenance == Provenance::kMeasured);
  CHECK(net.links[0].rtt_ms.source == bench_source(kRunMid, "father"));
  CHECK(net.father_egress_bytes_per_s.provenance == Provenance::kMeasured);
  CHECK(net.father_egress_bytes_per_s.value == 117e6);
  CHECK(validate(net).is_ok());
  CHECK_FALSE(net.synthetic_fixture);

  PeerLink loop{"sim", fake_link(0.05, 2e9, 2e9), false};
  const auto sim = build_network_profile("father", kRunMid, {loop}, nullptr);
  CHECK(weakest_provenance(sim) == Provenance::kSynthetic);
  CHECK(sim.synthetic_fixture);
  CHECK(sim.father_egress_bytes_per_s.provenance == Provenance::kSynthetic);

  // One real peer: link measured, concurrent egress not measured -> Synthetic.
  const auto single = build_network_profile("father", kRunMid, {a}, nullptr);
  CHECK(single.links[0].bandwidth_bytes_per_s.provenance == Provenance::kMeasured);
  CHECK(single.father_egress_bytes_per_s.provenance == Provenance::kSynthetic);
  CHECK(validate(single).is_ok());
}

TEST_CASE("network profile merge follows the same rules, including reversed link direction") {
  PeerLink a{"node-a", fake_link(0.4, 118e6, 112e6), true};
  auto base = build_network_profile("father", kRunMid, {a}, nullptr);
  // A newer run measured from the other end of the same link.
  PeerLink back{"father", fake_link(0.45, 100e6, 110e6), true};
  auto newer = build_network_profile("node-a", kRunNew, {back}, nullptr);
  merge_network_profile(base, newer);
  REQUIRE(base.links.size() == 1);
  CHECK(base.links[0].bandwidth_bytes_per_s.value == doctest::Approx(100e6));
  CHECK(base.links[0].rtt_ms.source == bench_source(kRunNew, "node-a"));
  // An older run does not replace the newer values.
  auto older = build_network_profile("father", kRunOld, {a}, nullptr);
  const auto report = merge_network_profile(base, older);
  CHECK(base.links[0].bandwidth_bytes_per_s.value == doctest::Approx(100e6));
  CHECK(report.kept_newer.size() == 3);
  // A Synthetic loopback measurement never replaces a Measured link.
  PeerLink loop{"node-a", fake_link(0.05, 2e9, 2e9), false};
  merge_network_profile(base, build_network_profile("father", kRunNew, {loop}, nullptr));
  CHECK(base.links[0].bandwidth_bytes_per_s.provenance == Provenance::kMeasured);
  // A new peer is appended.
  PeerLink b{"node-b", fake_link(0.5, 117e6, 119e6), true};
  merge_network_profile(base, build_network_profile("father", kRunNew, {b}, nullptr));
  CHECK(base.links.size() == 2);
}
