#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "commands.hpp"
#include "gpu_probe.hpp"
#include "schema_check.hpp"

using namespace clusterlm;
using namespace clusterlm::bench;
using nlohmann::json;

namespace {

// A GpuProbe returning canned numbers: lets the orchestration, metric emission and profile mapping be tested on
// a host without a GPU. Nothing here models real hardware.
class FakeProbe final : public GpuProbe {
 public:
  bool fail_gemm = false;
  int kernel_calls = 0;
  bool available() const override { return true; }
  std::string unavailable_reason() const override { return ""; }
  Result<std::vector<GpuDeviceInfo>> enumerate() override {
    GpuDeviceInfo d;
    d.name = "Fake GPU";
    d.total_bytes = 8ull << 30;
    d.free_bytes = 7ull << 30;
    d.compute_major = 8;
    d.compute_minor = 9;
    d.multiprocessors = 36;
    d.driver_version = 12040;
    d.runtime_version = 12000;
    return std::vector<GpuDeviceInfo>{d};
  }
  Result<std::vector<AllocOverheadPoint>> allocation_overhead(int, const std::vector<std::uint64_t>& sizes, unsigned reps) override {
    std::vector<AllocOverheadPoint> out;
    for (auto s : sizes) {
      AllocOverheadPoint p;
      p.bytes = s;
      for (unsigned i = 0; i < reps; ++i) {
        p.malloc_us.add(10);
        p.free_us.add(5);
      }
      out.push_back(p);
    }
    return out;
  }
  Result<std::vector<TransferPoint>> transfers(int, const std::vector<std::uint64_t>& rings, const std::vector<HostMemoryKind>& kinds,
                                               unsigned reps) override {
    std::vector<TransferPoint> out;
    for (auto k : kinds)
      for (auto r : rings) {
        TransferPoint p;
        p.bytes = r;
        p.kind = k;
        for (unsigned i = 0; i < reps; ++i) {
          p.h2d_bytes_per_s.add(k == HostMemoryKind::kPageable ? 6e9 : 22e9);
          p.d2h_bytes_per_s.add(k == HostMemoryKind::kPageable ? 5e9 : 21e9);
        }
        out.push_back(p);
      }
    return out;
  }
  Result<PinnedProbeResult> pinned_limit(int, std::uint64_t cap, std::uint64_t step) override {
    PinnedProbeResult r;
    r.cap_bytes = cap;
    r.largest_ok_bytes = cap / step * step;
    r.stopped_by_cap = true;
    return r;
  }
  Result<std::vector<GemmPoint>> gemm(int, const std::vector<GemmShape>& shapes, const std::vector<GemmPrecision>& precisions,
                                      unsigned reps) override {
    if (fail_gemm) return make_error(ErrorCode::kInternal, "cublas exploded");
    std::vector<GemmPoint> out;
    for (const auto& s : shapes)
      for (auto p : precisions) {
        GemmPoint g;
        g.shape = s;
        g.precision = p;
        for (unsigned i = 0; i < reps; ++i) g.ms.add(0.05);
        g.weight_bytes_per_s = 100e9;
        g.tflops = 1.5;
        out.push_back(g);
      }
    return out;
  }
  Result<std::vector<KernelTimePoint>> time_kernels(int, const std::vector<GpuKernelCase>& cases, unsigned reps) override {
    std::vector<KernelTimePoint> out;
    for (const auto& c : cases) {
      ++kernel_calls;
      KernelTimePoint k;
      k.name = c.name;
      for (unsigned i = 0; i < reps; ++i) k.ms.add(0.2);
      out.push_back(k);
    }
    return out;
  }
};

HostInfo host() {
  HostInfo h;
  h.os = "test";
  h.cpu_brand = "test";
  h.arch = "x86-64";
  return h;
}

json minimal_result() {
  BenchmarkResult r("dev-test", host());
  r.set_measured();
  r.metric("a", 1.0);
  return r.finish(0.1);
}

}  // namespace

TEST_CASE("this build without CUDA reports why, with every call failing cleanly") {
  auto probe = make_gpu_probe();
  if (gpu_probe_built_with_cuda()) {
    // A CUDA build on a host with no device must also report cleanly (no crash).
    if (!probe->available()) CHECK_FALSE(probe->unavailable_reason().empty());
    return;
  }
  CHECK_FALSE(probe->available());
  CHECK(probe->unavailable_reason().find("CLUSTERLM_BENCH_CUDA") != std::string::npos);
  CHECK(probe->enumerate().status().code() == ErrorCode::kHardwareUnavailable);
  const auto rep = run_gpu_bench(*probe, GpuBenchOptions{});
  CHECK_FALSE(rep.available);
  CHECK(rep.devices.empty());
  BenchmarkResult r("dev-gpu", host());
  emit_gpu_metrics(r, rep);
  const auto doc = r.finish(0);
  CHECK(doc["metrics"]["gpu.present"] == false);
  CHECK(doc["metrics"].contains("gpu.unavailable_reason"));
}

TEST_CASE("GPU orchestration over a probe: every sub-measurement, transfers pageable vs pinned, notes for failures") {
  FakeProbe probe;
  GpuKernelRegistry::instance().clear();
  GpuKernelRegistry::instance().register_case({"strata_expert_gemv", nullptr, [](void*) { return Status::ok(); }, nullptr});
  GpuBenchOptions o;
  o.ring_bytes = {64ull << 20, 256ull << 20};
  o.alloc_sizes = {1ull << 20};
  o.pinned_cap_bytes = 1ull << 30;
  o.pinned_step_bytes = 256ull << 20;
  o.reps = 5;
  auto rep = run_gpu_bench(probe, o);
  REQUIRE(rep.available);
  REQUIRE(rep.devices.size() == 1);
  const auto& d = rep.devices[0];
  CHECK(d.info.name == "Fake GPU");
  CHECK(d.alloc.size() == 1);
  CHECK(d.transfers.size() == 3 * 2);  // pageable, pinned alloc, registered x two rings
  CHECK(d.pinned.largest_ok_bytes == (1ull << 30));
  CHECK(d.gemm.size() == model_gemm_shapes(false).size() * 2);
  CHECK(d.kernels.size() == 1);
  CHECK(probe.kernel_calls == 1);
  CHECK(d.notes.empty());

  // Model-relevant shapes: Flash-Next expert, hidden square and attention projections at q=1/2/4 and prefill 512.
  bool expert_gate = false, square = false, gemv = false, prefill = false;
  for (const auto& s : model_gemm_shapes(false)) {
    expert_gate |= s.rows == 640 && s.cols == 2560;
    square |= s.rows == 2560 && s.cols == 2560;
    gemv |= s.batch == 1;
    prefill |= s.batch == 512;
  }
  CHECK((expert_gate && square && gemv && prefill));

  BenchmarkResult r("dev-gpu", host());
  r.set_measured();
  emit_gpu_metrics(r, rep);
  const auto doc = r.finish(0);
  CHECK(doc["metrics"]["gpu.present"] == true);
  CHECK(doc["metrics"].contains("gpu.0.pageable.67108864.h2d_bytes_per_s"));
  CHECK(doc["metrics"].contains("gpu.0.pinned.268435456.d2h_bytes_per_s"));
  CHECK(doc["metrics"].contains("gpu.0.kernel.strata_expert_gemv.ms"));
  CHECK(doc["metrics"].contains("gpu.0.gemm.f32.expert_gate_up.n1.weight_bytes_per_s"));

  probe.fail_gemm = true;
  const auto failed = run_gpu_bench(probe, o);
  REQUIRE(failed.devices.size() == 1);
  CHECK(failed.devices[0].gemm.empty());
  REQUIRE(failed.devices[0].notes.size() == 1);  // a failed sub-measurement is reported, never silently dropped
  CHECK(failed.devices[0].notes[0].find("cublas exploded") != std::string::npos);
  GpuKernelRegistry::instance().clear();
}

TEST_CASE("schema validator: the result schema accepts what BenchmarkResult emits and rejects violations") {
  const json schema = load_result_schema(CLUSTERLM_SOURCE_DIR);
  REQUIRE_FALSE(schema.is_null());
  json ok = minimal_result();
  CHECK(validate_against_schema(ok, schema).empty());
  CHECK(validate_result_semantics(ok).empty());

  SUBCASE("missing required key") {
    json bad = ok;
    bad.erase("provenance");
    const auto errs = validate_against_schema(bad, schema);
    REQUIRE_FALSE(errs.empty());
    CHECK(errs[0].find("provenance") != std::string::npos);
  }
  SUBCASE("enum violation") {
    json bad = ok;
    bad["provenance"] = "Verified";
    CHECK_FALSE(validate_against_schema(bad, schema).empty());
    json bad_role = ok;
    bad_role["environment"]["host_role"] = "laptop";
    CHECK_FALSE(validate_against_schema(bad_role, schema).empty());
  }
  SUBCASE("const and type violations") {
    json bad = ok;
    bad["schema_version"] = 2;
    CHECK_FALSE(validate_against_schema(bad, schema).empty());
    json bad2 = ok;
    bad2["checks"] = json::array({{{"name", "x"}, {"passed", "yes"}}});
    CHECK_FALSE(validate_against_schema(bad2, schema).empty());
    json bad3 = ok;
    bad3["metrics"]["m"] = json::object({{"n", "three"}});
    CHECK_FALSE(validate_against_schema(bad3, schema).empty());
  }
  SUBCASE("metric forms: number, bool, string, list, distribution") {
    json good = ok;
    good["metrics"]["s"] = "text";
    good["metrics"]["b"] = true;
    good["metrics"]["l"] = json::array({"avx2", "fma"});
    Distribution d;
    for (double v : {1.0, 2.0, 3.0, 4.0, 5.0}) d.add(v);
    good["metrics"]["d"] = d.to_json("ms");
    CHECK(validate_against_schema(good, schema).empty());
  }
  SUBCASE("semantic rules the schema cannot express") {
    json q = ok;
    q["provenance"] = "Qualified";
    CHECK_FALSE(validate_result_semantics(q).empty());
    json sim = ok;
    sim["simulated"] = {{"loopback", true}};
    CHECK_FALSE(validate_result_semantics(sim).empty());  // simulated but labelled Measured
    BenchmarkResult r("dev-sim", host());
    r.set_measured();
    r.mark_simulated("loopback", true);
    const auto doc = r.finish(0);
    CHECK(doc["provenance"] == "Synthetic");
    CHECK(validate_result_semantics(doc).empty());
  }
}

TEST_CASE("every command line in the qualification registry parses against the implemented command surface") {
  std::ifstream f(std::string(CLUSTERLM_SOURCE_DIR) + "/bench/qualification/experiments.json");
  REQUIRE(f);
  const json reg = json::parse(f);
  int n = 0;
  for (const auto& e : reg["experiments"]) {
    const std::string id = e["id"].get<std::string>();
    const std::string cmd = e["command"].get<std::string>();
    INFO(id << ": " << cmd);
    CHECK(validate_registry_command(cmd) == "");
    ++n;
  }
  CHECK(n >= 20);
  // The checker itself rejects what it should.
  CHECK(validate_registry_command("clusterlm-bench cpu --nonsense 1") != "");
  CHECK(validate_registry_command("clusterlm-bench nosuch") != "");
  CHECK(validate_registry_command("clusterlm-bench domain wrong-sub --backend strata") != "");
  CHECK(validate_registry_command("other-tool cpu") != "");
  CHECK(validate_registry_command("clusterlm-bench cpu --provider strata-cpu --q 1,2,4 --out <path>") == "");
  CHECK(validate_registry_command("python3 scripts/fetch_upstream.py --apply-patches strata && cmake --build build && build/bin/clusterlm-strata probe") == "");
  CHECK(validate_registry_command("for t in 1 2; do STRATA_FORCE_AVX2=1 build/bin/clusterlm-bench cpu --threads $t; done") == "");
  CHECK(validate_registry_command("python3 evil.py") != "");
  CHECK(validate_registry_command("rm -rf / && clusterlm-bench cpu") != "");
}

TEST_CASE("results emitted by the smoke commands validate against the schema") {
#ifdef CLUSTERLM_BENCH_RESULTS_DIR
  const json schema = load_result_schema(CLUSTERLM_SOURCE_DIR);
  REQUIRE_FALSE(schema.is_null());
  int seen = 0;
  for (const auto& entry : std::filesystem::directory_iterator(CLUSTERLM_BENCH_RESULTS_DIR)) {
    if (entry.path().extension() != ".json") continue;
    const std::string name = entry.path().filename().string();
    if (name.rfind("result-", 0) != 0) continue;  // only bench result documents (profiles are checked elsewhere)
    std::ifstream in(entry.path());
    const json doc = json::parse(in, nullptr, false);
    INFO(name);
    REQUIRE_FALSE(doc.is_discarded());
    const auto errs = validate_against_schema(doc, schema);
    CHECK_MESSAGE(errs.empty(), (errs.empty() ? "" : errs.front()));
    const auto sem = validate_result_semantics(doc);
    CHECK_MESSAGE(sem.empty(), (sem.empty() ? "" : sem.front()));
    ++seen;
  }
  CHECK(seen >= 6);
#endif
}
