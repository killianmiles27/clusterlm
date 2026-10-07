// WP20: OS/driver observers around a bench run (process memory series, NIC counters, power plan/governor, NVML
// telemetry, cache census) and the command-surface additions (faults --only, cluster --contexts, inter-token gaps).
#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/transport/transport.hpp"
#include "commands.hpp"
#include "nvml_probe.hpp"
#include "schema_check.hpp"
#include "storage_census.hpp"
#include "system_probe.hpp"

using namespace clusterlm;
using namespace clusterlm::bench;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

HostInfo test_host() {
  HostInfo h;
  h.os = "test";
  h.cpu_brand = "test";
  return h;
}

fs::path scratch(const std::string& name) {
  const fs::path p = fs::temp_directory_path() / ("clm-wp20-" + name + "-" + std::to_string(monotonic_ns()));
  fs::create_directories(p);
  return p;
}

void write_file(const fs::path& p, std::size_t bytes) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f << std::string(bytes, 'x');
}

}  // namespace

// ---- process memory ---------------------------------------------------------------------------------------

TEST_CASE("process status text parses RSS and the commit proxy, and a missing VmRSS is an error") {
  const std::string text = "Name:\tx\nVmSize:\t  9000 kB\nVmRSS:\t  1234 kB\nVmData:\t  5678 kB\nThreads:\t3\n";
  auto m = parse_proc_status_memory(text);
  REQUIRE(m.is_ok());
  CHECK(m->rss_bytes == 1234ull * 1024);
  CHECK(m->commit_bytes == 5678ull * 1024);
  CHECK_FALSE(parse_proc_status_memory("Name:\tkthreadd\nThreads:\t1\n").is_ok());
}

TEST_CASE("this process reports a plausible RSS; a pid that does not exist is an error, not zero") {
  auto self = sample_process_memory(0);
  REQUIRE_MESSAGE(self.is_ok(), self.status().to_string());
  CHECK(self->rss_bytes > 1024 * 1024);
  CHECK(self->commit_bytes > 0);
  CHECK_FALSE(sample_process_memory(0x7ffffff0).is_ok());
}

TEST_CASE("ResourceSampler records series per process, throttles per-round samples and survives a dead process") {
  ResourceSampler s;
  std::int64_t pid = 0;
  s.add_process("father", [] { return std::int64_t{0}; });
  s.add_process("node0", [&pid] { return pid; });
  pid = 0x7ffffff0;  // a pid that does not exist: its samples are null, not zero
  s.sample("start");
  s.sample("cycle");
  s.sample_throttled("round", 1e9);  // far inside the interval: not recorded
  CHECK(s.samples() == 2);
  s.sample_throttled("round", 0);    // zero spacing: always recorded
  CHECK(s.samples() == 3);

  BenchmarkResult r("dev-test", test_host());
  s.emit(r, "resources.");
  const json doc = r.finish(0);
  const auto& m = doc["metrics"];
  CHECK(m["resources.samples"] == 3);
  CHECK(m["resources.father.rss_bytes.series"].size() == 3);
  CHECK(m["resources.father.rss_bytes"]["n"] == 3);
  CHECK(m.contains("resources.father.rss_growth_bytes"));
  CHECK(m["resources.node0.unavailable"].get<std::string>().rfind("unavailable:", 0) == 0);
  CHECK(m["resources.phase.series"][0] == "start");
  const json schema = load_result_schema(CLUSTERLM_SOURCE_DIR);
  REQUIRE_FALSE(schema.is_null());
  const auto errs = validate_against_schema(doc, schema);
  CHECK_MESSAGE(errs.empty(), (errs.empty() ? std::string() : errs.front()));
}

TEST_CASE("ResourceSampler decimates long series to the cap and says so") {
  ResourceSampler s;
  s.add_process("self", [] { return std::int64_t{0}; });
  for (std::size_t i = 0; i < ResourceSampler::kMaxSeriesPoints * 2 + 10; ++i) s.sample("round");
  BenchmarkResult r("dev-test", test_host());
  s.emit(r, "r.");
  const json doc = r.finish(0);
  CHECK(doc["metrics"]["r.series_stride"] == 3);
  CHECK(doc["metrics"]["r.self.rss_bytes.series"].size() <= ResourceSampler::kMaxSeriesPoints);
  CHECK(doc["metrics"]["r.self.rss_bytes"]["n"] == ResourceSampler::kMaxSeriesPoints * 2 + 10);  // the distribution keeps every sample
}

// ---- NIC counters -----------------------------------------------------------------------------------------

namespace {
const char* kNetDev =
    "Inter-|   Receive                                                |  Transmit\n"
    " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
    "    lo:  1000      10    0    0    0     0          0         0     2000      20    1    2    0     0       0          0\n"
    "  eth0: 50000     400    3    4    0     0          0         0    60000     500    5    6    0     0       0          0\n";
}

TEST_CASE("/proc/net/dev parsing and counter deltas") {
  auto all = parse_proc_net_dev(kNetDev);
  REQUIRE(all.is_ok());
  REQUIRE(all->size() == 2);
  CHECK(all->at(1).name == "eth0");
  CHECK(all->at(1).rx_bytes == 50000);
  CHECK(all->at(1).tx_bytes == 60000);
  CHECK(all->at(1).rx_errors == 3);
  CHECK(all->at(1).tx_dropped == 6);
  NicCounters before = all->at(1), after = before;
  after.rx_bytes += 700;
  after.tx_bytes += 90;
  bool wrapped = false;
  auto d = diff_nic(before, after, &wrapped);
  CHECK(d.rx_bytes == 700);
  CHECK(d.tx_bytes == 90);
  CHECK_FALSE(wrapped);
  after.rx_bytes = 1;  // went backwards (counter reset)
  d = diff_nic(before, after, &wrapped);
  CHECK(d.rx_bytes == 0);
  CHECK(wrapped);
  CHECK_FALSE(parse_proc_net_dev("h1\nh2\n  eth0: 1 2 3\n").is_ok());
}

TEST_CASE("the route to a host picks the longest matching prefix, then the lowest metric") {
  // Destination/Mask are hex of the network-order bytes as printed on a little-endian host (0100A8C0 = 192.168.0.1).
  const std::string route =
      "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
      "eth0\t00000000\t0100A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0\n"
      "wlan0\t00000000\t0100A8C0\t0003\t0\t0\t50\t00000000\t0\t0\t0\n"
      "eth1\t0000A8C0\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n"
      "down0\t0000A8C0\t00000000\t0000\t0\t0\t0\t00FFFFFF\t0\t0\t0\n";
  CHECK(parse_route_interface(route, "192.168.7.9").value() == "eth1");   // /16, beats both defaults
  CHECK(parse_route_interface(route, "8.8.8.8").value() == "wlan0");      // default routes: lower metric wins
  CHECK(parse_route_interface(route, "127.0.0.1").value() == "lo");       // loopback is not in the table
  CHECK_FALSE(parse_route_interface(route, "not-an-ip").is_ok());
  CHECK_FALSE(parse_route_interface("Iface\tDestination\n", "10.0.0.1").is_ok());
}

TEST_CASE("NicWindow: unavailable interfaces are reported with a reason, never as zero counters") {
  NicWindow bad("no-such-nic-0", "");
  CHECK_FALSE(bad.available());
  BenchmarkResult r("dev-test", test_host());
  bad.begin();
  bad.end();
  bad.emit(r, "p.");
  const json doc = r.finish(0);
  CHECK(doc["metrics"]["p.nic.unavailable"].get<std::string>().rfind("unavailable:", 0) == 0);
  CHECK_FALSE(doc["metrics"].contains("p.nic.rx_bytes"));

  NicWindow no_peer("", "");  // no interface named and no Node to route to
  CHECK_FALSE(no_peer.available());
}

#ifndef _WIN32
TEST_CASE("NicWindow on the loopback route reads real OS counters that move with traffic") {
  NicWindow w("", "127.0.0.1");
  REQUIRE_MESSAGE(w.available(), w.unavailable_reason());
  CHECK(w.nic() == "lo");
  CHECK(default_nic_for("localhost").value() == "lo");
  w.begin();
  auto before = read_nic_counters("lo");
  REQUIRE(before.is_ok());
  // Real loopback traffic: a connected TCP pair moving some bytes.
  {
    transport::SecurityConfig sec;
    sec.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
    auto listener = transport::listen(transport::Endpoint{"127.0.0.1", 0}, sec);
    REQUIRE(listener.is_ok());
    auto client = transport::connect(listener.value()->local_endpoint(), sec, std::nullopt, std::chrono::seconds(5));
    REQUIRE(client.is_ok());
    auto server = listener.value()->accept(std::chrono::seconds(5));
    REQUIRE(server.is_ok());
    transport::Frame f;
    f.type = 1;
    f.channel = 1;
    f.payload.assign(256 * 1024, 7);
    REQUIRE(client.value()->send(f).is_ok());
    REQUIRE(server.value()->receive(std::chrono::seconds(5)).is_ok());
  }
  w.end();
  CHECK(w.delta().tx_bytes >= 256 * 1024);
  CHECK(w.delta().rx_bytes >= 256 * 1024);
}
#endif

// ---- power plan / governor --------------------------------------------------------------------------------

TEST_CASE("governors reduce to one value or a sorted mixed list") {
  CHECK(summarize_governors({}) == "");
  CHECK(summarize_governors({"performance", "performance"}) == "performance");
  CHECK(summarize_governors({"powersave", "performance", "powersave"}) == "mixed:performance,powersave");
}

TEST_CASE("the power environment is read where the OS offers it and explains itself where it does not") {
  const PowerEnvironment p = probe_power_environment();
  // Either something was read, or the reason says why not (a VM without cpufreq is the usual Linux CI case).
  CHECK((p.plan_name || p.plan_guid || p.governor || !p.unavailable.empty()));
  BenchmarkResult r("dev-test", test_host());
  emit_power_environment(r, p);
  PowerEnvironment fake;
  fake.plan_name = "Balanced";
  fake.plan_guid = "{381b4222-f694-41f0-9685-ff5bb260df2e}";
  CHECK(fake.summary() == "plan 'Balanced' {381b4222-f694-41f0-9685-ff5bb260df2e}");
  PowerEnvironment gov;
  gov.governor = "performance";
  CHECK(gov.summary() == "governor 'performance'");
  PowerEnvironment none;
  none.unavailable = "no cpufreq";
  CHECK(none.summary() == "unavailable: no cpufreq");
}

// ---- NVML -------------------------------------------------------------------------------------------------

namespace {
NvmlApi fake_nvml(unsigned devices, int init_rc = 0) {
  NvmlApi api;
  api.init = [init_rc] { return init_rc; };
  api.shutdown = [] { return 0; };
  api.device_count = [devices](unsigned* n) {
    *n = devices;
    return 0;
  };
  api.device_by_index = [](unsigned i, NvmlApi::Device* d) {
    *d = reinterpret_cast<NvmlApi::Device>(static_cast<std::uintptr_t>(i + 1));
    return 0;
  };
  api.device_name = [](NvmlApi::Device, char* buf, unsigned n) {
    std::snprintf(buf, n, "Fake GPU");
    return 0;
  };
  auto tick = std::make_shared<int>(0);
  api.clock_mhz = [tick](NvmlApi::Device, int type, unsigned* mhz) {
    *mhz = type == 1 ? 2000u - static_cast<unsigned>(*tick) * 100u : 9000u;
    return 0;
  };
  api.power_mw = [](NvmlApi::Device, unsigned* mw) {
    *mw = 85'000;
    return 0;
  };
  api.temperature_c = [tick](NvmlApi::Device, unsigned* c) {
    *c = 60u + static_cast<unsigned>((*tick)++);
    return 0;
  };
  api.memory = [](NvmlApi::Device, NvmlMemory* m) {
    m->total = 8ull << 30;
    m->used = 3ull << 30;
    m->free = 5ull << 30;
    return 0;
  };
  api.throttle_reasons = [tick](NvmlApi::Device, unsigned long long* mask) {
    *mask = *tick >= 2 ? 0x4 : 0x1;  // idle, then a software power cap
    return 0;
  };
  api.driver_version = [](char* buf, unsigned n) {
    std::snprintf(buf, n, "999.99");
    return 0;
  };
  api.error_string = [](int rc) { return std::string("fake error ") + std::to_string(rc); };
  return api;
}
}  // namespace

TEST_CASE("NVML throttle reason bits have names and the slowdown subset is identified") {
  CHECK(throttle_reason_names(0).empty());
  CHECK(throttle_reason_names(0x4 | 0x40) == std::vector<std::string>{"sw_power_cap", "hw_thermal_slowdown"});
  CHECK(throttle_reason_names(0x1000).front() == "0x1000");
  CHECK_FALSE(throttle_is_slowdown(0x1));  // idle
  CHECK_FALSE(throttle_is_slowdown(0x2));  // application clocks
  CHECK(throttle_is_slowdown(0x4));
  CHECK(throttle_is_slowdown(0x20));
}

TEST_CASE("NVML absent: loading reports why, opening fails as hardware-unavailable, the result says unavailable") {
  auto api = load_nvml_api("/nonexistent/libnvidia-ml.so.1");
  REQUIRE_FALSE(api.is_ok());
  CHECK(api.status().code() == ErrorCode::kHardwareUnavailable);
  CHECK(api.status().message().find("not found") != std::string::npos);

  auto failed_init = NvmlTelemetry::open_with(fake_nvml(1, /*init_rc=*/9));
  REQUIRE_FALSE(failed_init.is_ok());
  CHECK(failed_init.status().code() == ErrorCode::kHardwareUnavailable);
  CHECK(failed_init.status().message().find("fake error 9") != std::string::npos);
  CHECK_FALSE(NvmlTelemetry::open_with(fake_nvml(0)).is_ok());  // a driver with no devices

  BenchmarkResult r("dev-test", test_host());
  emit_gpu_telemetry_unavailable(r, api.status().message(), "nvml.");
  const json doc = r.finish(0);
  const std::string text = doc["metrics"]["nvml.unavailable"].get<std::string>();
  CHECK(text.rfind("unavailable: ", 0) == 0);
}

TEST_CASE("NVML present (fake function table): samples, recorder, series, throttle summary, schema-valid output") {
  auto opened = NvmlTelemetry::open_with(fake_nvml(2));
  REQUIRE_MESSAGE(opened.is_ok(), opened.status().to_string());
  auto& nvml = *opened.value();
  CHECK(nvml.device_names().size() == 2);
  CHECK(nvml.driver_version() == "999.99");
  auto one = nvml.sample();
  REQUIRE(one.is_ok());
  REQUIRE(one->size() == 2);
  CHECK(one->at(0).power_w.value() == doctest::Approx(85.0));
  CHECK(one->at(0).mem_used_bytes.value() == (3ull << 30));

  TelemetryRecorder rec(nvml, 0.01);
  rec.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  auto samples = rec.stop();
  CHECK(samples.size() >= 4);  // >= 2 devices x (>= 1 interval sample + the final one)

  BenchmarkResult r("dev-test", test_host());
  emit_gpu_telemetry(r, nvml, samples, "nvml.");
  const json doc = r.finish(0);
  const auto& m = doc["metrics"];
  CHECK(m["nvml.driver_version"] == "999.99");
  CHECK(m["nvml.gpu.0.name"] == "Fake GPU");
  CHECK(m["nvml.gpu.1.power_w"]["p50"] == doctest::Approx(85.0));
  CHECK(m["nvml.gpu.0.sm_clock_mhz.series"].size() >= 2);
  CHECK(m["nvml.gpu.0.mem_used_bytes.series"].size() >= 2);
  CHECK(m["nvml.gpu.0.mem_total_bytes"] == (8ull << 30));
  CHECK(m["nvml.gpu.0.slowdown_sample_fraction"].get<double>() > 0);
  bool power_cap = false;
  for (const auto& n : m["nvml.gpu.0.throttle_reasons_seen"]) power_cap |= n == "sw_power_cap";
  CHECK(power_cap);
  const json schema = load_result_schema(CLUSTERLM_SOURCE_DIR);
  REQUIRE_FALSE(schema.is_null());
  const auto errs = validate_against_schema(doc, schema);
  CHECK_MESSAGE(errs.empty(), (errs.empty() ? std::string() : errs.front()));

  // VRAM rides along in the resource sampler.
  ResourceSampler rs;
  rs.add_process("self", [] { return std::int64_t{0}; });
  rs.set_gpu(&nvml);
  rs.sample("cycle");
  rs.sample("cycle");
  BenchmarkResult r2("dev-test", test_host());
  rs.emit(r2, "res.");
  const json d2 = r2.finish(0);
  CHECK(d2["metrics"]["res.gpu0.vram_used_bytes"]["p50"] == doctest::Approx(static_cast<double>(3ull << 30)));
}

TEST_CASE("an NVML counter that fails becomes nullopt; a device with every counter failing is an error") {
  NvmlApi api = fake_nvml(1);
  api.power_mw = [](NvmlApi::Device, unsigned*) { return 3; };
  auto t = NvmlTelemetry::open_with(api);
  REQUIRE(t.is_ok());
  auto s = t.value()->sample();
  REQUIRE(s.is_ok());
  CHECK_FALSE(s->at(0).power_w.has_value());
  CHECK(s->at(0).sm_clock_mhz.has_value());

  NvmlApi dead = fake_nvml(1);
  dead.clock_mhz = [](NvmlApi::Device, int, unsigned*) { return 3; };
  dead.power_mw = [](NvmlApi::Device, unsigned*) { return 3; };
  dead.temperature_c = [](NvmlApi::Device, unsigned*) { return 3; };
  dead.memory = [](NvmlApi::Device, NvmlMemory*) { return 3; };
  dead.throttle_reasons = [](NvmlApi::Device, unsigned long long*) { return 3; };
  auto d = NvmlTelemetry::open_with(dead);
  REQUIRE(d.is_ok());
  CHECK_FALSE(d.value()->sample().is_ok());
}

// ---- cache / temp census ----------------------------------------------------------------------------------

TEST_CASE("census roots follow the platform conventions") {
  CensusEnvironment win;
  win.windows = true;
  win.vars = {{"APPDATA", "C:/Users/u/AppData/Roaming"}, {"LOCALAPPDATA", "C:/Users/u/AppData/Local"}, {"TEMP", "C:/Temp"}};
  std::map<std::string, std::string> label_path;
  for (const auto& r : census_roots_for(win)) label_path[r.label] = r.path.generic_string();
  CHECK(label_path["cuda_compute_cache"] == "C:/Users/u/AppData/Roaming/NVIDIA/ComputeCache");
  CHECK(label_path["nvidia_dx_cache"] == "C:/Users/u/AppData/Local/NVIDIA/DXCache");
  CHECK(label_path["clusterlm_local_appdata"] == "C:/Users/u/AppData/Local/ClusterLM");
  CHECK(label_path["os_temp"] == "C:/Temp");

  CensusEnvironment lin;
  lin.vars = {{"HOME", "/home/u"}, {"TMPDIR", "/var/tmp"}};
  label_path.clear();
  for (const auto& r : census_roots_for(lin)) label_path[r.label] = r.path.generic_string();
  CHECK(label_path["cuda_compute_cache"] == "/home/u/.nv/ComputeCache");
  CHECK(label_path["os_temp"] == "/var/tmp");
  lin.vars["CUDA_CACHE_PATH"] = "/data/cuda";  // the CUDA override wins
  for (const auto& r : census_roots_for(lin))
    if (r.label == "cuda_compute_cache") CHECK(r.path.generic_string() == "/data/cuda");
}

TEST_CASE("census diff reports new, changed and removed files by name and size, attributable to the window") {
  const fs::path root = scratch("census");
  write_file(root / "keep.bin", 100);
  write_file(root / "grow.bin", 100);
  write_file(root / "gone.bin", 10);
  write_file(root / "sub" / "deep.bin", 5);
  write_file(root / "skipme" / "ignored.bin", 5);
  CensusRoot cr;
  cr.label = "test";
  cr.path = root;
  cr.exclude = {root / "skipme"};
  CensusRoot absent;
  absent.label = "missing";
  absent.path = root / "does-not-exist";

  const auto before = take_census({cr, absent});
  REQUIRE(before.roots.size() == 2);
  CHECK(before.roots[0].files.size() == 4);  // skipme is excluded
  CHECK(before.roots[0].files.count("sub/deep.bin") == 1);
  CHECK_FALSE(before.roots[1].exists);

  write_file(root / "new.cache", 4096);
  write_file(root / "sub" / "new2.cache", 8);
  write_file(root / "grow.bin", 300);
  fs::remove(root / "gone.bin");
  write_file(root / "skipme" / "also-ignored.bin", 9);
  const auto after = take_census({cr, absent});
  const auto diff = diff_census(before, after, before.taken_ns, after.taken_ns);
  CHECK(diff.added.size() == 2);
  CHECK(diff.added_bytes == 4096 + 8);
  REQUIRE(diff.changed.size() == 1);
  CHECK(diff.changed[0].name == "grow.bin");
  CHECK(diff.changed[0].size_before == 100);
  CHECK(diff.changed[0].size_after == 300);
  CHECK(diff.changed[0].in_window);
  REQUIRE(diff.removed.size() == 1);
  CHECK(diff.removed[0].name == "gone.bin");
  CHECK(diff.absent_roots == std::vector<std::string>{"missing"});
  CHECK(diff.bytes_by_root.at("test") == 4096 + 8 + 300);

  // A window that ended before the change excludes a modified file from attribution (new files always count).
  const auto early = diff_census(before, after, 1, 2);
  REQUIRE(early.changed.size() == 1);
  CHECK_FALSE(early.changed[0].in_window);

  // JSON round trip keeps names, sizes and mtimes (and only those).
  auto back = census_from_json(census_to_json(after));
  REQUIRE(back.is_ok());
  CHECK(diff_census(after, back.value()).added.empty());
  CHECK(diff_census(after, back.value()).changed.empty());
  CHECK_FALSE(census_from_json(json::array()).is_ok());

  BenchmarkResult r("dev-test", test_host());
  emit_census_diff(r, diff, "x.");
  const json doc = r.finish(0);
  CHECK(doc["metrics"]["x.census.added_files"] == 2);
  CHECK(doc["metrics"]["x.census.diff"]["changed"][0]["name"] == "grow.bin");
  const json schema = load_result_schema(CLUSTERLM_SOURCE_DIR);
  const auto errs = validate_against_schema(doc, schema);
  CHECK_MESSAGE(errs.empty(), (errs.empty() ? std::string() : errs.front()));
  std::error_code ec;
  fs::remove_all(root, ec);
}

TEST_CASE("census honours depth and entry caps and says when it truncated") {
  const fs::path root = scratch("caps");
  for (int i = 0; i < 12; ++i) write_file(root / ("f" + std::to_string(i)), 1);
  write_file(root / "a" / "b" / "c" / "deep", 1);
  CensusRoot shallow;
  shallow.label = "shallow";
  shallow.path = root;
  shallow.max_depth = 2;
  CensusRoot capped = shallow;
  capped.label = "capped";
  capped.max_depth = 8;
  capped.max_entries = 5;
  const auto snap = take_census({shallow, capped});
  CHECK(snap.roots[0].files.count("a/b/c/deep") == 0);  // below the depth limit
  CHECK_FALSE(snap.roots[0].truncated);
  CHECK(snap.roots[1].files.size() == 5);
  CHECK(snap.roots[1].truncated);
  const auto diff = diff_census(snap, snap);
  CHECK(diff.truncated_roots == std::vector<std::string>{"capped"});
  std::error_code ec;
  fs::remove_all(root, ec);
}

// ---- faults --only ----------------------------------------------------------------------------------------

TEST_CASE("faults --only: scenarios, phases and the crash group select; unknown names are rejected") {
  const auto all = fault_scenario_names();
  CHECK(all.size() == 14);
  CHECK(select_fault_scenarios("").value() == all);
  CHECK(select_fault_scenarios("release_cycles").value() == std::vector<std::string>{"release_cycles"});
  // A bare lifecycle phase selects its crash scenario; output order is canonical, duplicates collapse.
  CHECK(select_fault_scenarios("stall,hashing,transfer,hashing").value() ==
        std::vector<std::string>{"crash_transfer", "crash_hashing", "stall"});
  CHECK(select_fault_scenarios("crash").value().size() == 9);
  CHECK(select_fault_scenarios("crash_ready,father_lost").value() == std::vector<std::string>{"crash_ready", "father_lost"});
  auto bad = select_fault_scenarios("stall,nonsense");
  REQUIRE_FALSE(bad.is_ok());
  CHECK(bad.status().code() == ErrorCode::kInvalidArgument);
  CHECK(bad.status().message().find("nonsense") != std::string::npos);
  CHECK_FALSE(select_fault_scenarios(",").is_ok());
}

// ---- command surface --------------------------------------------------------------------------------------

TEST_CASE("the new flags and commands are in the command spec registry") {
  CHECK(validate_registry_command("clusterlm-bench faults --only crash_ready,stall --release-cycles 3").empty());
  CHECK(validate_registry_command("clusterlm-bench cluster --tier ultra --contexts 8192,32768 --nic eth0 --census").empty());
  CHECK(validate_registry_command("clusterlm-bench nvml --minutes 30 --sample-s 5").empty());
  CHECK(validate_registry_command("clusterlm-bench storage-census --before --snapshot s.json").empty());
  CHECK(validate_registry_command("clusterlm-bench storage-census --after --snapshot s.json --out r.json").empty());
  CHECK_FALSE(validate_registry_command("clusterlm-bench faults --no-such-flag").empty());
}

TEST_CASE("Distribution reports p90 (inter-token gap summary)") {
  Distribution d;
  for (int i = 1; i <= 100; ++i) d.add(i);
  const auto j = d.to_json("ms");
  CHECK(j["p50"].get<double>() == doctest::Approx(50.5));
  CHECK(j["p90"].get<double>() == doctest::Approx(90.1));
  CHECK(j["p99"].get<double>() == doctest::Approx(99.01));
  CHECK(j["max"] == 100);
}

// ---- outputs of the smoke commands (tests/bench/CMakeLists.txt) ------------------------------------------------

#ifdef CLUSTERLM_BENCH_RESULTS_DIR
namespace {
json load_result(const char* name) {
  std::ifstream in(std::string(CLUSTERLM_BENCH_RESULTS_DIR) + "/" + name);
  REQUIRE_MESSAGE(in, "missing " << name);
  return json::parse(in);
}
void check_schema(const json& doc) {
  const json schema = load_result_schema(CLUSTERLM_SOURCE_DIR);
  REQUIRE_FALSE(schema.is_null());
  const auto errs = validate_against_schema(doc, schema);
  CHECK_MESSAGE(errs.empty(), (errs.empty() ? std::string() : errs.front()));
}
}  // namespace

TEST_CASE("cluster --contexts: one invocation, results keyed by context, observers and gap distribution recorded") {
  const json doc = load_result("result-cluster-contexts.json");
  check_schema(doc);
  CHECK(doc["provenance"] == "Synthetic");
  CHECK(doc["configuration"]["context_sweep"] == true);
  const auto& m = doc["metrics"];
  for (const char* ctx : {"ctx16", "ctx32"})
    for (const char* q : {"q1", "q2"})
      for (const char* k : {"decode_tok_s", "prefill_tok_s", "round_ms", "inter_token_gap_ms", "inter_delivery_gap_ms"}) {
        const std::string key = std::string(ctx) + "." + q + "." + k;
        CHECK_MESSAGE(m.contains(key), key);
      }
  const auto& gap = m["ctx32.q2.inter_token_gap_ms"];
  for (const char* f : {"p50", "p90", "p99", "max"}) CHECK_MESSAGE(gap.contains(f), f);
  CHECK(gap["n"].get<int>() > 0);
  CHECK(gap["max"].get<double>() >= gap["p99"].get<double>());
  // Per-context checks are keyed too, and all of them passed.
  bool saw_ctx32_check = false;
  for (const auto& c : doc["checks"]) {
    CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>());
    saw_ctx32_check |= c["name"].get<std::string>().rfind("ctx32.q2_tokens_match", 0) == 0;
  }
  CHECK(saw_ctx32_check);
  // Process series: Father and both Nodes of the strong tier (one Node), RSS and commit, with a series and growth.
  for (const char* k : {"resources.father.rss_bytes.series", "resources.node0.rss_bytes.series", "resources.node0.commit_bytes",
                        "resources.father.commit_growth_bytes", "resources.t_s.series"})
    CHECK_MESSAGE(m.contains(k), k);
  CHECK(m["resources.samples"].get<int>() >= 4 * 2);  // a start and an end per generation
  // NIC counters beside the boundary payload counters (loopback here); on an NVML-less host the reason is recorded.
  CHECK(m.contains("boundary.payload_bytes_total"));
  CHECK((m.contains("nic.name") || m.contains("nic.unavailable")));
  CHECK((m.contains("nvml.unavailable") || m.contains("nvml.gpu.0.name")));
  // Prepare breakdown: Father read/send and Node hash/write/build.
  for (const char* k : {"prepare.node.node0.father_source_read_ms", "prepare.node.node0.father_send_ms",
                        "prepare.node.node0.node_seal_hash_ms", "prepare.node.node0.node_chunk_write_ms",
                        "prepare.node.node0.node_build_ms", "prepare.node.node0.father_source_read_bytes_per_s"})
    CHECK_MESSAGE(m.contains(k), k);
  CHECK(m["prepare.node.node0.node_seal_hash_ms"]["p50"].get<double>() > 0);
  CHECK(m["prepare.node.node0.father_send_ms"]["p50"].get<double>() > 0);
  CHECK(m.contains("census.diff"));  // --census
}

TEST_CASE("faults --only runs exactly the selected scenarios and samples every release cycle") {
  const json doc = load_result("result-faults-subset.json");
  check_schema(doc);
  CHECK(doc["configuration"]["scenario_subset"] == true);
  CHECK(doc["configuration"]["scenarios_run"] == json::array({"crash_ready", "release_cycles"}));
  for (const auto& c : doc["checks"]) {
    const std::string name = c["name"].get<std::string>();
    CHECK_MESSAGE(c["passed"] == true, name);
    CHECK_MESSAGE((name.rfind("crash_ready", 0) == 0 || name.rfind("release_cycles", 0) == 0), name);  // nothing else ran
  }
  const auto& m = doc["metrics"];
  CHECK(m.contains("crash_ready.detect_ms"));
  CHECK_FALSE(m.contains("crash_hashing.detect_ms"));
  CHECK_FALSE(m.contains("stall.detect_ms"));
  CHECK(m["release_cycles.release_ms"]["n"] == 2);
  // Start + 2 cycles; RSS and commit series for Father and each Node.
  CHECK(m["release_cycles.resources.samples"] == 3);
  for (const char* p : {"father", "node0", "node1"}) {
    const std::string k = std::string("release_cycles.resources.") + p;
    CHECK(m[k + ".rss_bytes.series"].size() == 3);
    CHECK(m[k + ".commit_bytes.series"].size() == 3);
    CHECK(m[k + ".rss_bytes.series"][0].get<double>() > 0);
  }
  CHECK(m.contains("census.added_files"));
  CHECK(m.contains("release_cycles.prepare.node0.last_node_seal_hash_ms"));
}

TEST_CASE("nvml command without NVML records unavailable and a reason; with NVML it records series") {
  const json doc = load_result("result-nvml.json");
  check_schema(doc);
  const auto& m = doc["metrics"];
  if (m.contains("nvml.unavailable")) {
    CHECK(m["nvml.unavailable"].get<std::string>().rfind("unavailable: ", 0) == 0);
    CHECK(doc["provenance"] == "Synthetic");  // nothing measured
    bool gpu03 = false;
    for (const auto& id : doc["pending_qualification"]) gpu03 |= id == "HQ-GPU-03";
    CHECK(gpu03);
  } else {
    CHECK(doc["provenance"] == "Measured");
    CHECK(m.contains("nvml.gpu.0.sm_clock_mhz"));
  }
}

TEST_CASE("storage-census --before / --after: snapshot written, diff computed, names and sizes only") {
  const json before = load_result("result-census-before.json");
  check_schema(before);
  const json after = load_result("result-census-after.json");
  check_schema(after);
  CHECK(after["metrics"].contains("census.diff"));
  const std::string text = after["metrics"]["census.diff"].dump();
  CHECK(text.find("\"added\"") != std::string::npos);
  CHECK(after["metrics"].contains("census.added_bytes"));
  for (const auto& c : after["checks"]) CHECK_MESSAGE(c["passed"] == true, c["name"].get<std::string>());
}
#endif
