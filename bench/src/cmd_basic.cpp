// profile, placement, qualification and hardware-only commands.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/placement/placement.hpp"
#include "clusterlm/placement/profile.hpp"
#include "bench_common.hpp"
#include "commands.hpp"

namespace clusterlm::bench {

std::string source_dir() {
#ifdef CLUSTERLM_SOURCE_DIR
  return CLUSTERLM_SOURCE_DIR;
#else
  return ".";
#endif
}

int emit(const cli::Args& args, BenchmarkResult& result, double duration_s, int code_if_ok) {
  const bool ok = result.all_checks_passed();
  const auto doc = result.finish(duration_s);
  const std::string text = doc.dump(2) + "\n";
  if (args.has("out")) {
    std::ofstream f(args.get("out"), std::ios::binary);
    f << text;
    std::fprintf(stderr, "wrote %s (provenance %s)\n", args.get("out").c_str(), doc["provenance"].get<std::string>().c_str());
  } else {
    std::cout << text;
  }
  return ok ? code_if_ok : 1;
}

int cmd_profile(const cli::Args& args) {
  Stopwatch sw;
  const auto host = probe_host();
  const RunContext ctx = make_run_context(args);
  BenchmarkResult r(experiment_id(args, ctx, "HQ-PROF-01", "dev-profile"), host);
  apply_run_context(r, ctx);
  r.set_measured();
  const auto bytes = static_cast<std::uint64_t>(args.number("bandwidth-mib", 512)) << 20;
  const double bw = measure_read_bandwidth(bytes, static_cast<int>(args.integer("repeats", 3)));
  const MemoryInfo mem = probe_memory();
  r.metric("cpu.features", host.cpu_features);
  r.metric("cpu.logical_cpus", host.logical_cpus);
  r.metric("memory.ram_total", host.ram_bytes);
  r.metric("memory.ram_available", host.ram_available_bytes);
  if (mem.commit_limit) r.metric("memory.commit_limit", *mem.commit_limit);
  r.metric("memory.read_bandwidth_single_thread", bw);
  r.config("bandwidth_bytes", bytes);
  // Power source (Windows adapter); absent elsewhere.
  if (auto power = make_host_power_monitor()) {
    auto p = power->sample();
    if (p.is_ok()) {
      r.metric("power.on_ac", p->on_ac_power);
      r.metric("power.battery_saver", p->battery_saver);
    }
  }
  r.metric("gpu.present", host.has_cuda_device);
  // The fields below need the GPU build (`gpu`, `pcie`) or the hardware-backed backends; `calibrate` runs everything
  // that can be measured and writes the profile.
  for (const char* id : {"HQ-CPU-01", "HQ-GPU-01", "HQ-GPU-02", "HQ-PCIE-01"}) r.pending(id);
  r.check("cpu_features_detected", !host.cpu_features.empty() || host.arch != "x86-64",
          "CPUID and OS-enabled vector state");
  return emit(args, r, sw.elapsed_ms() / 1000.0);
}

int cmd_placement(const cli::Args& args) {
  Stopwatch sw;
  const auto host = probe_host();
  BenchmarkResult r("dev-placement", host);
  const std::string dir = args.get("profiles", source_dir() + "/fixtures/profiles");
  // A profile argument is a file name inside --profiles, or a path to an existing file.
  auto resolve = [&](const std::string& name) {
    return std::filesystem::exists(name) ? name : dir + "/" + name;
  };
  auto father = placement::load_hardware_profile(resolve(args.get("father", "Father-4060Ti-7600.json")));
  if (!father.is_ok()) {
    std::fprintf(stderr, "%s\n", father.status().to_string().c_str());
    return 1;
  }
  placement::PlacementRequest req;
  req.father = father.value();
  for (const auto& n : args.has("node") ? args.all("node")
                                        : std::vector<std::string>{"Node-G14-4070-8945HS.json", "Node-3060-5600.json"}) {
    auto p = placement::load_hardware_profile(resolve(n));
    if (!p.is_ok()) {
      std::fprintf(stderr, "%s\n", p.status().to_string().c_str());
      return 1;
    }
    req.nodes.push_back(p.value());
  }
  auto net = placement::load_network_profile(resolve(args.get("network", "network-gige-simulated.json")));
  if (!net.is_ok()) {
    std::fprintf(stderr, "%s\n", net.status().to_string().c_str());
    return 1;
  }
  req.network = net.value();
  req.model = placement::flash_next_planning_estimate();
  req.context_tokens = static_cast<std::uint32_t>(args.integer("context", 4096));
  req.q = static_cast<std::uint32_t>(args.integer("q", 4));
  auto result = placement::search_placements(req);
  if (!result.is_ok()) {
    std::fprintf(stderr, "%s\n", result.status().to_string().c_str());
    return 1;
  }
  // The result is only as trustworthy as its weakest input (docs/benchmark-methodology.md): any Synthetic quantity
  // in any profile makes the whole run Synthetic; fully Measured inputs give a Measured result.
  nlohmann::json synthetic_ids = nlohmann::json::array();
  auto note_profile = [&](placement::HardwareProfile p) {
    if (placement::weakest_provenance(p) == placement::Provenance::kSynthetic) synthetic_ids.push_back(p.id);
  };
  note_profile(req.father);
  for (const auto& n : req.nodes) note_profile(n);
  if (placement::weakest_provenance(req.network) == placement::Provenance::kSynthetic) synthetic_ids.push_back("network");
  if (!synthetic_ids.empty()) r.mark_simulated("synthetic_profiles", synthetic_ids);
  else r.set_measured();
  r.metric("placement.weakest_input_provenance",
           std::string(placement::to_string(std::min(placement::weakest_provenance(req.father),
                                                     placement::weakest_provenance(req.network)))));
  r.config("context_tokens", req.context_tokens);
  r.config("q", req.q);
  r.metric("placement.report", nlohmann::json::parse(placement::report_to_json(result.value())));
  r.pending("HQ-PLACE-01");
  return emit(args, r, sw.elapsed_ms() / 1000.0);
}

int cmd_qualification(const cli::Args& args) {
  std::ifstream f(args.get("registry", source_dir() + "/bench/qualification/experiments.json"));
  if (!f) {
    std::fprintf(stderr, "cannot read qualification registry\n");
    return 1;
  }
  const auto reg = nlohmann::json::parse(f);
  if (args.has("json")) {
    std::cout << reg.dump(2) << "\n";
    return 0;
  }
  for (const auto& e : reg["experiments"])
    std::printf("%-12s %-9s %s\n", e["id"].get<std::string>().c_str(), e["status"].get<std::string>().c_str(),
                e["title"].get<std::string>().c_str());
  return 0;
}

int cmd_hardware_only(const std::string& command, const cli::Args& args) {
  BenchmarkResult r("dev-" + command, probe_host());
  r.check("backend_available", false,
          "'" + command + "' experiments execute Strata/llama.cpp kernels on the target GPUs; this build has no "
          "hardware backend (configure with -DCLUSTERLM_ENABLE_STRATA=ON on a CUDA host). See "
          "HARDWARE-QUALIFICATION.md.");
  // Pending items: every registry entry whose command is this command (and subcommand, when given).
  std::string prefix = "clusterlm-bench " + command;
  if (!args.positional().empty()) prefix += " " + args.positional().front();
  std::ifstream f(args.get("registry", source_dir() + "/bench/qualification/experiments.json"));
  const auto reg = f ? nlohmann::json::parse(f, nullptr, false) : nlohmann::json();
  bool any = false;
  if (reg.is_object() && reg.contains("experiments"))
    for (const auto& e : reg["experiments"])
      if (e["command"].get<std::string>().rfind(prefix, 0) == 0) {
        r.pending(e["id"].get<std::string>());
        any = true;
      }
  if (!any)
    for (const char* id : {"HQ-GPU-01", "HQ-NUM-01", "HQ-P0A-01", "HQ-P0B-01"}) r.pending(id);
  (void)emit(args, r, 0);
  return 3;
}

}  // namespace clusterlm::bench
