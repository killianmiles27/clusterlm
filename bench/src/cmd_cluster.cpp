// clusterlm-bench cluster / faults: multi-process clusters on this machine through the real protocol.
//
// Father runs in this process (Coordinator); every Node is a separate clusterlm-node process reached over
// authenticated TCP. Results are Synthetic: localhost processes, a fixture model and optional simulated
// network conditions exercise the software, not the target hardware.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#include "clusterlm/backends/backend_factory.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/placement/placement.hpp"
#include "clusterlm/placement/profile.hpp"
#include "clusterlm/planning/planning.hpp"
#include "clusterlm/platform/fs_safety.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/protocol/messages.hpp"
#include "clusterlm/common/strcat.hpp"
#include "bench_backend.hpp"
#include "bench_common.hpp"
#include "cluster_pass.hpp"
#include "commands.hpp"
#include "corpus_prompts.hpp"
#include "local_cluster.hpp"
#include "nvml_probe.hpp"
#include "rank_agreement.hpp"
#include "storage_census.hpp"
#include "system_probe.hpp"
#include "cluster_pass.hpp"

namespace clusterlm::bench {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultPlan = "0-4@father,4-10@0,10-13@1,13-16@father";


// Layer split for a tier on a model with `n` layers: Ultra = Father + two Nodes, Strong = Father + one Node,
// Fast = Father only. Prefix and tail always stay on Father (token IDs never leave it).
std::string plan_for_tier(const std::string& tier, std::uint32_t n) {
  auto r = [](std::uint32_t a, std::uint32_t b) { return std::to_string(a) + "-" + std::to_string(b); };
  if (tier == "fast") return r(0, n / 2) + "@father," + r(n / 2, n) + "@father";
  if (tier == "strong") return r(0, n / 4) + "@father," + r(n / 4, n * 3 / 4) + "@0," + r(n * 3 / 4, n) + "@father";
  return r(0, n / 4) + "@father," + r(n / 4, n * 5 / 8) + "@0," + r(n * 5 / 8, n * 13 / 16) + "@1," + r(n * 13 / 16, n) +
         "@father";
}


}  // namespace

namespace {

// Exit code for a backend/hardware problem: 2 for a usage error (unknown backend), 3 for "this build or machine cannot do
// that" (a backend that is not built, a device that is not there), 1 otherwise.
int exit_code_for(const Status& st) {
  switch (st.code()) {
    case ErrorCode::kInvalidArgument: return 2;
    case ErrorCode::kUnimplemented:
    case ErrorCode::kHardwareUnavailable: return 3;
    default: return 1;
  }
}

// Application-owned bytes staged under a Node staging root (everything but the content-free journal), as the Node's own
// status line counts them; all-ones when the root cannot be read.
std::uint64_t staged_bytes(const fs::path& root) {
  auto total = platform::allocated_bytes_under(root);
  if (!total.is_ok()) return ~std::uint64_t{0};
  std::error_code ec;
  const auto journal = fs::file_size(root / "journal.log", ec);
  return total.value() - (ec ? 0 : std::min<std::uint64_t>(journal, total.value()));
}

void cleanup_work(const cli::Args& args, const fs::path& work) {
  if (args.has("keep-work") || args.has("work")) return;
  std::error_code ec;
  fs::remove_all(work, ec);
}

}  // namespace

int cmd_cluster(const cli::Args& args) {
  Stopwatch total;
  const RunContext rc = make_run_context(args);
  BenchmarkResult r(args.get("experiment", "dev-cluster"), probe_host());

  // A backend this binary was built without (or an unknown one) fails here, before a model is written or a process spawned.
  if (auto st = backends::check_backend_name(args.get("backend", "reference")); !st.is_ok()) {
    std::fprintf(stderr, "cluster: %s\n", st.to_string().c_str());
    return exit_code_for(st);
  }

  const fs::path work = args.has("work") ? fs::path(args.get("work")) : default_work_dir("cluster");
  auto model_dir = ensure_model(args, work);
  if (!model_dir.is_ok()) {
    std::fprintf(stderr, "model: %s\n", model_dir.status().to_string().c_str());
    cleanup_work(args, work);
    return 1;
  }
  std::uint32_t n_layers = 0;
  {
    auto store = objects::CanonicalModelStore::open(model_dir.value());
    if (!store.is_ok()) {
      std::fprintf(stderr, "%s\n", store.status().to_string().c_str());
      cleanup_work(args, work);
      return 1;
    }
    n_layers = store.value()->manifest().geometry.n_layers;
  }

  PassContext pc(args, r);
  pc.work = work;
  pc.model_dir = model_dir.value();
  pc.direct_peer = !args.has("relay");
  pc.fixture = !args.has("model");
  if (args.has("plan")) pc.plan_text = args.get("plan");
  else pc.plan_text = plan_for_tier(args.get("tier", "ultra"), n_layers);
  const std::size_t remote_stages = remote_stage_count(pc.plan_text);
  pc.max_new = static_cast<std::uint32_t>(args.integer("tokens", args.integer("max-new", 64)));
  pc.repeat = static_cast<int>(args.integer("repeat", 3));
  pc.prefill_chunk = static_cast<std::uint32_t>(args.integer("prefill-chunk", 16));
  pc.qs = args.has("q") ? parse_u32_list(args.get("q")) : std::vector<std::uint32_t>{1, 4};
  if (args.has("contexts") && args.has("context")) {
    std::fprintf(stderr, "--contexts and --context are mutually exclusive\n");
    cleanup_work(args, work);
    return 2;
  }
  // --contexts runs the whole sweep in one invocation with every result keyed ctx<N>.q<q>.* (even for one context);
  // --context keeps the older behaviour (keys are prefixed only when more than one context is given).
  pc.keyed_contexts = args.has("contexts");
  pc.contexts = args.has("contexts") ? parse_u32_list(args.get("contexts"))
                : args.has("context") ? parse_u32_list(args.get("context"))
                                      : std::vector<std::uint32_t>{static_cast<std::uint32_t>(args.integer("prompt-len", 48))};
  pc.resources = !args.has("no-resources");
  pc.sample_ms = args.number("sample-ms", 200);
  pc.telemetry_s = args.number("sample-s", 5);
  pc.census = args.has("census");
  if (pc.qs.empty() || pc.contexts.empty() || pc.repeat < 1) {
    std::fprintf(stderr, "q, context and repeat must be non-empty/positive\n");
    cleanup_work(args, work);
    return 2;
  }

  // --node: Nodes already running on other machines. The plan's node indices refer to them in the order given.
  auto external = parse_external_nodes(args);
  if (!external.is_ok() || (!external->empty() && external->size() < remote_stages)) {
    std::fprintf(stderr, "cluster: %s\n",
                 external.is_ok() ? ("the plan runs " + std::to_string(remote_stages) + " Node stage(s) but --node names " +
                                     std::to_string(external->size())).c_str()
                                  : external.status().to_string().c_str());
    cleanup_work(args, work);
    return 2;
  }
  auto backend = resolve_backend(args, pc.model_dir, remote_stages);
  if (!backend.is_ok()) {
    std::fprintf(stderr, "cluster: %s\n", backend.status().to_string().c_str());
    cleanup_work(args, work);
    return exit_code_for(backend.status());
  }
  pc.backend = &backend.value();
  if (backend->name == "llama" && std::any_of(pc.qs.begin(), pc.qs.end(), [](std::uint32_t q) { return q > 1; })) {
    std::fprintf(stderr, "cluster: the llama backend has no MTP drafter: use --q 1\n");
    cleanup_work(args, work);
    return 2;
  }

  // ---- provenance (docs/benchmark-methodology.md): anything simulated keeps the result Synthetic ----
  // Real = the converted model of the tier (--model), a real backend, no simulated network, no fixture drafter, and no
  // localhost Nodes standing in for machines: Nodes started by this command are localhost processes, --node endpoints that
  // are not loopback are real links, and a Father-only plan has no links at all. --on-target is the operator's assertion
  // that this is the named machine; localhost Nodes are never the target Nodes, so it is not honoured then.
  bool localhost_nodes = false;
  if (remote_stages > 0) {
    if (external->empty()) localhost_nodes = true;
    for (const auto& ep : external.value()) localhost_nodes = localhost_nodes || is_loopback_host(ep.endpoint.host);
  }
  RunContext effective = rc;
  effective.on_target = rc.on_target && !localhost_nodes;
  apply_run_context(r, effective);
  if (localhost_nodes) r.mark_simulated("localhost_cluster", true);
  r.config("external_nodes", external->size());
  if (pc.fixture) r.mark_simulated("fixture_model", true);
  if (!backend->real()) r.mark_simulated("reference_backend", true);
  if (args.has("impair")) r.mark_simulated("network_preset", args.get("impair"));
  if (args.get("drafter", "mtp").rfind("scripted", 0) == 0) r.mark_simulated("scripted_drafter", true);
  if (rc.on_target && localhost_nodes)
    r.config("on_target_not_applied", "the plan runs Nodes as localhost processes: not the target machines (use --node HOST:PORT@FINGERPRINT)");
  if (effective.on_target && !pc.fixture && backend->real()) r.set_measured();  // finish() overrides it when anything above was marked
  r.backend(backend->name, backend->build_hash);

  r.config("backend", backend->name);
  r.config("model", pc.fixture ? "fixture" : model_dir.value().filename().string());
  r.config("tier", args.get("tier", args.has("plan") ? "custom" : "ultra"));
  r.config("tls", !args.has("insecure"));
  r.config("context", pc.contexts);
  r.config("context_sweep", pc.keyed_contexts);
  r.config("q", pc.qs);
  r.config("max_new_tokens", pc.max_new);
  r.config("repeat", pc.repeat);
  r.config("interleave", args.has("interleave"));
  r.config("phase", args.get("phase", "all"));
  r.config("drafter", args.get("drafter", "mtp"));
  r.config("meets_min_repetitions", pc.repeat >= 5);  // qualification runs need >= 5

  // Prompts and Father-only reference outputs first: a real backend must not hold the reference model and the cluster's
  // Father domains in device memory together. --no-reference skips the references (a model that does not fit Father alone).
  if (args.get("phase") != "prepare") {
    auto sets = build_prompt_sets(args, pc.model_dir, backend.value().real() ? &backend.value() : nullptr, pc.contexts, pc.max_new,
                                  !args.has("no-reference"), r);
    if (!sets.is_ok()) {
      std::fprintf(stderr, "cluster: %s\n", sets.status().to_string().c_str());
      cleanup_work(args, work);
      return exit_code_for(sets.status());
    }
    pc.prompt_sets = std::move(sets).value();
  }

  Status st;
  if (args.has("compare-routing")) {
    // HQ-NET-02: the same plan with direct Node->Node forwarding and with Father relay.
    for (bool direct : {true, false}) {
      pc.prefix = direct ? "direct." : "relay.";
      pc.direct_peer = direct;
      st = run_cluster_pass(pc);
      if (!st.is_ok()) break;
    }
  } else {
    st = run_cluster_pass(pc);
  }
  if (!st.is_ok()) {
    std::fprintf(stderr, "cluster: %s\n", st.to_string().c_str());
    r.check("cluster_pass_completed", false, st.to_string());
  }
  for (const char* id : {"HQ-PERF-01", "HQ-NET-02", "HQ-MTP-01", "HQ-PROV-01"}) r.pending(id);
  int code = emit(args, r, total.elapsed_ms() / 1000.0);
  if (!st.is_ok() && exit_code_for(st) == 3) code = 3;  // e.g. the Strata engine without a usable CUDA device
  cleanup_work(args, work);
  return code;
}

// ---- faults ------------------------------------------------------------------------------------------------

int cmd_faults(const cli::Args& args) {
  Stopwatch total;
  if (auto st = backends::check_backend_name(args.get("backend", "reference")); !st.is_ok()) {
    std::fprintf(stderr, "faults: %s\n", st.to_string().c_str());
    return exit_code_for(st);
  }
  // --supervised adds the scenarios that run the Nodes under clusterlm-node-service to the default selection.
  auto selected = select_fault_scenarios(args.get("only"), args.has("supervised"));
  if (!selected.is_ok()) {
    std::fprintf(stderr, "faults: %s\n", selected.status().message().c_str());
    return 2;
  }
  auto want = [&](const std::string& name) {
    return std::find(selected->begin(), selected->end(), name) != selected->end();
  };
  BenchmarkResult r(args.get("experiment", "dev-faults"), probe_host());
  r.config("scenarios_run", nlohmann::json(*selected));
  r.config("scenario_subset", selected->size() != fault_scenario_names().size() + (args.has("supervised") ? supervised_fault_scenario_names().size() : 0));
  r.mark_simulated("localhost_cluster", true);
  if (!args.has("model")) r.mark_simulated("fixture_model", true);
  const std::string plan_text = args.get("plan", kDefaultPlan);
  const auto max_new = static_cast<std::uint32_t>(args.integer("max-new", 16));
  const fs::path work = args.has("work") ? fs::path(args.get("work")) : default_work_dir("faults");
  auto model_dir = ensure_model(args, work);
  if (!model_dir.is_ok()) {
    std::fprintf(stderr, "%s\n", model_dir.status().to_string().c_str());
    return 1;
  }
  auto backend = resolve_backend(args, model_dir.value(), remote_stage_count(plan_text));
  if (!backend.is_ok()) {
    std::fprintf(stderr, "faults: %s\n", backend.status().to_string().c_str());
    if (!args.has("keep-work") && !args.has("work")) {
      std::error_code ec;
      fs::remove_all(work, ec);
    }
    return exit_code_for(backend.status());
  }
  if (!backend->real()) r.mark_simulated("reference_backend", true);
  r.backend(backend->name, backend->build_hash);
  r.config("backend", backend->name);
  // HQ-STORE-01: cache/temp census around the whole run (the Node staging roots live in the excluded work dir and are
  // covered by the per-scenario census_bytes checks).
  const bool census_on = !args.has("no-census");
  std::vector<CensusRoot> census_roots;
  std::optional<CensusSnapshot> census_before;
  if (census_on) {
    census_roots = default_census_roots({}, {work});
    census_before = take_census(census_roots);
  }
  std::vector<std::int32_t> prompt;
  std::vector<std::int32_t> reference;
  const bool have_reference = !args.has("no-reference");  // a real model that does not fit Father alone has no reference
  {
    auto m = objects::CanonicalModelStore::open(model_dir.value());
    if (!m.is_ok()) return 1;
    prompt = synthetic_prompt_tokens(24, m.value()->manifest().geometry.vocab_size);
    if (have_reference) {
      auto ref = reference_tokens(model_dir.value(), &backend.value(), prompt, max_new);
      if (!ref.is_ok()) {
        std::fprintf(stderr, "faults: reference run: %s\n", ref.status().to_string().c_str());
        return exit_code_for(ref.status());
      }
      reference = ref.value();
    }
  }
  r.config("reference_check", have_reference);

  // Cooperative release deadline of a supervised Node: a worker that has not reached Busy and clean in this time is
  // terminated by its service (HQ-REL-01; the product default is 2 s).
  const auto deadline_ms = static_cast<std::uint32_t>(args.integer("deadline-ms", 1500));

  // One scenario = a fresh two-Node cluster. Returns the setup for follow-up checks.
  auto run_scenario = [&](const std::string& name, std::vector<LocalNodeOptions> node_opts,
                          const std::function<void(LocalCluster&, coordinator::Coordinator&,
                                                   const coordinator::ClusterPlan&)>& body,
                          bool connect_father = true, bool supervised = false) {
    LocalClusterOptions opts;
    opts.work_dir = work / name;
    opts.tls = !args.has("insecure");
    opts.supervised = supervised;
    opts.cooperative_deadline = std::chrono::milliseconds(deadline_ms);
    opts.node_args = backend->node_args;
    for (std::size_t i = 0; i < 2; ++i) {
      LocalNodeOptions n = i < node_opts.size() ? node_opts[i] : LocalNodeOptions{};
      if (n.name.empty()) n.name = "node" + std::to_string(i);
      opts.nodes.push_back(n);
    }
    auto cluster = LocalCluster::start(opts);
    if (!cluster.is_ok()) {
      r.check(name + "_setup", false, cluster.status().to_string());
      return;
    }
    ClusterSetup s;
    s.work = opts.work_dir;
    s.model_dir = model_dir.value();
    s.backend = backend->father;
    s.cluster = std::move(cluster).value();
    auto cfg = father_config(s, args);
    cfg.window_timeout = 3s;
    cfg.request_timeout = 5s;
    cfg.prepare_timeout = 10s;
    auto coord = coordinator::Coordinator::create(cfg);
    if (!coord.is_ok()) {
      r.check(name + "_setup", false, coord.status().to_string());
      return;
    }
    auto plan = coordinator::ClusterPlan::parse(plan_text, coord.value()->manifest().geometry.n_layers);
    if (!plan.is_ok()) return;
    if (connect_father) {
      if (auto st = coord.value()->connect(); !st.is_ok()) {
        r.check(name + "_connect", false, st.to_string());
        return;
      }
    }
    body(*s.cluster, *coord.value(), plan.value());
  };

  auto request = [&] {
    coordinator::GenerationRequest req;
    req.prompt = prompt;
    req.max_new_tokens = max_new;
    req.prefill_chunk = 8;
    return req;
  };

  // Recovery check shared by scenarios: restarted/released Nodes hold zero staged bytes and the cluster can
  // prepare and generate correctly again (on a new lease generation).
  auto recover_and_verify = [&](const std::string& name, LocalCluster& cl, coordinator::Coordinator& c,
                                const coordinator::ClusterPlan& plan) {
    (void)c.release();
    record_census(r, cl, name + "_census_zero_after_fault");
    if (auto st = c.connect(); !st.is_ok()) {
      r.check(name + "_reconnect", false, st.to_string());
      return;
    }
    auto prep = c.prepare(plan);
    if (!prep.is_ok()) {
      r.check(name + "_reprepare", false, prep.status().to_string());
      return;
    }
    auto gen = c.generate(request());
    r.check(name + (have_reference ? "_regenerate_matches_reference" : "_regenerate_completed"),
            gen.is_ok() && (!have_reference || gen->tokens == reference), gen.is_ok() ? "" : gen.status().to_string());
    (void)c.release();
    record_census(r, cl, name + "_census_zero_after_release");
  };

  // 1. Abrupt Node crash at each lifecycle phase, then restart with orphan recovery.
  for (const char* phase :
       {"transfer", "hashing", "mapping", "allocation", "ready", "prefill", "inference", "commit", "cleanup"}) {
    const std::string name = std::string("crash_") + phase;
    if (!want(name)) continue;
    LocalNodeOptions crashing;
    crashing.name = "node0";
    crashing.crash_at = phase;
    run_scenario(name, {crashing}, [&](LocalCluster& cl, coordinator::Coordinator& c, const auto& plan) {
      Stopwatch sw;
      auto prep = c.prepare(plan);
      Status outcome = prep.status();
      if (prep.is_ok()) outcome = c.generate(request()).status();
      if (outcome.is_ok() && std::string(phase) == "cleanup") outcome = c.release().status();
      const bool died = cl.wait_exit(0, 5s).is_ok();
      r.check(name + "_node_terminated", died);
      r.check(name + "_father_observed_failure_or_release", !outcome.is_ok() || std::string(phase) == "cleanup",
              outcome.to_string());
      r.metric(name + ".detect_ms", sw.elapsed_ms());
      auto st = cl.restart(0);
      r.check(name + "_restart", st.is_ok(), st.to_string());
      recover_and_verify(name, cl, c, plan);
    });
  }

  // 1b. Father disappears (process killed mid-generation): every Node releases its lease on control loss and
  // deletes all staged objects without any instruction.
  if (want("father_lost")) run_scenario("father_lost", {}, [&](LocalCluster& cl, coordinator::Coordinator&, const auto&) {
    const auto eps = cl.endpoints();
    std::vector<std::string> fargs = {"--model", model_dir.value().string(), "--plan", plan_text, "--max-new", "1500",
                                      "--prefill-chunk", "8"};
    for (const auto& flag : backend_flags())
      if (args.has(flag)) fargs.insert(fargs.end(), {"--" + flag, args.get(flag)});
    const auto id_dir = (work / "father_lost" / "father-id").string();
    if (!args.has("insecure")) fargs.insert(fargs.end(), {"--identity", id_dir});
    for (std::size_t i = 0; i < eps.size(); ++i)
      fargs.insert(fargs.end(), {"--node", eps[i].name + "=" + eps[i].endpoint.str() +
                                               (eps[i].device_id.empty() ? "" : "@" + eps[i].device_id)});
    auto father = platform::ChildProcess::spawn(platform::executable_dir() /
#ifdef _WIN32
                                                    "clusterlm-father.exe",
#else
                                                    "clusterlm-father",
#endif
                                                fargs);
    if (!father.is_ok()) {
      r.check("father_lost_spawn", false, father.status().to_string());
      return;
    }
    auto prepared = father.value()->read_until("prepared plan", 30s);
    r.check("father_lost_father_prepared", prepared.is_ok(), prepared.status().to_string());
    std::this_thread::sleep_for(50ms);  // let generation start
    Stopwatch sw;
    father.value()->kill();
    (void)father.value()->wait(5s);
    bool clean = false;
    while (!clean && sw.elapsed_ms() < 10000) {
      clean = true;
      for (std::size_t i = 0; i < cl.size(); ++i) {
        auto st = cl.status(i);
        clean = clean && st.is_ok() && st->census_bytes == 0 && st->state == "Available";
      }
      if (!clean) std::this_thread::sleep_for(20ms);
    }
    r.check("father_lost_nodes_released_and_clean", clean);
    r.metric("father_lost.cleanup_observed_ms", sw.elapsed_ms());
  }, /*connect_father=*/false);

  // 2. Local user activity while Ready: the Node revokes the lease without consulting Father.
  if (want("local_activity")) run_scenario("local_activity", {}, [&](LocalCluster& cl, coordinator::Coordinator& c, const auto& plan) {
    if (!c.prepare(plan).is_ok()) return;
    Stopwatch sw;
    (void)cl.local_activity(1);
    // Wait until the Node reports Busy with nothing staged.
    bool busy = false;
    for (int i = 0; i < 100 && !busy; ++i) {
      auto st = cl.status(1);
      busy = st.is_ok() && st->state == "Busy" && st->census_bytes == 0;
      if (!busy) std::this_thread::sleep_for(10ms);
    }
    r.metric("local_activity.release_observed_ms", sw.elapsed_ms());
    r.check("local_activity_node_busy_and_clean", busy);
    auto gen = c.generate(request());
    r.check("local_activity_father_refuses_stale_plan", !gen.is_ok(), gen.is_ok() ? "generated" : gen.status().to_string());
    (void)cl.local_idle(1);
    std::this_thread::sleep_for(100ms);
    recover_and_verify("local_activity", cl, c, plan);
  });

  // 3. Activation channel severed mid-generation (transport fault on the Node's 5th StageResult).
  if (want("link_loss")) {
    LocalNodeOptions faulty;
    faulty.name = "node1";
    faulty.faults = {"close-before:type=" + std::to_string(static_cast<int>(protocol::MessageType::kStageResult)) + ":nth=5"};
    run_scenario("link_loss", {LocalNodeOptions{}, faulty},
                 [&](LocalCluster& cl, coordinator::Coordinator& c, const auto& plan) {
                   if (!c.prepare(plan).is_ok()) return;
                   auto gen = c.generate(request());
                   r.check("link_loss_session_invalidated", !gen.is_ok(),
                           gen.is_ok() ? "generated" : gen.status().to_string());
                   recover_and_verify("link_loss", cl, c, plan);
                 });
  }

  // 4. Stalled Node: results never arrive; Father's window deadline invalidates the session.
  if (want("stall")) {
    LocalNodeOptions stalled;
    stalled.name = "node1";
    stalled.faults = {"stall:type=" + std::to_string(static_cast<int>(protocol::MessageType::kStageResult)) + ":nth=3"};
    run_scenario("stall", {LocalNodeOptions{}, stalled},
                 [&](LocalCluster& cl, coordinator::Coordinator& c, const auto& plan) {
                   if (!c.prepare(plan).is_ok()) return;
                   Stopwatch sw;
                   auto gen = c.generate(request());
                   r.check("stall_detected_by_deadline", !gen.is_ok() && gen.status().code() == ErrorCode::kDeadlineExceeded,
                           gen.is_ok() ? "generated" : gen.status().to_string());
                   r.metric("stall.detect_ms", sw.elapsed_ms());
                   cl.kill(1);
                   (void)cl.restart(1);
                   recover_and_verify("stall", cl, c, plan);
                 });
  }

  // 5. Repeated release / reprepare cycles within healthy leases (STORE-02 / PERF-04 shape).
  const int cycles = static_cast<int>(args.integer("release-cycles", 5));
  if (want("release_cycles")) run_scenario("release_cycles", {}, [&](LocalCluster& cl, coordinator::Coordinator& c, const auto& plan) {
    Distribution prep_ms, release_ms;
    bool ok = true;
    // HQ-PERF-04 / HQ-REL-01: RSS and commit of Father and each Node process (and VRAM when NVML is present) at the
    // start and after every prepare/generate/release cycle, recorded as series.
    ResourceSampler sampler;
    sampler.add_process("father", [] { return std::int64_t{0}; });
    for (std::size_t i = 0; i < cl.size(); ++i) sampler.add_process("node" + std::to_string(i), [&cl, i] { return cl.pid(i); });
    auto nvml = NvmlTelemetry::open();
    if (nvml.is_ok()) sampler.set_gpu(nvml.value().get());
    else emit_gpu_telemetry_unavailable(r, nvml.status().message(), "release_cycles.nvml.");
    sampler.sample("start");
    for (int i = 0; i < cycles && ok; ++i) {
      auto prep = c.prepare(plan);
      ok = prep.is_ok();
      if (!ok) break;
      prep_ms.add(prep->total_ms);
      for (const auto& n : prep->nodes) {
        const std::string k = "release_cycles.prepare." + n.node + ".";
        r.metric(k + "last_father_source_read_ms", static_cast<double>(n.father_source_read_ns) * 1e-6);
        r.metric(k + "last_father_send_ms", static_cast<double>(n.father_send_ns) * 1e-6);
        r.metric(k + "last_node_chunk_write_ms", static_cast<double>(n.node_chunk_write_ns) * 1e-6);
        r.metric(k + "last_node_seal_hash_ms", static_cast<double>(n.node_seal_hash_ns) * 1e-6);
      }
      // Two chats in one Ready lease reuse allocations: no reprovisioning between them.
      for (int chat = 0; chat < 2 && ok; ++chat) {
        auto gen = c.generate(request());
        ok = gen.is_ok() && (!have_reference || gen->tokens == reference);
      }
      Stopwatch sw;
      auto rel = c.release();
      release_ms.add(sw.elapsed_ms());
      ok = ok && rel.is_ok();
      if (rel.is_ok())
        for (const auto& n : rel->nodes) ok = ok && n.storage_cleaned && n.residual_bytes == 0;
      if (!c.connect().is_ok()) ok = false;
      sampler.sample("cycle");
    }
    if (!args.has("no-resources")) sampler.emit(r, "release_cycles.resources.");
    r.check("release_cycles_all_clean_and_correct", ok);
    r.metric("release_cycles.prepare_ms", prep_ms, "ms");
    r.metric("release_cycles.release_ms", release_ms, "ms");
    record_census(r, cl, "release_cycles_final_census_zero");
  });

  // 6. HQ-REL-01 under supervision. Node 0 runs under clusterlm-node-service (console mode, simulated user activity), which
  // supervises its worker the way the Windows service does (a Job Object there, a child process here):
  //   supervised_forced_termination  local activity arrives while the lease is Ready, but the worker hangs in its cleanup (a
  //                                  stuck driver): it misses the cooperative deadline, the service force-terminates it
  //                                  and relaunches it (event forced_termination; residual_bytes is what the relaunched
  //                                  worker's orphan recovery left staged);
  //   supervised_crash_recovery      Father releases the lease and the worker dies in its cleanup, with objects still staged:
  //                                  the service notices the exit and relaunches it (event worker_restarted), and the
  //                                  relaunched worker's orphan recovery must delete what the dead one left.
  // Either way the Node must be offered again and the next lease must generate correctly; two cycles each. The service's
  // events are counted (forced terminations, relaunches, revocations, offers).
  auto supervised_scenario = [&](const std::string& name, LocalNodeOptions faulty, bool expect_forced) {
    static constexpr int kCycles = 2;  // static: MSVC reports a constexpr used only in a nested lambda as unused (C4189)
    faulty.name = "node0";
    faulty.disk_gib = 1.0;  // Node 0 stages its objects in files under its staging root, so a dead worker leaves orphans behind
    run_scenario(name, {faulty}, [&](LocalCluster& cl, coordinator::Coordinator& c, const coordinator::ClusterPlan& base_plan) {
      coordinator::ClusterPlan plan = base_plan;
      {
        const auto& g = c.manifest().geometry;
        for (const auto& st : plan.stages) {
          if (st.domain != 0) continue;
          for (std::uint32_t layer = st.layers.begin; layer < st.layers.end; ++layer) {
            plan.targets.emplace_back(objects::dense_object_name(layer), objects::AllocationTarget::kTemporaryBacking);
            if (g.shared_expert_ff > 0)
              plan.targets.emplace_back(objects::shared_expert_object_name(layer), objects::AllocationTarget::kTemporaryBacking);
            for (std::uint32_t e = 0; e < g.n_experts; ++e)
              plan.targets.emplace_back(objects::expert_object_name(layer, e), objects::AllocationTarget::kTemporaryBacking);
          }
        }
      }
      Distribution terminate_ms, relaunch_ms;
      bool all_ok = true;
      std::uint64_t residual_at_relaunch = 0, staged_before_fault = 0;
      for (int cycle = 1; cycle <= kCycles && all_ok; ++cycle) {
        const std::string k = name + "_cycle" + std::to_string(cycle);
        auto prep = c.prepare(plan);
        auto gen = prep.is_ok() ? c.generate(request()) : Result<coordinator::GenerationResult>(prep.status());
        const bool generated = gen.is_ok() && (!have_reference || gen->tokens == reference);
        r.check(k + (have_reference ? "_generate_matches_reference" : "_generate_completed"), generated,
                gen.is_ok() ? "" : gen.status().to_string());
        if (!generated) {
          all_ok = false;
          break;
        }
        const std::size_t offered_before = cl.service_event_count(0, "offered");
        if (const auto staged = staged_bytes(cl.staging_root(0)); staged != ~std::uint64_t{0}) staged_before_fault = std::max(staged_before_fault, staged);
        Stopwatch sw;
        if (expect_forced) {
          (void)cl.local_activity(0);
          auto forced = cl.wait_service_event(0, "forced_termination", static_cast<std::size_t>(cycle), 30s);
          r.check(k + "_forced_termination_observed", forced.is_ok(), forced.status().to_string());
          if (!forced.is_ok()) {
            all_ok = false;
            break;
          }
          terminate_ms.add(sw.elapsed_ms());
          residual_at_relaunch += forced->residual_bytes;
          r.check(k + "_nothing_staged_at_relaunch", forced->residual_bytes == 0, std::to_string(forced->residual_bytes) + " bytes");
        } else {
          (void)c.release();  // the worker dies in its cleanup
          auto relaunched = cl.wait_service_event(0, "worker_restarted", static_cast<std::size_t>(cycle), 30s);
          r.check(k + "_worker_relaunched", relaunched.is_ok(), relaunched.status().to_string());
          if (!relaunched.is_ok()) {
            all_ok = false;
            break;
          }
          relaunch_ms.add(sw.elapsed_ms());
        }
        // Orphan recovery runs when the relaunched worker opens its lease store.
        bool clean = false;
        for (int i = 0; i < 100 && !clean; ++i) {
          clean = staged_bytes(cl.staging_root(0)) == 0;
          if (!clean) std::this_thread::sleep_for(50ms);
        }
        r.check(k + "_staging_clean_after_relaunch", clean);
        if (expect_forced) {
          (void)c.release();  // the lease is gone; release what is left on Father and the other Node
          (void)cl.local_idle(0);  // the simulated user is idle again: the service offers the Node once more
        }
        auto offered = cl.wait_service_event(0, "offered", offered_before + 1, 30s);
        r.check(k + "_offered_again", offered.is_ok(), offered.status().to_string());
        all_ok = all_ok && offered.is_ok() && c.connect().is_ok();
      }
      (void)c.release();
      const auto forced_count = cl.service_event_count(0, "forced_termination");
      const auto restart_count = cl.service_event_count(0, "worker_restarted");
      r.check(name + "_forced_terminations_counted", forced_count == (expect_forced ? std::size_t{kCycles} : std::size_t{0}),
              std::to_string(forced_count) + " forced terminations in " + std::to_string(kCycles) + " cycles");
      r.check(name + "_relaunches_counted", restart_count == (expect_forced ? std::size_t{0} : std::size_t{kCycles}),
              std::to_string(restart_count) + " relaunches after an exit");
      r.check(name + "_other_node_staging_clean", staged_bytes(cl.staging_root(1)) == 0);
      r.metric(name + ".forced_terminations", forced_count);
      r.metric(name + ".worker_relaunches", restart_count);
      r.metric(name + ".revocations", cl.service_event_count(0, "revoked"));
      r.metric(name + ".offers", cl.service_event_count(0, "offered"));
      r.metric(name + ".cooperative_deadline_ms", deadline_ms);
      r.metric(name + ".residual_bytes_at_relaunch", residual_at_relaunch);
      r.metric(name + ".staged_bytes_before_fault", staged_before_fault);  // what orphan recovery had to find (0: held in RAM)
      r.metric(name + ".activity_to_forced_termination_ms", terminate_ms, "ms");
      r.metric(name + ".release_to_relaunch_ms", relaunch_ms, "ms");
    }, true, /*supervised=*/true);
  };
  if (want("supervised_forced_termination")) {
    LocalNodeOptions hung;
    hung.hang_at = "cleanup";
    supervised_scenario("supervised_forced_termination", hung, true);
  }
  if (want("supervised_crash_recovery")) {
    LocalNodeOptions dying;
    dying.crash_at = "cleanup";
    supervised_scenario("supervised_crash_recovery", dying, false);
  }

  if (census_before) {
    const auto after = take_census(census_roots);
    emit_census_diff(r, diff_census(*census_before, after, census_before->taken_ns, after.taken_ns));
  }
  for (const char* id : {"HQ-REL-01", "HQ-STORE-01", "HQ-PERF-04"}) r.pending(id);
  if (!args.has("keep-work")) {
    std::error_code ec;
    fs::remove_all(work, ec);
  }
  return emit(args, r, total.elapsed_ms() / 1000.0);
}

// ---- placement-validate (HQ-PLACE-01) -----------------------------------------------------------------------------
//
// The placement search ranks candidate plans under its cost model. This runs the top-N of them through the local cluster
// harness (each converted to a ClusterPlan, a fresh LocalCluster per candidate) and puts the model's prediction next to
// the measurement: decode tok/s, prefill time and prepare time per candidate, the predicted/measured ratio, and how well
// the predicted order matches the measured order (Spearman rank correlation, share of concordant pairs, whether the
// predicted best is the measured best). On the fixture model with the Synthetic profiles this only exercises the machinery
// (localhost processes on one machine are not the target hardware); the same command on the real artifact, real backend and
// measured profiles is the qualification run once Nodes can be real machines.
int cmd_placement_validate(const cli::Args& args) {
  Stopwatch total;
  const RunContext rc = make_run_context(args);
  BenchmarkResult r(args.get("experiment", "dev-placement-validate"), probe_host());
  if (auto st = backends::check_backend_name(args.get("backend", "reference")); !st.is_ok()) {
    std::fprintf(stderr, "placement-validate: %s\n", st.to_string().c_str());
    return exit_code_for(st);
  }
  const fs::path work = args.has("work") ? fs::path(args.get("work")) : default_work_dir("placement-validate");
  auto fail = [&](const std::string& what, const Status& st) {
    std::fprintf(stderr, "placement-validate: %s: %s\n", what.c_str(), st.to_string().c_str());
    cleanup_work(args, work);
    return exit_code_for(st);
  };
  auto model_dir = ensure_model(args, work);
  if (!model_dir.is_ok()) return fail("model", model_dir.status());
  auto store = objects::CanonicalModelStore::open(model_dir.value());
  if (!store.is_ok()) return fail("model", store.status());
  const objects::ModelManifest& manifest = store.value()->manifest();

  // ---- the placement request, costed for the model that will actually run ----
  const std::string profiles_dir = args.get("profiles", source_dir() + "/fixtures/profiles");
  auto resolve = [&](const std::string& name) { return fs::exists(name) ? name : profiles_dir + "/" + name; };
  placement::PlacementRequest req;
  {
    auto father = placement::load_hardware_profile(resolve(args.get("father", "Father-4060Ti-7600.json")));
    if (!father.is_ok()) return fail("father profile", father.status());
    req.father = father.value();
    for (const auto& n : args.has("node") ? args.all("node")
                                          : std::vector<std::string>{"Node-G14-4070-8945HS.json", "Node-3060-5600.json"}) {
      auto p = placement::load_hardware_profile(resolve(n));
      if (!p.is_ok()) return fail("node profile", p.status());
      req.nodes.push_back(p.value());
    }
    auto net = placement::load_network_profile(resolve(args.get("network", "network-gige-simulated.json")));
    if (!net.is_ok()) return fail("network profile", net.status());
    req.network = net.value();
  }
  const auto max_new = static_cast<std::uint32_t>(args.integer("max-new", 16));
  const auto prompt_len = static_cast<std::uint32_t>(args.integer("prompt-len", 32));
  const auto prefill_chunk = static_cast<std::uint32_t>(args.integer("prefill-chunk", 16));
  const auto q = static_cast<std::uint32_t>(args.integer("q", 1));
  const int repeat = static_cast<int>(args.integer("repeat", 2));
  const auto top = static_cast<std::size_t>(args.integer("top", 4));
  if (top < 1 || repeat < 1 || prompt_len < 1 || q < 1) {
    std::fprintf(stderr, "placement-validate: --top, --repeat, --prompt-len and --q must be positive\n");
    cleanup_work(args, work);
    return 2;
  }
  {
    auto cost = planning::cost_inputs_from_manifest(manifest);
    if (!cost.is_ok()) return fail("model cost inputs", cost.status());
    req.model = cost.value();
  }
  // The fixture model's quantizations (f32, q8_0-fixture) have no entry in the shipped Synthetic profiles (they list the
  // product's IQ types): give each profile a Synthetic stand-in so the search can cost the fixture. A real model with a
  // quantization the profile lacks is a genuine input gap and is left to the search to reject.
  if (!args.has("model")) {
    auto backfill = [&](placement::HardwareProfile& p) {
      if (p.cpu.expert_bytes_per_s.empty()) return;
      const double any = p.cpu.expert_bytes_per_s.begin()->second.value;
      for (const auto& layer : req.model.layers)
        if (!p.cpu.expert_bytes_per_s.count(layer.quant))
          p.cpu.expert_bytes_per_s[layer.quant] = placement::Quantity::synthetic(any, "synthetic: fixture model stand-in");
    };
    backfill(req.father);
    for (auto& node : req.nodes) backfill(node);
  }
  const std::uint32_t max_context = prompt_len + max_new + 8;
  req.context_tokens = max_context;
  req.prompt_tokens = prompt_len;
  req.output_tokens = max_new;
  req.q = q;
  req.prefill_chunk = prefill_chunk;
  req.granularity = static_cast<std::uint32_t>(args.integer("granularity", 4));
  auto searched = placement::search_placements(req);
  if (!searched.is_ok()) return fail("placement search", searched.status());
  const auto& candidates = searched->candidates;
  if (candidates.empty()) {
    std::fprintf(stderr, "placement-validate: the placement search found no feasible plan for this model and these profiles (%zu rejected)\n",
                 searched->rejected.size());
    for (std::size_t i = 0; i < searched->rejected.size() && i < 3; ++i)
      for (const auto& reason : searched->rejected[i].reasons)
        std::fprintf(stderr, "  %s: %s\n", searched->rejected[i].key.c_str(), reason.c_str());
    cleanup_work(args, work);
    return 1;
  }
  const std::size_t n = std::min(top, candidates.size());

  // ---- backend, provenance, prompts ----
  std::size_t max_nodes = 0;
  for (std::size_t i = 0; i < n; ++i) max_nodes = std::max(max_nodes, candidates[i].node_ids().size());
  auto backend = resolve_backend(args, model_dir.value(), max_nodes);
  if (!backend.is_ok()) return fail("backend", backend.status());
  const bool fixture = !args.has("model");
  // --endpoint PROFILE_ID=HOST:PORT@FINGERPRINT: the Nodes of the profiles are already running (real links); a candidate uses
  // the endpoints of the Nodes it places, in its own order. Without them every candidate gets localhost Node processes.
  auto endpoint_list = parse_node_specs(args.all("endpoint"), "endpoint");
  if (!endpoint_list.is_ok()) return fail("endpoint", endpoint_list.status());
  std::map<std::string, coordinator::NodeEndpoint> endpoints;
  bool localhost_nodes = false;
  for (const auto& ep : endpoint_list.value()) {
    endpoints[ep.name] = ep;
    localhost_nodes = localhost_nodes || is_loopback_host(ep.endpoint.host);
  }
  if (endpoints.empty() && max_nodes > 0) localhost_nodes = true;
  RunContext effective = rc;
  effective.on_target = rc.on_target && !localhost_nodes;
  apply_run_context(r, effective);
  if (localhost_nodes) r.mark_simulated("localhost_cluster", true);
  r.config("external_nodes", endpoints.size());
  nlohmann::json synthetic_ids = nlohmann::json::array();
  auto note_profile = [&](const placement::HardwareProfile& p) {
    if (placement::weakest_provenance(p) == placement::Provenance::kSynthetic) synthetic_ids.push_back(p.id);
  };
  note_profile(req.father);
  for (const auto& node : req.nodes) note_profile(node);
  if (placement::weakest_provenance(req.network) == placement::Provenance::kSynthetic) synthetic_ids.push_back("network");
  if (!synthetic_ids.empty()) r.mark_simulated("synthetic_profiles", synthetic_ids);
  if (fixture) r.mark_simulated("fixture_model", true);
  if (!backend->real()) r.mark_simulated("reference_backend", true);
  if (args.has("impair")) r.mark_simulated("network_preset", args.get("impair"));
  if (effective.on_target && !fixture && backend->real() && synthetic_ids.empty()) r.set_measured();  // finish() overrides it when anything was marked
  r.backend(backend->name, backend->build_hash);
  r.config("backend", backend->name);
  r.config("candidates_requested", top);
  r.config("candidates_run", n);
  r.config("prompt_len", prompt_len);
  r.config("max_new_tokens", max_new);
  r.config("q", q);
  r.config("repeat", repeat);
  r.config("placement_provenance", std::string(placement::to_string(searched->provenance)));
  r.config("meets_min_repetitions", repeat >= 5);

  auto sets = build_prompt_sets(args, model_dir.value(), backend.value().real() ? &backend.value() : nullptr, {prompt_len}, max_new,
                                !args.has("no-reference"), r);
  if (!sets.is_ok()) return fail("prompts", sets.status());

  // ---- one pass per candidate ----
  std::vector<double> predicted_decode, measured_decode, predicted_prefill, measured_prefill, predicted_prepare, measured_prepare;
  std::size_t completed = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& cand = candidates[i];
    const std::string tag = "cand" + std::to_string(i);
    std::map<std::string, int> node_index;
    for (const auto& id : cand.node_ids())
      if (!node_index.count(id)) node_index.emplace(id, static_cast<int>(node_index.size()));
    auto plan = planning::to_cluster_plan(cand, manifest, req.father.id, node_index, max_context, std::max(q, prefill_chunk));
    if (!plan.is_ok()) {
      r.check(tag + "_plan_executable", false, plan.status().to_string());
      continue;
    }
    PassSummary summary;
    PassContext pc(args, r);
    pc.prefix = tag + ".";
    pc.work = work;
    pc.model_dir = model_dir.value();
    pc.plan_text = plan->describe();
    pc.plan = plan.value();
    pc.node_count = node_index.size();
    if (!endpoints.empty() && !node_index.empty()) {
      std::vector<coordinator::NodeEndpoint> chosen(node_index.size());
      bool all_present = true;
      for (const auto& [id, index] : node_index) {
        auto it = endpoints.find(id);
        if (it == endpoints.end()) {
          r.check(str_cat(tag, "_endpoint_for_", id), false, "no --endpoint for the node profile this candidate places");
          all_present = false;
          break;
        }
        chosen[static_cast<std::size_t>(index)] = it->second;
        chosen[static_cast<std::size_t>(index)].name = id;
      }
      if (!all_present) continue;
      pc.external = std::move(chosen);
    }
    pc.node_options.assign(node_index.size(), LocalNodeOptions{});
    for (const auto& ledger : cand.ledgers) {  // each Node offers the VRAM its domain is planned to use (plus slack)
      auto it = node_index.find(ledger.domain_id);
      if (it != node_index.end())
        pc.node_options[static_cast<std::size_t>(it->second)].vram_gib =
            static_cast<double>(ledger.vram_used()) / static_cast<double>(cli::kGiB) + 0.5;
    }
    pc.qs = {q};
    pc.contexts = {prompt_len};
    pc.prompt_sets = sets.value();
    pc.max_new = max_new;
    pc.prefill_chunk = prefill_chunk;
    pc.repeat = repeat;
    pc.direct_peer = !args.has("relay");
    pc.fixture = fixture;
    pc.resources = !args.has("no-resources");
    pc.backend = &backend.value();
    pc.summary = &summary;
    auto st = run_cluster_pass(pc);
    if (!st.is_ok()) r.check(tag + "_pass_completed", false, st.to_string());
    const auto& pm = cand.metrics;
    r.trace({{"kind", "placement_candidate"}, {"rank", i}, {"plan", cand.key}, {"nodes", node_index.size()},
             {"predicted_decode_tok_s", pm.decode_tok_s}, {"measured_decode_tok_s", summary.decode_tok_s},
             {"predicted_prefill_s", pm.prefill_s}, {"measured_prefill_s", summary.prefill_s},
             {"predicted_prepare_s", pm.prepare_s}, {"measured_prepare_s", summary.prepare_s}, {"completed", summary.completed}});
    r.metric(tag + ".predicted.decode_tok_s", pm.decode_tok_s);
    r.metric(tag + ".predicted.prefill_s", pm.prefill_s);
    r.metric(tag + ".predicted.prepare_s", pm.prepare_s);
    r.metric(tag + ".predicted.objective", pm.objective);
    r.metric(tag + ".measured.decode_tok_s", summary.decode_tok_s);
    r.metric(tag + ".measured.prefill_s", summary.prefill_s);
    r.metric(tag + ".measured.prepare_s", summary.prepare_s);
    if (pm.decode_tok_s > 0 && summary.decode_tok_s > 0) r.metric(tag + ".measured_over_predicted.decode", summary.decode_tok_s / pm.decode_tok_s);
    if (pm.prefill_s > 0 && summary.prefill_s > 0) r.metric(tag + ".measured_over_predicted.prefill", summary.prefill_s / pm.prefill_s);
    if (pm.prepare_s > 0 && summary.prepare_s > 0) r.metric(tag + ".measured_over_predicted.prepare", summary.prepare_s / pm.prepare_s);
    r.metric(tag + ".plan", cand.key);
    if (!summary.completed) continue;
    ++completed;
    predicted_decode.push_back(pm.decode_tok_s);
    measured_decode.push_back(summary.decode_tok_s);
    predicted_prefill.push_back(pm.prefill_s);
    measured_prefill.push_back(summary.prefill_s);
    predicted_prepare.push_back(pm.prepare_s);
    measured_prepare.push_back(summary.prepare_s);
  }
  r.check("every_candidate_executed", completed == n, std::to_string(completed) + " of " + std::to_string(n) + " candidates completed");

  // ---- predicted vs measured order ----
  auto emit_agreement = [&](const char* what, const std::vector<double>& pred, const std::vector<double>& meas, bool higher_is_better) {
    const RankAgreement a = compare_rankings(pred, meas, higher_is_better);
    const std::string k = std::string("rank.") + what;
    r.metric(k + ".candidates", a.n);
    r.metric(k + ".valid", a.valid);
    if (a.valid) {
      r.metric(k + ".spearman", a.spearman);
      r.metric(k + ".concordant_pair_fraction", a.concordant_fraction);
      r.metric(k + ".predicted_best_is_measured_best", a.best_agrees);
    }
  };
  emit_agreement("decode_tok_s", predicted_decode, measured_decode, true);
  emit_agreement("prefill_s", predicted_prefill, measured_prefill, false);
  emit_agreement("prepare_s", predicted_prepare, measured_prepare, false);
  r.pending("HQ-PLACE-01");
  int code = emit(args, r, total.elapsed_ms() / 1000.0);
  cleanup_work(args, work);
  return code;
}

}  // namespace clusterlm::bench
