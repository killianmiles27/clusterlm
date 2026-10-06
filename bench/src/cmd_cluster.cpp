// clusterlm-bench cluster / faults: multi-process clusters on this machine through the real protocol.
//
// Father runs in this process (Coordinator); every Node is a separate clusterlm-node process reached over
// authenticated TCP. Results are Synthetic: localhost processes, a fixture model and optional simulated
// network conditions exercise the software, not the target hardware.
#include <cstdio>
#include <filesystem>
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/protocol/messages.hpp"
#include "commands.hpp"
#include "local_cluster.hpp"

namespace clusterlm::bench {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultPlan = "0-4@father,4-10@0,10-13@1,13-16@father";

struct ClusterSetup {
  fs::path work;
  fs::path model_dir;
  std::unique_ptr<LocalCluster> cluster;
  std::optional<transport::NetworkConditions> impairment;
};

fs::path default_work_dir(const std::string& name) {
  return fs::temp_directory_path() / ("clusterlm-bench-" + name + "-" + std::to_string(monotonic_ns()));
}

Result<fs::path> ensure_model(const cli::Args& args, const fs::path& work) {
  if (args.has("model")) return fs::path(args.get("model"));
  const fs::path dir = work / "fixture-model";
  objects::FixtureSpec spec;
  spec.seed = args.integer("seed", spec.seed);
  CLM_RETURN_IF_ERROR(objects::write_fixture_model(spec, dir).status());
  return dir;
}

Result<ClusterSetup> setup(const cli::Args& args, const std::string& name, std::size_t nodes,
                           const std::vector<LocalNodeOptions>& overrides = {}) {
  ClusterSetup s;
  s.work = args.has("work") ? fs::path(args.get("work")) : default_work_dir(name);
  CLM_ASSIGN_OR_RETURN(s.model_dir, ensure_model(args, s.work));
  LocalClusterOptions opts;
  opts.work_dir = s.work;
  opts.tls = !args.has("insecure");
  for (std::size_t i = 0; i < nodes; ++i) {
    LocalNodeOptions n = i < overrides.size() ? overrides[i] : LocalNodeOptions{};
    if (n.name.empty()) n.name = "node" + std::to_string(i);
    if (args.has("impair")) n.impair = args.get("impair");
    opts.nodes.push_back(n);
  }
  if (args.has("impair")) {
    CLM_ASSIGN_OR_RETURN(auto preset, transport::network_preset(args.get("impair")));
    s.impairment = preset;
  }
  CLM_ASSIGN_OR_RETURN(s.cluster, LocalCluster::start(opts));
  return s;
}

coordinator::CoordinatorConfig father_config(const ClusterSetup& s, const cli::Args& args) {
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = s.model_dir;
  cfg.security = s.cluster ? s.cluster->father_security() : transport::SecurityConfig{};
  if (!s.cluster) cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  if (s.cluster) cfg.nodes = s.cluster->endpoints();
  cfg.direct_peer = !args.has("relay");
  cfg.impairment = s.impairment;
  cfg.window_timeout = std::chrono::milliseconds(args.integer("window-timeout-ms", 10'000));
  return cfg;
}

Result<std::shared_ptr<domain::Drafter>> make_drafter(const fs::path& model_dir, const objects::ModelManifest& m,
                                                       std::unique_ptr<objects::CanonicalModelStore>& keep) {
  CLM_ASSIGN_OR_RETURN(keep, objects::CanonicalModelStore::open(model_dir));
  CLM_ASSIGN_OR_RETURN(auto d, domain::MtpFixtureDrafter::create(m, *keep));
  return std::shared_ptr<domain::Drafter>(std::move(d));
}

std::vector<std::int32_t> prompt_tokens(std::uint32_t n, std::uint32_t vocab) {
  std::vector<std::int32_t> out;
  for (std::uint32_t i = 0; i < n; ++i) out.push_back(static_cast<std::int32_t>((i * 7919u + 13u) % vocab));
  return out;
}

// Father-only execution of the same model: the correctness reference for every distributed run.
Result<std::vector<std::int32_t>> reference_tokens(const fs::path& model_dir, const std::vector<std::int32_t>& prompt,
                                                   std::uint32_t max_new) {
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = model_dir;
  cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  CLM_ASSIGN_OR_RETURN(auto c, coordinator::Coordinator::create(cfg));
  const auto n = c->manifest().geometry.n_layers;
  CLM_ASSIGN_OR_RETURN(auto plan, coordinator::ClusterPlan::parse("0-" + std::to_string(n / 2) + "@father," +
                                                                      std::to_string(n / 2) + "-" + std::to_string(n) +
                                                                      "@father",
                                                                  n));
  CLM_RETURN_IF_ERROR(c->prepare(plan).status());
  coordinator::GenerationRequest req;
  req.prompt = prompt;
  req.max_new_tokens = max_new;
  CLM_ASSIGN_OR_RETURN(auto gen, c->generate(req));
  CLM_RETURN_IF_ERROR(c->release().status());
  return gen.tokens;
}

std::size_t remote_count(const coordinator::ClusterPlan& p) {
  std::size_t n = 0;
  for (const auto& s : p.stages) n += s.domain != coordinator::kFatherDomain;
  return n;
}

void record_census(BenchmarkResult& r, LocalCluster& cluster, const std::string& label) {
  bool clean = true;
  std::string detail;
  for (std::size_t i = 0; i < cluster.size(); ++i) {
    auto st = cluster.status(i);
    if (!st.is_ok()) {
      clean = false;
      detail += "node" + std::to_string(i) + ": " + st.status().to_string() + "; ";
      continue;
    }
    if (st->census_bytes != 0) clean = false;
    detail += "node" + std::to_string(i) + "=" + std::to_string(st->census_bytes) + "B(" + st->state + ") ";
  }
  r.check(label, clean, detail);
}

}  // namespace

int cmd_cluster(const cli::Args& args) {
  Stopwatch total;
  BenchmarkResult r(args.get("experiment", "dev-cluster"), probe_host());
  r.mark_simulated("localhost_cluster", true);
  r.mark_simulated("fixture_model", !args.has("model"));
  if (args.has("impair")) r.mark_simulated("network_preset", args.get("impair"));
  r.backend("reference", "reference");
  const std::string plan_text = args.get("plan", kDefaultPlan);
  const auto max_new = static_cast<std::uint32_t>(args.integer("max-new", 64));
  const auto prompt_len = static_cast<std::uint32_t>(args.integer("prompt-len", 48));
  const auto repeat = static_cast<int>(args.integer("repeat", 3));
  std::vector<std::uint32_t> qs = {1, 4};
  if (args.has("q")) {
    qs.clear();
    for (const auto& q : cli::split(args.get("q"), ',')) qs.push_back(static_cast<std::uint32_t>(std::stoul(q)));
  }

  // Count Nodes named in the plan.
  std::size_t nodes = 0;
  for (const auto& item : cli::split(plan_text, ','))
    if (item.find("@father") == std::string::npos) ++nodes;
  auto s = setup(args, "cluster", nodes);
  if (!s.is_ok()) {
    std::fprintf(stderr, "setup: %s\n", s.status().to_string().c_str());
    return 1;
  }
  auto coord = coordinator::Coordinator::create(father_config(s.value(), args));
  if (!coord.is_ok()) {
    std::fprintf(stderr, "%s\n", coord.status().to_string().c_str());
    return 1;
  }
  auto& c = *coord.value();
  const auto& m = c.manifest();
  r.model({{"artifact_id", m.artifact_id}, {"root_hash", m.root_hash().hex()}, {"fixture", !args.has("model")}});
  auto plan = coordinator::ClusterPlan::parse(plan_text, m.geometry.n_layers);
  if (!plan.is_ok()) {
    std::fprintf(stderr, "%s\n", plan.status().to_string().c_str());
    return 2;
  }
  r.config("plan", plan->describe());
  r.config("direct_peer", !args.has("relay"));
  r.config("tls", !args.has("insecure"));
  r.config("prompt_len", prompt_len);
  r.config("max_new_tokens", max_new);
  r.config("repeat", repeat);
  r.config("drafter", args.get("drafter", "mtp"));

  const auto prompt = prompt_tokens(prompt_len, m.geometry.vocab_size);
  auto reference = reference_tokens(s->model_dir, prompt, max_new);
  if (!reference.is_ok()) {
    std::fprintf(stderr, "reference: %s\n", reference.status().to_string().c_str());
    return 1;
  }
  std::unique_ptr<objects::CanonicalModelStore> drafter_store;
  auto drafter = make_drafter(s->model_dir, m, drafter_store);
  if (!drafter.is_ok()) {
    std::fprintf(stderr, "%s\n", drafter.status().to_string().c_str());
    return 1;
  }

  if (auto st = c.connect(); !st.is_ok()) {
    std::fprintf(stderr, "connect: %s\n", st.to_string().c_str());
    return 1;
  }
  Distribution prepare_ms;
  auto prep = c.prepare(plan.value());
  if (!prep.is_ok()) {
    std::fprintf(stderr, "prepare: %s\n", prep.status().to_string().c_str());
    return 1;
  }
  prepare_ms.add(prep->total_ms);
  std::uint64_t provisioned = 0;
  for (const auto& n : prep->nodes) {
    provisioned += n.bytes;
    r.metric("prepare.node." + n.node + ".bytes", n.bytes);
    r.metric("prepare.node." + n.node + ".objects", n.objects);
    r.metric("prepare.node." + n.node + ".ms", n.prepare_ms);
  }
  r.metric("prepare.total_ms", prep->total_ms);
  r.metric("prepare.father_resident_bytes", prep->father_resident_bytes);
  r.metric("model.total_object_bytes", [&] {
    std::uint64_t t = 0;
    for (const auto& o : m.objects) t += o.byte_size;
    return t;
  }());
  // Selected-object provisioning: Nodes together receive only their layers, never Father-only material.
  std::uint64_t expected = 0;
  for (const auto& st : plan->stages)
    if (st.domain != coordinator::kFatherDomain) expected += m.total_bytes(st.layers);
  r.check("provisioned_only_assigned_layers", provisioned == expected,
          std::to_string(provisioned) + " bytes provisioned, " + std::to_string(expected) + " assigned");

  const auto boundary = domain::BoundaryLayout::for_geometry(m.geometry);
  for (auto q : qs) {
    coordinator::GenerationRequest req;
    req.prompt = prompt;
    req.max_new_tokens = max_new;
    req.q = q;
    req.prefill_chunk = static_cast<std::uint32_t>(args.integer("prefill-chunk", 16));
    if (q > 1) {
      // --drafter mtp (default): the fixture MTP head — random weights, so it is almost never accepted.
      // --drafter scripted:RATE: replays the reference continuation with deterministic corruption at RATE,
      // exercising every acceptance length. Acceptance on the real model is HQ-MTP-01.
      const std::string kind = args.get("drafter", "mtp");
      if (kind.rfind("scripted", 0) == 0) {
        domain::ScriptedDrafter::Config dc;
        dc.vocab = m.geometry.vocab_size;
        dc.corruption_rate = kind.size() > 9 ? std::stod(kind.substr(9)) : 0.3;
        std::vector<std::int32_t> seq = prompt;
        seq.insert(seq.end(), reference->begin(), reference->end());
        req.drafter = std::make_shared<domain::ScriptedDrafter>(std::move(seq), dc);
      } else {
        req.drafter = drafter.value();
      }
    }
    Distribution round_ms, tok_s, ttft, accepted, bytes_per_token;
    bool tokens_ok = true, bounded_payload = true;
    for (int rep = 0; rep < repeat; ++rep) {
      auto gen = c.generate(req);
      if (!gen.is_ok()) {
        r.check("q" + std::to_string(q) + "_generate", false, gen.status().to_string());
        tokens_ok = false;
        break;
      }
      tokens_ok = tokens_ok && gen->tokens == reference.value();
      ttft.add(gen->first_token_ms);
      if (gen->decode_ms > 0) tok_s.add(1000.0 * gen->decode_tokens / gen->decode_ms);
      std::uint64_t decode_bytes = 0;
      for (const auto& rd : gen->rounds) {
        if (rd.prefill) continue;
        round_ms.add(rd.total_ms);
        accepted.add(rd.accepted);
        decode_bytes += rd.boundary_payload_bytes;
        // NET-01: every activation message is the boundary payload plus bounded fixed fields — there is no
        // room for token IDs or other per-position content.
        const std::uint64_t activation = std::uint64_t{rd.positions} * boundary.bytes_per_position();
        const std::uint64_t per_message_overhead = 512;
        if (rd.boundary_payload_bytes >
            rd.boundary_messages * (activation + per_message_overhead))
          bounded_payload = false;
        if (rep == 0 && q == qs.front())
          r.trace({{"q", q}, {"positions", rd.positions}, {"accepted", rd.accepted}, {"total_ms", rd.total_ms},
                   {"prefix_ms", rd.prefix_ms}, {"remote_ms", rd.remote_ms}, {"tail_ms", rd.tail_ms},
                   {"commit_ms", rd.commit_ms}, {"boundary_messages", rd.boundary_messages},
                   {"boundary_payload_bytes", rd.boundary_payload_bytes}});
      }
      if (gen->decode_tokens > 0) bytes_per_token.add(static_cast<double>(decode_bytes) / gen->decode_tokens);
    }
    const std::string k = "q" + std::to_string(q);
    r.check(k + "_tokens_match_father_only_reference", tokens_ok);
    r.check(k + "_activation_messages_bounded_by_boundary_abi", bounded_payload);
    r.metric(k + ".round_ms", round_ms, "ms");
    r.metric(k + ".decode_tok_s_fixture", tok_s, "tok/s");
    r.metric(k + ".ttft_ms", ttft, "ms");
    r.metric(k + ".accepted_per_round", accepted, "tokens");
    r.metric(k + ".boundary_bytes_per_emitted_token", bytes_per_token, "bytes");
  }
  r.metric("boundary.bytes_per_position", boundary.bytes_per_position());
  r.metric("boundary.remote_stages", remote_count(plan.value()));

  // Release: every Node must report resources released and storage cleaned with zero residual bytes.
  auto rel = c.release();
  if (!rel.is_ok()) {
    r.check("release", false, rel.status().to_string());
  } else {
    for (const auto& n : rel->nodes) {
      r.check("release_" + n.node + "_storage_cleaned", n.resources_released && n.storage_cleaned && n.residual_bytes == 0,
              n.errors);
      r.metric("release." + n.node + ".ms", n.release_ms);
    }
  }
  record_census(r, *s->cluster, "post_release_census_zero");
  for (const char* id : {"HQ-PERF-01", "HQ-NET-02", "HQ-MTP-01", "HQ-PROV-01"}) r.pending(id);
  return emit(args, r, total.elapsed_ms() / 1000.0);
}

// ---- faults ------------------------------------------------------------------------------------------------

int cmd_faults(const cli::Args& args) {
  Stopwatch total;
  BenchmarkResult r(args.get("experiment", "dev-faults"), probe_host());
  r.mark_simulated("localhost_cluster", true);
  r.mark_simulated("fixture_model", !args.has("model"));
  const std::string plan_text = args.get("plan", kDefaultPlan);
  const auto max_new = static_cast<std::uint32_t>(args.integer("max-new", 16));
  const fs::path work = args.has("work") ? fs::path(args.get("work")) : default_work_dir("faults");
  auto model_dir = ensure_model(args, work);
  if (!model_dir.is_ok()) {
    std::fprintf(stderr, "%s\n", model_dir.status().to_string().c_str());
    return 1;
  }
  std::vector<std::int32_t> prompt;
  std::vector<std::int32_t> reference;
  {
    auto m = objects::CanonicalModelStore::open(model_dir.value());
    if (!m.is_ok()) return 1;
    prompt = prompt_tokens(24, m.value()->manifest().geometry.vocab_size);
    auto ref = reference_tokens(model_dir.value(), prompt, max_new);
    if (!ref.is_ok()) return 1;
    reference = ref.value();
  }

  // One scenario = a fresh two-Node cluster. Returns the setup for follow-up checks.
  auto run_scenario = [&](const std::string& name, std::vector<LocalNodeOptions> node_opts,
                          const std::function<void(LocalCluster&, coordinator::Coordinator&,
                                                   const coordinator::ClusterPlan&)>& body) {
    cli::Args sub = args;
    LocalClusterOptions opts;
    opts.work_dir = work / name;
    opts.tls = !args.has("insecure");
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
    if (auto st = coord.value()->connect(); !st.is_ok()) {
      r.check(name + "_connect", false, st.to_string());
      return;
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
    r.check(name + "_regenerate_matches_reference", gen.is_ok() && gen->tokens == reference,
            gen.is_ok() ? "" : gen.status().to_string());
    (void)c.release();
    record_census(r, cl, name + "_census_zero_after_release");
  };

  // 1. Abrupt Node crash at each lifecycle phase, then restart with orphan recovery.
  for (const char* phase : {"transfer", "hashing", "mapping", "allocation", "ready", "inference", "cleanup"}) {
    const std::string name = std::string("crash_") + phase;
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

  // 2. Local user activity while Ready: the Node revokes the lease without consulting Father.
  run_scenario("local_activity", {}, [&](LocalCluster& cl, coordinator::Coordinator& c, const auto& plan) {
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
  {
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
  {
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
  run_scenario("release_cycles", {}, [&](LocalCluster& cl, coordinator::Coordinator& c, const auto& plan) {
    Distribution prep_ms, release_ms;
    bool ok = true;
    for (int i = 0; i < cycles && ok; ++i) {
      auto prep = c.prepare(plan);
      ok = prep.is_ok();
      if (!ok) break;
      prep_ms.add(prep->total_ms);
      // Two chats in one Ready lease reuse allocations: no reprovisioning between them.
      for (int chat = 0; chat < 2 && ok; ++chat) {
        auto gen = c.generate(request());
        ok = gen.is_ok() && gen->tokens == reference;
      }
      Stopwatch sw;
      auto rel = c.release();
      release_ms.add(sw.elapsed_ms());
      ok = ok && rel.is_ok();
      if (rel.is_ok())
        for (const auto& n : rel->nodes) ok = ok && n.storage_cleaned && n.residual_bytes == 0;
      if (!c.connect().is_ok()) ok = false;
    }
    r.check("release_cycles_all_clean_and_correct", ok);
    r.metric("release_cycles.prepare_ms", prep_ms, "ms");
    r.metric("release_cycles.release_ms", release_ms, "ms");
    record_census(r, cl, "release_cycles_final_census_zero");
  });

  for (const char* id : {"HQ-REL-01", "HQ-STORE-01", "HQ-PERF-04"}) r.pending(id);
  if (!args.has("keep-work")) {
    std::error_code ec;
    fs::remove_all(work, ec);
  }
  return emit(args, r, total.elapsed_ms() / 1000.0);
}

}  // namespace clusterlm::bench
