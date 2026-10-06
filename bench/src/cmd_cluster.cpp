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
#include <thread>

#include "clusterlm/common/clock.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/platform/process.hpp"
#include "clusterlm/protocol/messages.hpp"
#include "bench_common.hpp"
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

Result<ClusterSetup> setup(const cli::Args& args, const fs::path& work, const fs::path& model_dir, std::size_t nodes,
                           const std::vector<LocalNodeOptions>& overrides = {}) {
  ClusterSetup s;
  s.work = work;
  s.model_dir = model_dir;
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

coordinator::CoordinatorConfig father_config(const ClusterSetup& s, const cli::Args& args,
                                             std::optional<bool> direct_peer = std::nullopt) {
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = s.model_dir;
  cfg.security = s.cluster ? s.cluster->father_security() : transport::SecurityConfig{};
  if (!s.cluster) cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  if (s.cluster) cfg.nodes = s.cluster->endpoints();
  cfg.direct_peer = direct_peer.value_or(!args.has("relay"));
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
                                                   std::uint32_t max_new, std::uint32_t max_context = 2048) {
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = model_dir;
  cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  CLM_ASSIGN_OR_RETURN(auto c, coordinator::Coordinator::create(cfg));
  const auto n = c->manifest().geometry.n_layers;
  CLM_ASSIGN_OR_RETURN(auto plan, coordinator::ClusterPlan::parse("0-" + std::to_string(n / 2) + "@father," +
                                                                      std::to_string(n / 2) + "-" + std::to_string(n) +
                                                                      "@father",
                                                                  n));
  plan.max_context = std::max(plan.max_context, max_context);
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

namespace {

// Layer split for a tier on a model with `n` layers: Ultra = Father + two Nodes, Strong = Father + one Node,
// Fast = Father only. Prefix and tail always stay on Father (token IDs never leave it).
std::string plan_for_tier(const std::string& tier, std::uint32_t n) {
  auto r = [](std::uint32_t a, std::uint32_t b) { return std::to_string(a) + "-" + std::to_string(b); };
  if (tier == "fast") return r(0, n / 2) + "@father," + r(n / 2, n) + "@father";
  if (tier == "strong") return r(0, n / 4) + "@father," + r(n / 4, n * 3 / 4) + "@0," + r(n * 3 / 4, n) + "@father";
  return r(0, n / 4) + "@father," + r(n / 4, n * 5 / 8) + "@0," + r(n * 5 / 8, n * 13 / 16) + "@1," + r(n * 13 / 16, n) +
         "@father";
}

// Fixture tokenization of a corpus file: bytes folded into the vocabulary. Real corpora need the Father tokenizer
// (HQ-MTP-01 stays pending until the real model is in play).
std::vector<std::vector<std::int32_t>> corpus_prompts(const std::string& dir, std::uint32_t len, std::uint32_t vocab) {
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec))
    if (e.is_regular_file()) files.push_back(e.path());
  std::sort(files.begin(), files.end());
  std::vector<std::vector<std::int32_t>> out;
  for (const auto& f : files) {
    std::ifstream in(f, std::ios::binary);
    std::vector<std::int32_t> p;
    char c;
    while (p.size() < len && in.get(c)) p.push_back(static_cast<std::int32_t>(static_cast<unsigned char>(c)) % static_cast<std::int32_t>(vocab));
    const std::size_t n0 = p.size();
    while (n0 > 0 && p.size() < len) p.push_back(p[p.size() % n0]);
    if (!p.empty()) out.push_back(std::move(p));
  }
  return out;
}

struct Cfg {
  std::uint32_t q = 1;
  std::uint32_t context = 0;  // prompt length
  std::string key;
  std::vector<std::vector<std::int32_t>> prompts;
  std::vector<std::vector<std::int32_t>> references;  // Father-only tokens per prompt
  std::vector<std::shared_ptr<domain::Drafter>> drafters;
  Distribution round_ms, tok_s, ttft, ttft_cold, ttft_warm, accepted, bytes_per_token, prefill_ms, prefill_tok_s;
  Distribution draft_ms, verify_ms, commit_ms, wait_ms, father_compute_ms;
  std::map<std::size_t, Distribution> node_compute_ms, node_cpu_expert_ms;
  std::uint64_t proposed = 0, accepted_draft = 0, rounds = 0, tokens_emitted = 0;
  double wait_total_ms = 0, round_total_ms = 0;
  bool tokens_ok = true, bounded_payload = true, failed = false;
  coordinator::GenerationRequest request;
};

struct PassContext {
  const cli::Args& args;
  BenchmarkResult& r;
  std::string prefix;
  fs::path work, model_dir;
  std::string plan_text;
  std::vector<std::uint32_t> qs, contexts;
  std::uint32_t max_new = 64, prefill_chunk = 16;
  int repeat = 3;
  bool direct_peer = true;
  bool fixture = true;
};

// Folds one generation into a config's distributions.
void record_generation(Cfg& c, const coordinator::GenerationResult& gen, const domain::BoundaryLayout& boundary,
                       const std::vector<std::int32_t>& reference, int rep, bool first_config_first_rep, BenchmarkResult& r,
                       const std::string& trace_prefix) {
  c.tokens_ok = c.tokens_ok && gen.tokens == reference;
  c.ttft.add(gen.first_token_ms);
  (rep == 0 ? c.ttft_cold : c.ttft_warm).add(gen.first_token_ms);
  c.prefill_ms.add(gen.prefill_ms);
  if (gen.prefill_ms > 0) c.prefill_tok_s.add(1000.0 * static_cast<double>(c.context) / gen.prefill_ms);
  if (gen.decode_ms > 0) c.tok_s.add(1000.0 * gen.decode_tokens / gen.decode_ms);
  std::uint64_t decode_bytes = 0;
  for (const auto& rd : gen.rounds) {
    if (rd.prefill) continue;
    c.round_ms.add(rd.total_ms);
    c.accepted.add(rd.accepted);
    ++c.rounds;
    c.tokens_emitted += rd.accepted;
    // positions-1 draft tokens are proposed in a round; accepted-1 of them were verified correct (the first
    // position is the model's own next token and is always accepted).
    if (rd.positions > 0) c.proposed += rd.positions - 1;
    if (rd.accepted > 0) c.accepted_draft += rd.accepted - 1;
    c.draft_ms.add(rd.draft_ms);
    c.commit_ms.add(rd.commit_ms);
    c.verify_ms.add(rd.prefix_ms + rd.remote_ms + rd.tail_ms);
    decode_bytes += rd.boundary_payload_bytes;
    // Compute vs wait per stage: Father's prefix/tail are local compute; the remote section's wall time minus the
    // Nodes' reported compute is wait (link, queueing, pipeline dependency). Tolerant of missing timings.
    c.father_compute_ms.add(rd.prefix_ms + rd.tail_ms);
    double remote_compute = 0;
    for (std::size_t i = 0; i < rd.remote_timings.size(); ++i) {
      const double comp = static_cast<double>(rd.remote_timings[i].compute_ns) * 1e-6;
      c.node_compute_ms[i].add(comp);
      c.node_cpu_expert_ms[i].add(static_cast<double>(rd.remote_timings[i].cpu_expert_ns) * 1e-6);
      remote_compute += comp;
    }
    if (!rd.remote_timings.empty()) {
      const double wait = std::max(0.0, rd.remote_ms - remote_compute);
      c.wait_ms.add(wait);
      c.wait_total_ms += wait;
    }
    c.round_total_ms += rd.total_ms;
    // NET-01: every activation message is the boundary payload plus bounded fixed fields — there is no room for
    // token IDs or other per-position content.
    const std::uint64_t activation = std::uint64_t{rd.positions} * boundary.bytes_per_position();
    const std::uint64_t per_message_overhead = 512;
    if (rd.boundary_payload_bytes > rd.boundary_messages * (activation + per_message_overhead)) c.bounded_payload = false;
    if (first_config_first_rep)
      r.trace({{"kind", "round"}, {"q", c.q}, {"positions", rd.positions}, {"accepted", rd.accepted}, {"total_ms", rd.total_ms},
               {"draft_ms", rd.draft_ms}, {"prefix_ms", rd.prefix_ms}, {"remote_ms", rd.remote_ms}, {"tail_ms", rd.tail_ms},
               {"commit_ms", rd.commit_ms}, {"boundary_messages", rd.boundary_messages},
               {"boundary_payload_bytes", rd.boundary_payload_bytes}, {"scope", trace_prefix}});
  }
  if (gen.decode_tokens > 0) c.bytes_per_token.add(static_cast<double>(decode_bytes) / gen.decode_tokens);
}

void emit_cfg(BenchmarkResult& r, const std::string& prefix, const Cfg& c) {
  const std::string k = prefix + c.key;
  r.check(k + "_tokens_match_father_only_reference", c.tokens_ok);
  r.check(k + "_activation_messages_bounded_by_boundary_abi", c.bounded_payload);
  r.metric(k + ".round_ms", c.round_ms, "ms");
  r.metric(k + ".decode_tok_s_fixture", c.tok_s, "tok/s");
  r.metric(k + ".decode_tok_s", c.tok_s, "tok/s");
  r.metric(k + ".ttft_ms", c.ttft, "ms");
  r.metric(k + ".ttft_cold_ms", c.ttft_cold, "ms");
  r.metric(k + ".ttft_warm_ms", c.ttft_warm, "ms");
  r.metric(k + ".prefill_ms", c.prefill_ms, "ms");
  r.metric(k + ".prefill_tok_s", c.prefill_tok_s, "tok/s");
  r.metric(k + ".accepted_per_round", c.accepted, "tokens");
  r.metric(k + ".boundary_bytes_per_emitted_token", c.bytes_per_token, "bytes");
  r.metric(k + ".draft_ms", c.draft_ms, "ms");
  r.metric(k + ".verify_ms", c.verify_ms, "ms");
  r.metric(k + ".commit_ms", c.commit_ms, "ms");
  r.metric(k + ".mtp.proposed_positions", c.proposed);
  r.metric(k + ".mtp.accepted_positions", c.accepted_draft);
  r.metric(k + ".mtp.acceptance_rate", c.proposed > 0 ? static_cast<double>(c.accepted_draft) / static_cast<double>(c.proposed) : 0.0);
  r.metric(k + ".mtp.accepted_tokens_per_round", c.rounds > 0 ? static_cast<double>(c.tokens_emitted) / static_cast<double>(c.rounds) : 0.0);
  r.metric(k + ".stage.father.compute_ms", c.father_compute_ms, "ms");
  for (const auto& [i, d] : c.node_compute_ms) r.metric(k + ".stage.node" + std::to_string(i) + ".compute_ms", d, "ms");
  for (const auto& [i, d] : c.node_cpu_expert_ms) r.metric(k + ".stage.node" + std::to_string(i) + ".cpu_expert_ms", d, "ms");
  r.metric(k + ".stage.remote_wait_ms", c.wait_ms, "ms");
  r.metric(k + ".stage.wait_fraction", c.round_total_ms > 0 ? c.wait_total_ms / c.round_total_ms : 0.0);
}

// One pass over a freshly started cluster. Metric keys are prefixed with `ctx.prefix` ("" for a single pass,
// "direct." / "relay." when comparing routing).
Status run_cluster_pass(PassContext& pc) {
  BenchmarkResult& r = pc.r;
  const cli::Args& args = pc.args;
  const std::string& P = pc.prefix;

  std::size_t nodes = 0;
  for (const auto& item : cli::split(pc.plan_text, ','))
    if (item.find("@father") == std::string::npos) ++nodes;
  CLM_ASSIGN_OR_RETURN(auto s, setup(args, pc.work / (P.empty() ? "pass" : P.substr(0, P.size() - 1)), pc.model_dir, nodes));
  CLM_ASSIGN_OR_RETURN(auto coord, coordinator::Coordinator::create(father_config(s, args, pc.direct_peer)));
  auto& c = *coord;
  const auto& m = c.manifest();
  CLM_ASSIGN_OR_RETURN(auto plan, coordinator::ClusterPlan::parse(pc.plan_text, m.geometry.n_layers));
  std::uint32_t max_ctx = 0;
  for (auto ctxlen : pc.contexts) max_ctx = std::max(max_ctx, ctxlen);
  plan.max_context = std::max(plan.max_context, max_ctx + pc.max_new + 8);
  r.config(P + "plan", plan.describe());
  r.config(P + "direct_peer", pc.direct_peer);
  CLM_RETURN_IF_ERROR(c.connect());

  // ---- provisioning / load time per node ----
  const bool prepare_only = args.get("phase") == "prepare";
  Distribution prepare_total;
  std::map<std::string, Distribution> node_ms, node_prepare_ms, node_bytes_per_s;
  auto record_prepare = [&](const coordinator::PrepareReport& prep) {
    prepare_total.add(prep.total_ms);
    for (const auto& n : prep.nodes) {
      node_ms[n.node].add(n.prepare_ms);
      node_prepare_ms[n.node].add(static_cast<double>(n.node_prepare_ns) * 1e-6);
      if (n.prepare_ms > 0) node_bytes_per_s[n.node].add(static_cast<double>(n.bytes) / (n.prepare_ms / 1000.0));
    }
  };
  CLM_ASSIGN_OR_RETURN(auto prep, c.prepare(plan));
  record_prepare(prep);
  std::uint64_t provisioned = 0;
  for (const auto& n : prep.nodes) {
    provisioned += n.bytes;
    r.metric(P + "prepare.node." + n.node + ".bytes", n.bytes);
    r.metric(P + "prepare.node." + n.node + ".objects", n.objects);
  }
  r.metric(P + "prepare.father_resident_bytes", prep.father_resident_bytes);
  r.metric(P + "model.total_object_bytes", [&] {
    std::uint64_t t = 0;
    for (const auto& o : m.objects) t += o.byte_size;
    return t;
  }());
  std::uint64_t expected = 0;
  for (const auto& st : plan.stages)
    if (st.domain != coordinator::kFatherDomain) expected += m.total_bytes(st.layers);
  r.check(P + "provisioned_only_assigned_layers", provisioned == expected,
          std::to_string(provisioned) + " bytes provisioned, " + std::to_string(expected) + " assigned");

  if (prepare_only) {
    // PROV-01 shape: repeated release + prepare cycles; every cycle is a cold provisioning of the plan.
    for (int i = 1; i < pc.repeat; ++i) {
      CLM_RETURN_IF_ERROR(c.release().status());
      CLM_RETURN_IF_ERROR(c.connect());
      CLM_ASSIGN_OR_RETURN(auto again, c.prepare(plan));
      record_prepare(again);
    }
  }

  const auto boundary = domain::BoundaryLayout::for_geometry(m.geometry);
  std::unique_ptr<objects::CanonicalModelStore> drafter_store;
  std::shared_ptr<domain::Drafter> mtp_drafter;
  if (!prepare_only) {
    CLM_ASSIGN_OR_RETURN(mtp_drafter, make_drafter(pc.model_dir, m, drafter_store));
  }

  // ---- generation configurations ----
  std::vector<Cfg> cfgs;
  if (!prepare_only) {
    for (auto ctxlen : pc.contexts)
      for (auto q : pc.qs) {
        Cfg cfg;
        cfg.q = q;
        cfg.context = ctxlen;
        cfg.key = (pc.contexts.size() > 1 ? "ctx" + std::to_string(ctxlen) + "." : std::string()) + "q" + std::to_string(q);
        if (args.has("corpus")) {
          cfg.prompts = corpus_prompts(args.get("corpus"), ctxlen, m.geometry.vocab_size);
          if (cfg.prompts.empty()) return make_error(ErrorCode::kNotFound, "corpus '" + args.get("corpus") + "' has no readable files");
        } else {
          cfg.prompts.push_back(prompt_tokens(ctxlen, m.geometry.vocab_size));
        }
        for (const auto& pr : cfg.prompts) {
          CLM_ASSIGN_OR_RETURN(auto ref, reference_tokens(pc.model_dir, pr, pc.max_new, plan.max_context));
          cfg.references.push_back(std::move(ref));
          if (q > 1) {
            // --drafter mtp (default): the fixture MTP head — random weights, so it is almost never accepted.
            // --drafter scripted:RATE: replays the reference continuation with deterministic corruption at RATE,
            // exercising every acceptance length. Acceptance on the real model is HQ-MTP-01.
            const std::string kind = args.get("drafter", "mtp");
            if (kind.rfind("scripted", 0) == 0) {
              domain::ScriptedDrafter::Config dc;
              dc.vocab = m.geometry.vocab_size;
              dc.corruption_rate = kind.size() > 9 ? std::stod(kind.substr(9)) : 0.3;
              std::vector<std::int32_t> seq = pr;
              seq.insert(seq.end(), cfg.references.back().begin(), cfg.references.back().end());
              cfg.drafters.push_back(std::make_shared<domain::ScriptedDrafter>(std::move(seq), dc));
            } else {
              cfg.drafters.push_back(mtp_drafter);
            }
          } else {
            cfg.drafters.push_back(nullptr);
          }
        }
        cfg.request.max_new_tokens = pc.max_new;
        cfg.request.q = q;
        cfg.request.prefill_chunk = pc.prefill_chunk;
        cfgs.push_back(std::move(cfg));
      }
  }

  auto run_one = [&](Cfg& cfg, int rep, bool trace) -> Status {
    const std::size_t pi = static_cast<std::size_t>(rep) % cfg.prompts.size();
    coordinator::GenerationRequest req = cfg.request;
    req.prompt = cfg.prompts[pi];
    req.drafter = cfg.drafters[pi];
    auto gen = c.generate(req);
    if (!gen.is_ok()) {
      r.check(P + cfg.key + "_generate", false, gen.status().to_string());
      cfg.failed = true;
      cfg.tokens_ok = false;
      return gen.status();
    }
    record_generation(cfg, gen.value(), boundary, cfg.references[pi], rep, trace, r, P.empty() ? "cluster" : P);
    return Status::ok();
  };
  // Order: --interleave runs repetition r of every configuration before repetition r+1 of any (drift hits all
  // configurations alike); otherwise each configuration completes its repetitions in turn.
  if (args.has("interleave")) {
    for (int rep = 0; rep < pc.repeat; ++rep)
      for (std::size_t i = 0; i < cfgs.size(); ++i)
        if (!cfgs[i].failed) (void)run_one(cfgs[i], rep, rep == 0 && i == 0);
  } else {
    for (std::size_t i = 0; i < cfgs.size(); ++i)
      for (int rep = 0; rep < pc.repeat && !cfgs[i].failed; ++rep) (void)run_one(cfgs[i], rep, rep == 0 && i == 0);
  }
  for (const auto& cfg : cfgs) emit_cfg(r, P, cfg);

  // ---- sustained generation (--minutes): decode rate over time, thermal/leak derate ----
  if (!cfgs.empty() && args.has("minutes")) {
    const double minutes = args.number("minutes", 0);
    Cfg& cfg = cfgs.front();
    SustainedReport sus;
    sus.minutes = minutes;
    sus.sample_interval_s = 0;
    Stopwatch total;
    int rep = pc.repeat;
    while (total.elapsed_ms() / 1000.0 < minutes * 60.0) {
      const std::size_t before = cfg.tok_s.samples.size();
      if (!run_one(cfg, rep++, false).is_ok()) break;
      if (cfg.tok_s.samples.size() > before) sus.samples.push_back({total.elapsed_ms() / 1000.0, cfg.tok_s.samples.back(), std::nullopt});
    }
    summarize_sustained(sus);
    r.metric(P + "sustained.minutes", minutes);
    r.metric(P + "sustained.samples", sus.samples.size());
    r.metric(P + "sustained.initial_decode_tok_s", sus.initial_bytes_per_s);
    r.metric(P + "sustained.final_decode_tok_s", sus.final_bytes_per_s);
    r.metric(P + "sustained.factor", sus.sustained_factor);
    r.metric(P + "sustained.slope_pct_per_min", sus.slope_pct_per_min);
    r.metric(P + "sustained.time_to_equilibrium_s", sus.time_to_equilibrium_s);
    r.check(P + "sustained_all_generations_correct", cfg.tokens_ok);
    if (minutes < kMinSustainedMinutes) r.pending("HQ-PERF-04");
  }

  r.metric(P + "prepare.total_ms", prepare_total, "ms");
  for (const auto& [node, d] : node_ms) r.metric(P + "prepare.node." + node + ".ms", d, "ms");
  for (const auto& [node, d] : node_prepare_ms) r.metric(P + "prepare.node." + node + ".node_side_ms", d, "ms");
  for (const auto& [node, d] : node_bytes_per_s) r.metric(P + "prepare.node." + node + ".provision_bytes_per_s", d, "bytes/s");
  r.metric(P + "boundary.bytes_per_position", boundary.bytes_per_position());
  r.metric(P + "boundary.remote_stages", remote_count(plan));

  // Release: every Node must report resources released and storage cleaned with zero residual bytes.
  auto rel = c.release();
  if (!rel.is_ok()) {
    r.check(P + "release", false, rel.status().to_string());
  } else {
    Distribution release_ms;
    for (const auto& n : rel->nodes) {
      r.check(P + "release_" + n.node + "_storage_cleaned", n.resources_released && n.storage_cleaned && n.residual_bytes == 0, n.errors);
      r.metric(P + "release." + n.node + ".ms", n.release_ms);
      release_ms.add(n.release_ms);
    }
  }
  record_census(r, *s.cluster, P + "post_release_census_zero");
  return Status::ok();
}

}  // namespace

int cmd_cluster(const cli::Args& args) {
  Stopwatch total;
  const RunContext rc = make_run_context(args);
  BenchmarkResult r(args.get("experiment", "dev-cluster"), probe_host());
  apply_run_context(r, RunContext{rc.run_id, rc.machine_id, rc.role, false});  // localhost: never a target machine
  r.mark_simulated("localhost_cluster", true);
  r.mark_simulated("fixture_model", !args.has("model"));
  if (args.has("impair")) r.mark_simulated("network_preset", args.get("impair"));
  r.backend("reference", "reference");

  const fs::path work = args.has("work") ? fs::path(args.get("work")) : default_work_dir("cluster");
  auto model_dir = ensure_model(args, work);
  if (!model_dir.is_ok()) {
    std::fprintf(stderr, "model: %s\n", model_dir.status().to_string().c_str());
    return 1;
  }
  std::uint32_t n_layers = 0;
  {
    auto store = objects::CanonicalModelStore::open(model_dir.value());
    if (!store.is_ok()) {
      std::fprintf(stderr, "%s\n", store.status().to_string().c_str());
      return 1;
    }
    n_layers = store.value()->manifest().geometry.n_layers;
  }

  PassContext pc{args, r, "", work, model_dir.value(), {}, {}, {}, 64, 16, 3, !args.has("relay"), !args.has("model")};
  if (args.has("plan")) pc.plan_text = args.get("plan");
  else pc.plan_text = plan_for_tier(args.get("tier", "ultra"), n_layers);
  pc.max_new = static_cast<std::uint32_t>(args.integer("tokens", args.integer("max-new", 64)));
  pc.repeat = static_cast<int>(args.integer("repeat", 3));
  pc.prefill_chunk = static_cast<std::uint32_t>(args.integer("prefill-chunk", 16));
  pc.qs = args.has("q") ? parse_u32_list(args.get("q")) : std::vector<std::uint32_t>{1, 4};
  pc.contexts = args.has("context") ? parse_u32_list(args.get("context"))
                                    : std::vector<std::uint32_t>{static_cast<std::uint32_t>(args.integer("prompt-len", 48))};
  if (pc.qs.empty() || pc.contexts.empty() || pc.repeat < 1) {
    std::fprintf(stderr, "q, context and repeat must be non-empty/positive\n");
    return 2;
  }
  r.config("tier", args.get("tier", args.has("plan") ? "custom" : "ultra"));
  r.config("tls", !args.has("insecure"));
  r.config("context", pc.contexts);
  r.config("q", pc.qs);
  r.config("max_new_tokens", pc.max_new);
  r.config("repeat", pc.repeat);
  r.config("interleave", args.has("interleave"));
  r.config("phase", args.get("phase", "all"));
  r.config("drafter", args.get("drafter", "mtp"));
  r.config("meets_min_repetitions", pc.repeat >= 5);  // qualification runs need >= 5

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
  const int code = emit(args, r, total.elapsed_ms() / 1000.0);
  if (!args.has("keep-work") && !args.has("work")) {
    std::error_code ec;
    fs::remove_all(work, ec);
  }
  return code;
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
                                                   const coordinator::ClusterPlan&)>& body,
                          bool connect_father = true) {
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
    r.check(name + "_regenerate_matches_reference", gen.is_ok() && gen->tokens == reference,
            gen.is_ok() ? "" : gen.status().to_string());
    (void)c.release();
    record_census(r, cl, name + "_census_zero_after_release");
  };

  // 1. Abrupt Node crash at each lifecycle phase, then restart with orphan recovery.
  for (const char* phase :
       {"transfer", "hashing", "mapping", "allocation", "ready", "prefill", "inference", "commit", "cleanup"}) {
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

  // 1b. Father disappears (process killed mid-generation): every Node releases its lease on control loss and
  // deletes all staged objects without any instruction.
  run_scenario("father_lost", {}, [&](LocalCluster& cl, coordinator::Coordinator&, const auto&) {
    const auto eps = cl.endpoints();
    std::vector<std::string> fargs = {"--model", model_dir.value().string(), "--plan", plan_text, "--max-new", "1500",
                                      "--prefill-chunk", "8"};
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
