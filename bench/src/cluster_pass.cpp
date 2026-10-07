// One measured pass over a freshly started LocalCluster (see cluster_pass.hpp), plus the helpers `cluster`, `faults` and
// `placement-validate` share: starting a cluster for a plan, the Father CoordinatorConfig, prompts and Father-only
// references, and the per-pass metric emission. Lives in the harness library so tests can drive a pass with an injected
// backend (the Strata fake engine) without spawning the bench executable.
#include "cluster_pass.hpp"

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

namespace clusterlm::bench {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

std::vector<std::int32_t> synthetic_prompt_tokens(std::uint32_t n, std::uint32_t vocab) {
  std::vector<std::int32_t> out;
  for (std::uint32_t i = 0; i < n; ++i) out.push_back(static_cast<std::int32_t>((i * 7919u + 13u) % vocab));
  return out;
}

std::size_t remote_stage_count(const std::string& plan_text) {
  std::size_t n = 0;
  for (const auto& item : cli::split(plan_text, ','))
    if (item.find("@father") == std::string::npos) ++n;
  return n;
}

// Father-only execution of the same model on the same backend: the correctness reference for every distributed run.
Result<std::vector<std::int32_t>> reference_tokens(const fs::path& model_dir, const BackendChoice* backend,
                                                   const std::vector<std::int32_t>& prompt, std::uint32_t max_new,
                                                   std::uint32_t max_context) {
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = model_dir;
  if (backend) cfg.backend = backend->father;
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

Result<std::vector<PromptSet>> build_prompt_sets(const cli::Args& args, const fs::path& model_dir, const BackendChoice* backend,
                                                 const std::vector<std::uint32_t>& contexts, std::uint32_t max_new,
                                                 bool with_reference, BenchmarkResult& r) {
  if (args.has("tokenizer-gguf") && !args.has("corpus"))
    return make_error(ErrorCode::kInvalidArgument, "--tokenizer-gguf tokenizes a corpus: give --corpus FILE|DIR as well");
  CLM_ASSIGN_OR_RETURN(auto store, objects::CanonicalModelStore::open(model_dir));
  const std::uint32_t vocab = store->manifest().geometry.vocab_size;
  std::shared_ptr<father::Tokenizer> tokenizer;
  if (args.has("tokenizer-gguf")) {
    CLM_ASSIGN_OR_RETURN(tokenizer, load_corpus_tokenizer(args.get("tokenizer-gguf")));
  }
  const bool fixture = !args.has("model");
  std::uint32_t max_ctx = 0;
  for (auto c : contexts) max_ctx = std::max(max_ctx, c);
  nlohmann::json per_context = nlohmann::json::array();
  std::vector<PromptSet> sets;
  for (auto ctxlen : contexts) {
    PromptSet set;
    set.context = ctxlen;
    if (args.has("corpus")) {
      CLM_ASSIGN_OR_RETURN(auto cp, make_corpus_prompts(args.get("corpus"), ctxlen, vocab, tokenizer.get(), fixture));
      set.prompts = std::move(cp.prompts);
      if (cp.ids_folded) r.mark_simulated("tokenizer_ids_folded", true);  // a real vocabulary on the fixture model
      r.config("corpus_tokenizer", cp.tokenizer);
      per_context.push_back({{"context", ctxlen}, {"prompts", set.prompts.size()}, {"source_tokens", cp.source_tokens}});
    } else {
      set.prompts.push_back(synthetic_prompt_tokens(ctxlen, vocab));
    }
    if (with_reference)
      for (const auto& prompt : set.prompts) {
        CLM_ASSIGN_OR_RETURN(auto ref, reference_tokens(model_dir, backend, prompt, max_new, max_ctx + max_new + 8));
        set.references.push_back(std::move(ref));
      }
    sets.push_back(std::move(set));
  }
  if (args.has("corpus")) r.config("corpus", per_context);  // counts only, never text or token ids
  r.config("reference_check", with_reference);
  return sets;
}

bool is_loopback_host(const std::string& host) {
  return host == "localhost" || host == "::1" || host == "[::1]" || host.rfind("127.", 0) == 0;
}

Result<std::vector<coordinator::NodeEndpoint>> parse_external_nodes(const cli::Args& args) {
  return parse_node_specs(args.all("node"), "node");
}

Result<std::vector<coordinator::NodeEndpoint>> parse_node_specs(const std::vector<std::string>& specs, const std::string& flag) {
  std::vector<coordinator::NodeEndpoint> out;
  for (const auto& spec : specs) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos || eq == 0)
      return make_error(ErrorCode::kInvalidArgument, str_cat("--", flag, " expects NAME=HOST:PORT[@FINGERPRINT], got '", spec, "'"));
    coordinator::NodeEndpoint n;
    n.name = spec.substr(0, eq);
    std::string rest = spec.substr(eq + 1);
    if (auto at = rest.find('@'); at != std::string::npos) {
      n.device_id = rest.substr(at + 1);
      rest = rest.substr(0, at);
    }
    CLM_ASSIGN_OR_RETURN(n.endpoint, transport::Endpoint::parse(rest));
    out.push_back(std::move(n));
  }
  return out;
}

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
                           const BackendChoice* backend, const std::vector<LocalNodeOptions>& overrides, bool supervised,
                           const std::vector<coordinator::NodeEndpoint>* external) {
  ClusterSetup s;
  s.work = work;
  s.model_dir = model_dir;
  if (backend) s.backend = backend->father;
  LocalClusterOptions opts;
  opts.work_dir = s.work;
  opts.tls = !args.has("insecure");
  opts.supervised = supervised;
  if (backend) opts.node_args = backend->node_args;
  if (external) {
    opts.external = *external;
  } else {
    CLM_ASSIGN_OR_RETURN(opts.external, parse_external_nodes(args));
  }
  if (!opts.external.empty()) {
    if (supervised) return make_error(ErrorCode::kInvalidArgument, "supervised Nodes are started by the bench; they cannot be --node endpoints");
    if (opts.external.size() < nodes)
      return make_error(ErrorCode::kInvalidArgument, "the plan needs " + std::to_string(nodes) + " Nodes but --node names " + std::to_string(opts.external.size()));
    opts.father_identity_dir = args.get("identity");
  }
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

coordinator::CoordinatorConfig father_config(const ClusterSetup& s, const cli::Args& args, std::optional<bool> direct_peer) {
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = s.model_dir;
  cfg.backend = s.backend;
  cfg.security = s.cluster ? s.cluster->father_security() : transport::SecurityConfig{};
  if (!s.cluster) cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  if (s.cluster) cfg.nodes = s.cluster->endpoints();
  cfg.direct_peer = direct_peer.value_or(!args.has("relay"));
  cfg.impairment = s.impairment;
  cfg.window_timeout = std::chrono::milliseconds(args.integer("window-timeout-ms", 10'000));
  return cfg;
}

std::size_t remote_count(const coordinator::ClusterPlan& p) {
  std::size_t n = 0;
  for (const auto& s : p.stages) n += s.domain != coordinator::kFatherDomain;
  return n;
}

void record_census(BenchmarkResult& r, LocalCluster& cluster, const std::string& label) {
  if (cluster.external()) {
    // The staging roots are on other machines: the Nodes' own release reports (resources released, storage cleaned,
    // residual bytes) are the evidence; a file census there is `clusterlm-bench storage-census` on that machine.
    r.metric(label + ".unavailable", "external Nodes: their staging is on their machines (see the release_*_storage_cleaned checks)");
    return;
  }
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

namespace {

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
  // Inter-token gaps of the streamed output (on_tokens deliveries): per token (members of one delivery are 0 apart)
  // and per delivery (the stall a reader of the stream actually sees). HQ-MTP-01 / HQ-PERF-01.
  Distribution token_gap_ms, delivery_gap_ms;
  std::uint64_t payload_bytes_total = 0;  // boundary payload over every round incl. prefill (for the NIC comparison)
  std::uint64_t proposed = 0, accepted_draft = 0, rounds = 0, tokens_emitted = 0;
  double wait_total_ms = 0, round_total_ms = 0;
  bool tokens_ok = true, bounded_payload = true, failed = false;
  bool have_reference = true;  // false with --no-reference: the correctness check is not made
  coordinator::GenerationRequest request;
};

// Folds one generation into a config's distributions.
void record_generation(Cfg& c, const coordinator::GenerationResult& gen, const domain::BoundaryLayout& boundary,
                       const std::vector<std::int32_t>* reference, int rep, bool first_config_first_rep, BenchmarkResult& r,
                       const std::string& trace_prefix) {
  if (reference) c.tokens_ok = c.tokens_ok && gen.tokens == *reference;
  c.ttft.add(gen.first_token_ms);
  (rep == 0 ? c.ttft_cold : c.ttft_warm).add(gen.first_token_ms);
  c.prefill_ms.add(gen.prefill_ms);
  if (gen.prefill_ms > 0) c.prefill_tok_s.add(1000.0 * static_cast<double>(c.context) / gen.prefill_ms);
  if (gen.decode_ms > 0) c.tok_s.add(1000.0 * gen.decode_tokens / gen.decode_ms);
  std::uint64_t decode_bytes = 0;
  for (const auto& rd : gen.rounds) {
    c.payload_bytes_total += rd.boundary_payload_bytes;
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
  if (c.have_reference) r.check(k + "_tokens_match_father_only_reference", c.tokens_ok);
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
  r.metric(k + ".inter_token_gap_ms", c.token_gap_ms, "ms");
  r.metric(k + ".inter_delivery_gap_ms", c.delivery_gap_ms, "ms");
  r.metric(k + ".stage.father.compute_ms", c.father_compute_ms, "ms");
  for (const auto& [i, d] : c.node_compute_ms) r.metric(k + ".stage.node" + std::to_string(i) + ".compute_ms", d, "ms");
  for (const auto& [i, d] : c.node_cpu_expert_ms) r.metric(k + ".stage.node" + std::to_string(i) + ".cpu_expert_ms", d, "ms");
  r.metric(k + ".stage.remote_wait_ms", c.wait_ms, "ms");
  r.metric(k + ".stage.wait_fraction", c.round_total_ms > 0 ? c.wait_total_ms / c.round_total_ms : 0.0);
}

}  // namespace


// One pass over a freshly started cluster. Metric keys are prefixed with `ctx.prefix` ("" for a single pass,
// "direct." / "relay." when comparing routing).
Status run_cluster_pass(PassContext& pc) {
  BenchmarkResult& r = pc.r;
  const cli::Args& args = pc.args;
  const std::string& P = pc.prefix;

  std::size_t nodes = pc.node_count;
  if (nodes == 0 && !pc.plan) nodes = remote_stage_count(pc.plan_text);
  CLM_ASSIGN_OR_RETURN(auto s, setup(args, pc.work / (P.empty() ? "pass" : P.substr(0, P.size() - 1)), pc.model_dir, nodes,
                                     pc.backend, pc.node_options, false, pc.external ? &*pc.external : nullptr));
  CLM_ASSIGN_OR_RETURN(auto coord, coordinator::Coordinator::create(father_config(s, args, pc.direct_peer)));
  auto& c = *coord;

  // ---- OS-level observers around this pass (HQ-NET-02, HQ-PERF-04, HQ-PERF-02, HQ-GPU-03, HQ-STORE-01) ----
  // NIC byte counters of the interface routed to the first Node (or --nic), over the whole pass and over generation
  // only; process RSS/commit of Father and every Node process; NVML when the driver library is present.
  const std::string first_host = s.cluster->size() > 0 ? s.cluster->endpoints().front().endpoint.host : std::string();
  NicWindow nic_pass(args.get("nic"), first_host), nic_gen(args.get("nic"), first_host);
  nic_pass.begin();
  ResourceSampler sampler;
  sampler.add_process("father", [] { return std::int64_t{0}; });
  if (!s.cluster->external())  // an external Node is another machine's process: its memory is not this machine's to read
    for (std::size_t i = 0; i < s.cluster->size(); ++i)
      sampler.add_process("node" + std::to_string(i), [&s, i] { return s.cluster->pid(i); });
  auto nvml = NvmlTelemetry::open();
  std::unique_ptr<TelemetryRecorder> telemetry;
  if (nvml.is_ok()) {
    sampler.set_gpu(nvml.value().get());
    if (args.has("minutes")) {
      telemetry = std::make_unique<TelemetryRecorder>(*nvml.value(), pc.telemetry_s);
      telemetry->start();
    }
  } else {
    emit_gpu_telemetry_unavailable(r, nvml.status().message(), P + "nvml.");
  }
  std::optional<CensusSnapshot> census_before;
  std::vector<CensusRoot> census_roots;
  std::int64_t census_start_ns = 0;
  if (pc.census) {
    std::vector<CensusRoot> staging;
    for (std::size_t i = 0; i < s.cluster->size() && !s.cluster->external(); ++i) {
      CensusRoot root;
      root.label = "node_staging_" + std::to_string(i);
      root.path = s.cluster->staging_root(i);
      staging.push_back(std::move(root));
    }
    census_roots = default_census_roots(staging, {pc.work});
    census_before = take_census(census_roots);
    census_start_ns = census_before->taken_ns;
  }
  if (pc.resources) sampler.sample("start");
  const auto& m = c.manifest();
  coordinator::ClusterPlan plan;
  if (pc.plan) {
    plan = *pc.plan;
  } else {
    CLM_ASSIGN_OR_RETURN(plan, coordinator::ClusterPlan::parse(pc.plan_text, m.geometry.n_layers));
  }
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
  // HQ-PROV-01: where a Node's prepare time goes. Father: source read vs. digest vs. send (transfer, including
  // transport backpressure). Node: chunk verify+write vs. whole-object hash at seal vs. domain build.
  struct Breakdown {
    Distribution father_read_ms, father_digest_ms, father_send_ms, node_write_ms, node_hash_ms, node_build_ms;
    Distribution father_read_bytes_per_s, father_send_bytes_per_s;
  };
  std::map<std::string, Breakdown> breakdown;
  auto record_prepare = [&](const coordinator::PrepareReport& prep) {
    prepare_total.add(prep.total_ms);
    for (const auto& n : prep.nodes) {
      node_ms[n.node].add(n.prepare_ms);
      node_prepare_ms[n.node].add(static_cast<double>(n.node_prepare_ns) * 1e-6);
      if (n.prepare_ms > 0) node_bytes_per_s[n.node].add(static_cast<double>(n.bytes) / (n.prepare_ms / 1000.0));
      auto& b = breakdown[n.node];
      b.father_read_ms.add(static_cast<double>(n.father_source_read_ns) * 1e-6);
      b.father_digest_ms.add(static_cast<double>(n.father_chunk_digest_ns) * 1e-6);
      b.father_send_ms.add(static_cast<double>(n.father_send_ns) * 1e-6);
      b.node_write_ms.add(static_cast<double>(n.node_chunk_write_ns) * 1e-6);
      b.node_hash_ms.add(static_cast<double>(n.node_seal_hash_ns) * 1e-6);
      b.node_build_ms.add(static_cast<double>(n.node_build_ns) * 1e-6);
      if (n.father_source_read_ns > 0) b.father_read_bytes_per_s.add(static_cast<double>(n.bytes) / (static_cast<double>(n.father_source_read_ns) * 1e-9));
      if (n.father_send_ns > 0) b.father_send_bytes_per_s.add(static_cast<double>(n.bytes) / (static_cast<double>(n.father_send_ns) * 1e-9));
    }
    if (pc.resources) sampler.sample("prepare_cycle");
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
  // The MTP drafter of Father's backend, bound to the prepared plan (the reference fixture head, or Strata's MTP head on
  // the Father tail); made only when some configuration speculates and does not use the scripted drafter.
  const std::string drafter_kind = args.get("drafter", "mtp");
  const bool scripted = drafter_kind.rfind("scripted", 0) == 0;
  std::shared_ptr<domain::Drafter> mtp_drafter;
  if (!prepare_only && !scripted && std::any_of(pc.qs.begin(), pc.qs.end(), [](std::uint32_t q) { return q > 1; })) {
    CLM_ASSIGN_OR_RETURN(mtp_drafter, c.make_drafter());
  }

  // ---- generation configurations ----
  std::vector<Cfg> cfgs;
  if (!prepare_only) {
    for (std::size_t ci = 0; ci < pc.contexts.size(); ++ci)
      for (auto q : pc.qs) {
        const auto ctxlen = pc.contexts[ci];
        const PromptSet& set = pc.prompt_sets.at(ci);
        Cfg cfg;
        cfg.q = q;
        cfg.context = ctxlen;
        cfg.key = (pc.contexts.size() > 1 || pc.keyed_contexts ? "ctx" + std::to_string(ctxlen) + "." : std::string()) + "q" + std::to_string(q);
        cfg.prompts = set.prompts;
        cfg.have_reference = !set.references.empty();
        if (cfg.have_reference) cfg.references = set.references;
        if (scripted && q > 1 && !cfg.have_reference)
          return make_error(ErrorCode::kInvalidArgument, "--drafter scripted replays the reference continuation: it cannot be used with --no-reference");
        for (std::size_t pi = 0; pi < cfg.prompts.size(); ++pi) {
          const auto& pr = cfg.prompts[pi];
          if (q > 1) {
            // --drafter mtp (default): the backend's MTP head (the fixture's has random weights, so it is almost never
            // accepted; the real head's acceptance is HQ-MTP-01). --drafter scripted:RATE: replays the reference
            // continuation with deterministic corruption at RATE, exercising every acceptance length (fixture only).
            if (scripted) {
              domain::ScriptedDrafter::Config dc;
              dc.vocab = m.geometry.vocab_size;
              dc.corruption_rate = drafter_kind.size() > 9 ? std::stod(drafter_kind.substr(9)) : 0.3;
              std::vector<std::int32_t> seq = pr;
              seq.insert(seq.end(), cfg.references[pi].begin(), cfg.references[pi].end());
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
    // Streamed-token timing: counts and timestamps only, never the tokens themselves.
    struct StreamClock {
      bool have = false;
      std::uint64_t last_ns = 0;
    } stream;
    req.on_tokens = [&](std::span<const std::int32_t> tokens) {
      const std::uint64_t now = monotonic_ns();
      if (!tokens.empty()) {
        if (stream.have) {
          const double gap = static_cast<double>(now - stream.last_ns) * 1e-6;
          cfg.delivery_gap_ms.add(gap);
          cfg.token_gap_ms.add(gap);
        } else {
          stream.have = true;
        }
        for (std::size_t i = 1; i < tokens.size(); ++i) cfg.token_gap_ms.add(0.0);  // same delivery: no wait between
        stream.last_ns = now;
      }
      if (pc.resources) sampler.sample_throttled("round", pc.sample_ms);
    };
    if (pc.resources) sampler.sample("generation_start");
    auto gen = c.generate(req);
    if (pc.resources) sampler.sample("generation_end");
    if (!gen.is_ok()) {
      r.check(P + cfg.key + "_generate", false, gen.status().to_string());
      cfg.failed = true;
      cfg.tokens_ok = false;
      return gen.status();
    }
    record_generation(cfg, gen.value(), boundary, cfg.have_reference ? &cfg.references[pi] : nullptr, rep, trace, r,
                      P.empty() ? "cluster" : P);
    return Status::ok();
  };
  // Order: --interleave runs repetition r of every configuration before repetition r+1 of any (drift hits all
  // configurations alike); otherwise each configuration completes its repetitions in turn.
  nic_gen.begin();
  if (args.has("interleave")) {
    for (int rep = 0; rep < pc.repeat; ++rep)
      for (std::size_t i = 0; i < cfgs.size(); ++i)
        if (!cfgs[i].failed) (void)run_one(cfgs[i], rep, rep == 0 && i == 0);
  } else {
    for (std::size_t i = 0; i < cfgs.size(); ++i)
      for (int rep = 0; rep < pc.repeat && !cfgs[i].failed; ++rep) (void)run_one(cfgs[i], rep, rep == 0 && i == 0);
  }
  for (const auto& cfg : cfgs) emit_cfg(r, P, cfg);
  if (!cfgs.empty() && !args.has("minutes")) nic_gen.end();

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
    nic_gen.end();
    r.metric(P + "sustained.minutes", minutes);
    r.metric(P + "sustained.samples", sus.samples.size());
    r.metric(P + "sustained.initial_decode_tok_s", sus.initial_bytes_per_s);
    r.metric(P + "sustained.final_decode_tok_s", sus.final_bytes_per_s);
    r.metric(P + "sustained.factor", sus.sustained_factor);
    r.metric(P + "sustained.slope_pct_per_min", sus.slope_pct_per_min);
    r.metric(P + "sustained.time_to_equilibrium_s", sus.time_to_equilibrium_s);
    if (cfg.have_reference) r.check(P + "sustained_all_generations_correct", cfg.tokens_ok);
    else if (!cfg.tokens_ok) r.check(P + "sustained_all_generations_completed", false);
    if (minutes < kMinSustainedMinutes) r.pending("HQ-PERF-04");
  }

  r.metric(P + "prepare.total_ms", prepare_total, "ms");
  for (const auto& [node, d] : node_ms) r.metric(str_cat(P, "prepare.node.", node, ".ms"), d, "ms");
  for (const auto& [node, d] : node_prepare_ms) r.metric(str_cat(P, "prepare.node.", node, ".node_side_ms"), d, "ms");
  for (const auto& [node, d] : node_bytes_per_s) r.metric(str_cat(P, "prepare.node.", node, ".provision_bytes_per_s"), d, "bytes/s");
  r.metric(P + "boundary.bytes_per_position", boundary.bytes_per_position());
  r.metric(P + "boundary.remote_stages", remote_count(plan));
  for (const auto& [node, b] : breakdown) {
    const std::string k = str_cat(P, "prepare.node.", node, ".");
    r.metric(k + "father_source_read_ms", b.father_read_ms, "ms");
    r.metric(k + "father_chunk_digest_ms", b.father_digest_ms, "ms");
    r.metric(k + "father_send_ms", b.father_send_ms, "ms");
    r.metric(k + "node_chunk_write_ms", b.node_write_ms, "ms");
    r.metric(k + "node_seal_hash_ms", b.node_hash_ms, "ms");
    r.metric(k + "node_build_ms", b.node_build_ms, "ms");
    r.metric(k + "father_source_read_bytes_per_s", b.father_read_bytes_per_s, "bytes/s");
    r.metric(k + "father_send_bytes_per_s", b.father_send_bytes_per_s, "bytes/s");
  }
  {
    std::uint64_t payload_total = 0;
    for (const auto& cfg : cfgs) payload_total += cfg.payload_bytes_total;
    r.metric(P + "boundary.payload_bytes_total", payload_total);
    nic_gen.emit(r, P + "generation.");
    if (nic_gen.available() && payload_total > 0)
      r.metric(P + "generation.nic.wire_bytes_per_boundary_payload_byte",
               static_cast<double>(nic_gen.delta().rx_bytes + nic_gen.delta().tx_bytes) / static_cast<double>(payload_total));
  }

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
    // HQ-PERF-02: sequence state (KV / recurrent / PLE) each domain actually had allocated, as the domain reports it:
    // the peak over the lease, sized for the plan's max_context. Sizes only.
    std::uint64_t state_total = 0, window_total = 0;
    auto record_domain = [&](const std::string& owner, const coordinator::DomainStateReport& d) {
      const std::string k = P + "state." + owner + ".stage" + std::to_string(d.stage) + ".";
      r.metric(k + "bytes", d.state_bytes_peak);
      r.metric(k + "window_bytes", d.window_bytes_peak);
      state_total += d.state_bytes_peak;
      window_total += d.window_bytes_peak;
    };
    for (const auto& d : rel->father_domain_state) record_domain("father", d);
    for (const auto& n : rel->nodes)
      for (const auto& d : n.domain_state) record_domain(n.node, d);
    r.metric(P + "state.total_bytes", state_total);
    r.metric(P + "state.total_window_bytes", window_total);
    r.metric(P + "state.max_context", plan.max_context);
  }
  record_census(r, *s.cluster, P + "post_release_census_zero");
  if (pc.resources) {
    sampler.sample("released");
    sampler.emit(r, P + "resources.");
  }
  nic_pass.end();
  nic_pass.emit(r, P);
  if (telemetry && nvml.is_ok()) {
    const auto samples = telemetry->stop();
    emit_gpu_telemetry(r, *nvml.value(), samples, P + "nvml.");
  }
  if (census_before) {
    const auto after = take_census(census_roots);
    emit_census_diff(r, diff_census(*census_before, after, census_start_ns, after.taken_ns), P);
  }
  if (pc.summary) {
    pc.summary->completed = !cfgs.empty() && !cfgs.front().failed;
    pc.summary->remote_stages = static_cast<std::uint32_t>(remote_count(plan));
    pc.summary->prepare_s = prepare_total.mean() / 1000.0;
    if (!cfgs.empty()) {
      pc.summary->prefill_s = cfgs.front().prefill_ms.mean() / 1000.0;
      pc.summary->decode_tok_s = cfgs.front().tok_s.mean();
    }
  }
  return Status::ok();
}

}  // namespace clusterlm::bench
