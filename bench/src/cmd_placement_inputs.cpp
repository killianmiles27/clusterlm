// clusterlm-bench placement-inputs: measures the model/workload-side placement inputs that do not depend on
// the cluster layout — aggregate routing frequencies (layer x expert counts only), Father's per-round draft/
// verify overhead, and MTP acceptance per verification width — by running a corpus through a Father-only plan.
//
// On the fixture model this is Synthetic. On the real artifact with the real backend and a held-out corpus it
// produces Measured routing aggregates that `planning::load_routing_aggregates` feeds into placement (HQ-PLACE-02).
#include <cstdio>
#include <fstream>

#include "bench_common.hpp"
#include "clusterlm/common/clock.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/father/tokenizer.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "clusterlm/objects/fixture_model.hpp"
#include "clusterlm/planning/planning.hpp"
#include "commands.hpp"

namespace clusterlm::bench {

namespace fs = std::filesystem;

int cmd_placement_inputs(const cli::Args& args) {
  Stopwatch total;
  BenchmarkResult r(args.get("experiment", "dev-placement-inputs"), probe_host());
  const bool fixture = !args.has("model");
  fs::path model_dir;
  if (fixture) {
    model_dir = fs::path(args.get("work", (fs::temp_directory_path() / "clusterlm-placement-inputs").string())) / "model";
    auto m = objects::write_fixture_model(objects::FixtureSpec{}, model_dir);
    if (!m.is_ok()) {
      std::fprintf(stderr, "%s\n", m.status().to_string().c_str());
      return 1;
    }
    r.mark_simulated("fixture_model", true);
  } else {
    model_dir = args.get("model");
  }
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = model_dir;
  cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  auto coord = coordinator::Coordinator::create(cfg);
  if (!coord.is_ok()) {
    std::fprintf(stderr, "%s\n", coord.status().to_string().c_str());
    return 1;
  }
  auto& c = *coord.value();
  const auto& m = c.manifest();
  const auto L = m.geometry.n_layers;
  auto plan = coordinator::ClusterPlan::parse("0-" + std::to_string(L) + "@father," + std::to_string(L) + "-" +
                                                  std::to_string(L) + "@father",
                                              L);
  if (!plan.is_ok() || !c.prepare(plan.value()).is_ok()) {
    std::fprintf(stderr, "cannot prepare a Father-only plan for this model\n");
    return 1;
  }
  if (auto st = c.enable_routing_aggregation(true); !st.is_ok()) {
    r.check("routing_aggregation_supported", false, st.to_string());
    return emit(args, r, total.elapsed_ms() / 1000.0);
  }
  // Corpus: one prompt per line. The fixture uses the byte-level fixture tokenizer; a real model needs its own
  // tokenizer from the backend.
  std::vector<std::string> corpus;
  if (args.has("corpus")) {
    std::ifstream f(args.get("corpus"));
    for (std::string line; std::getline(f, line);)
      if (!line.empty()) corpus.push_back(line);
  } else {
    corpus = {"The quick brown fox jumps over the lazy dog.", "Distributed inference across idle machines.",
              "int main() { return 0; }", "Explain why the sky is blue in two sentences."};
  }
  if (!fixture) {
    r.check("tokenizer_available", false,
            "the real model's tokenizer is provided by its backend; this build has no tokenizer for it");
    r.pending("HQ-PLACE-02");
    return emit(args, r, total.elapsed_ms() / 1000.0, 3);
  }
  auto tok = father::FixtureByteTokenizer::create(m.geometry.vocab_size);
  auto store = objects::CanonicalModelStore::open(model_dir);
  auto drafter = store.is_ok() ? domain::MtpFixtureDrafter::create(m, *store.value())
                               : Result<std::unique_ptr<domain::MtpFixtureDrafter>>(store.status());
  if (!tok.is_ok() || !drafter.is_ok()) return 1;
  std::shared_ptr<domain::Drafter> d = std::move(drafter).value();
  const auto max_new = static_cast<std::uint32_t>(args.integer("max-new", 32));
  std::vector<std::uint32_t> qs = {1, 2, 3, 4};
  if (args.has("q")) {
    qs.clear();
    for (const auto& q : cli::split(args.get("q"), ',')) qs.push_back(static_cast<std::uint32_t>(std::stoul(q)));
  }
  for (auto q : qs) {
    Distribution accepted, round_overhead_ms, draft_ms;
    for (const auto& text : corpus) {
      coordinator::GenerationRequest g;
      g.prompt = tok.value()->encode(text);
      g.max_new_tokens = max_new;
      g.q = q;
      if (q > 1) g.drafter = d;
      auto out = c.generate(g);
      if (!out.is_ok()) {
        r.check("generate_q" + std::to_string(q), false, out.status().to_string());
        break;
      }
      for (const auto& rd : out->rounds) {
        if (rd.prefill) continue;
        accepted.add(rd.accepted);
        draft_ms.add(rd.draft_ms);
        // Father-side work per round outside layer stages is approximated by draft + commit bookkeeping here;
        // prefix/tail layer time is part of the stage model, not of this overhead.
        round_overhead_ms.add(rd.draft_ms + rd.commit_ms);
      }
    }
    const std::string k = "q" + std::to_string(q);
    r.metric(k + ".accepted_per_round", accepted, "positions");
    r.metric(k + ".draft_ms", draft_ms, "ms");
    r.metric(k + ".father_round_overhead_ms", round_overhead_ms, "ms");
  }
  auto agg = c.routing_aggregates();
  if (!agg.is_ok()) {
    r.check("routing_aggregates", false, agg.status().to_string());
    return emit(args, r, total.elapsed_ms() / 1000.0);
  }
  planning::RoutingAggregates out;
  out.n_layers = L;
  out.n_experts = m.geometry.n_experts;
  out.n_active = m.geometry.n_active_experts;
  out.provenance = placement::Provenance::kSynthetic;  // fixture model; a real run on target would be Measured
  out.source = "bench:placement-inputs fixture";
  out.frequencies.assign(L, std::vector<double>(m.geometry.n_experts, 0.0));
  std::uint64_t positions = 0;
  for (const auto& a : agg.value()) {
    for (std::size_t li = 0; li < a.counts.size(); ++li)
      for (std::size_t e = 0; e < a.counts[li].size(); ++e)
        out.frequencies[a.first_layer + li][e] =
            a.positions ? static_cast<double>(a.counts[li][e]) / static_cast<double>(a.positions) : 0.0;
    positions = std::max(positions, a.positions);
  }
  out.positions = positions;
  r.metric("routing.positions", positions);
  bool rows_ok = positions > 0;
  for (const auto& row : out.frequencies) {
    double s = 0;
    for (double v : row) s += v;
    rows_ok = rows_ok && std::abs(s - out.n_active) < 1e-9;
  }
  r.check("routing_rows_sum_to_active_experts", rows_ok);
  const std::string routing_path = args.get("routing-out", "routing-aggregates.json");
  {
    std::ofstream f(routing_path, std::ios::binary);
    f << planning::to_json(out) << "\n";
  }
  // The file must load back through the placement input path.
  auto back = planning::load_routing_aggregates(routing_path);
  r.check("routing_aggregates_load_for_placement", back.is_ok(), back.is_ok() ? routing_path : back.status().to_string());
  r.pending("HQ-PLACE-02");
  return emit(args, r, total.elapsed_ms() / 1000.0);
}

}  // namespace clusterlm::bench
