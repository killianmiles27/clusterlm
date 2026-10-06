// clusterlm-father: command-line ClusterLM Father for development and qualification runs.
//
//   clusterlm-father --model DIR --node a=127.0.0.1:7001 --node b=127.0.0.1:7002
//                    --plan 0-4@father,4-10@0,10-13@1,13-16@father --prompt 1,2,3,4 --max-new 32 --q 4
//
// The user-facing Father application wraps the same Coordinator; this tool exposes it without UI.
#include <cstdio>
#include <string>

#include "cli.hpp"
#include "clusterlm/common/log.hpp"
#include "clusterlm/coordinator/coordinator.hpp"
#include "clusterlm/domain/drafter.hpp"
#include "clusterlm/objects/canonical_store.hpp"

using namespace clusterlm;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: clusterlm-father --model DIR --plan PLAN [--node NAME=HOST:PORT[@FINGERPRINT]]...\n"
               "                        [--prompt T1,T2,...] [--max-new N] [--q N] [--prefill-chunk N] [--relay]\n"
               "                        [--insecure-loopback | --identity DIR] [--impair PRESET]\n");
  return 2;
}

int fail(const Status& s) {
  std::fprintf(stderr, "error: %s\n", s.to_string().c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  if (!args.has("model") || !args.has("plan")) return usage();
  log::set_component("father");
  coordinator::CoordinatorConfig cfg;
  cfg.model_dir = args.get("model");
  cfg.direct_peer = !args.has("relay");
  for (const auto& spec : args.all("node")) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos) return usage();
    coordinator::NodeEndpoint n;
    n.name = spec.substr(0, eq);
    std::string rest = spec.substr(eq + 1);
    if (auto at = rest.find('@'); at != std::string::npos) {
      n.device_id = rest.substr(at + 1);
      rest = rest.substr(0, at);
    }
    auto ep = transport::Endpoint::parse(rest);
    if (!ep.is_ok()) return fail(ep.status());
    n.endpoint = ep.value();
    cfg.nodes.push_back(n);
  }
  if (args.has("identity")) {
    auto id = transport::DeviceIdentity::load_or_generate(args.get("identity"), "father");
    if (!id.is_ok()) return fail(id.status());
    cfg.security.mode = transport::SecurityConfig::Mode::kMutualTls;
    cfg.security.identity = std::make_shared<const transport::DeviceIdentity>(std::move(id).value());
    for (const auto& n : cfg.nodes) cfg.security.trust(n.device_id);
  } else {
    cfg.security.mode = transport::SecurityConfig::Mode::kInsecureLoopbackOnly;
  }
  if (args.has("impair")) {
    auto preset = transport::network_preset(args.get("impair"));
    if (!preset.is_ok()) return fail(preset.status());
    cfg.impairment = preset.value();
  }

  auto coord = coordinator::Coordinator::create(cfg);
  if (!coord.is_ok()) return fail(coord.status());
  auto& c = *coord.value();
  auto plan = coordinator::ClusterPlan::parse(args.get("plan"), c.manifest().geometry.n_layers);
  if (!plan.is_ok()) return fail(plan.status());
  if (auto st = c.connect(); !st.is_ok()) return fail(st);
  auto prep = c.prepare(plan.value());
  if (!prep.is_ok()) return fail(prep.status());
  std::printf("prepared plan %s in %.1f ms\n", plan->describe().c_str(), prep->total_ms);

  coordinator::GenerationRequest req;
  for (const auto& t : cli::split(args.get("prompt", "1,2,3,4,5,6,7,8"), ',')) req.prompt.push_back(std::stoi(t));
  req.max_new_tokens = static_cast<std::uint32_t>(args.integer("max-new", 32));
  req.q = static_cast<std::uint32_t>(args.integer("q", 1));
  req.prefill_chunk = static_cast<std::uint32_t>(args.integer("prefill-chunk", 128));
  std::unique_ptr<objects::CanonicalModelStore> drafter_store;
  if (req.q > 1) {
    auto store = objects::CanonicalModelStore::open(cfg.model_dir);
    if (!store.is_ok()) return fail(store.status());
    drafter_store = std::move(store).value();
    auto drafter = domain::MtpFixtureDrafter::create(c.manifest(), *drafter_store);
    if (!drafter.is_ok()) return fail(drafter.status());
    req.drafter = std::move(drafter).value();
  }
  auto gen = c.generate(req);
  if (!gen.is_ok()) return fail(gen.status());
  std::printf("tokens:");
  for (auto t : gen->tokens) std::printf(" %d", t);
  std::printf("\nprefill %.2f ms, decode %.2f ms over %u rounds\n", gen->prefill_ms, gen->decode_ms, gen->decode_rounds);
  auto rel = c.release();
  if (!rel.is_ok()) return fail(rel.status());
  for (const auto& n : rel->nodes)
    std::printf("release %s: resources=%d storage_cleaned=%d residual_bytes=%llu\n", n.node.c_str(),
                n.resources_released ? 1 : 0, n.storage_cleaned ? 1 : 0,
                static_cast<unsigned long long>(n.residual_bytes));
  return 0;
}
