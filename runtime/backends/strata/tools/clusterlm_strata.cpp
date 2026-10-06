// clusterlm-strata: the Strata backend's own tool (Father side and qualification). Subcommands:
//
//   probe                      this host: CPU kernel tier, CUDA devices, backend build
//   cpu-experts                CPU expert kernel throughput on synthetic blobs (real Strata kernels)      HQ-CPU-01
//   convert                    Strata pack + model GGUF -> a ClusterLM model directory whose dense objects
//                              are strata-dense containers and whose experts are GGUF slices (ADR 0200)
//   requirements   [CUDA]      describe_requirements per domain of a plan; --measure also prepares each
//                              domain on this GPU and records the device-memory delta                    HQ-GPU-02
//   numerics       [CUDA]      split vs reference execution, every rejection length, abort_window
//                              restoration, optional MTP acceptance                          HQ-NUM-01/GPU-05/MTP-02
//
// Results are JSON in the ClusterLM Bench result shape (provenance Measured on this host; never Qualified).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "clusterlm/backends/strata/convert.hpp"
#include "clusterlm/backends/strata/cpu_expert_kernel.hpp"
#include "clusterlm/backends/strata/object_map.hpp"
#include "clusterlm/common/digest.hpp"
#include "clusterlm/objects/canonical_store.hpp"

#if defined(CLUSTERLM_STRATA_HAVE_CUDA)
#include <cuda_runtime.h>

#include "clusterlm/backends/strata_backend.hpp"
#include "clusterlm/domain/sampling.hpp"
#endif

#ifndef CLUSTERLM_STRATA_PIN
#define CLUSTERLM_STRATA_PIN "unknown"
#endif

namespace {

using namespace clusterlm;
namespace bs = clusterlm::backends::strata;
using json = nlohmann::json;
namespace fs = std::filesystem;

struct Args {
  std::string cmd;
  std::map<std::string, std::string> kv;
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? def : it->second;
  }
  bool has(const std::string& k) const { return kv.contains(k); }
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) continue;
    k = k.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a.kv[k] = argv[++i];
    else a.kv[k] = "1";
  }
  return a;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep))
    if (!item.empty()) out.push_back(item);
  return out;
}

std::vector<std::uint32_t> uints(const std::string& s) {
  std::vector<std::uint32_t> out;
  for (const auto& x : split(s, ',')) out.push_back(static_cast<std::uint32_t>(std::stoul(x)));
  return out;
}

int fail(const std::string& m) {
  std::fprintf(stderr, "clusterlm-strata: %s\n", m.c_str());
  return 1;
}

json result_header(const std::string& experiment) {
  json r;
  r["tool"] = "clusterlm-strata";
  r["experiment"] = experiment;
  r["provenance"] = "Measured";  // on THIS host; target-machine qualification is the HQ entry's job
  r["build"] = std::string("strata@") + CLUSTERLM_STRATA_PIN + "+clusterlm-patches";
  r["cpu_expert_isa"] = bs::cpu_expert_isa();
  return r;
}

int write_result(const Args& a, const json& r) {
  const std::string out = a.get("out");
  if (out.empty()) {
    std::printf("%s\n", r.dump(2).c_str());
    return 0;
  }
  std::ofstream f(out);
  if (!f) return fail("cannot write " + out);
  f << r.dump(2) << "\n";
  std::printf("wrote %s\n", out.c_str());
  return 0;
}

// ---------------------------------------------------------------- probe

int cmd_probe(const Args& a) {
  json r = result_header("probe");
  r["cpu_expert_kernels"] = bs::cpu_expert_kernels_available();
#if defined(CLUSTERLM_STRATA_HAVE_CUDA)
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
  (void)cudaGetLastError();
  json devs = json::array();
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
    devs.push_back({{"index", i}, {"name", p.name}, {"sm", std::to_string(p.major) + std::to_string(p.minor)},
                    {"memory_bytes", p.totalGlobalMem}});
  }
  r["cuda_devices"] = devs;
  r["backend_hardware_available"] = backends::make_strata_backend({})->info().hardware_available;
#else
  r["cuda_devices"] = "not built (CLUSTERLM_ENABLE_STRATA=OFF)";
#endif
  return write_result(a, r);
}

// ---------------------------------------------------------------- cpu-experts

std::uint16_t fp16_bits(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t e = ((x >> 23) & 0xff) - 127 + 15, m = (x >> 13) & 0x3ff;
  return static_cast<std::uint16_t>((e << 10) | m);
}

Bytes synthetic_expert(const bs::CpuExpertKernel& k, std::uint32_t seed) {
  // valid i-quant blocks: random codes, sane fp16 block scales (see tests/backends/test_strata_cpu_kernels.cpp)
  std::mt19937 rng(seed);
  Bytes b(k.blob_bytes());
  for (auto& x : b) x = static_cast<std::uint8_t>(rng());
  auto scales = [&](std::size_t off, const bs::GgmlType& t, std::uint64_t elems, std::uint64_t rows, float s) {
    const std::uint64_t blocks = elems / t.block_elems * rows;
    for (std::uint64_t i = 0; i < blocks; ++i) {
      const std::uint16_t h = fp16_bits(s * (0.5f + static_cast<float>(rng() % 1000) / 1000.0f));
      std::memcpy(b.data() + off + i * t.block_bytes, &h, 2);
    }
  };
  const bs::GgmlType& gu = *bs::ggml_type_by_name(k.gate_up_type());
  const bs::GgmlType& d = *bs::ggml_type_by_name(k.down_type());
  const std::uint64_t gate = bs::ggml_bytes(gu, k.hidden()) * k.ff();
  scales(0, gu, k.hidden(), k.ff(), 0.004f);
  scales(gate, gu, k.hidden(), k.ff(), 0.004f);
  scales(2 * gate, d, k.ff(), k.hidden(), 0.02f);
  return b;
}

int cmd_cpu_experts(const Args& a) {
  const auto types = split(a.get("types", "iq3_s+iq4_nl,iq2_xs+iq4_nl"), ',');
  const auto tokens = uints(a.get("tokens", "1,2,4,8"));
  const std::uint32_t H = static_cast<std::uint32_t>(std::stoul(a.get("hidden", "2560")));
  const std::uint32_t FF = static_cast<std::uint32_t>(std::stoul(a.get("ff", "640")));
  const unsigned threads = static_cast<unsigned>(std::stoul(a.get("threads", "1")));
  const int reps = std::stoi(a.get("reps", "20"));
  if (!bs::cpu_expert_kernels_available()) return fail("this build has no Strata CPU kernels");
  json r = result_header("HQ-CPU-01");
  r["note"] = "synthetic expert blobs (valid blocks, random codes) at the given geometry; every expert is a separate "
              "blob so the weights stream from memory as in decode";
  r["threads"] = threads;
  json rows = json::array();
  for (const auto& t : types) {
    const auto plus = t.find('+');
    const std::string gu = t.substr(0, plus), dn = plus == std::string::npos ? gu : t.substr(plus + 1);
    auto k = bs::make_cpu_expert_kernel(gu, dn, H, FF);
    if (!k.is_ok()) return fail(k.status().to_string());
    // enough distinct experts per thread to defeat the caches (~256 MiB per thread)
    const std::size_t n_exp = std::max<std::size_t>(8, (256ull << 20) / (*k)->blob_bytes());
    std::vector<std::vector<Bytes>> blobs(threads);
    for (unsigned th = 0; th < threads; ++th)
      for (std::size_t e = 0; e < n_exp; ++e) blobs[th].push_back(synthetic_expert(**k, static_cast<std::uint32_t>(th * 100003 + e)));
    for (std::uint32_t T : tokens) {
      std::vector<float> x(std::size_t{T} * H, 0.5f);
      std::vector<std::vector<float>> y(threads, std::vector<float>(std::size_t{T} * H));
      auto body = [&](unsigned th) {
        for (int rep = 0; rep < reps; ++rep)
          for (const Bytes& b : blobs[th]) (void)(*k)->run(b, x, T, y[th]);
      };
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<std::thread> pool;
      for (unsigned th = 0; th < threads; ++th) pool.emplace_back(body, th);
      for (auto& p : pool) p.join();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double experts = static_cast<double>(reps) * static_cast<double>(n_exp) * threads;
      rows.push_back({{"quant", t},
                      {"tokens", T},
                      {"path", std::string((*k)->path(T))},
                      {"ms_per_expert_per_thread", 1e3 * s * threads / experts},
                      {"expert_bytes_per_s", experts * static_cast<double>((*k)->blob_bytes()) / s},
                      {"blob_bytes", (*k)->blob_bytes()}});
      std::fprintf(stderr, "%s q=%u %s: %.3f ms/expert/thread, %.2f GB/s\n", t.c_str(), T,
                   std::string((*k)->path(T)).c_str(), 1e3 * s * threads / experts,
                   experts * static_cast<double>((*k)->blob_bytes()) / s / 1e9);
    }
  }
  r["results"] = rows;
  return write_result(a, r);
}

// ---------------------------------------------------------------- convert

int cmd_convert(const Args& a) {
  if (!a.has("pack") || !a.has("gguf"))
    return fail("convert needs --pack <strata-pack-dir> --gguf <first-shard> [--out <dir> (default: the GGUF's dir)]");
  bs::ConvertOptions o;
  o.pack_dir = a.get("pack");
  o.gguf = a.get("gguf");
  o.out = a.get("out");
  auto m = bs::convert_model(o, [](std::string_view what) { std::fprintf(stderr, "converted %.*s\n", static_cast<int>(what.size()), what.data()); });
  if (!m.is_ok()) return fail(m.status().to_string());
  std::printf("converted %zu objects (geometry %u layers, %u experts); manifest.json + strata-dense.bin written beside "
              "the GGUF; routed experts reference the GGUF in place\n",
              m->objects.size(), m->geometry.n_layers, m->geometry.n_experts);
  return 0;
}

#if defined(CLUSTERLM_STRATA_HAVE_CUDA)
// ---------------------------------------------------------------- plans and domains (CUDA)

struct Stage {
  domain::StageRole role;
  objects::LayerRange layers;
};

Result<std::vector<Stage>> parse_plan(const std::string& s, std::uint32_t n_layers) {
  std::vector<Stage> out;
  for (const auto& part : split(s, ',')) {
    const auto dash = part.find('-');
    if (dash == std::string::npos) return make_error(ErrorCode::kInvalidArgument, "plan ranges are A-B");
    out.push_back({domain::StageRole::kMiddle,
                   {static_cast<std::uint32_t>(std::stoul(part.substr(0, dash))),
                    static_cast<std::uint32_t>(std::stoul(part.substr(dash + 1)))}});
  }
  if (out.size() < 2 || out.front().layers.begin != 0 || out.back().layers.end != n_layers)
    return make_error(ErrorCode::kInvalidArgument, "a plan covers 0..n_layers in at least a prefix and a tail");
  out.front().role = domain::StageRole::kPrefix;
  out.back().role = domain::StageRole::kTail;
  return out;
}

backends::StrataBackendOptions backend_options(const Args& a) {
  backends::StrataBackendOptions o;
  o.cuda_device = std::stoi(a.get("device", "0"));
  o.ple_table_gguf = a.get("ple-gguf");
  o.mtp_dir = a.get("mtp");
  o.cpu_threads = static_cast<std::uint32_t>(std::stoul(a.get("cpu-threads", "0")));
  return o;
}

domain::DomainSpec spec_for(const Stage& s, std::uint32_t ctx, std::uint32_t q) {
  domain::DomainSpec sp;
  sp.role = s.role;
  sp.layers = s.layers;
  sp.max_context = ctx;
  sp.max_window = q;
  sp.max_sessions = 1;
  return sp;
}

std::uint64_t device_used() {
  std::size_t f = 0, t = 0;
  if (cudaMemGetInfo(&f, &t) != cudaSuccess) return 0;
  return t - f;
}

int cmd_requirements(const Args& a) {
  auto store = objects::CanonicalModelStore::open(a.get("model"));
  if (!store.is_ok()) return fail("--model: " + store.status().to_string());
  const auto& m = (*store)->manifest();
  auto plan = parse_plan(a.get("plan"), m.geometry.n_layers);
  if (!plan.is_ok()) return fail(plan.status().to_string());
  const std::uint32_t ctx = static_cast<std::uint32_t>(std::stoul(a.get("context", "32768")));
  auto backend = backends::make_strata_backend(backend_options(a));
  json r = result_header("HQ-GPU-02");
  json stages = json::array();
  for (const Stage& s : *plan) {
    auto d = backend->create_domain(m, spec_for(s, ctx, 8));
    if (!d.is_ok()) return fail(d.status().to_string());
    auto req = (*d)->describe_requirements();
    if (!req.is_ok()) return fail(req.status().to_string());
    json st = {{"role", std::string(domain::to_string(s.role))},
               {"layers", std::to_string(s.layers.begin) + "-" + std::to_string(s.layers.end)},
               {"gpu_weight_bytes", req->gpu_weight_bytes},
               {"cpu_weight_bytes", req->cpu_weight_bytes},
               {"state_bytes", req->state_bytes},
               {"window_bytes", req->window_bytes},
               {"scratch_bytes", req->scratch_bytes},
               {"staging_bytes", req->staging_bytes}};
    if (a.has("measure")) {
      if (!backend->info().hardware_available) return fail("--measure needs a CUDA device");
      const std::uint64_t before = device_used();
      Status ps = (*d)->prepare(**store);
      if (ps.is_ok()) ps = (*d)->open_session(Epoch{1}, SessionId{1});
      st["measured_device_bytes_after_prepare_and_open"] = device_used() - before;
      st["prepare_status"] = ps.to_string();
      st["resident_weight_bytes"] = (*d)->read_metrics().resident_weight_bytes;
      (void)(*d)->release();
      st["device_bytes_after_release"] = device_used() - before;
    }
    stages.push_back(st);
  }
  r["stages"] = stages;
  r["context"] = ctx;
  return write_result(a, r);
}

// A whole pipeline of domains in one process: the prefix gets tokens, the tail returns logits.
struct Pipeline {
  std::vector<std::unique_ptr<domain::ExecutionDomain>> d;
  std::uint64_t base = 0, window = 0;
  StateVersion state{0};
  Epoch epoch{1};
  SessionId session{1};

  Status build(domain::BackendAdapter& b, const objects::ModelManifest& m, const std::vector<Stage>& plan,
               const objects::ObjectResolver& store, std::uint32_t ctx, std::uint32_t q) {
    for (const Stage& s : plan) {
      CLM_ASSIGN_OR_RETURN(auto dom, b.create_domain(m, spec_for(s, ctx, q)));
      CLM_RETURN_IF_ERROR(dom->prepare(store));
      CLM_RETURN_IF_ERROR(dom->open_session(epoch, session));
      d.push_back(std::move(dom));
    }
    return Status::ok();
  }
  Result<domain::Logits> run(std::span<const std::int32_t> tokens) {
    const domain::WindowRequest req{epoch, session, WindowId{++window}, base, state,
                                    static_cast<std::uint32_t>(tokens.size())};
    CLM_ASSIGN_OR_RETURN(domain::StageActivations a, d.front()->run_prefix(req, tokens));
    for (std::size_t i = 1; i + 1 < d.size(); ++i) CLM_ASSIGN_OR_RETURN(a, d[i]->run_window(req, a));
    return d.back()->run_tail(req, a);
  }
  Status commit(std::uint32_t accepted) {
    const domain::CommitRequest c{epoch, session, WindowId{window}, accepted, state};
    for (auto& x : d) CLM_RETURN_IF_ERROR(x->commit_window(c).status());
    base += accepted;
    state = state.next();
    return Status::ok();
  }
  Status abort() {
    for (auto& x : d) CLM_RETURN_IF_ERROR(x->abort_window(epoch, session, WindowId{window}).status());
    return Status::ok();
  }
};

int cmd_numerics(const Args& a) {
  auto store = objects::CanonicalModelStore::open(a.get("model"));
  if (!store.is_ok()) return fail("--model: " + store.status().to_string());
  const auto& m = (*store)->manifest();
  auto plan = parse_plan(a.get("plan"), m.geometry.n_layers);
  if (!plan.is_ok()) return fail(plan.status().to_string());
  // the reference: the smallest legal split (prefix + tail) at the plan's first boundary
  const std::vector<Stage> ref = {{domain::StageRole::kPrefix, plan->front().layers},
                                  {domain::StageRole::kTail, {plan->front().layers.end, m.geometry.n_layers}}};
  std::vector<std::int32_t> tokens;
  {
    std::ifstream in(a.get("tokens"));
    for (std::int32_t t; in >> t;) tokens.push_back(t);
  }
  if (tokens.size() < 16) return fail("--tokens needs a file of at least 16 token ids");
  const auto qs = uints(a.get("q", "1,2,4,8"));
  const std::uint32_t ctx = static_cast<std::uint32_t>(std::stoul(a.get("context", "8192")));
  auto backend = backends::make_strata_backend(backend_options(a));
  if (!backend->info().hardware_available) return fail("numerics needs a CUDA device (and the model)");
  json r = result_header("HQ-NUM-01");
  json runs = json::array();
  for (std::uint32_t q : qs) {
    Pipeline split, refp;
    if (Status s = split.build(*backend, m, *plan, **store, ctx, q); !s.is_ok()) return fail("split: " + s.to_string());
    if (Status s = refp.build(*backend, m, ref, **store, ctx, q); !s.is_ok()) return fail("reference: " + s.to_string());
    double max_abs = 0;
    std::uint64_t windows = 0, greedy_agree = 0, positions = 0, abort_ok = 0, abort_checked = 0;
    std::uint32_t reject = 1;
    for (std::size_t p = 0; p + q <= tokens.size(); p = split.base) {
      const std::span<const std::int32_t> win(tokens.data() + p, q);
      auto ls = split.run(win);
      auto lr = refp.run(win);
      if (!ls.is_ok() || !lr.is_ok()) return fail("window: " + (ls.is_ok() ? lr.status() : ls.status()).to_string());
      // abort_window restores the committed state: the same window again gives the same logits, bit for bit
      if (windows % 4 == 0) {
        if (Status s = split.abort(); !s.is_ok()) return fail("abort_window: " + s.to_string());
        auto again = split.run(win);
        if (!again.is_ok()) return fail(again.status().to_string());
        ++abort_checked;
        abort_ok += again->data == ls->data ? 1 : 0;
      }
      for (std::size_t i = 0; i < ls->data.size(); ++i)
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(ls->data[i] - lr->data[i])));
      for (std::uint32_t t = 0; t < q; ++t) {
        const std::span<const float> a1(ls->data.data() + std::size_t{t} * ls->vocab, ls->vocab);
        const std::span<const float> a2(lr->data.data() + std::size_t{t} * lr->vocab, lr->vocab);
        greedy_agree += domain::argmax_token(a1) == domain::argmax_token(a2) ? 1 : 0;
        ++positions;
      }
      // every rejection length in turn (1..q accepted)
      const std::uint32_t accepted = reject;
      reject = reject % q + 1;
      if (Status s = split.commit(accepted); !s.is_ok()) return fail("commit: " + s.to_string());
      if (Status s = refp.commit(accepted); !s.is_ok()) return fail("commit: " + s.to_string());
      ++windows;
    }
    runs.push_back({{"q", q},
                    {"windows", windows},
                    {"max_abs_logit_diff", max_abs},
                    {"greedy_agreement", positions ? static_cast<double>(greedy_agree) / static_cast<double>(positions) : 0.0},
                    {"abort_window_restores", abort_ok},
                    {"abort_window_checked", abort_checked}});
    std::fprintf(stderr, "q=%u: %llu windows, max |dlogit| %.3g, abort restores %llu/%llu\n", q,
                 static_cast<unsigned long long>(windows), max_abs, static_cast<unsigned long long>(abort_ok),
                 static_cast<unsigned long long>(abort_checked));
  }
  r["plan"] = a.get("plan");
  r["runs"] = runs;
  if (a.has("mtp")) {
    // greedy generation on the reference pipeline with the tail's MTP drafter
    Pipeline g;
    const std::uint32_t q = qs.back();
    if (Status s = g.build(*backend, m, *plan, **store, ctx, q); !s.is_ok()) return fail("mtp: " + s.to_string());
    auto drafter = backends::make_strata_mtp_drafter(*g.d.back());
    if (!drafter.is_ok()) return fail(drafter.status().to_string());
    std::vector<std::int32_t> hist(tokens.begin(), tokens.end() - 1);
    std::int32_t next = tokens.back();
    // prefill (one token per window keeps it within the Verifier's width)
    for (std::size_t p = 0; p < hist.size(); ++p) {
      const std::int32_t t = hist[p];
      if (!g.run(std::span<const std::int32_t>(&t, 1)).is_ok() || !g.commit(1).is_ok()) return fail("mtp prefill failed");
    }
    std::uint64_t proposed = 0, accepted = 0;
    const std::uint32_t gen = static_cast<std::uint32_t>(std::stoul(a.get("generate", "128")));
    for (std::uint32_t produced = 0; produced < gen;) {
      std::vector<std::int32_t> w{next};
      const auto drafts = (*drafter)->draft(hist, next, q - 1);
      w.insert(w.end(), drafts.begin(), drafts.end());
      auto lg = g.run(w);
      if (!lg.is_ok()) return fail(lg.status().to_string());
      std::uint32_t n = 1;
      std::int32_t pick = domain::argmax_token(std::span<const float>(lg->data.data(), lg->vocab));
      while (n < q && pick == w[n]) {
        pick = domain::argmax_token(std::span<const float>(lg->data.data() + std::size_t{n} * lg->vocab, lg->vocab));
        ++n;
      }
      proposed += q - 1;
      accepted += n - 1;
      hist.insert(hist.end(), w.begin(), w.begin() + n);
      next = pick;
      produced += n;
      if (!g.commit(n).is_ok()) return fail("mtp commit failed");
    }
    r["mtp"] = {{"q", q}, {"drafts_proposed", proposed}, {"drafts_accepted", accepted},
                {"acceptance", proposed ? static_cast<double>(accepted) / static_cast<double>(proposed) : 0.0}};
  }
  return write_result(a, r);
}
#endif

int usage() {
  std::fprintf(stderr,
               "usage: clusterlm-strata <command> [options]\n"
               "  probe        [--out F]\n"
               "  cpu-experts  [--types iq3_s+iq4_nl,iq2_xs+iq4_nl] [--tokens 1,2,4,8] [--threads N] [--reps N] [--out F]\n"
               "  convert      --pack <strata-pack-dir> --gguf <first-shard> [--out <model-dir>]\n"
#if defined(CLUSTERLM_STRATA_HAVE_CUDA)
               "  requirements --model <model-dir> --plan 0-12,12-24,24-36,36-48 [--context N] [--measure] [--ple-gguf G] [--out F]\n"
               "  numerics     --model <model-dir> --plan ... --tokens <ids.txt> --ple-gguf G [--q 1,2,4,8] [--mtp <rt-dir>]\n"
               "               [--generate N] [--context N] [--out F]\n"
#endif
  );
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  if (a.cmd == "probe") return cmd_probe(a);
  if (a.cmd == "cpu-experts") return cmd_cpu_experts(a);
  if (a.cmd == "convert") return cmd_convert(a);
#if defined(CLUSTERLM_STRATA_HAVE_CUDA)
  if (a.cmd == "requirements") return cmd_requirements(a);
  if (a.cmd == "numerics") return cmd_numerics(a);
#else
  if (a.cmd == "requirements" || a.cmd == "numerics")
    return fail(a.cmd + " needs the CUDA backend (configure with -DCLUSTERLM_ENABLE_STRATA=ON)");
#endif
  return usage();
}
