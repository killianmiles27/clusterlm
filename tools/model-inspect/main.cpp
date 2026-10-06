// clusterlm-model-inspect <gguf...> [--manifest out.json] [--hash] [--hash-shards] [--summary]
//                         [--allow-unclassified] [--artifact-id ID] [--threads N]
//
// Opens the GGUF file(s) of a model (a shard of a "-NNNNN-of-MMMMM.gguf" set expands to the whole set), builds
// the canonical ModelManifest from the tensor directory and prints what Father and the planner would see:
// geometry, tensor inventory by kind and type, bytes per layer, per-layer quant types, Father-only bytes and the
// planning cost inputs. Tensor data is read only with --hash/--hash-shards (streamed, bounded buffers).
//
// This tool reports structure and sizes. It never emits provenance beyond what the files themselves state and
// never marks anything Qualified; the real-model run is hardware-qualification step HQ-MODEL-01.
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "clusterlm/objects/gguf_manifest.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;

namespace {

constexpr const char* kUsage =
    "usage: clusterlm-model-inspect <gguf...> [--manifest out.json] [--hash] [--hash-shards] [--summary]\n"
    "                               [--allow-unclassified] [--artifact-id ID] [--threads N]\n";

unsigned long long ull(std::uint64_t v) { return static_cast<unsigned long long>(v); }

std::string human(std::uint64_t b) {
  char buf[48];
  const double v = static_cast<double>(b);
  if (b >= (1ull << 30)) std::snprintf(buf, sizeof buf, "%.2f GiB", v / static_cast<double>(1ull << 30));
  else if (b >= (1ull << 20)) std::snprintf(buf, sizeof buf, "%.2f MiB", v / static_cast<double>(1ull << 20));
  else if (b >= (1ull << 10)) std::snprintf(buf, sizeof buf, "%.2f KiB", v / static_cast<double>(1ull << 10));
  else std::snprintf(buf, sizeof buf, "%llu B", ull(b));
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::filesystem::path> inputs;
  std::string manifest_out, artifact_id;
  bool hash = false, hash_shards = false, summary = false, allow_unclassified = false;
  unsigned threads = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (a == "--manifest") {
      const char* v = value();
      if (!v) { std::fprintf(stderr, "--manifest needs a path\n"); return 2; }
      manifest_out = v;
    } else if (a == "--artifact-id") {
      const char* v = value();
      if (!v) { std::fprintf(stderr, "--artifact-id needs a value\n"); return 2; }
      artifact_id = v;
    } else if (a == "--threads") {
      const char* v = value();
      if (!v) { std::fprintf(stderr, "--threads needs a value\n"); return 2; }
      threads = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
    } else if (a == "--hash") {
      hash = true;
    } else if (a == "--hash-shards") {
      hash_shards = true;
    } else if (a == "--summary") {
      summary = true;
    } else if (a == "--allow-unclassified") {
      allow_unclassified = true;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown option %s\n%s", a.c_str(), kUsage);
      return 2;
    } else {
      inputs.emplace_back(a);
    }
  }
  if (inputs.empty()) {
    std::fprintf(stderr, "%s", kUsage);
    return 2;
  }

  // A single "-00001-of-0000N.gguf" argument stands for the whole set.
  std::vector<std::filesystem::path> paths;
  for (const auto& in : inputs) {
    auto ex = expand_split_paths(in);
    if (!ex.is_ok()) {
      std::fprintf(stderr, "error: %s\n", ex.status().to_string().c_str());
      return 1;
    }
    for (auto& p : *ex)
      if (std::find(paths.begin(), paths.end(), p) == paths.end()) paths.push_back(std::move(p));
  }

  auto files = open_gguf_model(paths);
  if (!files.is_ok()) {
    std::fprintf(stderr, "error: %s\n", files.status().to_string().c_str());
    return 1;
  }

  ManifestBuildOptions opts;
  opts.hash_objects = hash;
  opts.hash_shards = hash_shards;
  opts.allow_unclassified = allow_unclassified;
  opts.artifact_id = artifact_id;
  opts.hash_threads = threads;
  std::uint64_t last_pct = 101;
  if (hash || hash_shards) {
    opts.progress = [&](const BuildProgress& p) {
      const std::uint64_t pct = p.bytes_total == 0 ? 100 : p.bytes_done * 100 / p.bytes_total;
      if (pct != last_pct) {
        last_pct = pct;
        std::fprintf(stderr, "\rhashing %s: %3llu%% (%s / %s)   ", p.phase == BuildProgress::Phase::kHashingShards ? "shards " : "objects",
                     ull(pct), human(p.bytes_done).c_str(), human(p.bytes_total).c_str());
      }
      return true;
    };
  }
  auto built = build_manifest(*files, opts);
  if (hash || hash_shards) std::fprintf(stderr, "\n");
  if (!built.is_ok()) {
    std::fprintf(stderr, "error: %s\n", built.status().to_string().c_str());
    return 1;
  }
  const ModelManifest& m = built->manifest;
  const ManifestBuildReport& rep = built->report;
  const ModelGeometry& g = m.geometry;

  std::printf("artifact        %s\n", m.artifact_id.c_str());
  std::printf("architecture    %s\n", rep.architecture.c_str());
  std::printf("license         %s\n", m.license.empty() ? "(none stated)" : m.license.c_str());
  std::printf("files           %zu%s\n", m.shards.size(), files->is_split ? " (split GGUF)" : "");
  for (std::size_t s = 0; s < m.shards.size(); ++s)
    std::printf("  [%zu] %s  %s  gguf v%u  %zu tensors  digest %s\n", s, m.shards[s].file_name.c_str(), human(m.shards[s].byte_size).c_str(),
                files->shards[s].version, files->shards[s].tensors.size(),
                m.shards[s].digest.is_zero() ? "(not computed)" : m.shards[s].digest.hex().substr(0, 16).c_str());
  std::printf("manifest root   %s%s\n", m.root_hash().hex().c_str(), rep.hashed_objects ? "" : "  (object digests NOT computed; rerun with --hash)");

  std::printf("\ngeometry\n");
  std::uint32_t n_qsa = 0;
  for (LayerKind k : g.layer_kinds) n_qsa += k == LayerKind::kFullAttention ? 1 : 0;
  std::printf("  layers %u (%u linear-attention / %u full-attention)  hidden %u  residual_streams %u\n", g.n_layers, g.n_layers - n_qsa, n_qsa,
              g.hidden_size, g.residual_streams);
  std::printf("  experts %u routed, %u active, expert_ff %u, shared_expert_ff %u\n", g.n_experts, g.n_active_experts, g.expert_ff, g.shared_expert_ff);
  std::printf("  heads %u q / %u kv, head_dim %u  vocab %u\n", g.n_heads, g.n_kv_heads, g.head_dim, g.vocab_size);
  std::printf("  PLE layer %u, n-gram <= %u, rows %u  MTP layers %u\n", g.ple_layer, g.ple_ngram, g.ple_rows, g.mtp_layers);

  // ---- inventory by kind and type (every tensor in the files, manifest or not) ----------------------------------
  struct KindStat {
    std::uint64_t tensors = 0, bytes = 0;
    std::map<std::string, std::uint64_t> by_type;
  };
  std::map<std::string, KindStat> kinds;
  for (const GgufFile& f : files->shards)
    for (const GgufTensorInfo& t : f.tensors) {
      const TensorClassification c = classify_tensor(t.name, g.n_layers);
      KindStat& k = kinds[std::string(to_string(c.cls))];
      ++k.tensors;
      k.bytes += t.n_bytes;
      k.by_type[std::string(ggml_type_info(t.type).name)] += t.n_bytes;
    }
  std::printf("\ntensor inventory (%llu tensors, %s)\n", ull(rep.tensors_total), human(rep.tensor_bytes_total).c_str());
  for (const auto& [name, k] : kinds) {
    std::printf("  %-15s %6llu tensors  %12s   ", name.c_str(), ull(k.tensors), human(k.bytes).c_str());
    for (const auto& [type, bytes] : k.by_type) std::printf("%s:%s ", type.c_str(), human(bytes).c_str());
    std::printf("\n");
  }
  if (!rep.unclassified.empty()) {
    std::printf("\nUNCLASSIFIED tensors (%zu, not in the manifest):\n", rep.unclassified.size());
    for (const UnclassifiedTensor& u : rep.unclassified)
      std::printf("  %s  %s  %s  (%s)\n", u.name.c_str(), std::string(ggml_type_info(u.type).name).c_str(), human(u.bytes).c_str(), u.reason.c_str());
  }

  // ---- per layer ---------------------------------------------------------------------------------------------
  struct LayerStat {
    std::uint64_t dense = 0, shared = 0, expert_one = 0, experts = 0;
    std::string dense_q, expert_q;
  };
  std::vector<LayerStat> layers(g.n_layers);
  std::uint64_t father_embd = 0, father_ple = 0, father_head = 0, father_mtp = 0;
  for (const ManifestObject& o : m.objects) {
    switch (o.kind) {
      case ObjectKind::kEmbedding: father_embd += o.byte_size; break;
      case ObjectKind::kPleLookup: father_ple += o.byte_size; break;
      case ObjectKind::kOutputHead: father_head += o.byte_size; break;
      case ObjectKind::kMtp: father_mtp += o.byte_size; break;
      case ObjectKind::kLayerDense: layers[*o.layer].dense += o.byte_size; layers[*o.layer].dense_q = o.representation.quant_type; break;
      case ObjectKind::kSharedExpert: layers[*o.layer].shared += o.byte_size; break;
      case ObjectKind::kRoutedExpert:
        layers[*o.layer].experts += o.byte_size;
        layers[*o.layer].expert_one = std::max(layers[*o.layer].expert_one, o.byte_size);
        layers[*o.layer].expert_q = o.representation.quant_type;
        break;
    }
  }
  std::uint64_t layers_total = 0, dense_total = 0, expert_total = 0;
  if (!summary) {
    std::printf("\nper layer (bytes as stored; expert quant = gate|up|down types of expert 0)\n");
    std::printf("  %5s %-5s %12s %12s %14s %12s  %s\n", "layer", "kind", "dense", "shared-exp", "experts(all)", "per-expert", "expert quant");
  }
  for (std::uint32_t L = 0; L < g.n_layers; ++L) {
    const LayerStat& s = layers[L];
    layers_total += s.dense + s.shared + s.experts;
    dense_total += s.dense + s.shared;
    expert_total += s.experts;
    if (!summary)
      std::printf("  %5u %-5s %12s %12s %14s %12s  %s\n", L, g.layer_kinds[L] == LayerKind::kFullAttention ? "QSA" : "GDN", human(s.dense).c_str(),
                  human(s.shared).c_str(), human(s.experts).c_str(), human(s.expert_one).c_str(), s.expert_q.c_str());
  }
  std::map<std::string, std::uint32_t> expert_quant_layers;
  for (const LayerStat& s : layers) ++expert_quant_layers[s.expert_q];
  std::printf("\nexpert quant types by layer count:");
  for (const auto& [q, n] : expert_quant_layers) std::printf("  %s x%u", q.c_str(), n);
  std::printf("\n");

  std::printf("\ntotals\n");
  std::printf("  layers (dense+shared)  %s\n", human(dense_total).c_str());
  std::printf("  layers (routed)        %s\n", human(expert_total).c_str());
  std::printf("  layers total           %s\n", human(layers_total).c_str());
  std::printf("  father-only            %s  (embedding %s, PLE lookup %s, head %s, MTP %s)\n", human(father_embd + father_ple + father_head + father_mtp).c_str(),
              human(father_embd).c_str(), human(father_ple).c_str(), human(father_head).c_str(), human(father_mtp).c_str());
  std::printf("  in manifest            %s of %s on disk\n", human(rep.tensor_bytes_in_manifest).c_str(), human(rep.tensor_bytes_total).c_str());

  // ---- planning cost inputs (structural facts; routing statistics and timings are measured elsewhere) ----------
  const std::uint64_t boundary = (std::uint64_t{g.residual_streams} * g.hidden_size + g.hidden_size + g.residual_streams) * 4;
  std::uint64_t max_expert = 0, max_dense = 0;
  for (const LayerStat& s : layers) {
    max_expert = std::max(max_expert, s.expert_one);
    max_dense = std::max(max_dense, s.dense + s.shared);
  }
  std::printf("\nplanning inputs (from this manifest)\n");
  std::printf("  boundary ABI bytes/position (kResidualHandoffF32V1)  %s\n", human(boundary).c_str());
  std::printf("  KV elements/token/full-attention layer (2*kv*head_dim)  %llu  (%s at f16, %s at f32)\n",
              ull(std::uint64_t{2} * g.n_kv_heads * g.head_dim), human(std::uint64_t{2} * g.n_kv_heads * g.head_dim * 2).c_str(),
              human(std::uint64_t{2} * g.n_kv_heads * g.head_dim * 4).c_str());
  std::printf("  largest per-layer dense+shared object  %s   largest routed expert  %s\n", human(max_dense).c_str(), human(max_expert).c_str());
  std::printf("  fully-resident bytes for all %u layers  %s (plus Father-only %s)\n", g.n_layers, human(layers_total).c_str(),
              human(father_embd + father_head + father_mtp).c_str());

  if (!manifest_out.empty()) {
    const std::string json = m.to_json();
    std::ofstream f(manifest_out, std::ios::binary | std::ios::trunc);
    f.write(json.data(), static_cast<std::streamsize>(json.size()));
    f.flush();
    if (!f) {
      std::fprintf(stderr, "error: cannot write %s\n", manifest_out.c_str());
      return 1;
    }
    std::printf("\nwrote manifest (%zu objects) to %s\n", m.objects.size(), manifest_out.c_str());
  }
  return rep.unclassified.empty() || allow_unclassified ? 0 : 1;
}
