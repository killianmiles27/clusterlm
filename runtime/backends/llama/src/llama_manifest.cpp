#include "clusterlm/backends/llama_manifest.hpp"

#include <algorithm>
#include <fstream>

#include "clusterlm/objects/gguf.hpp"
#include "clusterlm/objects/gguf_manifest.hpp"

namespace clusterlm::backends {
namespace {

using namespace objects;

Status bad(std::string m) { return make_error(ErrorCode::kInvalidArgument, "llama manifest: " + std::move(m)); }


// Integer metadata, or the maximum of an array of integers (per-layer head counts and similar).
std::optional<std::uint64_t> meta_u64(const GgufFile& f, const std::string& key) {
  const GgufValue* v = f.find(key);
  if (v == nullptr) return std::nullopt;
  if (v->is_array()) {
    std::optional<std::uint64_t> best;
    for (const GgufValue& item : v->items)
      if (auto x = item.as_u64()) best = std::max(best.value_or(0), *x);
    return best;
  }
  return v->as_u64();
}

std::string meta_string(const GgufFile& f, const std::string& key) {
  const GgufValue* v = f.find(key);
  return v != nullptr && v->is_string() ? v->s : std::string{};
}

struct Ref {
  std::uint32_t shard;
  const GgufTensorInfo* info;
};

void add_ranges(ManifestObject& o, const std::vector<Ref>& refs) {
  std::vector<GgmlType> types;
  for (const Ref& r : refs) {
    o.source_ranges.push_back({r.shard, r.info->file_offset, r.info->n_bytes});
    o.byte_size += r.info->n_bytes;
    types.push_back(r.info->type);
  }
  o.representation.quant_type = encode_quant_types(types);
  const GgmlTypeInfo& first = ggml_type_info(types.front());
  o.representation.block_size = first.block_elems > 1 ? first.block_elems : 0;
}

}  // namespace

Result<std::vector<std::filesystem::path>> llama_model_files(const std::filesystem::path& any_shard) {
  return expand_split_paths(any_shard);
}

Result<ModelManifest> build_llama_manifest(const std::vector<std::filesystem::path>& paths,
                                           const LlamaManifestOptions& options, LlamaModelReport* report) {
  if (paths.empty()) return bad("no GGUF files");
  for (const auto& p : paths)
    if (p.parent_path() != paths.front().parent_path()) return bad("all shards must be in one directory");
  CLM_ASSIGN_OR_RETURN(GgufModelFiles files, open_gguf_model(paths));
  const GgufFile& meta = files.meta();

  const std::string arch = meta_string(meta, "general.architecture");
  if (arch.empty()) return bad("general.architecture is missing");
  const auto key = [&](const char* suffix) { return arch + "." + suffix; };

  ModelManifest m;
  m.artifact_id = !options.artifact_id.empty() ? options.artifact_id
                  : !meta_string(meta, "general.name").empty() ? meta_string(meta, "general.name")
                                                               : paths.front().filename().string();
  m.license = meta_string(meta, "general.license");

  ModelGeometry& g = m.geometry;
  g.family = "llama.cpp:" + arch;
  const auto block_count = meta_u64(meta, key("block_count"));
  const auto hidden = meta_u64(meta, key("embedding_length"));
  if (!block_count || !hidden || *block_count == 0 || *hidden == 0) return bad(arch + ".block_count / .embedding_length missing");
  const std::uint64_t nextn = meta_u64(meta, key("nextn_predict_layers")).value_or(0);
  if (nextn >= *block_count) return bad("nextn_predict_layers >= block_count");
  g.n_layers = static_cast<std::uint32_t>(*block_count - nextn);
  g.hidden_size = static_cast<std::uint32_t>(*hidden);
  g.residual_streams = 1;
  g.n_experts = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, meta_u64(meta, key("expert_count")).value_or(1)));
  g.n_active_experts = static_cast<std::uint32_t>(
      std::clamp<std::uint64_t>(meta_u64(meta, key("expert_used_count")).value_or(1), 1, g.n_experts));
  const std::uint64_t ff = meta_u64(meta, key("feed_forward_length")).value_or(0);
  g.expert_ff = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(1, meta_u64(meta, key("expert_feed_forward_length")).value_or(ff ? ff : *hidden)));
  g.shared_expert_ff = static_cast<std::uint32_t>(meta_u64(meta, key("expert_shared_feed_forward_length")).value_or(0));
  g.n_heads = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, meta_u64(meta, key("attention.head_count")).value_or(1)));
  g.n_kv_heads = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(1, meta_u64(meta, key("attention.head_count_kv")).value_or(g.n_heads)));
  if (g.n_heads % g.n_kv_heads != 0) g.n_kv_heads = g.n_heads;  // geometry requires a whole GQA ratio
  g.head_dim = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(1, meta_u64(meta, key("attention.key_length")).value_or(*hidden / g.n_heads)));
  g.ple_layer = 0;
  g.ple_ngram = 1;
  g.ple_rows = 1;
  g.mtp_layers = static_cast<std::uint32_t>(nextn);

  std::uint32_t vocab = static_cast<std::uint32_t>(meta_u64(meta, key("vocab_size")).value_or(0));
  if (vocab == 0)
    if (const GgufValue* toks = meta.find("tokenizer.ggml.tokens"); toks != nullptr && toks->is_array())
      vocab = static_cast<std::uint32_t>(toks->count);

  // ---- tensors -> objects ----------------------------------------------------------------------------------
  std::vector<Ref> embedding, head;
  std::vector<std::vector<Ref>> per_layer(g.n_layers);
  std::vector<bool> recurrent(g.n_layers, false);
  LlamaModelReport rep;
  for (std::uint32_t s = 0; s < files.shards.size(); ++s) {
    for (const GgufTensorInfo& t : files.shards[s].tensors) {
      const Ref ref{s, &t};
      ++rep.tensor_count;
      rep.tensor_bytes += t.n_bytes;
      auto& st = rep.tensor_types[std::string(ggml_type_info(t.type).name)];
      ++st.tensors;
      st.bytes += t.n_bytes;
      const TensorClassification c = classify_tensor(t.name, g.n_layers);
      switch (c.cls) {
        case TensorClass::kEmbedding: embedding.push_back(ref); break;
        case TensorClass::kLayerDense:
        case TensorClass::kSharedExpert:
        case TensorClass::kExpertGate:
        case TensorClass::kExpertUp:
        case TensorClass::kExpertDown:
        case TensorClass::kExpertGateUp:
          per_layer[*c.layer].push_back(ref);
          if (t.name.find(".ssm_") != std::string::npos) recurrent[*c.layer] = true;
          break;
        // Output head, final norm, MTP/nextn tensors, rope tables and anything else without a layer: Father-only.
        default: head.push_back(ref); break;
      }
    }
  }
  if (embedding.size() != 1) return bad("expected exactly one token_embd.weight");
  const GgufTensorInfo& emb = *embedding[0].info;
  if (vocab == 0 && emb.n_dims == 2) vocab = static_cast<std::uint32_t>(emb.dims[1]);
  if (vocab == 0) return bad("cannot determine the vocabulary size");
  g.vocab_size = vocab;
  g.layer_kinds.assign(g.n_layers, LayerKind::kFullAttention);
  for (std::uint32_t L = 0; L < g.n_layers; ++L)
    if (recurrent[L]) g.layer_kinds[L] = LayerKind::kRecurrent;

  for (const auto& f : files.paths) {
    ShardInfo si;
    si.file_name = f.filename().string();
    std::error_code ec;
    si.byte_size = std::filesystem::file_size(f, ec);
    if (ec) return make_error(ErrorCode::kNotFound, "cannot stat " + f.string() + ": " + ec.message());
    m.shards.push_back(std::move(si));
  }

  auto add_object = [&](std::string name, ObjectKind kind, std::optional<std::uint32_t> layer,
                        const std::vector<Ref>& refs) {
    if (refs.empty()) return;
    ManifestObject o;
    o.name = std::move(name);
    o.kind = kind;
    o.layer = layer;
    o.alignment = static_cast<std::uint32_t>(std::min<std::uint64_t>(files.meta().alignment, 64));
    if ((o.alignment & (o.alignment - 1)) != 0 || o.alignment == 0) o.alignment = 32;
    add_ranges(o, refs);
    m.objects.push_back(std::move(o));
  };
  add_object(std::string(kEmbeddingObjectName), ObjectKind::kEmbedding, std::nullopt, embedding);
  for (std::uint32_t L = 0; L < g.n_layers; ++L) add_object(dense_object_name(L), ObjectKind::kLayerDense, L, per_layer[L]);
  add_object(std::string(kHeadObjectName), ObjectKind::kOutputHead, std::nullopt, head);

  if (options.hash_objects || options.hash_shards) {
    HashOptions h;
    h.hash_objects = options.hash_objects;
    h.hash_shards = options.hash_shards;
    CLM_RETURN_IF_ERROR(compute_manifest_digests(m, files.paths, h));
  }
  CLM_RETURN_IF_ERROR(m.validate());

  if (report != nullptr) {
    rep.architecture = arch;
    rep.name = meta_string(meta, "general.name");
    rep.n_layers = g.n_layers;
    rep.hidden_size = g.hidden_size;
    rep.vocab_size = g.vocab_size;
    *report = std::move(rep);
  }
  return m;
}

Result<ModelManifest> write_llama_model_dir(const std::vector<std::filesystem::path>& paths,
                                            const LlamaManifestOptions& options) {
  CLM_ASSIGN_OR_RETURN(ModelManifest m, build_llama_manifest(paths, options));
  const auto file = paths.front().parent_path() / "manifest.json";
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) return make_error(ErrorCode::kUnavailable, "cannot write " + file.string());
  out << m.to_json();
  out.close();
  if (!out) return make_error(ErrorCode::kUnavailable, "short write to " + file.string());
  return m;
}

}  // namespace clusterlm::backends
