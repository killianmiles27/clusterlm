#include "clusterlm/objects/gguf_manifest.hpp"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>

namespace clusterlm::objects {

namespace {

Status bad(const std::string& m) { return make_error(ErrorCode::kInvalidArgument, "model: " + m); }

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends_with(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.substr(s.size() - p.size()) == p; }

// "blk.<n>.<rest>" -> n, rest. Canonical decimal only (no sign, no leading zeros).
bool parse_blk(std::string_view name, std::uint32_t& layer, std::string_view& rest) {
  if (!starts_with(name, "blk.")) return false;
  std::size_t i = 4, start = 4;
  std::uint64_t v = 0;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    v = v * 10 + static_cast<std::uint64_t>(name[i] - '0');
    if (++i - start > 9) return false;
  }
  if (i == start || i >= name.size() || name[i] != '.') return false;
  if (name[start] == '0' && i - start > 1) return false;
  layer = static_cast<std::uint32_t>(v);
  rest = name.substr(i + 1);
  return !rest.empty();
}

using Ref = GgufModelFiles::TensorRef;

struct LayerTensors {
  std::vector<Ref> dense, shared;
  std::optional<Ref> gate, up, down, gate_up;
};

bool physical_less(const Ref& a, const Ref& b) {
  if (a.shard != b.shard) return a.shard < b.shard;
  return a.info->file_offset < b.info->file_offset;
}

Result<std::uint32_t> meta_u32(const GgufFile& f, const std::string& key, bool required, std::uint32_t dflt = 0) {
  const GgufValue* v = f.find(key);
  if (v == nullptr) {
    if (required) return bad("required metadata key '" + key + "' is missing");
    return dflt;
  }
  auto u = v->as_u64();
  if (!u || *u > 0xFFFFFFFFull) return bad("metadata key '" + key + "' must be a non-negative integer that fits in 32 bits");
  return static_cast<std::uint32_t>(*u);
}

std::string dims_str(const GgufTensorInfo& t) {
  std::string s = "[";
  for (std::uint32_t i = 0; i < t.n_dims; ++i) s += (i ? "," : "") + std::to_string(t.dims[i]);
  return s + "]";
}

struct ObjectBuilder {
  ModelManifest& m;
  ManifestBuildReport& rep;
  const GgufModelFiles& files;
  bool layouts;

  // Appends one object made of `parts` (a tensor, or an expert slice of a stacked tensor).
  struct Part {
    Ref ref;
    std::uint64_t file_offset;
    std::uint64_t length;
    std::vector<std::uint64_t> dims;
  };

  Status add(std::string name, ObjectKind kind, std::optional<std::uint32_t> layer, std::optional<std::uint32_t> expert,
             const std::vector<Part>& parts) {
    ManifestObject o;
    o.name = std::move(name);
    o.kind = kind;
    o.layer = layer;
    o.expert = expert;
    std::vector<GgmlType> types;
    ObjectLayout lay;
    bool same_block = true;
    std::uint32_t block = 0;
    for (const Part& p : parts) {
      o.source_ranges.push_back({p.ref.shard, p.file_offset, p.length});
      types.push_back(p.ref.info->type);
      const std::uint32_t be = ggml_type_info(p.ref.info->type).block_elems;
      if (block == 0) block = be;
      same_block = same_block && be == block;
      if (layouts) lay.entries.push_back({p.ref.info->name, p.dims, p.ref.info->type, o.byte_size, p.length});
      if (o.byte_size > ~std::uint64_t{0} - p.length) return bad("object '" + o.name + "' size overflows");
      o.byte_size += p.length;
    }
    o.representation.quant_type = encode_quant_types(types);
    o.representation.block_size = same_block && block > 1 ? block : 0;
    o.representation.conversion_version = 0;
    o.representation.little_endian = true;
    o.alignment = 64;
    rep.tensor_bytes_in_manifest += o.byte_size;
    m.objects.push_back(std::move(o));
    if (layouts) rep.layouts.push_back(std::move(lay));
    return Status::ok();
  }

  static Part whole(const Ref& r) {
    return {r, r.info->file_offset, r.info->n_bytes, std::vector<std::uint64_t>(r.info->dims.begin(), r.info->dims.begin() + r.info->n_dims)};
  }

  Status add_whole(std::string name, ObjectKind kind, std::optional<std::uint32_t> layer, std::vector<Ref> refs) {
    std::sort(refs.begin(), refs.end(), physical_less);
    std::vector<Part> parts;
    parts.reserve(refs.size());
    for (const Ref& r : refs) parts.push_back(whole(r));
    return add(std::move(name), kind, layer, std::nullopt, parts);
  }
};

}  // namespace

std::string_view to_string(TensorClass c) {
  switch (c) {
    case TensorClass::kEmbedding: return "embedding";
    case TensorClass::kPleLookup: return "ple_lookup";
    case TensorClass::kOutputHead: return "output_head";
    case TensorClass::kMtp: return "mtp";
    case TensorClass::kLayerDense: return "layer_dense";
    case TensorClass::kSharedExpert: return "shared_expert";
    case TensorClass::kExpertGate: return "expert_gate";
    case TensorClass::kExpertUp: return "expert_up";
    case TensorClass::kExpertDown: return "expert_down";
    case TensorClass::kExpertGateUp: return "expert_gate_up";
    case TensorClass::kUnclassified: return "unclassified";
  }
  return "?";
}

TensorClassification classify_tensor(std::string_view name, std::uint32_t n_layers) {
  auto make = [](TensorClass c, std::optional<std::uint32_t> layer = std::nullopt, std::string reason = {}) {
    return TensorClassification{c, layer, std::move(reason)};
  };
  if (name == "token_embd.weight") return make(TensorClass::kEmbedding);
  if (name == "per_layer_token_embd.weight") return make(TensorClass::kPleLookup);
  if (name == "output.weight" || name == "output_norm.weight" || (starts_with(name, "output_hc_") && ends_with(name, ".weight")))
    return make(TensorClass::kOutputHead);
  if (starts_with(name, "mtp.") || starts_with(name, "nextn.")) return make(TensorClass::kMtp);
  std::uint32_t layer = 0;
  std::string_view rest;
  if (parse_blk(name, layer, rest)) {
    if (layer >= n_layers) return make(TensorClass::kMtp, layer);
    if (starts_with(rest, "nextn.")) return make(TensorClass::kMtp, layer);
    if (rest == "ffn_gate_exps.weight") return make(TensorClass::kExpertGate, layer);
    if (rest == "ffn_up_exps.weight") return make(TensorClass::kExpertUp, layer);
    if (rest == "ffn_down_exps.weight") return make(TensorClass::kExpertDown, layer);
    if (rest == "ffn_gate_up_exps.weight") return make(TensorClass::kExpertGateUp, layer);
    if (rest.find("_exps") != std::string_view::npos)
      return make(TensorClass::kUnclassified, layer, "unrecognized stacked-expert tensor");
    if (rest == "ffn_gate_shexp.weight" || rest == "ffn_up_shexp.weight" || rest == "ffn_down_shexp.weight" ||
        rest == "ffn_gate_inp_shexp.weight")
      return make(TensorClass::kSharedExpert, layer);
    return make(TensorClass::kLayerDense, layer);
  }
  return make(TensorClass::kUnclassified, std::nullopt, "name matches no known tensor family");
}

std::string encode_quant_types(std::span<const GgmlType> per_range) {
  if (per_range.empty()) return {};
  const bool uniform = std::all_of(per_range.begin(), per_range.end(), [&](GgmlType t) { return t == per_range[0]; });
  std::string out;
  for (std::size_t i = 0; i < (uniform ? 1 : per_range.size()); ++i) {
    if (i) out += '|';
    out += ggml_type_info(per_range[i]).name;
  }
  return out;
}

Result<std::vector<GgmlType>> decode_quant_types(std::string_view q, std::size_t n_ranges) {
  std::vector<GgmlType> out;
  std::size_t pos = 0;
  while (true) {
    const std::size_t bar = q.find('|', pos);
    const std::string_view tok = q.substr(pos, bar == std::string_view::npos ? std::string_view::npos : bar - pos);
    const GgmlTypeInfo* t = ggml_type_by_name(tok);
    if (t == nullptr) return make_error(ErrorCode::kInvalidArgument, "quant_type: unknown ggml type '" + std::string(tok) + "'");
    out.push_back(t->type);
    if (bar == std::string_view::npos) break;
    pos = bar + 1;
  }
  if (out.size() == 1) out.assign(n_ranges, out[0]);
  if (out.size() != n_ranges)
    return make_error(ErrorCode::kInvalidArgument, "quant_type lists " + std::to_string(out.size()) + " types for " + std::to_string(n_ranges) + " ranges");
  return out;
}

// ---- hashing ------------------------------------------------------------------------------------------------------

Status compute_manifest_digests(ModelManifest& manifest, const std::vector<std::filesystem::path>& shard_paths, const HashOptions& opts) {
  if (shard_paths.size() != manifest.shards.size()) return bad("shard path count does not match the manifest");
  for (std::size_t s = 0; s < shard_paths.size(); ++s) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(shard_paths[s], ec);
    if (ec) return make_error(ErrorCode::kNotFound, "cannot stat " + shard_paths[s].string() + ": " + ec.message());
    if (size != manifest.shards[s].byte_size)
      return make_error(ErrorCode::kDataLoss, "shard " + shard_paths[s].filename().string() + " changed size since the manifest was built");
  }
  struct Task {
    bool shard;
    std::size_t index;
  };
  std::vector<Task> tasks;
  std::uint64_t total_objects = 0, total_shards = 0;
  if (opts.hash_objects)
    for (std::size_t i = 0; i < manifest.objects.size(); ++i) {
      tasks.push_back({false, i});
      total_objects += manifest.objects[i].byte_size;
    }
  if (opts.hash_shards)
    for (std::size_t s = 0; s < manifest.shards.size(); ++s) {
      tasks.push_back({true, s});
      total_shards += manifest.shards[s].byte_size;
    }

  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex mu;  // guards first_error, progress callback and the counters
  Status first_error = Status::ok();
  std::uint64_t done_objects = 0, done_shards = 0;
  const std::size_t buf_bytes = std::max<std::size_t>(opts.buffer_bytes, 4096);

  auto fail = [&](Status st) {
    std::lock_guard<std::mutex> lk(mu);
    if (first_error.is_ok()) first_error = std::move(st);
    failed = true;
  };
  auto report = [&](bool shard, std::uint64_t delta, std::string_view current) {
    std::lock_guard<std::mutex> lk(mu);
    (shard ? done_shards : done_objects) += delta;
    if (opts.progress) {
      BuildProgress p;
      p.phase = shard ? BuildProgress::Phase::kHashingShards : BuildProgress::Phase::kHashingObjects;
      p.bytes_done = shard ? done_shards : done_objects;
      p.bytes_total = shard ? total_shards : total_objects;
      p.current = current;
      if (!opts.progress(p) && first_error.is_ok()) {
        first_error = make_error(ErrorCode::kCancelled, "manifest hashing cancelled");
        failed = true;
      }
    }
  };

  auto worker = [&]() {
    std::vector<std::ifstream> files(shard_paths.size());
    Bytes buf(buf_bytes);
    auto stream = [&](std::uint32_t shard, std::uint64_t offset, std::uint64_t length, Sha256& h, bool is_shard,
                      std::string_view current) -> Status {
      std::ifstream& f = files[shard];
      if (!f.is_open()) {
        f.open(shard_paths[shard], std::ios::binary);
        if (!f) return make_error(ErrorCode::kUnavailable, "cannot open " + shard_paths[shard].string());
      }
      f.clear();
      f.seekg(static_cast<std::streamoff>(offset));
      while (length > 0) {
        if (failed) return make_error(ErrorCode::kCancelled, "aborted");
        const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(length, buf.size()));
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
        if (!f || static_cast<std::size_t>(f.gcount()) != n)
          return make_error(ErrorCode::kDataLoss, "short read from " + shard_paths[shard].filename().string() + " at offset " + std::to_string(offset));
        h.update(ByteSpan(buf.data(), n));
        length -= n;
        offset += n;
        report(is_shard, n, current);
      }
      return Status::ok();
    };
    for (std::size_t t = next++; t < tasks.size() && !failed; t = next++) {
      const Task& task = tasks[t];
      Sha256 h;
      if (task.shard) {
        ShardInfo& s = manifest.shards[task.index];
        Status st = stream(static_cast<std::uint32_t>(task.index), 0, s.byte_size, h, true, s.file_name);
        if (!st.is_ok()) return fail(std::move(st));
        s.digest = h.finish();
      } else {
        ManifestObject& o = manifest.objects[task.index];
        for (const SourceRange& r : o.source_ranges) {
          Status st = stream(r.shard, r.offset, r.length, h, false, o.name);
          if (!st.is_ok()) return fail(std::move(st));
        }
        o.source_digest = h.finish();
        o.object_digest = o.source_digest;  // conversion_version 0: provisioned bytes == source bytes
      }
    }
  };

  const unsigned n_threads = std::max(1u, std::min<unsigned>(opts.threads, 64));
  if (n_threads == 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < n_threads; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  }
  return first_error;
}

// ---- build --------------------------------------------------------------------------------------------------------

Result<BuiltManifest> build_manifest(const std::vector<std::filesystem::path>& paths, const ManifestBuildOptions& opts) {
  CLM_ASSIGN_OR_RETURN(GgufModelFiles files, open_gguf_model(paths, opts.limits));
  return build_manifest(files, opts);
}

Result<BuiltManifest> build_manifest(const GgufModelFiles& files, const ManifestBuildOptions& opts) {
  BuiltManifest out;
  ModelManifest& m = out.manifest;
  ManifestBuildReport& rep = out.report;
  const GgufFile& meta = files.meta();

  const GgufValue* arch_v = meta.find("general.architecture");
  if (arch_v == nullptr || !arch_v->is_string() || arch_v->s.empty()) return bad("general.architecture is missing");
  const std::string arch = arch_v->s;
  rep.architecture = arch;
  if (!opts.require_architecture.empty() && arch != opts.require_architecture)
    return bad("architecture is '" + arch + "', expected '" + opts.require_architecture + "'");
  const auto key = [&](const char* suffix) { return arch + "." + suffix; };

  ModelGeometry& g = m.geometry;
  g.family = arch;
  CLM_ASSIGN_OR_RETURN(g.n_layers, meta_u32(meta, key("block_count"), true));
  CLM_ASSIGN_OR_RETURN(g.hidden_size, meta_u32(meta, key("embedding_length"), true));
  CLM_ASSIGN_OR_RETURN(g.n_experts, meta_u32(meta, key("expert_count"), true));
  CLM_ASSIGN_OR_RETURN(g.n_active_experts, meta_u32(meta, key("expert_used_count"), true));
  CLM_ASSIGN_OR_RETURN(g.n_heads, meta_u32(meta, key("attention.head_count"), true));
  CLM_ASSIGN_OR_RETURN(g.n_kv_heads, meta_u32(meta, key("attention.head_count_kv"), true));
  CLM_ASSIGN_OR_RETURN(g.ple_ngram, meta_u32(meta, key("ple.ngram_size"), true));
  if (g.n_layers == 0 || g.n_layers > (1u << 16)) return bad("block_count out of range");

  // ---- classify every tensor -------------------------------------------------------------------------------
  std::vector<LayerTensors> layers(g.n_layers);
  std::vector<Ref> embedding, ple, head, mtp;
  std::map<std::uint32_t, int> mtp_groups;  // distinct MTP layer ids (mtp.layers.N / blk.N>=n_layers)
  bool mtp_ungrouped = false;
  for (std::uint32_t s = 0; s < files.shards.size(); ++s) {
    for (const GgufTensorInfo& t : files.shards[s].tensors) {
      ++rep.tensors_total;
      rep.tensor_bytes_total += t.n_bytes;
      const Ref ref{s, &t};
      const TensorClassification c = classify_tensor(t.name, g.n_layers);
      switch (c.cls) {
        case TensorClass::kEmbedding: embedding.push_back(ref); break;
        case TensorClass::kPleLookup: ple.push_back(ref); break;
        case TensorClass::kOutputHead: head.push_back(ref); break;
        case TensorClass::kMtp:
          mtp.push_back(ref);
          if (c.layer) {
            mtp_groups[*c.layer] = 1;
          } else if (starts_with(t.name, "mtp.layers.")) {
            std::uint64_t n = 0;
            std::size_t i = 11;
            while (i < t.name.size() && t.name[i] >= '0' && t.name[i] <= '9' && i < 20) n = n * 10 + static_cast<std::uint64_t>(t.name[i++] - '0');
            mtp_groups[static_cast<std::uint32_t>(n) | 0x80000000u] = 1;
          } else {
            mtp_ungrouped = true;
          }
          break;
        case TensorClass::kLayerDense: layers[*c.layer].dense.push_back(ref); break;
        case TensorClass::kSharedExpert: layers[*c.layer].shared.push_back(ref); break;
        case TensorClass::kExpertGate: layers[*c.layer].gate = ref; break;
        case TensorClass::kExpertUp: layers[*c.layer].up = ref; break;
        case TensorClass::kExpertDown: layers[*c.layer].down = ref; break;
        case TensorClass::kExpertGateUp: layers[*c.layer].gate_up = ref; break;
        case TensorClass::kUnclassified:
          rep.unclassified.push_back({t.name, s, t.type, t.n_bytes, c.reason});
          break;
      }
    }
  }
  if (!rep.unclassified.empty() && !opts.allow_unclassified) {
    std::string names;
    for (std::size_t i = 0; i < rep.unclassified.size() && i < 8; ++i) names += (i ? ", " : "") + rep.unclassified[i].name;
    return bad(std::to_string(rep.unclassified.size()) + " tensor(s) cannot be classified (" + names +
               (rep.unclassified.size() > 8 ? ", ..." : "") + "); fix the mapping or set allow_unclassified");
  }
  if (embedding.size() != 1) return bad("expected exactly one token_embd.weight, found " + std::to_string(embedding.size()));
  if (ple.size() != 1) return bad("expected exactly one per_layer_token_embd.weight, found " + std::to_string(ple.size()));
  if (head.empty()) return bad("no output head tensors (output.weight, output_norm.weight, output_hc_*) found");
  if (!std::any_of(head.begin(), head.end(), [](const Ref& r) { return r.info->name == "output.weight"; })) return bad("output.weight is missing");

  // ---- geometry from tensor shapes, cross-checked against optional metadata ----------------------------------
  const GgufTensorInfo& emb = *embedding[0].info;
  if (emb.n_dims != 2 || emb.dims[0] != g.hidden_size)
    return bad("token_embd.weight " + dims_str(emb) + " must be [embedding_length, vocab]");
  if (emb.dims[1] > 0xFFFFFFFFull) return bad("vocab size does not fit in 32 bits");
  g.vocab_size = static_cast<std::uint32_t>(emb.dims[1]);
  const GgufTensorInfo& pt = *ple[0].info;
  if (pt.n_dims != 2 || pt.dims[1] > 0xFFFFFFFFull) return bad("per_layer_token_embd.weight " + dims_str(pt) + " must be [row_width, rows]");
  g.ple_rows = static_cast<std::uint32_t>(pt.dims[1]);

  g.layer_kinds.assign(g.n_layers, LayerKind::kRecurrent);
  std::optional<std::uint32_t> ple_layer, head_dim_from_tensor;
  std::optional<std::uint32_t> hc, ff, sff;
  const auto find_dense = [&](const LayerTensors& lt, std::string_view suffix, std::uint32_t L) -> const GgufTensorInfo* {
    const std::string full = "blk." + std::to_string(L) + "." + std::string(suffix);
    for (const Ref& r : lt.dense)
      if (r.info->name == full) return r.info;
    return nullptr;
  };
  for (std::uint32_t L = 0; L < g.n_layers; ++L) {
    LayerTensors& lt = layers[L];
    const std::string ctx = "layer " + std::to_string(L) + ": ";
    if (lt.dense.empty()) return bad(ctx + "no dense tensors found");
    const bool qsa = find_dense(lt, "attn_q.weight", L) != nullptr;
    const bool gdn = find_dense(lt, "attn_qkv.weight", L) != nullptr;
    if (qsa == gdn) return bad(ctx + (qsa ? "has both attn_q.weight and attn_qkv.weight" : "has neither attn_q.weight (full attention) nor attn_qkv.weight (linear attention)"));
    g.layer_kinds[L] = qsa ? LayerKind::kFullAttention : LayerKind::kRecurrent;
    if (qsa && !head_dim_from_tensor) {
      if (const GgufTensorInfo* qn = find_dense(lt, "attn_q_norm.weight", L); qn != nullptr && qn->n_dims == 1) head_dim_from_tensor = static_cast<std::uint32_t>(qn->dims[0]);
    }
    if (const GgufTensorInfo* inj = find_dense(lt, "hc_attn_inject.weight", L)) {
      if (inj->n_dims != 2 || inj->dims[1] > 64 || inj->dims[0] % g.hidden_size != 0)
        return bad(ctx + "hc_attn_inject.weight " + dims_str(*inj) + " must be [k*embedding_length, residual_streams]");
      const std::uint32_t v = static_cast<std::uint32_t>(inj->dims[1]);
      if (hc && *hc != v) return bad(ctx + "residual stream count differs from earlier layers");
      hc = v;
    }
    if (find_dense(lt, "ple_key.weight", L) != nullptr) {
      if (ple_layer) return bad("PLE tensors (blk.N.ple_key.weight) exist in more than one layer; geometry carries one ple_layer");
      ple_layer = L;
    }

    // Experts: (gate, up, down) or (gate_up, down).
    const bool fused = lt.gate_up.has_value();
    if (!lt.down || (fused ? (lt.gate || lt.up) : !(lt.gate && lt.up)))
      return bad(ctx + "routed expert tensors must be {ffn_gate,ffn_up,ffn_down}_exps or {ffn_gate_up,ffn_down}_exps");
    const GgufTensorInfo& a = fused ? *lt.gate_up->info : *lt.gate->info;
    const GgufTensorInfo& d = *lt.down->info;
    for (const GgufTensorInfo* t : {&a, &d})
      if (t->n_dims != 3 || t->dims[2] != g.n_experts)
        return bad(ctx + "stacked expert tensor '" + t->name + "' " + dims_str(*t) + " must have 3 dims with n_experts = " + std::to_string(g.n_experts) + " last");
    if (!fused && (lt.up->info->dims != a.dims)) return bad(ctx + "ffn_up_exps and ffn_gate_exps shapes differ");
    if (a.dims[0] != g.hidden_size || d.dims[1] != g.hidden_size) return bad(ctx + "expert tensors do not match embedding_length");
    const std::uint64_t layer_ff = fused ? a.dims[1] / 2 : a.dims[1];
    if (fused && a.dims[1] % 2 != 0) return bad(ctx + "ffn_gate_up_exps has an odd output width");
    if (d.dims[0] != layer_ff) return bad(ctx + "ffn_down_exps input width differs from the gate/up output width");
    if (ff && *ff != layer_ff) return bad(ctx + "expert_ff differs from earlier layers");
    ff = static_cast<std::uint32_t>(layer_ff);

    // Shared expert: all of gate/up/down or none.
    std::uint64_t layer_sff = 0;
    if (!lt.shared.empty()) {
      const GgufTensorInfo *sg = nullptr, *su = nullptr, *sd = nullptr;
      for (const Ref& r : lt.shared) {
        const std::string_view n = r.info->name;
        if (ends_with(n, "ffn_gate_shexp.weight")) sg = r.info;
        if (ends_with(n, "ffn_up_shexp.weight")) su = r.info;
        if (ends_with(n, "ffn_down_shexp.weight")) sd = r.info;
      }
      if (!sg || !su || !sd || sg->n_dims != 2 || sg->dims[0] != g.hidden_size) return bad(ctx + "incomplete or malformed shared expert tensors");
      layer_sff = sg->dims[1];
    }
    if (sff && *sff != layer_sff) return bad(ctx + "shared expert width differs from earlier layers");
    sff = static_cast<std::uint32_t>(layer_sff);
  }
  if (!hc) return bad("residual stream count cannot be determined: no blk.N.hc_attn_inject.weight tensor");
  {
    // The PLE block's layer is where blk.N.ple_key.weight lives; <arch>.ple.layer (a ClusterLM extension used by
    // generated fixtures that carry no PLE block tensors) is the fallback and must agree when both exist.
    CLM_ASSIGN_OR_RETURN(std::uint32_t meta_ple_layer, meta_u32(meta, key("ple.layer"), false, 0xFFFFFFFFu));
    if (ple_layer && meta_ple_layer != 0xFFFFFFFFu && meta_ple_layer != *ple_layer)
      return bad(arch + ".ple.layer disagrees with the layer holding ple_key.weight");
    if (!ple_layer && meta_ple_layer != 0xFFFFFFFFu) ple_layer = meta_ple_layer;
  }
  if (!ple_layer) return bad("PLE block tensors (blk.N.ple_key.weight) not found and " + key("ple.layer") + " is not set; ple_layer cannot be determined");
  g.residual_streams = *hc;
  g.expert_ff = *ff;
  g.shared_expert_ff = *sff;
  g.ple_layer = *ple_layer;
  if (g.n_experts != 0 && g.n_active_experts > g.n_experts) return bad("expert_used_count exceeds expert_count");
  CLM_ASSIGN_OR_RETURN(std::uint32_t key_len, meta_u32(meta, key("attention.key_length"), false));
  if (key_len != 0 && head_dim_from_tensor && key_len != *head_dim_from_tensor)
    return bad(arch + ".attention.key_length disagrees with attn_q_norm.weight");
  g.head_dim = key_len != 0 ? key_len : head_dim_from_tensor.value_or(0);
  if (g.head_dim == 0) return bad("head_dim cannot be determined: set " + key("attention.key_length") + " or provide attn_q_norm.weight on a full-attention layer");
  CLM_ASSIGN_OR_RETURN(std::uint32_t meta_ff, meta_u32(meta, key("expert_feed_forward_length"), false));
  if (meta_ff != 0 && meta_ff != g.expert_ff) return bad(arch + ".expert_feed_forward_length disagrees with the expert tensor shapes");
  g.mtp_layers = mtp.empty() ? 0 : std::max<std::uint32_t>(1, static_cast<std::uint32_t>(mtp_groups.size() + (mtp_ungrouped && mtp_groups.empty() ? 1 : 0)));
  CLM_RETURN_IF_ERROR(g.validate());

  // ---- manifest metadata -----------------------------------------------------------------------------------
  if (!opts.artifact_id.empty()) {
    m.artifact_id = opts.artifact_id;
  } else if (const GgufValue* n = meta.find("general.name"); n != nullptr && n->is_string() && !n->s.empty()) {
    m.artifact_id = n->s;
  } else {
    m.artifact_id = files.paths.front().filename().string();
  }
  if (!opts.license.empty()) {
    m.license = opts.license;
  } else if (const GgufValue* l = meta.find("general.license"); l != nullptr && l->is_string()) {
    m.license = l->s;
  }
  if (m.artifact_id.size() > 4096) m.artifact_id.resize(4096);
  if (m.license.size() > 4096) m.license.resize(4096);
  for (std::size_t s = 0; s < files.shards.size(); ++s) {
    ShardInfo si;
    si.file_name = files.paths[s].filename().string();
    si.byte_size = files.shards[s].file_size;
    m.shards.push_back(std::move(si));
    out.shard_paths.push_back(files.paths[s]);
  }

  // ---- objects (deterministic order: father-only, then per layer dense, shared, experts) -----------------------
  ObjectBuilder ob{m, rep, files, opts.collect_layouts};
  CLM_RETURN_IF_ERROR(ob.add_whole(std::string(kEmbeddingObjectName), ObjectKind::kEmbedding, std::nullopt, embedding));
  CLM_RETURN_IF_ERROR(ob.add_whole(std::string(kPleObjectName), ObjectKind::kPleLookup, std::nullopt, ple));
  CLM_RETURN_IF_ERROR(ob.add_whole(std::string(kHeadObjectName), ObjectKind::kOutputHead, std::nullopt, head));
  if (!mtp.empty()) CLM_RETURN_IF_ERROR(ob.add_whole(std::string(kMtpObjectName), ObjectKind::kMtp, std::nullopt, mtp));

  for (std::uint32_t L = 0; L < g.n_layers; ++L) {
    LayerTensors& lt = layers[L];
    {
      std::vector<Ref> dense = lt.dense;
      CLM_RETURN_IF_ERROR(ob.add_whole(dense_object_name(L), ObjectKind::kLayerDense, L, std::move(dense)));
    }
    if (!lt.shared.empty()) CLM_RETURN_IF_ERROR(ob.add_whole(shared_expert_object_name(L), ObjectKind::kSharedExpert, L, lt.shared));

    std::vector<Ref> stacked;
    if (lt.gate_up) {
      stacked = {*lt.gate_up, *lt.down};
    } else {
      stacked = {*lt.gate, *lt.up, *lt.down};
    }
    for (std::uint32_t e = 0; e < g.n_experts; ++e) {
      std::vector<ObjectBuilder::Part> parts;
      for (const Ref& r : stacked) {
        const GgufTensorInfo& t = *r.info;
        const GgmlTypeInfo& ti = ggml_type_info(t.type);
        // Slice = row_bytes * ne1 bytes; it must start and end on a block boundary of the stacked tensor.
        auto row = ggml_row_bytes(t.type, t.dims[0]);
        if (!row.is_ok()) return row.status();
        const std::uint64_t slice = *row * t.dims[1];
        if (t.dims[0] % ti.block_elems != 0 || slice % ti.block_bytes != 0 || slice * t.dims[2] != t.n_bytes)
          return bad("expert slice of '" + t.name + "' does not fall on block boundaries");
        parts.push_back({r, t.file_offset + slice * e, slice, {t.dims[0], t.dims[1]}});
      }
      CLM_RETURN_IF_ERROR(ob.add(expert_object_name(L, e), ObjectKind::kRoutedExpert, L, e, parts));
    }
  }
  CLM_RETURN_IF_ERROR(m.validate());

  if (opts.hash_objects || opts.hash_shards) {
    HashOptions h;
    h.hash_objects = opts.hash_objects;
    h.hash_shards = opts.hash_shards;
    h.threads = opts.hash_threads;
    h.buffer_bytes = opts.hash_buffer_bytes;
    h.progress = opts.progress;
    CLM_RETURN_IF_ERROR(compute_manifest_digests(m, out.shard_paths, h));
    rep.hashed_objects = opts.hash_objects;
    rep.hashed_shards = opts.hash_shards;
  }
  return out;
}

}  // namespace clusterlm::objects
