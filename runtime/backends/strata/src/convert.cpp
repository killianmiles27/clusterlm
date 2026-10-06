// Strata pack + model GGUF -> ClusterLM model directory. See clusterlm/backends/strata/convert.hpp.
#include "clusterlm/backends/strata/convert.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "clusterlm/backends/strata/object_map.hpp"
#include "clusterlm/common/digest.hpp"
#include "clusterlm/objects/canonical_store.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"

namespace clusterlm::backends::strata {
namespace {

namespace fs = std::filesystem;
using objects::ManifestObject;
using objects::ObjectKind;

Status invalid(std::string m) { return make_error(ErrorCode::kInvalidArgument, "strata convert: " + std::move(m)); }

struct ShardWriter {
  std::ofstream f;
  std::uint64_t at = 0;
  Status put(ByteSpan b, objects::SourceRange& range) {
    static const char zeros[64] = {};
    const std::uint64_t pad = (64 - at % 64) % 64;
    f.write(zeros, static_cast<std::streamsize>(pad));
    at += pad;
    range = {0, at, b.size()};
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    at += b.size();
    return f ? Status::ok() : make_error(ErrorCode::kDataLoss, "strata convert: write failed");
  }
};

std::uint32_t meta_u32(const ::strata::GgufModel& model, const std::string& key, std::uint32_t def) {
  const ::strata::MetaValue* v = model.meta().get(key);
  return v != nullptr && v->is_num() && v->u > 0 ? static_cast<std::uint32_t>(v->u) : def;
}

}  // namespace

Result<objects::ModelManifest> convert_model(const ConvertOptions& options,
                                             const std::function<void(std::string_view)>& progress) {
  CLM_ASSIGN_OR_RETURN(StrataPackReader pack, StrataPackReader::open(options.pack_dir));
  try {
    const std::vector<std::string> paths = ::strata::gguf_split_paths(options.gguf.string());
    const ::strata::GgufModel model(paths);
    const fs::path out = options.out.empty() ? options.gguf.parent_path() : options.out;
    for (const std::string& p : paths)
      if (fs::weakly_canonical(fs::path(p).parent_path()) != fs::weakly_canonical(out))
        return invalid("the GGUF shards must be in the output directory (" + out.string() +
                       "): the manifest names them relative to it");
    const ::strata::TensorInfo* exps = model.find("blk.0.ffn_gate_exps.weight");
    const ::strata::TensorInfo* head = model.find("output.weight");
    if (exps == nullptr || head == nullptr || exps->shape.size() != 3 || head->shape.size() != 2)
      return invalid("the GGUF has no stacked routed experts or no output.weight");

    objects::ModelManifest m;
    m.artifact_id = "strata-converted:" + fs::path(options.gguf).filename().string();
    m.license = "see the model's license";
    objects::ModelGeometry& g = m.geometry;
    const ::strata::MetaValue* arch_v = model.meta().get("general.architecture");
    const std::string arch = arch_v != nullptr ? arch_v->s : "";
    g.family = arch.empty() ? "strata" : arch;
    while (model.find("blk." + std::to_string(g.n_layers) + ".ffn_gate_exps.weight") != nullptr) ++g.n_layers;
    g.hidden_size = static_cast<std::uint32_t>(exps->shape[0]);
    g.expert_ff = static_cast<std::uint32_t>(exps->shape[1]);
    g.n_experts = static_cast<std::uint32_t>(exps->shape[2]);
    g.residual_streams = 4;  // Strata's hyper-connection streams (layout.hpp hc)
    g.n_active_experts = std::min(meta_u32(model, arch + ".expert_used_count", 10), g.n_experts);
    const ::strata::TensorInfo* shg = model.find("blk.0.ffn_gate_shexp.weight");
    g.shared_expert_ff = shg != nullptr && shg->shape.size() == 2 ? static_cast<std::uint32_t>(shg->shape[1]) : 0;
    g.n_heads = meta_u32(model, arch + ".attention.head_count", 24);
    g.n_kv_heads = meta_u32(model, arch + ".attention.head_count_kv", 2);
    g.head_dim = meta_u32(model, arch + ".attention.key_length", 256);
    g.vocab_size = static_cast<std::uint32_t>(head->shape[1]);
    g.ple_layer = 1;  // Strata's PLE block is a layer-1 module (blk.1.ple_*)
    g.ple_ngram = 3;
    g.ple_rows = 1;   // the n-gram table stays a Father file (ADR 0200), not an object
    g.mtp_layers = 1;
    for (std::uint32_t l = 0; l < g.n_layers; ++l)
      g.layer_kinds.push_back(l % 4 == 3 ? objects::LayerKind::kFullAttention : objects::LayerKind::kRecurrent);

    // shard 0: the converted objects; shards 1..: the GGUF files (routed experts in place)
    ShardWriter w;
    const fs::path dense_path = out / "strata-dense.bin";
    w.f.open(dense_path, std::ios::binary | std::ios::trunc);
    if (!w.f) return make_error(ErrorCode::kPermissionDenied, "strata convert: cannot write " + dense_path.string());
    m.shards.push_back({"strata-dense.bin", 0, {}});
    for (const std::string& p : paths) m.shards.push_back({fs::path(p).filename().string(), fs::file_size(p), {}});

    const NativeTensorProvider native = [&](std::string_view name) -> Result<std::optional<NativeTensor>> {
      std::size_t at = 0;
      const ::strata::TensorInfo* t = model.find(std::string(name), &at);
      if (t == nullptr || t->shape.size() != 2) return std::optional<NativeTensor>{};
      if (!model.in_bounds(*t, at)) return make_error(ErrorCode::kDataLoss, "strata convert: truncated " + std::string(name));
      NativeTensor nt;
      nt.type = static_cast<std::int32_t>(t->type);
      nt.ne0 = t->shape[0];
      nt.ne1 = t->shape[1];
      const std::uint8_t* p = model.shard(at).tensor_data(*t);
      nt.bytes.assign(p, p + ::strata::tensor_payload_bytes(*t));
      return std::optional<NativeTensor>(std::move(nt));
    };
    auto dense = [&](const std::string& name, ObjectKind kind, std::optional<std::uint32_t> layer) -> Status {
      ManifestObject o;
      o.name = name;
      o.kind = kind;
      o.layer = layer;
      o.representation = {std::string(kDenseQuantType), 0, kDenseConversionVersion, true};
      CLM_ASSIGN_OR_RETURN(Bytes b, convert_object(o, pack, native));
      objects::SourceRange r;
      CLM_RETURN_IF_ERROR(w.put(b, r));
      o.source_ranges.push_back(r);
      o.byte_size = b.size();
      o.object_digest = o.source_digest = Sha256::of(b);  // the shard holds the converted bytes themselves
      m.objects.push_back(std::move(o));
      return Status::ok();
    };

    CLM_RETURN_IF_ERROR(dense(std::string(objects::kEmbeddingObjectName), ObjectKind::kEmbedding, {}));
    for (std::uint32_t L = 0; L < g.n_layers; ++L) {
      CLM_RETURN_IF_ERROR(dense(objects::dense_object_name(L), ObjectKind::kLayerDense, L));
      if (g.shared_expert_ff > 0) CLM_RETURN_IF_ERROR(dense(objects::shared_expert_object_name(L), ObjectKind::kSharedExpert, L));
      static constexpr const char* kRoles[3] = {"gate", "up", "down"};
      const ::strata::TensorInfo* t[3];
      std::size_t sh[3] = {0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        t[i] = model.find("blk." + std::to_string(L) + ".ffn_" + kRoles[i] + "_exps.weight", &sh[i]);
        if (t[i] == nullptr || t[i]->shape.size() != 3 || !model.in_bounds(*t[i], sh[i]))
          return invalid("layer " + std::to_string(L) + ": " + kRoles[i] + " experts missing or truncated");
      }
      const GgmlType* gu = ggml_type_by_id(static_cast<int>(t[0]->type));
      const GgmlType* dn = ggml_type_by_id(static_cast<int>(t[2]->type));
      if (gu == nullptr || dn == nullptr || t[1]->type != t[0]->type)
        return invalid("layer " + std::to_string(L) + ": expert types are not supported");
      const objects::Representation rep{expert_quant_type(*gu, *dn), gu->block_elems, kExpertConversionVersion, true};
      CLM_ASSIGN_OR_RETURN(ExpertFormat f, expert_format(rep, g));
      const std::uint64_t per[3] = {f.gate_bytes, f.up_bytes, f.down_bytes};
      for (int i = 0; i < 3; ++i)
        if (::strata::tensor_payload_bytes(*t[i]) != per[i] * g.n_experts)
          return invalid("layer " + std::to_string(L) + ": " + kRoles[i] + " experts are not n_experts slices");
      for (std::uint32_t e = 0; e < g.n_experts; ++e) {
        ManifestObject o;
        o.name = objects::expert_object_name(L, e);
        o.kind = ObjectKind::kRoutedExpert;
        o.layer = L;
        o.expert = e;
        o.representation = rep;
        Sha256 d;
        for (int i = 0; i < 3; ++i) {
          const ::strata::GgufFile& file = model.shard(sh[i]);
          o.source_ranges.push_back(
              {static_cast<std::uint32_t>(1 + sh[i]), file.data_start() + t[i]->offset + per[i] * e, per[i]});
          d.update(ByteSpan(file.tensor_data(*t[i]) + per[i] * e, static_cast<std::size_t>(per[i])));
        }
        o.byte_size = f.total();
        o.object_digest = o.source_digest = d.finish();
        m.objects.push_back(std::move(o));
      }
      if (progress) progress("layer " + std::to_string(L) + " (" + rep.quant_type + " experts)");
    }
    CLM_RETURN_IF_ERROR(dense(std::string(objects::kHeadObjectName), ObjectKind::kOutputHead, {}));
    w.f.close();
    m.shards[0].byte_size = w.at;
    {
      std::ifstream in(dense_path, std::ios::binary);
      Sha256 d;
      std::vector<char> buf(1 << 20);
      while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        d.update(ByteSpan(reinterpret_cast<const std::uint8_t*>(buf.data()), static_cast<std::size_t>(in.gcount())));
      }
      m.shards[0].digest = d.finish();
    }
    CLM_RETURN_IF_ERROR(m.validate());
    std::ofstream(out / objects::CanonicalModelStore::kManifestFileName) << m.to_json();
    return m;
  } catch (const std::exception& e) {
    return make_error(ErrorCode::kDataLoss, std::string("strata convert: GGUF: ") + e.what());
  }
}

}  // namespace clusterlm::backends::strata
