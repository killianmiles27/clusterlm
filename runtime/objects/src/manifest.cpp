#include "clusterlm/objects/manifest.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace clusterlm::objects {

namespace {

constexpr std::size_t kMaxName = 256;
constexpr std::uint32_t kMaxShards = 4096;
constexpr std::uint32_t kMaxObjects = 1u << 22;
constexpr std::uint32_t kMaxRanges = 1u << 16;
constexpr std::uint32_t kMaxDeps = 1u << 12;
constexpr std::size_t kMaxText = 4096;

bool is_layer_kind(ObjectKind k) {
  return k == ObjectKind::kLayerDense || k == ObjectKind::kRoutedExpert || k == ObjectKind::kSharedExpert;
}

Status invalid(const std::string& m) { return make_error(ErrorCode::kInvalidArgument, "manifest: " + m); }

void put_digest(ByteWriter& w, const Digest256& d) { w.raw(d.bytes); }

bool get_digest(ByteReader& r, Digest256& d) {
  ByteSpan s;
  if (!r.raw(32, s)) return false;
  std::copy(s.begin(), s.end(), d.bytes.begin());
  return true;
}

void encode_object(ByteWriter& w, const ManifestObject& o) {
  w.str(o.name);
  w.u8(static_cast<std::uint8_t>(o.kind));
  w.boolean(o.layer.has_value());
  w.u32(o.layer.value_or(0));
  w.boolean(o.expert.has_value());
  w.u32(o.expert.value_or(0));
  w.str(o.representation.quant_type);
  w.u32(o.representation.block_size);
  w.u32(o.representation.conversion_version);
  w.boolean(o.representation.little_endian);
  w.u32(static_cast<std::uint32_t>(o.source_ranges.size()));
  for (const SourceRange& s : o.source_ranges) {
    w.u32(s.shard);
    w.u64(s.offset);
    w.u64(s.length);
  }
  w.u64(o.byte_size);
  w.u32(o.alignment);
  w.u64(o.peak_workspace);
  put_digest(w, o.source_digest);
  put_digest(w, o.object_digest);
  w.u32(static_cast<std::uint32_t>(o.dependencies.size()));
  for (const std::string& d : o.dependencies) w.str(d);
}

Result<ManifestObject> decode_object(ByteReader& r) {
  ManifestObject o;
  std::uint8_t kind = 0;
  bool has_layer = false, has_expert = false;
  std::uint32_t layer = 0, expert = 0, n_ranges = 0, n_deps = 0;
  r.str(o.name, kMaxName);
  r.u8(kind);
  r.boolean(has_layer);
  r.u32(layer);
  r.boolean(has_expert);
  r.u32(expert);
  r.str(o.representation.quant_type, kMaxName);
  r.u32(o.representation.block_size);
  r.u32(o.representation.conversion_version);
  r.boolean(o.representation.little_endian);
  r.u32(n_ranges);
  if (!r.ok() || kind > static_cast<std::uint8_t>(ObjectKind::kMtp) || n_ranges > kMaxRanges)
    return make_error(ErrorCode::kProtocolError, "manifest: malformed object header");
  // Each range is 20 bytes on the wire; bound the allocation by what the input can actually hold.
  if (r.remaining() < std::uint64_t{n_ranges} * 20) return make_error(ErrorCode::kProtocolError, "manifest: truncated");
  o.kind = static_cast<ObjectKind>(kind);
  if (has_layer) o.layer = layer;
  if (has_expert) o.expert = expert;
  o.source_ranges.resize(n_ranges);
  for (SourceRange& s : o.source_ranges) {
    r.u32(s.shard);
    r.u64(s.offset);
    r.u64(s.length);
  }
  r.u64(o.byte_size);
  r.u32(o.alignment);
  r.u64(o.peak_workspace);
  get_digest(r, o.source_digest);
  get_digest(r, o.object_digest);
  r.u32(n_deps);
  if (!r.ok() || n_deps > kMaxDeps) return make_error(ErrorCode::kProtocolError, "manifest: malformed object");
  for (std::uint32_t i = 0; i < n_deps && r.ok(); ++i) {
    std::string d;
    r.str(d, kMaxName);
    o.dependencies.push_back(std::move(d));
  }
  if (!r.ok()) return make_error(ErrorCode::kProtocolError, "manifest: truncated or malformed object");
  return o;
}

// ---- JSON ----
using json = nlohmann::json;

json range_to_json(const SourceRange& s) {
  return json{{"shard", s.shard}, {"offset", s.offset}, {"length", s.length}};
}

json object_to_json(const ManifestObject& o) {
  json j;
  j["name"] = o.name;
  j["kind"] = std::string(to_string(o.kind));
  if (o.layer) j["layer"] = *o.layer;
  if (o.expert) j["expert"] = *o.expert;
  j["representation"] = {{"quant_type", o.representation.quant_type},
                         {"block_size", o.representation.block_size},
                         {"conversion_version", o.representation.conversion_version},
                         {"little_endian", o.representation.little_endian}};
  j["source_ranges"] = json::array();
  for (const SourceRange& s : o.source_ranges) j["source_ranges"].push_back(range_to_json(s));
  j["byte_size"] = o.byte_size;
  j["alignment"] = o.alignment;
  j["peak_workspace"] = o.peak_workspace;
  j["source_digest"] = o.source_digest.hex();
  j["object_digest"] = o.object_digest.hex();
  j["dependencies"] = o.dependencies;
  return j;
}

Result<ObjectKind> kind_from_string(const std::string& s) {
  for (std::uint8_t i = 0; i <= static_cast<std::uint8_t>(ObjectKind::kMtp); ++i)
    if (to_string(static_cast<ObjectKind>(i)) == s) return static_cast<ObjectKind>(i);
  return invalid("unknown object kind '" + s + "'");
}

Result<ManifestObject> object_from_json(const json& j) {
  ManifestObject o;
  o.name = j.at("name").get<std::string>();
  CLM_ASSIGN_OR_RETURN(o.kind, kind_from_string(j.at("kind").get<std::string>()));
  if (j.contains("layer")) o.layer = j.at("layer").get<std::uint32_t>();
  if (j.contains("expert")) o.expert = j.at("expert").get<std::uint32_t>();
  const json& rep = j.at("representation");
  o.representation.quant_type = rep.at("quant_type").get<std::string>();
  o.representation.block_size = rep.at("block_size").get<std::uint32_t>();
  o.representation.conversion_version = rep.at("conversion_version").get<std::uint32_t>();
  o.representation.little_endian = rep.at("little_endian").get<bool>();
  for (const json& s : j.at("source_ranges"))
    o.source_ranges.push_back({s.at("shard").get<std::uint32_t>(), s.at("offset").get<std::uint64_t>(),
                               s.at("length").get<std::uint64_t>()});
  o.byte_size = j.at("byte_size").get<std::uint64_t>();
  o.alignment = j.at("alignment").get<std::uint32_t>();
  o.peak_workspace = j.at("peak_workspace").get<std::uint64_t>();
  CLM_ASSIGN_OR_RETURN(o.source_digest, Digest256::from_hex(j.at("source_digest").get<std::string>()));
  CLM_ASSIGN_OR_RETURN(o.object_digest, Digest256::from_hex(j.at("object_digest").get<std::string>()));
  o.dependencies = j.at("dependencies").get<std::vector<std::string>>();
  return o;
}

json geometry_to_json(const ModelGeometry& g) {
  json kinds = json::array();
  for (LayerKind k : g.layer_kinds) kinds.push_back(k == LayerKind::kRecurrent ? "R" : "A");
  return json{{"family", g.family},
              {"n_layers", g.n_layers},
              {"hidden_size", g.hidden_size},
              {"residual_streams", g.residual_streams},
              {"n_experts", g.n_experts},
              {"n_active_experts", g.n_active_experts},
              {"expert_ff", g.expert_ff},
              {"shared_expert_ff", g.shared_expert_ff},
              {"n_heads", g.n_heads},
              {"n_kv_heads", g.n_kv_heads},
              {"head_dim", g.head_dim},
              {"vocab_size", g.vocab_size},
              {"ple_layer", g.ple_layer},
              {"ple_ngram", g.ple_ngram},
              {"ple_rows", g.ple_rows},
              {"mtp_layers", g.mtp_layers},
              {"layer_kinds", kinds}};
}

Result<ModelGeometry> geometry_from_json(const json& j) {
  ModelGeometry g;
  g.family = j.at("family").get<std::string>();
  g.n_layers = j.at("n_layers").get<std::uint32_t>();
  g.hidden_size = j.at("hidden_size").get<std::uint32_t>();
  g.residual_streams = j.at("residual_streams").get<std::uint32_t>();
  g.n_experts = j.at("n_experts").get<std::uint32_t>();
  g.n_active_experts = j.at("n_active_experts").get<std::uint32_t>();
  g.expert_ff = j.at("expert_ff").get<std::uint32_t>();
  g.shared_expert_ff = j.at("shared_expert_ff").get<std::uint32_t>();
  g.n_heads = j.at("n_heads").get<std::uint32_t>();
  g.n_kv_heads = j.at("n_kv_heads").get<std::uint32_t>();
  g.head_dim = j.at("head_dim").get<std::uint32_t>();
  g.vocab_size = j.at("vocab_size").get<std::uint32_t>();
  g.ple_layer = j.at("ple_layer").get<std::uint32_t>();
  g.ple_ngram = j.at("ple_ngram").get<std::uint32_t>();
  g.ple_rows = j.at("ple_rows").get<std::uint32_t>();
  g.mtp_layers = j.at("mtp_layers").get<std::uint32_t>();
  for (const json& k : j.at("layer_kinds")) {
    const std::string s = k.get<std::string>();
    if (s != "R" && s != "A") return invalid("bad layer kind '" + s + "'");
    g.layer_kinds.push_back(s == "R" ? LayerKind::kRecurrent : LayerKind::kFullAttention);
  }
  return g;
}

}  // namespace

std::string dense_object_name(std::uint32_t layer) { return "blk." + std::to_string(layer) + ".dense"; }
std::string expert_object_name(std::uint32_t layer, std::uint32_t expert) {
  return "blk." + std::to_string(layer) + ".exp." + std::to_string(expert);
}
std::string shared_expert_object_name(std::uint32_t layer) { return "blk." + std::to_string(layer) + ".shared"; }

std::string_view to_string(ObjectKind kind) {
  switch (kind) {
    case ObjectKind::kEmbedding: return "embedding";
    case ObjectKind::kPleLookup: return "ple_lookup";
    case ObjectKind::kLayerDense: return "layer_dense";
    case ObjectKind::kRoutedExpert: return "routed_expert";
    case ObjectKind::kSharedExpert: return "shared_expert";
    case ObjectKind::kOutputHead: return "output_head";
    case ObjectKind::kMtp: return "mtp";
  }
  return "unknown";
}

std::string_view to_string(AllocationTarget target) {
  switch (target) {
    case AllocationTarget::kGpuResident: return "gpu_resident";
    case AllocationTarget::kCpuResident: return "cpu_resident";
    case AllocationTarget::kTemporaryBacking: return "temporary_backing";
    case AllocationTarget::kFatherOnly: return "father_only";
  }
  return "unknown";
}

Digest256 ModelManifest::root_hash() const {
  ByteWriter w;
  encode(w);
  Sha256 h;
  h.update(std::string_view("clusterlm.manifest.root.v1"));
  h.update(w.bytes());
  return h.finish();
}

const ManifestObject* ModelManifest::find(std::string_view name) const {
  for (const ManifestObject& o : objects)
    if (o.name == name) return &o;
  return nullptr;
}

std::vector<const ManifestObject*> ModelManifest::layer_objects(LayerRange range) const {
  std::vector<const ManifestObject*> out;
  for (const ManifestObject& o : objects)
    if (o.layer && !o.father_only() && range.contains(*o.layer)) out.push_back(&o);
  return out;
}

std::uint64_t ModelManifest::total_bytes(LayerRange range) const {
  std::uint64_t n = 0;
  for (const ManifestObject* o : layer_objects(range)) n += o->byte_size;
  return n;
}

Status ModelManifest::validate() const {
  CLM_RETURN_IF_ERROR(geometry.validate());
  if (shards.empty() || shards.size() > kMaxShards) return invalid("shard count out of range");
  for (const ShardInfo& s : shards) {
    const std::string& f = s.file_name;
    if (f.empty() || f.size() > kMaxName || f.front() == '/' || f.front() == '\\' || f.find("..") != std::string::npos ||
        f.find(':') != std::string::npos)
      return invalid("shard file name must be a plain relative name: '" + f + "'");
  }
  std::unordered_set<std::string> names;
  for (const ManifestObject& o : objects) names.insert(o.name);
  if (names.size() != objects.size()) return invalid("duplicate or empty object names");

  for (const ManifestObject& o : objects) {
    const std::string ctx = "object '" + o.name + "': ";
    if (o.name.empty() || o.name.size() > kMaxName) return invalid(ctx + "bad name");
    if (is_layer_kind(o.kind)) {
      if (!o.layer || *o.layer >= geometry.n_layers) return invalid(ctx + "layer missing or out of range");
    }
    if (o.kind == ObjectKind::kRoutedExpert && (!o.expert || *o.expert >= geometry.n_experts))
      return invalid(ctx + "expert missing or out of range");
    if (o.source_ranges.empty()) return invalid(ctx + "no source ranges");
    if (o.alignment == 0 || (o.alignment & (o.alignment - 1)) != 0) return invalid(ctx + "alignment not a power of two");
    std::uint64_t total = 0;
    for (const SourceRange& s : o.source_ranges) {
      if (s.shard >= shards.size()) return invalid(ctx + "range names unknown shard");
      if (s.length == 0) return invalid(ctx + "empty range");
      if (s.offset > shards[s.shard].byte_size || s.length > shards[s.shard].byte_size - s.offset)
        return invalid(ctx + "range outside shard");
      if (s.length > std::numeric_limits<std::uint64_t>::max() - total) return invalid(ctx + "range lengths overflow");
      total += s.length;
    }
    // Ranges of one object may not alias each other: that would make the object larger than its source.
    // Sorted sweep (O(n log n)): a pairwise scan is quadratic in a wire-controlled range count.
    if (o.source_ranges.size() > 1) {
      std::vector<const SourceRange*> sorted;
      sorted.reserve(o.source_ranges.size());
      for (const SourceRange& s : o.source_ranges) sorted.push_back(&s);
      std::sort(sorted.begin(), sorted.end(), [](const SourceRange* a, const SourceRange* b) {
        return a->shard != b->shard ? a->shard < b->shard : a->offset < b->offset;
      });
      for (std::size_t i = 1; i < sorted.size(); ++i) {
        const SourceRange &a = *sorted[i - 1], &b = *sorted[i];
        // offset + length cannot wrap: both were checked against the shard size above.
        if (a.shard == b.shard && b.offset < a.offset + a.length) return invalid(ctx + "overlapping source ranges");
      }
    }
    if (o.representation.conversion_version == 0 && o.byte_size != total)
      return invalid(ctx + "byte_size != sum of source ranges for an unconverted object");
    if (o.byte_size == 0) return invalid(ctx + "zero byte_size");
    for (const std::string& d : o.dependencies)
      if (!names.contains(d)) return invalid(ctx + "unknown dependency '" + d + "'");
  }
  return Status::ok();
}

void ModelManifest::encode(ByteWriter& w) const {
  w.u32(kFormatVersion);
  w.str(artifact_id);
  w.str(license);
  geometry.encode(w);
  w.u32(static_cast<std::uint32_t>(shards.size()));
  for (const ShardInfo& s : shards) {
    w.str(s.file_name);
    w.u64(s.byte_size);
    put_digest(w, s.digest);
  }
  w.u32(static_cast<std::uint32_t>(objects.size()));
  for (const ManifestObject& o : objects) encode_object(w, o);
}

Result<ModelManifest> ModelManifest::decode(ByteReader& r) {
  std::uint32_t version = 0;
  r.u32(version);
  if (!r.ok()) return make_error(ErrorCode::kProtocolError, "manifest: truncated");
  if (version != kFormatVersion) return make_error(ErrorCode::kVersionMismatch, "manifest: unsupported format version");
  ModelManifest m;
  r.str(m.artifact_id, kMaxText);
  r.str(m.license, kMaxText);
  if (!r.ok()) return make_error(ErrorCode::kProtocolError, "manifest: truncated header");
  CLM_ASSIGN_OR_RETURN(m.geometry, ModelGeometry::decode(r));
  std::uint32_t n_shards = 0;
  r.u32(n_shards);
  if (!r.ok() || n_shards > kMaxShards) return make_error(ErrorCode::kProtocolError, "manifest: bad shard count");
  for (std::uint32_t i = 0; i < n_shards; ++i) {
    ShardInfo s;
    r.str(s.file_name, kMaxName);
    r.u64(s.byte_size);
    get_digest(r, s.digest);
    if (!r.ok()) return make_error(ErrorCode::kProtocolError, "manifest: truncated shard");
    m.shards.push_back(std::move(s));
  }
  std::uint32_t n_objects = 0;
  r.u32(n_objects);
  if (!r.ok() || n_objects > kMaxObjects) return make_error(ErrorCode::kProtocolError, "manifest: bad object count");
  for (std::uint32_t i = 0; i < n_objects; ++i) {
    CLM_ASSIGN_OR_RETURN(ManifestObject o, decode_object(r));
    m.objects.push_back(std::move(o));
  }
  CLM_RETURN_IF_ERROR(m.validate());
  return m;
}

std::string ModelManifest::to_json() const {
  json j;
  j["format_version"] = kFormatVersion;
  j["artifact_id"] = artifact_id;
  j["license"] = license;
  j["root_hash"] = root_hash().hex();  // informational; recomputed (and checked) on load
  j["geometry"] = geometry_to_json(geometry);
  j["shards"] = json::array();
  for (const ShardInfo& s : shards)
    j["shards"].push_back({{"file_name", s.file_name}, {"byte_size", s.byte_size}, {"digest", s.digest.hex()}});
  j["objects"] = json::array();
  for (const ManifestObject& o : objects) j["objects"].push_back(object_to_json(o));
  return j.dump(1);
}

Result<ModelManifest> ModelManifest::from_json(std::string_view text) {
  json j = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) return invalid("not a JSON object");
  try {
    if (j.at("format_version").get<std::uint32_t>() != kFormatVersion)
      return make_error(ErrorCode::kVersionMismatch, "manifest: unsupported format version");
    ModelManifest m;
    m.artifact_id = j.at("artifact_id").get<std::string>();
    m.license = j.at("license").get<std::string>();
    CLM_ASSIGN_OR_RETURN(m.geometry, geometry_from_json(j.at("geometry")));
    for (const json& s : j.at("shards")) {
      ShardInfo si;
      si.file_name = s.at("file_name").get<std::string>();
      si.byte_size = s.at("byte_size").get<std::uint64_t>();
      CLM_ASSIGN_OR_RETURN(si.digest, Digest256::from_hex(s.at("digest").get<std::string>()));
      m.shards.push_back(std::move(si));
    }
    for (const json& o : j.at("objects")) {
      CLM_ASSIGN_OR_RETURN(ManifestObject mo, object_from_json(o));
      m.objects.push_back(std::move(mo));
    }
    CLM_RETURN_IF_ERROR(m.validate());
    if (j.contains("root_hash") && j.at("root_hash").get<std::string>() != m.root_hash().hex())
      return make_error(ErrorCode::kDataLoss, "manifest: root_hash does not match content");
    return m;
  } catch (const std::exception& e) {
    return invalid(std::string("malformed JSON: ") + e.what());
  }
}

}  // namespace clusterlm::objects
