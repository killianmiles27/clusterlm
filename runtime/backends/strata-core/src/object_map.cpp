#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS  // std::sscanf reads Strata's index.txt rows exactly as Strata's loader does
#endif
#include "clusterlm/backends/strata/object_map.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace clusterlm::backends::strata {
namespace {

using objects::LayerKind;
using objects::ManifestObject;
using objects::ObjectKind;

Status invalid(std::string m) { return make_error(ErrorCode::kInvalidArgument, std::move(m)); }
Status data_loss(std::string m) { return make_error(ErrorCode::kDataLoss, std::move(m)); }

// ggml_type values and block geometry at the Strata ggml pin (ggml/include/ggml.h, ggml/src/ggml-common.h of
// llama.cpp 3cf03257). Only the types Strata's packs and the target GGUFs use.
constexpr std::array<GgmlType, 23> kTypes = {{
    {0, "f32", 1, 4},
    {1, "f16", 1, 2},
    {2, "q4_0", 32, 18},
    {3, "q4_1", 32, 20},
    {6, "q5_0", 32, 22},
    {7, "q5_1", 32, 24},
    {8, "q8_0", 32, 34},
    {10, "q2_k", 256, 84},
    {11, "q3_k", 256, 110},
    {12, "q4_k", 256, 144},
    {13, "q5_k", 256, 176},
    {14, "q6_k", 256, 210},
    {16, "iq2_xxs", 256, 66},
    {17, "iq2_xs", 256, 74},
    {18, "iq3_xxs", 256, 98},
    {19, "iq1_s", 256, 50},
    {20, "iq4_nl", 32, 18},
    {21, "iq3_s", 256, 110},
    {22, "iq2_s", 256, 82},
    {23, "iq4_xs", 256, 136},
    {29, "iq1_m", 256, 56},
    {30, "bf16", 1, 2},
    {42, "q2_0", 64, 18},
}};

constexpr std::string_view kSharedSuffixes[] = {"ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
                                                "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
constexpr std::string_view kExpertSuffixes[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
// NativeDense's eligible projections (strata src/core/native_dense.cpp `eligible`), the PLE key aside.
constexpr std::string_view kNativeSuffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
                                                ".attn_q.weight",   ".attn_k.weight",    ".attn_v.weight",
                                                ".attn_output.weight", ".ffn_gate_shexp.weight",
                                                ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};

bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

// "blk.<L>.<suffix>" -> (L, suffix)
std::optional<std::pair<std::uint32_t, std::string_view>> split_block(std::string_view name) {
  if (name.substr(0, 4) != "blk.") return std::nullopt;
  const std::size_t dot = name.find('.', 4);
  if (dot == std::string_view::npos || dot == 4) return std::nullopt;
  std::uint32_t layer = 0;
  const auto [p, ec] = std::from_chars(name.data() + 4, name.data() + dot, layer);
  if (ec != std::errc() || p != name.data() + dot) return std::nullopt;
  return std::make_pair(layer, name.substr(dot + 1));
}

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

// Plane sizes must account for the canonical bytes exactly as Strata's loader checks them (weights.cpp `segs`).
Status check_row(const DenseRow& r, std::uint64_t payload) {
  const std::string ctx = "strata-dense row '" + r.name + "': ";
  if (r.name.empty() || r.name.size() > 255 || r.name.find_first_of(" \t\r\n") != std::string::npos)
    return data_loss(ctx + "bad tensor name");
  if (r.kind < 0 || r.kind > 5) return data_loss(ctx + "unknown pack index kind");
  if (r.payload_offset > payload || r.src_bytes > payload - r.payload_offset) return data_loss(ctx + "outside the payload");
  if (r.has_native() && (r.native_offset > payload || r.native_bytes > payload - r.native_offset))
    return data_loss(ctx + "native copy outside the payload");
  if (r.has_native()) {
    const GgmlType* t = ggml_type_by_id(r.native_type);
    if (t == nullptr) return data_loss(ctx + "native copy of an unknown ggml type");
    if (r.native_ne0 == 0 || r.native_ne1 == 0 || ggml_bytes(*t, r.native_ne0) * r.native_ne1 != r.native_bytes)
      return data_loss(ctx + "native copy size disagrees with its type and shape");
  }
  if (r.src_bytes == 0) {
    if (!r.has_native()) return data_loss(ctx + "neither canonical nor native bytes");
    return Status::ok();
  }
  if (r.codes_bytes != 0) {
    const std::uint64_t scales_src = r.scales_fp16 != 0 ? r.scales_bytes / 2 : r.scales_bytes;
    if (r.codes_bytes + scales_src + r.offset_bytes != r.src_bytes)
      return data_loss(ctx + "plane sizes do not add up to the source bytes");
    if (r.codes_bytes + r.scales_bytes + r.offset_bytes != r.dst_bytes)
      return data_loss(ctx + "plane sizes do not add up to the arena bytes");
  }
  return Status::ok();
}

void put_row(ByteWriter& w, const DenseRow& r) {
  w.str(r.name);
  w.i32(r.kind);
  w.u64(r.payload_offset);
  w.u64(r.src_bytes);
  w.u64(r.dst_bytes);
  w.i64(r.ne0);
  w.i64(r.ne1);
  for (std::int32_t v : {r.code_bits, r.code_bias, r.group_elems, r.codebook, r.has_offset}) w.i32(v);
  w.u64(r.codes_bytes);
  w.u64(r.scales_bytes);
  w.u64(r.offset_bytes);
  w.i32(r.scales_fp16);
  w.i32(r.act_kind);
  w.i32(r.source_file);
  w.i32(r.native_type);
  w.u64(r.native_ne0);
  w.u64(r.native_ne1);
  w.u64(r.native_offset);
  w.u64(r.native_bytes);
}

bool get_row(ByteReader& rd, DenseRow& r) {
  return rd.str(r.name, 255) && rd.i32(r.kind) && rd.u64(r.payload_offset) && rd.u64(r.src_bytes) &&
         rd.u64(r.dst_bytes) && rd.i64(r.ne0) && rd.i64(r.ne1) && rd.i32(r.code_bits) && rd.i32(r.code_bias) &&
         rd.i32(r.group_elems) && rd.i32(r.codebook) && rd.i32(r.has_offset) && rd.u64(r.codes_bytes) &&
         rd.u64(r.scales_bytes) && rd.u64(r.offset_bytes) && rd.i32(r.scales_fp16) && rd.i32(r.act_kind) &&
         rd.i32(r.source_file) && rd.i32(r.native_type) && rd.u64(r.native_ne0) && rd.u64(r.native_ne1) &&
         rd.u64(r.native_offset) && rd.u64(r.native_bytes);
}

constexpr std::uint32_t kMaxRows = 4096;
constexpr std::uint32_t kSegmentAlign = 64;

}  // namespace

// ---------------------------------------------------------------- ggml types

const GgmlType* ggml_type_by_name(std::string_view name) {
  for (const GgmlType& t : kTypes)
    if (t.name == name) return &t;
  return nullptr;
}

const GgmlType* ggml_type_by_id(int id) {
  for (const GgmlType& t : kTypes)
    if (t.id == id) return &t;
  return nullptr;
}

std::uint64_t ggml_bytes(const GgmlType& t, std::uint64_t elems) {
  if (t.block_elems == 0 || elems % t.block_elems != 0) return 0;
  return elems / t.block_elems * t.block_bytes;
}

// ---------------------------------------------------------------- experts

std::string expert_quant_type(const GgmlType& gate_up, const GgmlType& down) {
  return gate_up.id == down.id ? std::string(gate_up.name) : std::string(gate_up.name) + "+" + std::string(down.name);
}

namespace {
// Types the pinned Strata VRAM-tier expert kernels accept (upstream src/kernels/cuda/iq_kernels.cu native_expert_grouped:
// gate/up list at :611-613, down list at :615). Any other type reaches std::exit(1) inside the engine, which would
// kill a Node mid-run, so it is refused here as a Status instead (workstream F review, ADR 0408).
constexpr std::string_view kGpuGateUp[] = {"iq2_xxs", "iq2_xs", "iq3_xxs", "iq3_s", "iq2_s", "iq4_xs", "iq1_m",
                                           "q2_0",    "q4_k",   "q5_k",    "q5_0",  "q4_0",  "q4_1",   "q8_0"};
constexpr std::string_view kGpuDown[] = {"iq4_nl", "q2_0", "q5_1", "q5_0", "q4_0", "q4_1", "q8_0"};
bool in_list(std::string_view n, const std::string_view* b, const std::string_view* e) { return std::find(b, e, n) != e; }
}  // namespace

Result<ExpertFormat> expert_format(const objects::Representation& rep, const objects::ModelGeometry& g) {
  if (rep.conversion_version != kExpertConversionVersion)
    return invalid("routed expert representation version " + std::to_string(rep.conversion_version) +
                   ": Strata consumes the GGUF slices unchanged (version 0)");
  const std::string_view q = rep.quant_type;
  const std::size_t plus = q.find('+');
  ExpertFormat f;
  f.gate_up = ggml_type_by_name(q.substr(0, plus));
  f.down = plus == std::string_view::npos ? f.gate_up : ggml_type_by_name(q.substr(plus + 1));
  if (f.gate_up == nullptr || f.down == nullptr) return invalid("routed expert quant_type '" + rep.quant_type + "' is unknown");
  const std::uint64_t H = g.hidden_size, ff = g.expert_ff;
  const std::uint64_t gu_row = ggml_bytes(*f.gate_up, H), d_row = ggml_bytes(*f.down, ff);
  if (gu_row == 0 || d_row == 0)
    return invalid("routed expert quant_type '" + rep.quant_type + "' does not tile the expert rows (H " +
                   std::to_string(H) + ", ff " + std::to_string(ff) + ")");
  if (!in_list(f.gate_up->name, std::begin(kGpuGateUp), std::end(kGpuGateUp)))
    return make_error(ErrorCode::kVersionMismatch, "routed expert gate/up type '" + std::string(f.gate_up->name) +
                                                       "' has no Strata GPU expert kernel");
  if (!in_list(f.down->name, std::begin(kGpuDown), std::end(kGpuDown)))
    return make_error(ErrorCode::kVersionMismatch, "routed expert down type '" + std::string(f.down->name) +
                                                       "' has no Strata GPU expert kernel");
  f.gate_bytes = gu_row * ff;
  f.up_bytes = gu_row * ff;
  f.down_bytes = d_row * H;
  return f;
}

// ---------------------------------------------------------------- strata-dense container

Result<DenseObject> parse_dense_object(ByteSpan bytes) {
  ByteReader rd(bytes);
  ByteSpan magic;
  std::uint32_t version = 0, n_rows = 0, align = 0, reserved = 0;
  if (!rd.raw(sizeof kDenseMagic, magic) || !std::equal(magic.begin(), magic.end(), kDenseMagic))
    return data_loss("not a strata-dense object (magic)");
  if (!rd.u32(version) || !rd.u32(n_rows) || !rd.u32(align) || !rd.u32(reserved)) return data_loss("strata-dense header truncated");
  if (version != kDenseConversionVersion)
    return make_error(ErrorCode::kVersionMismatch, "strata-dense container version " + std::to_string(version));
  if (n_rows == 0 || n_rows > kMaxRows) return data_loss("strata-dense row count out of range");
  if (align == 0 || (align & (align - 1)) != 0) return data_loss("strata-dense alignment is not a power of two");
  DenseObject out;
  out.align = align;
  out.rows.resize(n_rows);
  for (DenseRow& r : out.rows)
    if (!get_row(rd, r)) return data_loss("strata-dense row table truncated");
  std::uint64_t payload_bytes = 0;
  if (!rd.u64(payload_bytes) || payload_bytes != rd.remaining()) return data_loss("strata-dense payload size mismatch");
  if (!rd.raw(static_cast<std::size_t>(payload_bytes), out.payload)) return data_loss("strata-dense payload truncated");
  for (const DenseRow& r : out.rows) CLM_RETURN_IF_ERROR(check_row(r, payload_bytes));
  for (std::size_t i = 0; i < out.rows.size(); ++i)
    for (std::size_t k = i + 1; k < out.rows.size(); ++k)
      if (out.rows[i].name == out.rows[k].name) return data_loss("strata-dense duplicate row '" + out.rows[i].name + "'");
  return out;
}

Result<Bytes> encode_dense_object(std::uint32_t align, std::vector<DenseRow> rows, const std::vector<ByteSpan>& canonical,
                                  const std::vector<ByteSpan>& native) {
  if (rows.empty() || rows.size() > kMaxRows || canonical.size() != rows.size() || native.size() != rows.size())
    return invalid("strata-dense: rows and byte spans disagree");
  if (align == 0 || (align & (align - 1)) != 0) return invalid("strata-dense: alignment is not a power of two");
  std::uint64_t at = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    DenseRow& r = rows[i];
    r.src_bytes = canonical[i].size();
    r.payload_offset = r.src_bytes ? at : 0;
    at = align_up(at + r.src_bytes, kSegmentAlign);
    if (r.has_native()) {
      r.native_bytes = native[i].size();
      r.native_offset = at;
      at = align_up(at + r.native_bytes, kSegmentAlign);
    } else if (!native[i].empty()) {
      return invalid("strata-dense: native bytes for a row without a native type");
    }
  }
  const std::uint64_t payload = at;
  for (const DenseRow& r : rows) CLM_RETURN_IF_ERROR(check_row(r, payload));
  ByteWriter w;
  w.raw(ByteSpan(reinterpret_cast<const std::uint8_t*>(kDenseMagic), sizeof kDenseMagic));
  w.u32(kDenseConversionVersion);
  w.u32(static_cast<std::uint32_t>(rows.size()));
  w.u32(align);
  w.u32(0);
  for (const DenseRow& r : rows) put_row(w, r);
  w.u64(payload);
  Bytes out = std::move(w).take();
  const std::size_t base = out.size();
  out.resize(base + payload, 0);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::copy(canonical[i].begin(), canonical[i].end(), out.begin() + static_cast<std::ptrdiff_t>(base + rows[i].payload_offset));
    if (rows[i].has_native())
      std::copy(native[i].begin(), native[i].end(), out.begin() + static_cast<std::ptrdiff_t>(base + rows[i].native_offset));
  }
  return out;
}

std::string index_line(const DenseRow& r, std::int32_t file_id, std::uint64_t src_off, std::uint64_t dst_off) {
  char buf[512];
  std::snprintf(buf, sizeof buf, "%s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d\n",
                r.name.c_str(), file_id, r.kind, static_cast<unsigned long long>(src_off),
                static_cast<unsigned long long>(r.src_bytes), static_cast<unsigned long long>(dst_off),
                static_cast<unsigned long long>(r.dst_bytes), static_cast<long long>(r.ne0),
                static_cast<long long>(r.ne1), r.code_bits, r.code_bias, r.group_elems, r.codebook, r.has_offset,
                static_cast<unsigned long long>(r.codes_bytes), static_cast<unsigned long long>(r.scales_bytes),
                static_cast<unsigned long long>(r.offset_bytes), r.scales_fp16, r.act_kind);
  return buf;
}

// ---------------------------------------------------------------- placement

TensorPlacement place_tensor(std::string_view name) {
  TensorPlacement p;
  if (name == "token_embd.weight") {
    p.home = TensorHome::kEmbedding;
    p.object_name = std::string(objects::kEmbeddingObjectName);
    return p;
  }
  if (name == "output.weight" || name == "output_norm.weight" || name.substr(0, 10) == "output_hc_") {
    p.home = TensorHome::kHead;
    p.object_name = std::string(objects::kHeadObjectName);
    return p;
  }
  const auto blk = split_block(name);
  if (!blk) return p;
  p.layer = blk->first;
  const std::string_view suffix = blk->second;
  for (std::string_view s : kExpertSuffixes)
    if (suffix == s) {
      p.home = TensorHome::kRoutedExperts;
      return p;
    }
  for (std::string_view s : kSharedSuffixes)
    if (suffix == s) {
      p.home = TensorHome::kSharedExpert;
      p.object_name = objects::shared_expert_object_name(blk->first);
      return p;
    }
  p.home = TensorHome::kLayerDense;
  p.object_name = objects::dense_object_name(blk->first);
  return p;
}

bool served_natively(std::string_view name) {
  if (name == "output.weight" || name == "token_embd.weight") return true;
  if (name.substr(0, 4) != "blk." || name == "blk.1.ple_key.weight") return false;
  for (std::string_view s : kNativeSuffixes)
    if (ends_with(name, s)) return true;
  return false;
}

// ---------------------------------------------------------------- pack reader

Result<StrataPackReader> StrataPackReader::open(const std::filesystem::path& pack_dir) {
  StrataPackReader r;
  r.dir_ = pack_dir;
  std::ifstream in(pack_dir / "index.txt", std::ios::binary);
  if (!in) return make_error(ErrorCode::kNotFound, "cannot open " + (pack_dir / "index.txt").string());
  std::string line;
  bool have_header = false;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      int a = 0, t = 0;
      unsigned long long p = 0;
      if (std::sscanf(line.c_str(), "# align %d pool %llu tensors %d", &a, &p, &t) == 3) {
        if (a <= 0 || (a & (a - 1)) != 0) return data_loss("index.txt: bad alignment");
        r.align_ = static_cast<std::uint32_t>(a);
        have_header = true;
      }
      continue;
    }
    PackRow pr;
    DenseRow& d = pr.row;
    char name[256] = {0};
    unsigned long long src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0, cb = 0, sb = 0, ob = 0;
    long long ne0 = 0, ne1 = 0;
    const int n = std::sscanf(line.c_str(), "%255s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d",
                              name, &pr.file, &d.kind, &src_off, &src_bytes, &dst_off, &dst_bytes, &ne0, &ne1, &d.code_bits,
                              &d.code_bias, &d.group_elems, &d.codebook, &d.has_offset, &cb, &sb, &ob, &d.scales_fp16,
                              &d.act_kind);
    if (n != 19) return data_loss("index.txt: could not parse a row (" + std::to_string(n) + " of 19 fields)");
    d.name = name;
    d.payload_offset = src_off;
    d.src_bytes = src_bytes;
    d.dst_bytes = dst_bytes;
    d.ne0 = ne0;
    d.ne1 = ne1;
    d.codes_bytes = cb;
    d.scales_bytes = sb;
    d.offset_bytes = ob;
    d.source_file = pr.file;
    pr.dst_off = dst_off;
    r.rows_.push_back(std::move(pr));
  }
  if (!have_header || r.rows_.empty()) return data_loss("index.txt has no '# align ... pool ...' header or no rows");
  return r;
}

const PackRow* StrataPackReader::find(std::string_view name) const {
  for (const PackRow& r : rows_)
    if (r.row.name == name) return &r;
  return nullptr;
}

Result<Bytes> StrataPackReader::read(const PackRow& row) const {
  static constexpr const char* kFiles[] = {"dense.bin", "embd.bin", "experts.bin", "extra.bin"};
  if (row.file < 0 || row.file > 3) return data_loss("pack row '" + row.row.name + "' names an unknown pack file");
  if (row.row.src_bytes == 0) return Bytes{};
  const std::filesystem::path p = dir_ / kFiles[row.file];
  std::ifstream in(p, std::ios::binary);
  if (!in) return make_error(ErrorCode::kNotFound, "cannot open " + p.string());
  in.seekg(static_cast<std::streamoff>(row.row.payload_offset));
  Bytes out(static_cast<std::size_t>(row.row.src_bytes));
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  if (static_cast<std::uint64_t>(in.gcount()) != row.row.src_bytes)
    return data_loss("short read of '" + row.row.name + "' from " + p.string());
  return out;
}

// ---------------------------------------------------------------- conversion

Result<Bytes> convert_object(const objects::ManifestObject& object, const StrataPackReader& pack,
                             const NativeTensorProvider& native) {
  switch (object.kind) {
    case ObjectKind::kLayerDense:
    case ObjectKind::kSharedExpert:
    case ObjectKind::kEmbedding:
    case ObjectKind::kOutputHead:
      break;
    case ObjectKind::kRoutedExpert:
      return invalid("routed experts are provisioned as their GGUF slices unchanged; nothing to convert");
    default:
      return invalid("object '" + object.name + "' (" + std::string(objects::to_string(object.kind)) +
                     ") is file-backed in Strata and never converted");
  }
  if (object.representation.quant_type != kDenseQuantType ||
      object.representation.conversion_version != kDenseConversionVersion)
    return invalid("object '" + object.name + "' does not declare the strata-dense representation");
  std::vector<DenseRow> rows;
  std::vector<Bytes> canon_store, native_store;
  for (const PackRow& pr : pack.rows()) {
    const TensorPlacement place = place_tensor(pr.row.name);
    if (place.object_name != object.name) continue;
    DenseRow row = pr.row;
    CLM_ASSIGN_OR_RETURN(Bytes canon, pack.read(pr));
    row.source_file = pr.file;
    std::optional<NativeTensor> nt;
    if (served_natively(row.name) && native) {
      CLM_ASSIGN_OR_RETURN(nt, native(row.name));
    }
    if (nt) {
      row.native_type = nt->type;
      row.native_ne0 = nt->ne0;
      row.native_ne1 = nt->ne1;
    } else if (canon.empty()) {
      return data_loss("tensor '" + row.name + "' has neither pack bytes nor a GGUF-form copy");
    }
    rows.push_back(std::move(row));
    canon_store.push_back(std::move(canon));
    native_store.push_back(nt ? std::move(nt->bytes) : Bytes{});
  }
  if (rows.empty()) return make_error(ErrorCode::kNotFound, "the pack has no tensors for object '" + object.name + "'");
  std::vector<ByteSpan> canon_spans, native_spans;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    canon_spans.emplace_back(canon_store[i]);
    native_spans.emplace_back(native_store[i]);
  }
  return encode_dense_object(pack.align(), std::move(rows), canon_spans, native_spans);
}

// ---------------------------------------------------------------- object sets

std::vector<std::string> required_objects(const objects::ModelGeometry& g, const domain::DomainSpec& spec,
                                          const StrataObjectOptions& options) {
  std::vector<std::string> names;
  if (spec.role == domain::StageRole::kPrefix || (spec.role == domain::StageRole::kTail && options.with_mtp))
    names.emplace_back(objects::kEmbeddingObjectName);
  for (std::uint32_t L = spec.layers.begin; L < spec.layers.end; ++L) {
    names.push_back(objects::dense_object_name(L));
    if (g.shared_expert_ff > 0) names.push_back(objects::shared_expert_object_name(L));
    for (std::uint32_t e = 0; e < g.n_experts; ++e) names.push_back(objects::expert_object_name(L, e));
  }
  if (spec.role == domain::StageRole::kTail) names.emplace_back(objects::kHeadObjectName);
  return names;
}

}  // namespace clusterlm::backends::strata
