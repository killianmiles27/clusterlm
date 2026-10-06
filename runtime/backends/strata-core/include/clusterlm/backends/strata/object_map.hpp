#pragma once
// How ClusterLM manifest objects map to what the pinned Strata engine consumes (docs/backends/strata-port.md §6,
// ADR 0200). Platform-independent and Strata-header-free: Father provisioning, the domain adapter and the tests use
// it without CUDA.
//
// Object kinds and their Strata representation:
//
//   kRoutedExpert   "blk.L.exp.E"   Strata's NATIVE expert blob is the three GGUF slices back to back,
//                                   [gate rows | up rows | down rows] (strata native_expert.hpp). That is exactly
//                                   the manifest's three source ranges concatenated in order, so the object is the
//                                   source bytes unchanged: conversion_version 0, object_digest == source_digest.
//                                   quant_type names the ggml types: "<gate/up>" or "<gate/up>+<down>"
//                                   (e.g. "iq3_s+iq4_nl").
//   kLayerDense     "blk.L.dense"   A STRATA-DENSE container (below): the layer's rows of Strata's pack index
//   kSharedExpert   "blk.L.shexp"   (`index.txt`, engine-form planes as tools/pack_index.py wrote them) with their
//   kEmbedding      "token_embd"    bytes, plus - for tensors the engine serves in GGUF form (NativeDense, NativeHead,
//   kOutputHead     "output_head"   NativeEmbed) - the GGUF blocks. quant_type "strata-dense", conversion_version 1;
//                                   object_digest is over the container bytes. Father builds it with
//                                   `convert_object` from its Strata pack + model GGUF: a byte-exact re-layout, never
//                                   a requantization (every plane is copied as the pack/GGUF stores it).
//   kPleLookup / kMtp               Father-only and file-backed in Strata (PleTable reads the n-gram table from the
//                                   model's GGUF with direct I/O; MtpDrafter loads tools/mtp_rt.py's directory).
//                                   Never provisioned to a Node; the Father domains open them from Father's
//                                   canonical model directory (StrataBackendOptions).
//
// Which tensor goes into which object: `place_tensor`. A layer's shared-expert tensors (blk.L.ffn_*_shexp.*) form
// the shared-expert object, its stacked routed tensors (blk.L.ffn_*_exps.weight) are split into the expert objects,
// every other blk.L.* tensor (mixer, router, norms, hyper-connections, and on layer 1 the PLE projections) is the
// dense object. token_embd.weight is the embedding object; output.weight, output_norm.weight and output_hc_*.weight
// the head object.
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clusterlm/common/bytes.hpp"
#include "clusterlm/common/status.hpp"
#include "clusterlm/domain/execution_domain.hpp"
#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::backends::strata {

// ---------------------------------------------------------------- ggml types (the subset Strata's packs use)

struct GgmlType {
  int id = -1;                 // ggml_type enum value at the Strata ggml pin (third_party/upstream.json strata-ggml)
  std::string_view name;       // lower-case GGUF type name
  std::uint32_t block_elems = 0;
  std::uint32_t block_bytes = 0;
};
const GgmlType* ggml_type_by_name(std::string_view name);
const GgmlType* ggml_type_by_id(int id);
// Bytes of `elems` values of `t`; 0 when `elems` is not a whole number of blocks.
std::uint64_t ggml_bytes(const GgmlType& t, std::uint64_t elems);

// ---------------------------------------------------------------- routed experts

inline constexpr std::uint32_t kExpertConversionVersion = 0;  // the GGUF slices unchanged

struct ExpertFormat {
  const GgmlType* gate_up = nullptr;
  const GgmlType* down = nullptr;
  std::uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;  // gate/up: ff rows of H; down: H rows of ff
  std::uint64_t total() const { return gate_bytes + up_bytes + down_bytes; }
};
std::string expert_quant_type(const GgmlType& gate_up, const GgmlType& down);
// Parses an expert Representation against the geometry. Refuses a conversion_version other than 0 and a
// quant_type whose blocks do not tile the expert's rows.
Result<ExpertFormat> expert_format(const objects::Representation& rep, const objects::ModelGeometry& g);

// ---------------------------------------------------------------- the strata-dense container

inline constexpr std::string_view kDenseQuantType = "strata-dense";
inline constexpr std::uint32_t kDenseConversionVersion = 1;
inline constexpr char kDenseMagic[8] = {'C', 'L', 'M', 'S', 'T', 'R', 'D', '1'};

// One tensor of a strata-dense object. The canonical fields are a row of Strata's pack index (weights.cpp IndexRow,
// tools/pack_index.py), with the source offset relative to the container's payload. `native_*` describe the
// tensor's GGUF-form blocks when the engine serves it natively (NativeDense projection, NativeHead, NativeEmbed);
// such a row may have no canonical bytes (`src_bytes == 0`): the engine then skips it in the canonical arena.
struct DenseRow {
  std::string name;
  std::int32_t kind = 0;  // pack index kind 0..5 (WeightKind, plus 4/5 = raw 16-bit in a native pack)
  std::uint64_t payload_offset = 0, src_bytes = 0, dst_bytes = 0;
  std::int64_t ne0 = 0, ne1 = 0;
  std::int32_t code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0;
  std::uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
  std::int32_t scales_fp16 = 0, act_kind = 0;
  std::int32_t source_file = 0;  // the pack file the bytes came from (0 dense.bin, 1 embd.bin, 3 extra.bin)
  std::int32_t native_type = -1;  // ggml type of the GGUF-form copy, -1 = none
  std::uint64_t native_ne0 = 0, native_ne1 = 0, native_offset = 0, native_bytes = 0;
  bool has_native() const { return native_type >= 0; }
  friend bool operator==(const DenseRow&, const DenseRow&) = default;
};

// A parsed container. `payload` points into the bytes it was parsed from.
struct DenseObject {
  std::uint32_t align = 256;  // the pack index's arena alignment
  std::vector<DenseRow> rows;
  ByteSpan payload;
  ByteSpan canonical(const DenseRow& r) const { return payload.subspan(r.payload_offset, r.src_bytes); }
  ByteSpan native(const DenseRow& r) const { return payload.subspan(r.native_offset, r.native_bytes); }
};

// Bounds-checked parse (every row's planes inside the payload, plane sizes consistent with src/dst bytes).
Result<DenseObject> parse_dense_object(ByteSpan bytes);
// Serializes rows + their bytes (canonical[i] / native[i] belong to rows[i]; offsets are assigned here, each segment
// 64-byte aligned). Deterministic: the same inputs give the same bytes, hence the same object digest.
Result<Bytes> encode_dense_object(std::uint32_t align, std::vector<DenseRow> rows, const std::vector<ByteSpan>& canonical,
                                  const std::vector<ByteSpan>& native);
// The row as a line of Strata's index.txt (19 fields), with the given source and arena offsets.
std::string index_line(const DenseRow& row, std::int32_t file_id, std::uint64_t src_off, std::uint64_t dst_off);

// ---------------------------------------------------------------- tensor -> object

enum class TensorHome : std::uint8_t { kLayerDense, kSharedExpert, kRoutedExperts, kEmbedding, kHead, kOther };
struct TensorPlacement {
  TensorHome home = TensorHome::kOther;
  std::optional<std::uint32_t> layer;
  std::string object_name;  // empty for routed experts (split per expert) and kOther
};
TensorPlacement place_tensor(std::string_view tensor_name);

// ---------------------------------------------------------------- Father-side conversion

// A row of a Strata pack's index.txt, with its absolute location.
struct PackRow {
  DenseRow row;  // payload_offset holds the absolute src_off inside the pack file
  std::int32_t file = 0;
  std::uint64_t dst_off = 0;
};

// Reads a Strata pack directory (index.txt + dense.bin / embd.bin / extra.bin). Father-only: it is part of the
// canonical model store.
class StrataPackReader {
 public:
  static Result<StrataPackReader> open(const std::filesystem::path& pack_dir);
  std::uint32_t align() const { return align_; }
  const std::vector<PackRow>& rows() const { return rows_; }
  const PackRow* find(std::string_view name) const;
  Result<Bytes> read(const PackRow& row) const;

 private:
  std::filesystem::path dir_;
  std::uint32_t align_ = 256;
  std::vector<PackRow> rows_;
};

// A tensor's GGUF-form blocks (from the model GGUF; Father's GGUF reader supplies them).
struct NativeTensor {
  std::int32_t type = -1;
  std::uint64_t ne0 = 0, ne1 = 0;
  Bytes bytes;
};
// Returns the GGUF-form tensor when the engine serves `tensor_name` natively, std::nullopt when it does not.
using NativeTensorProvider = std::function<Result<std::optional<NativeTensor>>(std::string_view tensor_name)>;

// Which of a tensor's GGUF forms the engine loads natively (Strata NativeDense::served_names' rule plus the head
// and the embedding): the mixer/attention/shared-expert projections, output.weight, token_embd.weight.
bool served_natively(std::string_view tensor_name);

// The provisioned bytes of one strata-dense object (dense, shared expert, embedding, head): every pack row the
// object owns, plus native copies from `native`. Routed experts need no conversion (their object is the source
// bytes) and are refused here.
Result<Bytes> convert_object(const objects::ManifestObject& object, const StrataPackReader& pack,
                             const NativeTensorProvider& native);

// ---------------------------------------------------------------- domain object sets

struct StrataObjectOptions {
  bool with_mtp = false;  // a tail domain hosting the MTP drafter also needs the embedding (Father-local)
};
// The objects a Strata domain binds: its layers' dense + shared + routed experts; the prefix adds the embedding
// (token_embd; PLE stays a Father file), the tail the head (+ the embedding with MTP).
std::vector<std::string> required_objects(const objects::ModelGeometry& g, const domain::DomainSpec& spec,
                                          const StrataObjectOptions& options = {});

}  // namespace clusterlm::backends::strata
