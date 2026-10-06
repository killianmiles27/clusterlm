# Model manifest from GGUF

Father builds the canonical `ModelManifest` (`runtime/objects/include/clusterlm/objects/manifest.hpp`) directly from the
real GGUF artifact(s). Nothing is repacked or requantized: every object names exact byte ranges inside the GGUF files,
and the digests are SHA-256 over those bytes. Nodes never see a GGUF; they receive plan-assigned objects only.

Code: `ggml_types.hpp` (type table), `gguf.hpp` (reader), `gguf_manifest.hpp` (builder, hashing, classification),
`gguf_writer.hpp` / `fixture_gguf.hpp` (test and fixture writers). Tool: `clusterlm-model-inspect`. Decisions:
`docs/adr/0100`, `0101`. Real-model qualification step: `HQ-MODEL-01`.

## Pipeline

```
GGUF file(s) --open_gguf_model--> GgufModelFiles --build_manifest--> ModelManifest (+ ManifestBuildReport)
 (header only)  (validated, split-aware)            (ranges, types)      |
                                                                          +-- compute_manifest_digests (optional, streamed)
```

`build_manifest(paths, ManifestBuildOptions{hash_objects, hash_shards, allow_unclassified, ...})` reports progress through
a callback that can cancel. Hashing can also be done afterwards (`compute_manifest_digests`), on another thread, with a
bounded buffer per thread (default 4 MiB) and N worker threads; an Ultra-size model is ~83 GB, so structure-only builds
(`hash_objects = false`, zero digests) are fast and hashing is an explicit second step. A manifest with zero digests is
valid for planning, not for provisioning.

## GGUF reader limits

Only the header region is read (magic, metadata, tensor directory); tensor data is never read or mapped except when
hashing. Reads come from the head of the file in 1 MiB chunks. Defaults (`GgufLimits`): 1,048,576 tensors, 1,048,576
metadata entries, 64 KiB strings, 16,777,216 array elements, 256 MiB header, alignment up to 1 GiB, 4096 shards. Counts and
lengths are checked before allocation; all offset/size arithmetic is overflow-checked; each tensor must lie inside the file,
start on the alignment, and not overlap another. Versions 2 and 3, little-endian only. See ADR 0101 for the full rule list.

Split GGUF: `split.no`, `split.count`, `split.tensors.count` are cross-validated; shards are ordered by `split.no`
(manifest shard index = split order). `clusterlm-model-inspect` expands any `-NNNNN-of-MMMMM.gguf` argument to the whole set.
Files without split keys passed together are **companions** (for example a separate MTP GGUF): the first supplies the
metadata, tensors are merged by name, duplicates are errors.

## ggml types

`ggml_types.cpp` carries id, name, block elements and bytes per block for every type in the pinned llama.cpp
(`6753a033f058fbf778d282556ed9b16c78de7c71`; values verified by compiling `sizeof(block_*)` from `ggml-common.h`), e.g.
`iq3_s` 256/110, `iq2_xs` 256/74, `q8_0` 32/34, `q4_k` 256/144, `bf16` 1/2. Unknown or removed ids are errors. A row must be
a whole number of blocks.

## Metadata keys and geometry

`<arch>` is `general.architecture` (Flash-Next: `qwen4exp`). `ModelGeometry` is built from:

| Field | Source |
|---|---|
| `family` | `general.architecture` |
| `n_layers` | `<arch>.block_count` (required) |
| `hidden_size` | `<arch>.embedding_length` (required) |
| `n_experts`, `n_active_experts` | `<arch>.expert_count`, `<arch>.expert_used_count` (required) |
| `n_heads`, `n_kv_heads` | `<arch>.attention.head_count`, `<arch>.attention.head_count_kv` (required, scalar) |
| `ple_ngram` | `<arch>.ple.ngram_size` (required) |
| `head_dim` | `<arch>.attention.key_length` if present, else `blk.L.attn_q_norm.weight` of a full-attention layer; must agree if both |
| `expert_ff` | shape of the stacked expert tensors (`ffn_gate_exps` dim 1; `ffn_gate_up_exps` dim 1 / 2); `<arch>.expert_feed_forward_length` must agree if present |
| `shared_expert_ff` | `ffn_gate_shexp.weight` dim 1 (0 = none; must be the same in every layer) |
| `residual_streams` | dim 1 of `blk.N.hc_attn_inject.weight` (`[k*hidden, hc]`); same in every layer |
| `vocab_size` | dim 1 of `token_embd.weight` (`[hidden, vocab]`) |
| `ple_rows` | dim 1 of `per_layer_token_embd.weight` (`[row_width, rows]`) |
| `ple_layer` | the layer holding `ple_key.weight` (exactly one); optional `<arch>.ple.layer` is a ClusterLM extension used when no PLE block tensors exist (fixtures) and must agree otherwise |
| `layer_kinds` | `attn_q.weight` present = full attention (QSA); `attn_qkv.weight` present = linear attention (GDN); both or neither is an error |
| `mtp_layers` | distinct MTP layer ids among `mtp.layers.N.*` / `blk.N.*` with N >= block_count; 1 for ungrouped `mtp.*` tensors; 0 if none |

Missing required keys fail with the key name. The geometry passes `ModelGeometry::validate()`.

Note on the PLE layer: Strata at the pin places the PLE block on layer **1** (`blk.1.ple_*`, six tensors); the product brief
says "layer id 2". The manifest takes the layer from the artifact, so `geometry.ple_layer` is whatever the files say
(`HQ-MODEL-01` records it). The fixture models keep `ple_layer = 2`.

## Objects and naming

| Kind | Name | Source tensors | Notes |
|---|---|---|---|
| `kEmbedding` | `token_embd` | `token_embd.weight` | Father only |
| `kPleLookup` | `ple_lookup` | `per_layer_token_embd.weight` | Father only (SSD-resident; in the real artifact it is its own shard) |
| `kOutputHead` | `output_head` | `output.weight`, `output_norm.weight`, `output_hc_*.weight` | Father only |
| `kMtp` | `mtp_drafter` | `mtp.*`, `nextn.*`, `blk.N.*` with N >= block_count, `blk.L.nextn.*` | Father only; absent if the artifact has none |
| `kLayerDense` | `blk.L.dense` | every `blk.L.*` tensor except experts and shared expert (router `ffn_gate_inp`, GDN/QSA projections, indexer, norms, gated-residual `hc_*`, PLE block, ...) | one range per tensor, physical order |
| `kSharedExpert` | `blk.L.shared` | `ffn_{gate,up,down}_shexp.weight`, `ffn_gate_inp_shexp.weight` | omitted if the layer has none |
| `kRoutedExpert` | `blk.L.exp.E` | slice E of `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (or `ffn_gate_up_exps`, `ffn_down_exps`) | 3 (or 2) ranges |

Object order in `manifest.objects` is deterministic: Father-only objects, then for each layer dense, shared, experts 0..E-1.
Names come from `dense_object_name` / `expert_object_name` / `shared_expert_object_name`; they are structural identities,
never vocabulary token IDs.

**Expert slicing.** Stacked expert tensors have ggml dims `[in, out, n_experts]`, expert axis slowest, so expert E of a tensor
is the contiguous run `[offset + E*slice, +slice)` with `slice = row_bytes(ne0) * ne1`. Slices start and end on block
boundaries by construction (rows are whole blocks); the builder re-checks and fails otherwise. Gate/up/down of one expert are
three ranges; their byte sizes follow their own types.

**Ranges and sizes.** Every range names `(shard, absolute file offset, length)`; the object `byte_size` is the sum, and
`conversion_version` is 0, so the provisioned bytes are the source bytes. Ranges of one object never overlap
(`ModelManifest::validate`). Together the objects cover every classified tensor byte exactly once
(`ManifestBuildReport::tensor_bytes_in_manifest`).

**Unclassified tensors.** A tensor the mapping does not know is an error naming it, unless `allow_unclassified`; then it is left
out of the manifest and listed in `ManifestBuildReport::unclassified` (name, shard, type, bytes, reason). A layer that lacks
experts, has an incomplete expert set, inconsistent shapes or mismatched counts is always an error.

## Representation identity

`Representation::quant_type` is the lower-case ggml type name if all ranges of the object have the same type; otherwise one
name per range in range order, `|`-separated (`iq3_s|iq3_s|iq4_xs`, `q8_0|iq3_s|q4_k`). `block_size` is the common block
size when all ranges share one (> 1), else 0. `decode_quant_types(quant_type, n_ranges)` recovers the per-range types. The
representation is part of the manifest root hash, so a Node cannot be handed a different quantization under the same name.

`ManifestBuildReport::layouts[i]` (parallel to `manifest.objects`) lists, per range, the tensor name, per-object dims, type and
offset inside the object. It is not serialized into the manifest (ADR 0100).

## Digests

`source_digest` = SHA-256 over the concatenated source ranges in range order; `object_digest` is the same value for unconverted
objects. `ShardInfo::digest` (optional, `hash_shards`) is the SHA-256 of the whole file. The root hash covers geometry, shards and
every object entry. The `CanonicalModelStore` reads the GGUF shards directly (a GGUF file is a shard; ranges are absolute file
offsets) and verifies object digests on load; provisioning streams the same ranges to Nodes in bounded chunks.

## How Nodes receive subsets

Father builds a plan-scoped manifest per Node from this manifest (geometry plus only that Node's layer objects: dense, shared,
routed experts it was assigned). Father-only objects (embedding, PLE lookup, head, MTP) are never included, and a Node never
learns a GGUF file name or path (`ShardInfo::file_name` is Father-relative). Nodes hold objects only in the ephemeral lease
store and delete them on release.

## Tooling

- `clusterlm-model-inspect <gguf...> [--manifest out.json] [--hash] [--hash-shards] [--summary] [--allow-unclassified]
  [--artifact-id ID] [--threads N]`: geometry, tensor inventory by class and type, per-layer bytes and expert quant types,
  Father-only bytes, boundary ABI bytes per position, KV elements per token, largest dense/expert objects. Exit code 1 if tensors
  are unclassified (unless allowed). It reports structure and sizes only; it never labels anything Qualified.
- `clusterlm-fixture-model --format gguf [--f32-experts]`: the deterministic fixture as a two-shard split GGUF with Flash-Next
  style names; its manifest comes from `build_manifest`. The reference backend runs from it unchanged (dense tensors are written in
  the reference layout order; experts are F32 or ggml Q8_0, decoded by the new `q8_0` codec in `tensor_codec`).
- `GgufWriter` writes synthetic GGUF files with any metadata and any ggml type (random payloads) for tests and fuzz seeds.
- `tests/fuzz/fuzz_gguf` (libFuzzer, `-DCLUSTERLM_FUZZ=ON`, clang): `gen_gguf_seeds <dir>` writes a seed corpus of header-only
  files; the target pads the file with virtual zeros so small headers reach the manifest builder.
