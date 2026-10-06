# 0100. GGUF to manifest object mapping, range order and the `quant_type` grammar

Status: accepted (WP1)

## Context

Father builds the `ModelManifest` straight from the real GGUF artifact(s): no repacking, no requantization, exact byte
ranges. `ManifestObject` has one `Representation` and an ordered list of `SourceRange`s, but a real object is several
tensors of different ggml types (a layer's dense object mixes F32 norms, BF16 gates and K-quant projections; a routed
expert may use different types for gate, up and down). The manifest wire format carries no tensor names or shapes, and
changing it would break every other module.

## Decision

- **Range order is physical order**: ranges of a dense, shared-expert, head or MTP object are sorted by (shard, file
  offset), one range per tensor, never merged across alignment padding (merging would add padding bytes to the
  object). Experts use fixed order gate, up, down (or gate_up, down for a fused tensor). The order is therefore a
  property of the artifact, which also lets a generated fixture control its layout (the fixture GGUF writes tensors in
  `DenseLayout` order so the reference backend runs from a GGUF-derived manifest unchanged).
- **`quant_type`** is the ggml type name (lower case) when every range has the same type, otherwise one name per range in
  range order joined by `|` (`iq3_s|iq3_s|iq4_xs`). A single name means "all ranges"; `decode_quant_types(q, n_ranges)`
  is the inverse and rejects any other count. `block_size` is the common block size when all ranges share one block
  size > 1, else 0 (consumers then use the per-range type list). Mixed types are never collapsed or approximated.
- **Tensor identity stays outside the manifest**: `build_manifest` returns an `ObjectLayout` per object (tensor name,
  per-object dims, type, offset inside the object). Backends that need tensor boundaries use this sidecar (and,
  later, plan-scoped layout records); the manifest format and its root hash are unchanged.
- **Classification** is by name (`classify_tensor`). Anything unrecognized is an error unless `allow_unclassified`, in
  which case it is listed in the report and left out of the manifest. Shared-expert gate/up/down and
  `ffn_gate_inp_shexp` form the shared-expert object; the router and everything else of a layer is dense.
- **PLE layer**: taken from the layer that holds `ple_key.weight` (Strata's artifact: layer 1, six `ple_*` tensors);
  `<arch>.ple.layer` is an optional ClusterLM fallback for fixtures and must agree if both exist.

## Consequences

- Manifests are exact and mixed quantization survives. Object identity depends on the artifact's tensor order, which is
  fine because the root hash already covers ranges.
- Consumers must parse `quant_type` with `decode_quant_types`, not compare it with `==` against a single name unless
  they know the object is uniform. The planner today only copies the label of an expert object.
- A node-side backend adapter will need `ObjectLayout` shipped with (or derivable from) plan-scoped manifests; this is
  tracked as follow-up work, not solved here.
