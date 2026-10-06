# ADR 0200: Strata domains bind weights from provisioned objects, not a pack directory

Status: accepted (WP6)

## Context

Pinned Strata loads its dense weights from a pack directory (`<pack>/index.txt` + `dense.bin`/`embd.bin`/`extra.bin`,
`WeightTable::load`), its GGUF-form projections, head and embedding from the model's GGUF shards (`NativeDense`,
`NativeHead`, `NativeEmbed`), its routed experts from `experts.bin` or the GGUF in place (`FileExpertSource`), and the
CPU expert layout from `<pack>/native_experts.txt`. A Node receives only plan-assigned objects into an ephemeral
LeaseStore and must never need the full model, a pack directory or the GGUF. Two options were on the table: a loader
over `ObjectResolver` byte spans, or a temporary partition (write the assigned objects as a minimal pack under the
lease directory and let Strata's path loaders read it).

## Decision

Memory-backed loading. Strata patch 0005 adds a `WeightSource` interface to `WeightTable` (the pack directory is one
implementation; parsing, plane checks and arena layout are unchanged), span loaders for `NativeDense`
(`load_tensors`), `NativeHead`/`NativeEmbed` (`load_tensor`) and `expert_layout_set`. ClusterLM objects map to
Strata as follows (`runtime/backends/strata-core/.../object_map.hpp`):

* **Routed expert** = Strata's native expert blob, `[gate | up | down]` GGUF slices: the manifest's three source
  ranges concatenated, unchanged (conversion_version 0, `quant_type` "iq3_s+iq4_nl"). The CPU pool reads it in place
  from the lease store (`ResolverExpertSource`); GPU-resident ones are uploaded into an `ExpertCache` slot.
* **Layer dense, shared expert, embedding, head** = a *strata-dense* container (conversion_version 1): the object's
  rows of Strata's pack index (engine-form planes as `tools/pack_index.py` wrote them) with their bytes, plus the
  GGUF-form copies Strata serves natively. Father builds it once with `convert_object(pack, gguf)`: a byte-exact
  re-layout, never a requantization; `object_digest` is over the container.
* **PLE n-gram table and MTP runtime** stay Father-only and file-backed (Strata reads the table with direct I/O
  from the GGUF; the MTP loader takes a directory). The Father prefix/tail domains get their paths from
  `StrataBackendOptions`; no other role ever sees them.

## Consequences

* No weight file is written on a Node; release frees device arenas and unmaps nothing Strata owns.
* Five upstream files change (`weights`, `native_dense`, `native_head`, `expert_layout`); the path loaders keep
  their behaviour and are now thin wrappers, which keeps the patch upstreamable.
* Father provisioning must serve converted objects. `clusterlm-strata convert` writes a ClusterLM model directory
  whose shards already hold converted objects (source == object bytes), so the existing CanonicalModelStore and
  provisioning pipeline serve them unchanged.
* `describe_requirements` reports container sizes for dense objects before binding; the exact engine-form arena
  (bf16 re-rounding shrinks, fp16 scale widening grows) is known after `prepare` (`read_metrics`).
