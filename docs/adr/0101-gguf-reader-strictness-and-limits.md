# 0101. GGUF reader: header-only bounded reads, strict validation, split and companion files

Status: accepted (WP1)

## Context

The reader runs on Father against an ~83 GB artifact it does not control (a download, possibly corrupt or tampered),
and must never read or map tensor data, allocate from an attacker-controlled count, or accept a file whose tensors do
not lie inside it. ggml's own loader is permissive in places (it tolerates overlaps, accepts any nonzero bool, and
leaves some validation to later asserts). The real Ultra/Strong artifacts are split (the PLE table alone is a 26.8 GiB
shard) and the MTP head may ship as a separate GGUF.

## Decision

- Parse through a `GgufSource` (file or memory) with a sequential cursor that fetches 1 MiB chunks from the head of the
  file; at most one chunk beyond the end of the header is ever fetched and it is never interpreted. Every count and
  length is checked against `GgufLimits` before allocation (1M tensors, 1M KV, 64 KiB strings, 16M array elements,
  256 MiB header, shard and alignment caps); array and tensor vectors grow with data actually present.
- v2 and v3, little-endian only. v1 and big-endian files are `kVersionMismatch`. Nested arrays are rejected; arrays keep
  only their first 64 elements (the count is exact). Duplicate keys, empty keys, bool values other than 0/1, zero or
  non-power-of-two alignment, duplicate tensor names, `n_dims` outside 1..4, zero dimensions, unknown ggml type ids,
  rows that are not whole blocks, unaligned tensor offsets, tensors outside the data section and **overlapping
  tensors** are all errors, with the tensor named in the message. All size arithmetic is overflow-checked.
- Split sets must agree on `split.count` / `split.tensors.count`, cover `split.no` 0..count-1 exactly once, sum to
  `split.tensors.count`, and shard 0 carries `general.architecture`; shards are ordered by `split.no`, not argument
  order. A tensor name may live in one file only. Files without split keys, given together, are treated as companion
  files (the primary supplies metadata); mixing keyed and unkeyed files is an error. A single shard of a split set
  opened alone is an error.

## Consequences

- Unusual but ggml-loadable files (overlapping tensors, bool = 2) are refused; none are produced by llama.cpp or gguf-py.
- A fuzz target (`fuzz_gguf`) exercises the parser and manifest builder; accepted inputs must satisfy the reader
  invariants. Companion handling means a separate MTP GGUF can be passed on the command line without a new code path.
