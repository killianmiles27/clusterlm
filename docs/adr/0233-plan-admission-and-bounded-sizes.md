# 0233 Bounded geometry, saturating plan sums and working-memory admission

## Context

A Node sizes real allocations from numbers it receives: manifest geometry (hidden width, streams, experts,
heads, vocab ...), `StageAssignment.max_context` / `max_window`, object byte sizes, and counts of ranges,
objects, stages and timings. The audit and the fuzzers found places where such a number could reach `resize`,
`assign` or a product before any check against what the input or the Node can actually back:

* geometry had lower bounds only, so a manifest could ask a domain for gigabytes of scratch;
* `StageActivations` bounded each dimension to 2^20 but `hc*H + H + hc` is a 32-bit expression and wrapped;
* `PreparePlan` summed object sizes and stage state without overflow checks and never admitted the stages'
  working memory (KV, window snapshots, scratch) against the RAM allowance;
* some repeated fields allocated up to a protocol maximum before verifying that the payload held that many
  elements, and the frame reader allocated `payload_len` bytes on a 24-byte header.

## Decision

* Geometry has sanity ceilings far above any real model (Flash-Next fits with 1-3 orders of magnitude to spare);
  activation record size is bounded in 64-bit arithmetic.
* Repeated fields are sized only after checking `remaining_input >= count * element_size`.
* The frame reader grows its payload buffer in 1 MiB steps as bytes arrive. `max_payload` still bounds a frame, but a
  peer must deliver bytes to make the receiver allocate them.
* A Node validates a plan before creating anything: saturating sums of object sizes, no object assigned twice,
  no Father-only object or target, only middle stages, and a probe domain's `describe_requirements()` (state +
  window + scratch) must fit the RAM allowance (`kResourceExhausted` otherwise).
* Range validation in the manifest is O(n log n) and overflow-checked.

## Consequences

* A hostile or corrupt plan fails early with a precise status instead of exhausting memory or CPU on the Node.
* Plans from the real backend must report realistic `describe_requirements()`; the admission is only as good as
  that report (DEVELOPMENT-STATUS gap 6 already asks for a backend-reported ledger).
* The ceilings are constants in `geometry.cpp`; a future model that exceeds one needs a deliberate change plus a
  security-review note.
* Tests: `tests/security/test_hardening.cpp`, fuzz targets `manifest_decode`, `stage_activations`,
  `protocol_decode`, `frame_stream`.
