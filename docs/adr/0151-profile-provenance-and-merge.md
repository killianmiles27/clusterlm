# 0151 Provenance and merge rules for bench-written profiles

## Context

`clusterlm-bench calibrate` writes `HardwareProfile` / `NetworkProfile` files that placement loads directly. Profiles
are updated over time (a CPU re-run, a new GPU driver, a second network pass) and may start from a synthetic
fixture. The tool must never let a number look better-established than it is.

## Decision

- A field the run measured is `Measured` with source `bench:<run-id>@<machine-id>`. The run id is
  `run-<UTC yyyymmddThhmmssZ>-<4 hex>`, so lexicographic order is chronological order.
- A field the run did not measure is a Synthetic placeholder, or keeps its value from the base/existing file.
  The profile schema has no "absent" state, so placeholders carry `synthetic: not measured by clusterlm-bench`.
  `--base fixtures/profiles/X.json` starts a new profile from realistic Synthetic values instead of zeros.
- The tool never writes `Qualified` and never promotes beyond `Measured`. `mark_qualified` stays the only path and
  fails while any field is Synthetic.
- Merge, per quantity: incoming Synthetic never replaces anything; Synthetic existing is replaced by Measured;
  Measured existing is replaced only by a Measured value from a newer-or-equal run id; Measured with an unparseable
  source (hand-entered) is kept and reported; Qualified existing is never touched. Incoming `Qualified` is clamped to
  `Measured`. Map keys (`expert_bytes_per_s`, `dense_layer_ms`) are added when measured, never invented.
- After a merge, inherited Synthetic placeholders that would contradict a fresh measurement
  (`ram_safe_allowance > ram_total`, `vram_budget > vram_total`) are clamped so the profile still validates.
- `cpu.sustained_factor` is populated only by a sustained run of at least 30 minutes (HQ-CPU-02); shorter runs are
  recorded in the result file but leave the factor Synthetic.
- Result `host_role` is `development-host` unless the operator passes `--on-target`, which asserts the machine is the
  named target. The tool cannot verify this; qualification review does.

## Consequences

- A profile made on the development host is Measured but labelled with a `dev-<hostname>` machine id and the note
  names the kernel provider (`reference` is not the production CPU path). It must not be copied to a target.
- Loopback or impaired network measurements produce Synthetic link quantities and never replace Measured ones.
- Merge is deterministic and idempotent for a repeated run id, and unit-tested (`tests/bench/test_profile_merge.cpp`).
