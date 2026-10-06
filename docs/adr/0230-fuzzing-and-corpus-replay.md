# 0230 Fuzzing with libFuzzer and corpus replay in ctest

## Context

Every decoder that reads bytes from the LAN, from a model download or from a local file is a trust boundary:
frame headers, all 25 protocol messages, `StageActivations`, the binary and JSON manifest, the lease journal, the
tier catalog, placement profiles, the small text parsers (endpoint, plan, fault rule, hex) and the experimental
expert-domain messages. Unit tests cover the intended inputs; bugs like an uncaught exception or a count that
sizes an allocation before it is checked are found by hostile ones. The CI matrix already has a clang
ASan+UBSan job that runs the whole suite, but no fuzzing.

## Decision

* One translation unit per target in `tests/fuzz/` exporting `LLVMFuzzerTestOneInput`. Each target also checks
  invariants of what it accepted (decode -> encode -> decode is a fixed point, accepted manifests validate,
  journal names are store-generated, ...) and aborts on violation, so a finding is a saved reproducer.
* `-DCLUSTERLM_FUZZ=ON` (clang only) builds `fuzz_<name>` libFuzzer executables. The top-level `CMakeLists.txt`
  adds `-fsanitize=fuzzer-no-link,address,undefined` to every library when the option is on (it has to precede
  the module directories), so the fuzzers see inside the code under test.
* The same source always builds as `fuzz_replay_<name>`, registered in ctest (label `fuzz`). It replays the
  committed corpus in `tests/fuzz/corpus/<name>/` and a bounded deterministic mutation pass (48 mutations per seed,
  fixed PRNG). In CI's existing sanitizer job this means the corpus - including every fixed reproducer - runs
  under ASan+UBSan on every push without a new workflow step.
* Seeds are generated from valid encodings by `clusterlm_fuzz_gen_corpus` (all 25 messages, a fixture manifest,
  journals, the shipped catalog and profiles, parser inputs) and committed. Reproducers of fixed findings are
  added as `crash-*` and are not touched by the generator.
* Transport framing is reached through a small public seam, `transport::testing::make_byte_source_connection`,
  which runs the real framed reader over an in-memory byte string.

## Consequences

* A regression in a decoder fails an ordinary `ctest` run, not only a manual fuzzing session.
* The corpus is a few hundred KiB and must be regenerated (and the fuzzers re-run) when a wire format changes.
* Adding a protocol message requires a seed (the generator iterates the sample set, which has a `static_assert`
  on the number of variant alternatives).
* The replay mutation pass is not a substitute for a long fuzzing run; the review records the runs that were done.
