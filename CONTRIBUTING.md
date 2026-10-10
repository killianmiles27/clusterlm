# Contributing to ClusterLM

ClusterLM is developed in parallel workstreams (0, A to G, Z) described in [docs/plan.md](docs/plan.md). Read these first:

1. [docs/status.md](docs/status.md): the requirement ledger (what is done, what is not).
2. [docs/interfaces/](docs/interfaces/README.md): frozen contracts. A change to one needs an ADR ([ADR 0406](docs/adr/0406-interface-freeze-process.md)).
3. [docs/cross-module-requests.md](docs/cross-module-requests.md): how to ask for a change in code another workstream owns.
4. The latest note in [docs/handoff/](docs/handoff/).

## Build and test

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -LE slow --output-on-failure    # everything CI runs, minus tests labelled "slow"
python3 -I tests/conformance/check_docs.py             # links + licensing policy
python3 -I tests/conformance/check_offline.py          # no internet clients in product code
python3 -I docs/interfaces/validate_examples.py       # needs: pip install jsonschema
```

Sanitizer, clang-tidy, llama.cpp, Strata and Windows variants are CI jobs; see [docs/testing.md](docs/testing.md) for how to
reproduce each one locally. A change is ready when the jobs relevant to the files it touches are green **on CI**. Do not
describe a test you did not run as passing.

## Rules that are not negotiable

* **Honest results.** Never present a fixture, a mock or a synthetic estimate as a benchmark, a CUDA result or a
  qualification. Every numeric result carries a provenance (`synthetic`, `measured`, `qualified`; [docs/provenance.md](docs/provenance.md)).
  When hardware is not available, write the test and mark it pending in [HARDWARE-QUALIFICATION.md](HARDWARE-QUALIFICATION.md)
  (generated: edit `bench/qualification/experiments.json`, run `python3 scripts/gen_qualification_md.py`).
* **Compatibility labels** are exactly `Supported and qualified`, `Supported, awaiting hardware qualification`,
  `Experimental`, `Unsupported`.
* **Privacy.** Prompts, completions, token IDs, tool arguments and private paths never go to logs, diagnostics, exports or
  Worker-bound messages.
* **Bounded.** Every parser and allocation has a stated limit; new parsers get a fuzz target and corpus
  ([ADR 0230](docs/adr/0230-fuzzing-and-corpus-replay.md)).
* **No silent substitution** of a model or quantization.
* **Licensing is the owner's decision.** Do not add a `LICENSE` file, SPDX headers or wording that asserts a license for the project. CI rejects a license file. See [docs/licensing/license-comparison.md](docs/licensing/license-comparison.md).
* **No telemetry, accounts, or cloud calls** in product code.

## Code style

Match the surrounding code. C++20, no exceptions across module boundaries (the code base uses `Result`/`ErrorCode`), product
code is clean under `.clang-tidy` (warnings are errors). Windows-specific code must compile under the MinGW syntax check
(`scripts/check_windows_compile.sh`) and MSVC.

## Pull requests

Describe what a user would see before and after, then how you checked it (name the ctest targets or CI jobs and say what
was **not** run). Update the ledger row(s) in `docs/status.md` for the requirement you touched. Two workstreams do not edit
the same files concurrently ([ownership table](docs/plan.md)).
