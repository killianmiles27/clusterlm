# Result provenance

Every number ClusterLM shows or returns (tokens per second, memory, time to ready, latency, predicted placement cost) has one of
three provenances. The code type is `placement::Provenance`; the rule is in
[interfaces/README.md](interfaces/README.md#conventions-shared-by-all-contracts).

| Provenance | Meaning | Examples | May be called a benchmark? |
|---|---|---|---|
| `synthetic` | Computed from a model, a fixture or a declared parameter, not observed on the machine | hardware profiles in `fixtures/profiles/`, `--impair gige-simulated`, cost-model predictions | No |
| `measured` | Observed by running the real code on a real machine, with the command and environment recorded | `clusterlm-bench` results on the user's own hardware | Yes, for that machine and configuration only |
| `qualified` | Measured on the target hardware following a procedure in [HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md) and accepted | none today | Yes |

Rules:

* A result derived from several inputs carries the **weakest** input's provenance. A measurement is never promoted
  automatically.
* The reference backend and generated fixture models test correctness (ownership, bitwise-identical logits, protocol), not
  speed or quality. Their timings are not benchmarks.
* Strata CUDA code is compiled in CI but has never run on a GPU. Anything about GPU speed, VRAM margins or CUDA correctness is
  `pending`, not `synthetic` and not `measured`.
* A documentation or UI sentence quoting a performance figure must say which provenance it has. The 20 tok/s Ultra figure is
  a design target (`HQ-PERF-01`), not a result.
* Compatibility labels follow the same discipline: `Supported and qualified` requires qualification evidence; code that
  merely runs on a developer machine is `Supported, awaiting hardware qualification` at best.

When contributing a result: record the command, the commit, the machine (CPU, RAM, GPU, driver, OS), and the provenance. See
[benchmark-methodology.md](benchmark-methodology.md).
