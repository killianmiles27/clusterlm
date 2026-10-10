# Qualification evidence

A backend descriptor (or one of its family entries) may carry the label `Supported and qualified` only when every id in
its `qualification.hardware_experiments` has a passing record here. Record = `<HQ-id>.<backend-id>.<date>.evidence.json`:

```json
{ "schema": "clusterlm.qualification-evidence", "version": 1, "experiment": "HQ-FAST-01", "backend": "llama-local",
  "result": "pass", "benchmark_result": "results/perf-fast.json", "benchmark_sha256": "<hex>",
  "build_hash": "<backend build hash>", "reviewed_by": "<human>", "date": "2026-11-01" }
```

`scripts/check_backend_descriptors.py` verifies: the referenced benchmark result exists (stored next to the record), its
hash matches, its `experiment` equals the record's, and its `provenance` is `Qualified` (set by human review, never by
tools). `Synthetic` or `Measured`-only results never qualify. No records exist yet: every descriptor is below
`Supported and qualified` and the check says so.
