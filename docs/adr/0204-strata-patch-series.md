# ADR 0204: Upstream changes as an ordered, content-verified patch series

Status: accepted (WP6)

## Context

The Strata domain needs six changes to the pinned Strata commit (CUDA 12.0 build, token-free stages, window abort,
domain-local state, memory-backed weights, allocation-free sizing). Upstream sources are fetched, never committed
(`third_party/upstream/`, gitignored). A fork would hide the delta; ad-hoc edits would not be reproducible.

## Decision

* Each change is one `third_party/patches/strata/NNNN-*.patch` (`git diff` format), minimal, commented with
  `ClusterLM patch NNNN`, written so it could be proposed upstream. Patches apply in file-name order.
* `scripts/fetch_upstream.py --apply-patches` fetches the pin and applies the series. It is idempotent and
  content-addressed: the expected tree is computed by applying every patch to `HEAD` in a temporary index; a
  checkout whose working tree hashes to it is "pin + patches" and is left alone; a clean pin is patched; any other
  modification is refused and never patched over; a failing patch leaves the pin clean. `--check` verifies without
  touching anything; ctest runs it in every Strata build (`strata_checkout_is_pin_plus_patches`).
* Strata's ggml dependency is pinned as `strata-ggml` (llama.cpp at the commit the Strata pin names) and passed as
  `STRATA_GGML_DIR`, so a Strata build needs no configure-time network fetch.
* Configure refuses an unpatched checkout (marker check); the build is otherwise Strata's own CMake, with its
  targets excluded from `all` and its headers as SYSTEM includes.

## Consequences

* Moving the pin means re-generating the patches against the new commit; `fetch_upstream.py` reports exactly which
  patch no longer applies.
* `scripts/test_fetch_upstream.py` (ctest) covers idempotence, refusal of foreign edits and failure cleanup on a
  synthetic upstream, in every build.
