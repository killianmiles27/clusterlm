# llama.cpp local backend (Fast tier)

Pin: llama.cpp `6753a033f058fbf778d282556ed9b16c78de7c71`. Design: ADR 0300. Code: `runtime/backends/llama/`.

## Build
`python3 scripts/fetch_upstream.py llama.cpp`, then `cmake -S . -B build -G Ninja -DCLUSTERLM_ENABLE_LLAMA=ON`.
The pinned library is built from source as static libraries (CPU backend, `GGML_NATIVE=OFF`, RPC client/server
compiled in). `-DCLUSTERLM_LLAMA_CUDA=ON` adds `GGML_CUDA` (archs 86;89; with nvcc 12.0 the host compiler is forced to
`g++-12`). The CUDA build is compile-checked here but cannot run here (no GPU); behaviour on the RTX 4060 Ti is
HQ-TIER-01. Default builds (option OFF) do not need the checkout. CI job `linux-llama-cpu` builds and tests the CPU
variant.

## How Fast runs
`make_llama_backend({model_dir, n_gpu_layers, ...})` is passed to the Coordinator as `CoordinatorConfig::backend`
with the plan `0-L@father,L-L@father`. `build_llama_manifest` / `write_llama_model_dir` produce `manifest.json` from
the GGUF files (exact byte ranges; stored tensor types are reported in `LlamaModelReport`).

| Contract item | llama.cpp mapping |
|---|---|
| model load | `llama_model_load_from_file/splits` on Father's canonical files, mmap by default |
| sequence state | one `llama_seq_id` per session; KV for `max_context` x `max_sessions` (unified) |
| window | one `llama_batch`, positions `base..base+q-1`, logits for every position (up to 32 positions) |
| commit(n) | `llama_memory_seq_rm(seq, base+n, -1)` |
| abort_window | `llama_memory_seq_rm(seq, base, -1)` |
| state versions, stale epochs | `WindowLedger` in each domain |
| release | frees context, batch and model; `llama_live_counts()` returns to zero |

## Limitations
- Father only; middle stages are refused. MTP drafting is independent of this backend.
- `describe_requirements` gives estimates; exact memory is only in llama.cpp's own log.
- Prefill windows return only the last position's logits.
- A refused partial KV removal re-decodes the prefix (slow, correct); not tested here on a recurrent architecture.
- GPU offload (`n_gpu_layers`) needs the CUDA build; unverified on target hardware.

## Tests
`tests/backends_llama/` (ctest `test_llama_backend`) uses a tiny random-weight llama GGUF (`write_tiny_llama_gguf`, F32
or F16): Coordinator greedy generation equals llama.cpp's own decode; q=1..4 with every acceptance length equals q=1
tokens; commit/abort/replay/stale epoch; logits bitwise equal to llama.cpp for the same batch and within 1e-4
(observed about 1e-6) for other batch shapes; conversation KV reuse; two sessions; release.
