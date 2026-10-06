# Strata port analysis

Pin: `https://github.com/Niko1221/Strata` @ `1735d6471df29b42c26170efaac1f1446a58640f` (MIT, commit date
2026-10-06, version 0.1.40). All `file:line` references are at this pin, relative to
`third_party/upstream/strata/` (fetched by `scripts/fetch_upstream.py`, never committed).

## 1. Headline findings

1. **Hand-off buffer layout differs from `boundary.hpp` (ABI mismatch, adapter must convert).**
   Strata's hand-off is `handoff_floats(g) = hc*n_embd + n_embd + hc` floats per token
   (`include/strata/core/verify.hpp:126`), which matches `BoundaryLayout::floats_per_position()`. The field
   *order* also matches (streams, pending block output, injections). But the buffer is **field-major per
   window, not position-major**: for a window of `T` rows the buffer is
   `[ R: T x (hc x n_embd) ][ bo: T x n_embd ][ inj: T x hc ]`
   (reader `src/core/verify.cpp:682-688`, writer `src/core/verify.cpp:1335-1340`; offsets `T*HC*N` and
   `T*(HC+1)*N`). `boundary.hpp` specifies position-major records (`[streams|pending|inj]` repeated per
   position). Within the streams block, stream index is the slow axis: `Rt(t) = R_ + t*HC*N`
   (`verify.cpp:649`), i.e. token, then stream, then channel.
   Consequence: the Strata adapter must transpose between the wire layout (position-major, ClusterLM) and the
   mapped hand-off buffer (field-major, depends on `T`). Cost is one host copy of 51,216 B/position for
   Flash-Next; no precision change. Alternative (changing the ABI to field-major) is worse: the wire layout
   would depend on window length and partial-window slicing/resend gets harder. Recommendation: keep
   `boundary.hpp` as is. Treat the transpose as adapter responsibility with a round-trip unit test.
   The `hrow0*HB` base offset (`verify.cpp:682`) only applies to batch slot groups, not to speculative windows.
2. **PLE needs token IDs at the stage that holds layer 1.** `ple_stage()` is `lb_ <= 1 && 1 < le_`
   (`verify.hpp:344`); that stage hashes the window's tokens plus `ss.ple_prev` into n-gram rows
   (`verify.cpp:1695-1705`, `ngram_rows`) and advances `ple_prev` from token IDs at commit
   (`verify.cpp:2036-2040`). A middle stage that does not hold layer 1 never uses token IDs. Therefore
   "token IDs never leave Father" holds only if **the Father prefix owns layers 0 and 1** (PLE is a layer-1
   module, `session.hpp:67-88`). The placement planner must enforce `prefix.layers.end >= 2`, and the n-gram/PLE
   table (tens of GB per the comments at `expert_source.hpp:690-697`) stays Father-only.
3. **No abort-without-commit for a window.** `Verifier::commit(n_keep)` requires `1 <= n_keep <= last_t_`
   (`verify.cpp:2012`). The window itself *mutates* indexer tails (`verify.cpp:986`) and PLE history
   (`ple_history_advance`, `verify.cpp:795`). Snapshots exist for the indexer tail (`tail_snap_`, `verify.cpp:942`)
   and for PLE history *after each token* (`hist_snap_[t]`, `verify.cpp:800`), but **not for the PLE history before
   the window**. Strata never needs an abort because a verify window always accepts at least token 0.
   ClusterLM's `abort_session` is fine (discard everything), but any "drop this window, keep the session"
   path (stale epoch after a partial window, failed downstream stage) needs a new snapshot of pre-window PLE
   history plus an indexer-tail restore. Until then the adapter must map such a case to `abort_session`.
4. **Commit is not idempotent.** `commit()` replays GDN state from stored inputs and advances `ple_prev`
   (`verify.cpp:2010-2042`); calling it twice with the same count advances twice. ClusterLM's idempotent-commit
   contract must be implemented in the adapter (cache the `CommitAck` by window ID; never re-call).
5. **Middle stages need a token-free entry point.** `Verifier::run(T, tokens, pos0, ...)` always calls
   `stage_inputs`, which dereferences `tokens[t]` unconditionally (`verify.cpp:1615-1636`, `1693`) and every
   stage in the chain receives the token array (`verify.cpp:1844`, `next_->run(T, tokens, ...)`).
   `stage_inputs` copies tokens into `h_tok_` (consumed only by the stage holding layer 0) and into
   `last_tokens_` (consumed only by `ple_stage()` commit). Positions, step counts and RoPE position rows are
   derived from `pos0` alone (`verify.cpp:1621-1628`). So for a middle stage the copy is dead data, but the
   pointer must be non-null. Minimal patch: let `run` accept `tokens == nullptr` when `lb_ > 1`
   (skip the token copies). Without the patch the adapter must pass a zero-filled dummy array.
   **Verified: `Verifier::stage_inputs` copies tokens for intermediate stages** (it does not look at the stage
   range at all).

## 2. Build and test record (Linux, CPU only)

Environment: Linux x86_64, 4 cores, GCC 13.3, CMake + Ninja, no CUDA toolkit installed, no GPU.
Out-of-tree build dirs under the session scratchpad (the checkout stays clean).

`cmake -S third_party/upstream/strata -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF`

* Default configuration: `STRATA_BUILD_TESTS` defaults OFF because the published tree has no
  `tests/CMakeLists.txt` and no `bench/micro` (`CMakeLists.txt:95-99`). Targets available: `strata-gguf`,
  `strata-dequant`, `strata-plan`, `router_dot_parity`, `iq_avx2_parity` (plus ggml-base/ggml-cpu,
  `strata_kernels_cpu`). All built, 0 compiler warnings in the log. `router_dot_parity`: 0 failures;
  `iq_avx2_parity`: 0 failures; `strata-plan` ran and printed a memory plan; `strata-gguf`/`strata-dequant`
  print usage (need a GGUF).
* With `-DSTRATA_BUILD_TESTS=ON` (forced): 23 ctest tests configured and built (91 ninja steps); **21 passed, 2
  failed**. Failures: `expert_parity` and `pool_test`, both with
  `cannot read expert 0 (layer 0) from pack/full/experts.bin` — they need the ~34 GB model pack, which is
  absent here. They are environment failures, not code failures; not investigated further.
  Passing: gguf_reader_test, gguf_split_test, tuning_header_arch_family_test, arch_defaults_test,
  suffix_drafter_test, controller_test, draft_policy_test, message_boundary_test, conv_cache_test,
  coupled_draft_test, bf16_bits_test, dequant_q5_1_test, router_dot_parity(+_avx), iq_avx2_parity(+_iq3s_mt1),
  pool_affinity_test, expert_multi_test, q2_bitplane_parity, pool_stress, direct_file_async_test.
* **Not built, cannot be built here:** everything under `if(STRATA_ENABLE_CUDA ...)` — `strata_core`
  (device.cu, pinned.cu, verify.cpp, session.cpp, layer.cpp, ...), `strata_kernels`, the prefill library,
  `generate.cpp` (the `strata` executable), the Verifier, and all CUDA parity tests. verify.cpp et al. include
  `<cuda_runtime.h>`; the Verifier/session seams below were therefore analysed by reading only, not by running.
* Nothing about GPU performance or correctness of the CUDA path was measured.

## 3. Seam table

| Seam | Where (pin) | What it does | Change needed for ClusterLM domain extraction |
|---|---|---|---|
| **session** | `include/strata/core/session.hpp:31-89` (`SessionState`), `:98-104` (`session_bytes`/`session_init`), `:111` (`session_release`) | Owns per-layer sequence state: GDN recurrent state, QSA KV + indexer, scratch, PLE history/`ple_prev`. Already layer-range carved (`layer_lo/hi`, `qsa_ord0`, `gdn_ord0`, lines 50-62); `session_bytes` is pure arithmetic and prices a candidate range. | Use `session_bytes(g, max_ctx, k, lb, le)` to produce `DomainRequirements::state_bytes`. One `SessionState` per ClusterLM session; `max_sessions>1` needs multiple sessions (Strata has a slot mechanism, `init_slots`, `verify.cpp:2058`, greedy only/no MTP). Wrap `session_init` memory in the domain; never expose pointers. |
| **verify** | `include/strata/core/verify.hpp:57-345`; `src/core/verify.cpp:365` (`init`), `:1679` (`run`), `:2010` (`commit`), `:1615` (`stage_inputs`) | The speculative window: T tokens (`T <= kVerifyMaxT = 8`, `include/strata/kernels/verify_kernels.hpp:21`) through a layer range `[lb_, le_)` as a captured CUDA graph; `set_stage(lb, le, handoff_in, handoff_out)` (`verify.hpp:119`) makes stage-split windows; `set_next` chains stages in-process (`verify.cpp:1844`, `2042`). | This is the core of the `ExecutionDomain` adapter: `run_window` -> `run` with host-side hand-off buffers (convert layout, finding 1); `commit_window` -> `commit` (idempotency + abort in adapter/patch, findings 3-4); drop `set_next` chaining (each domain drives only its own Verifier; the transport carries the hand-off). `q <= 8` bounds `max_window`. Token-free entry for middle stages (finding 5). Replace `std::string& err` with `Status`. |
| **expert_source** | `include/strata/core/expert_source.hpp:124` (`ExpertSource::blob`), `:433` (`FileExpertSource::open(pack_dir,...)`), `:702` (`ArenaExpertSource::open`); `src/core/expert_source.cpp:541-575` (`CreateFileMappingW`/`MapViewOfFile` and `mmap`) | Supplies per-(layer, expert) blobs to the CPU expert pool, from `<pack>/experts.bin` or from mapped GGUF shards. Windows path already exists. Read-only file mapping. | Experts are addressed by filesystem path. For the lease store, give the domain a directory (or a `ByteSource`) holding only plan-assigned objects; add a constructor taking object-store handles rather than `pack_dir`; ensure the lease store deletes (and unmaps) on release; `ArenaExpertSource` (anonymous memory) is the better fit for ephemeral leases. Expert sets are a per-layer-range subset, not the full model. |
| **expert_cache** | `include/strata/core/expert_cache.hpp:81` (`ExpertCache`), `:68-79` profile ranking; `src/core/expert_cache.cpp` | Device-resident expert slots with a residency table; learned profile (`STRP` file, `write_expert_profile`) chooses which experts live in VRAM. | Cache must be sized from the domain's own free VRAM and restricted to the domain's layer range (Strata already builds per-stage caches, `GpuStage::cache`, `generate.cpp:990-1006`). Profile files are expert-selection data: under the privacy rule they must not be logged or leave the owning machine. |
| **pinned** | `include/strata/core/pinned.hpp:24-31` (`PageBacking`, `PinnedArena`), `:74` (`LoadStats`); `src/core/pinned.cu` | Host arena for experts: large pages (`VirtualAlloc(MEM_LARGE_PAGES)` on Windows, hugetlbfs on Linux) with `cudaHostRegister`, best-effort and reported. | Mostly reusable. Windows large pages need `SeLockMemoryPrivilege`; the fallback is the common case and must be reported in `BackendInfo`/diagnostics. Budget `staging_bytes` from this arena. |
| **remote_experts** | `include/strata/core/remote_experts.hpp:18` (`RemoteExperts`), `peer_experts.hpp:35` | A static expert tier on *another CUDA device in the same process* (CUDA1..3); results return through pinned CPU rows. Not a network feature. | Out of scope for the first adapter (single GPU per node). Do not confuse with cross-machine experts; keep disabled (`remote_count = 0`, `expert_source.hpp:257`). Multi-GPU nodes could later become several domains or one domain with an internal tier. |
| **exchange_storage** | `include/strata/core/exchange_storage.hpp:17-114` | Slot-ownership exchange so that experts can swap between the resident complement and the exchange area without copying (non-owning views, uniform mapped slots). | Internal to `FileExpertSource`; no cross-domain exposure needed. Verify it only reports in-process counters. Keep inside the adapter. |
| **weights** | `include/strata/core/weights.hpp:110` (`WeightTable`), `:131` `load(pack_dir, arena, bytes, err, skip)`, `:116` `pool_bytes`; `src/core/weights.cpp:92,124,148` (reads `<pack>/index.txt` with `std::fopen`) | Loads dense weights into one device arena from a pack directory index; `skip` set + `pool_bytes` already allow loading only a layer range (`generate.cpp:2916-2947`, `add_foreign`, `NativeDense::set_layer_range`). | Loader is path-based and uses ANSI `fopen` (Windows-unsafe for non-ASCII; comment at `weights.cpp:75`). Add a loader taking lease-store object handles/streams; reuse `skip` to restrict to the domain's range. Dense weights are the `gpu_weight_bytes` in `DomainRequirements`; `pool_bytes` (`weights.hpp`, "readable without loading anything") gives admission numbers before allocation. |
| **mtp** | `include/strata/core/mtp.hpp:36` (`MtpDrafter`), `:45` `load(rt_dir,...)`, `:76` `bind(wt, head, window_R, ...)`, `:92` `draft(...)`; `src/core/mtp.cpp` | Draft layer on the GPU: consumes token IDs and the main model's **final** residual (`final_R_all`, `verify.hpp:238`), produces drafts; its own K/V. | Father-tail-only: it needs token IDs and the last-layer residual, so it lives on the tail domain beside the head. ClusterLM only needs the tail's `final_R` to cross into the drafter locally; nothing about MTP crosses the network. The drafter allocates ~0.9 GB before the expert tier is sized (`mtp.hpp` comment on `load`) — account in tail `DomainRequirements`. `rt_dir` path loading must go through the lease/object store (Father-local). |
| **generate.cpp: `GpuStage`** | `src/program/generate.cpp:990-1010` | One later layer-split stage on its own CUDA device: own `WeightTable`, `NativeDense`, `NativeHead`, `SessionState`, stream, `ExpertCache`, `Verifier ver` (+`ver_b` for pipelined windows), `Prefill sp`. This is the closest existing analogue to a ClusterLM domain. | Lift this struct (minus global `Drive`/CLI coupling) into the `strata` backend's `StrataDomain`. Stage construction order matters: weights before the host arena is mapped, session after layer range is known (comments at `generate.cpp:2543`, `2960-2965`, `3003`). Prefill (`Prefill sp`) is a separate path from `Verifier`; ClusterLM prefill chunks need to go through whichever is used, and the hand-off format for prefill must be checked separately (not verified here). |
| **generate.cpp: `SplitDrive`** | `src/program/generate.cpp:964-987` (`SplitDrive`, `drive_pool_split`) | Selects, per layer, which stage's GPU plan, cache base and PCIe share the (single, shared) CPU expert pool uses; all stages share one `Drive` (counters, failure flags) and one process-wide CPU pool. Hand-offs are `cudaHostAlloc(Mapped|Portable)` buffers, `kVerifyMaxT * handoff_floats` floats each (`generate.cpp:6040-6052`). | Cross-machine: no shared pool/`Drive`; each domain needs its own `Drive`-equivalent (CPU pool, counters). The stage chaining (`set_stage`/`set_next`, `generate.cpp:6055-6087`) is replaced by the ClusterLM transport; hand-off buffers stay local per domain (mapped pinned memory is device-visible). `drive_pool_split`'s layer->stage lookup becomes trivial (one stage per domain). |

## 4. Window state commit and rollback (verified by reading)

Contract (`verify.hpp:3-20`, `verify.cpp:2010-2042`, `capture_commit` `verify.cpp:1543-1612`):

* **Window run** (`record_window`): K/V for all T positions appended at `pos0..pos0+T-1`; indexer keys appended
  for all T positions (`verify.cpp:986`), so indexer tail/pooled state is changed; GDN conv/recurrent state is
  **left untouched** — the per-row `qkv`, `h`, `gate`, `beta` are stored in per-layer buffers (`qkv_L_`, `h_L_`,
  `gate_L_`, `beta_L_`, `idx_raw_L_`); PLE history advances in place (`verify.cpp:795`) with per-token snapshots
  (`hist_snap_`, `verify.cpp:800`). The indexer tail before the window is snapshotted (`tail_snap_`, `:942`).
* **Commit(n_keep)**: a captured graph (`commit_exec_`) fed by host array `h_commit_ = [n_keep, n_keep-1, pos or -1 ...]`
  (`verify.cpp:2014-2016`). GDN: conv history and state advanced by replaying the first n_keep rows
  (`gdn_conv_commit`, `gdn_step_norm_multi`, `:1571-1575`). Indexer: tail restored from `tail_snap_` then the
  accepted keys re-appended (`:1581-1590`) — a rejected key may sit in a slot the current block still needs.
  PLE history: set to `hist_snap_[n_keep-1]` via `copy_indexed` (`:1595`). Host side: `ple_prev` advanced
  from `last_tokens_` (`:2036-2040`).
* **Rollback of rejected rows**: implicit. K/V cells past the accepted prefix are overwritten when those
  positions are processed again before any query reads them (`verify.hpp` header comment). There is no explicit
  rollback API (see findings 3 and 4).
* One-token windows with `one_token_self_commit()` advance state during the window and skip the commit graph
  (`verify.cpp:2017-2018`); a ClusterLM commit of such a window must still be accepted as a no-op acknowledged
  commit (and cannot be un-done).
* `commit` is asynchronous when `g_commit_async` and no `next_` (`verify.cpp:2026`); `wait_commit`
  (`:2048`) must be called before any other reader of the session. The adapter must wait before answering
  `commit_window` or exporting state.

Mapping to ClusterLM `ExecutionDomain` contract (`execution_domain.hpp`): one uncommitted window per session
(matches Strata's single `last_t_`/staged state), `window_id` and `StateVersion` are adapter-level bookkeeping
(Strata has no such concepts), stale-epoch rejection is adapter-level.

## 5. Open items / not verified

* CUDA-only code was read, not compiled or run. Layout statements (finding 1) are from source; a GPU-side
  round-trip test of the hand-off conversion is required before any claim of bit-exactness across a machine
  boundary.
* Prefill path (`src/prefill/prefill.cpp`, `generate.cpp` prompt chunking) uses a different code path than
  `Verifier::run`; its stage hand-off, if any, was not analysed here.
* `init_slots`/batch windows (`verify.cpp:2058-2477`) were skimmed only.
* Strata packs weights in its own "pack" format (`index.txt` + `experts.bin`), produced by `tools/` scripts;
  how ClusterLM model objects map to pack files was not analysed (belongs to the objects module).
