# ClusterLM Bench methodology

`clusterlm-bench` measures the numbers the placement scheduler consumes and writes them as `HardwareProfile` /
`NetworkProfile` files. This document says what each measurement means, how it is repeated, how a result becomes a
placement input and which provenance rules apply. The experiments that still have to be run on the real machines are
listed in [HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md); decisions behind the tool are in
`docs/adr/0150`–`0154`.

## Ground rules

- **Measured means measured on the machine that ran it.** Every result names `environment.host_role`:
  `development-host` unless the operator passed `--on-target` (an assertion that the machine is the named
  Father/Node; the tool cannot check it, qualification review does).
- **Provenance** in a result is `Synthetic` or `Measured`; the tool never emits `Qualified`. Anything simulated
  (loopback peers, `--impair` presets, localhost clusters, fixture models, synthetic profiles) forces `Synthetic`.
- **No fabricated numbers.** If a measurement is impossible here (no CUDA device, no Strata kernels) the command
  records a failed check, lists the pending HQ ids and exits 3. It does not substitute estimates.
- **Privacy.** Results contain sizes, counts and timings only: no token IDs, text, logits, prompts, activations or
  routing sequences. Expert selection in the CPU benchmark uses synthetic weights and synthetic indices.
- Exit codes: 0 success, 1 failed checks/errors, 2 usage (unknown command or flag), 3 hardware/backend absent.

## Repetition, warm-up and ordering

| Rule | Where |
|---|---|
| At least 5 measured repetitions per configuration (qualification runs); fewer is allowed for smoke runs and is recorded as `configuration.meets_min_repetitions = false` | all commands (`--reps`, `--iterations`, `--repeat`) |
| 1 warm-up repetition discarded for CPU/memory; first allocation/transfer discarded for GPU; 5 warm-up RTT exchanges per size | `cpu`, `memory`, `gpu`, `transport` |
| Repetitions are **interleaved** across configurations (thread counts, q, patterns; with `cluster --interleave`, contexts and q) so thermal drift and background load hit all of them alike | `cpu`, `cluster` |
| Each repetition lasts about `--target-s` (default 0.5 s) so timer resolution and scheduling noise are small | `cpu` |
| 512-token generations for throughput and acceptance claims (`--tokens 512`) | `cluster` (HQ-MTP-01, HQ-PERF-*) |
| Sustained runs are 30 minutes with a sample every `--sample-s` (default 10 s); shorter runs are recorded but do not populate `cpu.sustained_factor` | `cpu --sustained`, `cluster --minutes` |
| 20 release cycles per the addendum §14 | `faults --release-cycles 20` (HQ-REL-01) |
| Distributions are reported as n/mean/min/p10/p50/p90/p95/p99/max/stddev; profiles use medians | all |

## Measurements

### CPU (`cpu`, part of `calibrate`)

The workload is one MoE layer's routed-expert work for a verification window of `q` positions. Each position selects
`--active` experts (10 for Flash-Next) out of a bank sized to exceed the last-level cache (`--bank-mib`, default
1 GiB, at most 512 experts); the round executes the **union** of the selected experts once, each over the positions
that chose it, spread across worker threads (one expert per thread at a time).

- `cpu.expert_bytes_per_s.<representation>`: stored bytes of the union's experts per second of round time at q=1 on
  the best thread count. This includes dequantization and GEMV, which is what `HardwareProfile.cpu.expert_bytes_per_s`
  documents.
- `cpu.<rep>.dequant_fraction`, `.dequant_bytes_per_s_per_thread`, `.gemv_bytes_per_s_per_thread`: dequant cost
  separated from GEMV (per-thread rates; 0 for representations consumed in place).
- `cpu.<rep>.thread_scaling.t<N>`: bytes/s by thread count (sweep 1,2,3,4,6,8,12,16,... up to the logical CPUs).
  `best_threads` maximizes the median; `usable_threads` is the smallest count reaching 95% of the best, minus one CPU
  kept free when the machine has at least four, so the service and the user's session stay responsive.
- `cpu.<rep>.q<q>.<pattern>.*`: round time, bytes/s and mean union size for q = 1/2/4 and patterns `same`
  (union = active), `overlap` (half shared with the previous position) and `disjoint`.
- `cpu.q_scaling`: mean of `(t_q/t_1 - 1)/(q-1)` over q > 1 on the `same` pattern, the factor in
  `time *= 1 + q_scaling*(q-1)`. Weight traffic for the larger union is a separate term in the cost model and is
  visible in the `overlap`/`disjoint` rows. Placement carries one value, so the profile takes the largest over the
  representations measured.
- `--provider` selects the kernels: `reference` (scalar FP32 reference-backend math, always available) or
  `strata-cpu` (registered later by the Strata workstream; a stub exits 3 until then). `--isa` records the ISA the
  operator expects; a check fails if the provider selected another.
- **Sustained** (`--sustained --minutes M --sample-s S`): the q=1 workload on `usable_threads` for M minutes. Reports
  initial rate (median of the first up to 5 samples), final rate (median of the last quarter), `sustained_factor =
  min(1, final/initial)`, the slope in %/minute, `time_to_equilibrium_s` (first time the rate stays within 3% of the
  final rate for at least 3 samples; 0 if it never settles) and, on Windows, the AC/battery state with every sample.
  Ctrl-C stops early and the samples so far are reported.

### Memory (`memory`, part of `calibrate`)

- `memory.ram_total`, `ram_available`, page size; `commit_limit` (Windows `ullTotalPageFile`, Linux `CommitLimit`)
  and `commit_available`.
- **Allocation probe**: ascending single allocations (`--step-mib`) up to `min(--max-gib, hard cap 64 GiB,
  available - reserve)`, every page touched so overcommit cannot hide failure. The reserve is at least 2 GiB or 10% of
  RAM. `memory.alloc.stopped_by_policy` says whether the probe stopped at the ceiling or at a failed allocation.
  `largest_ok_bytes` is a probe result under policy, not a recommendation; `ram_safe_allowance` is a policy value and
  stays a Synthetic placeholder in profiles.
- Peak tracking: process peak working set before/after (`memory.peak_*`).
- `memory.read_bandwidth.t<N>`: read bandwidth with N threads over one buffer (each thread its own slice, buffer at
  most a quarter of available RAM); `memory.ram_bandwidth` is the best median.
- Pinned allocation limit: part of `gpu`/`pcie` (needs the CUDA runtime).

### GPU and PCIe (`gpu`, `pcie`, part of `calibrate`; CUDA build only)

Build with `-DCLUSTERLM_BENCH_CUDA=ON` and the CUDA toolkit installed (`cudart`, `cublas`; no nvcc needed).

- Device list: name, VRAM total/free (`cudaMemGetInfo`), compute capability, driver and runtime versions; on Windows
  the DXGI budget and current usage (`platform::GpuBudgetProbe`).
- Allocation overhead: `cudaMalloc`/`cudaFree` time for 1 MiB … 1 GiB.
- Transfers: H2D and D2H bytes/s for ring sizes 64 MiB, 256 MiB, 1 GiB, for pageable memory, `cudaHostAlloc` and
  `cudaHostRegister`, bounded by 80% of free VRAM and by host memory policy.
- Pinned limit: pin memory in steps up to a cap (`--pinned-cap`, default 4 GiB) and report the largest success and
  whether the cap or a failure stopped it.
- cuBLAS GEMV/GEMM for 640x2560, 2560x640, 2560x2560, 6144x2560, 512x2560, 2560x6144 at 1, 2, 4 and 512 positions, in
  FP32 and FP16: milliseconds, weight bytes/s and TFLOPS. These bound what the GPU can do; they are **not** the
  Strata layer kernels.
- Backend kernels register `GpuKernelCase`s in `GpuKernelRegistry` and are timed with device events.
- Without CUDA or without a device: `gpu.present=false`, the reason, exit 3, pending HQ-GPU-01/02 and HQ-PCIE-01.

### Network (`transport`, part of `calibrate`)

- Per message size (64 B, 51,216 B ≈ one Flash-Next position, 204,864 B ≈ a q=4 window, 1 MiB): RTT distribution,
  jitter (sample standard deviation) and the tail (`p99 - p50`), plus throughput per direction (`tx` this machine
  to peer, `rx` peer to this machine) from acknowledged bursts.
- Several simultaneous peers (`--peer a=HOST:PORT --peer b=HOST:PORT`): after the per-peer pass, bursts to and from
  all peers at once give `concurrent.egress_bytes_per_s` (Father serving two Nodes through one NIC).
- Real links use mutual TLS with pinned fingerprints by default (`--identity`, `--trust`, `--peer-id`); `--no-tls`
  exists for diagnostics. A `--peer` on this machine (loopback) is not a link: the result and profile are Synthetic.
- Node to Node: run `transport --serve` on Node B and `transport --peer B` on Node A. `--node-to-node` runs both on
  localhost as an orchestration check (always Synthetic).
- `--duration S` keeps sampling RTT for at least S seconds, split across the sizes.

### Runtime (`cluster`)

Father in-process, Nodes as separate `clusterlm-node` processes, fixture model, reference backend: always Synthetic.
It measures the software path and gives the harness for the real runs; the same command, pointed at the real artifact and
backend, is the real run (WP21, docs/adr/0350-bench-drives-real-backends.md):

- `--backend reference|strata|llama` with the option flags of `clusterlm-father`/`clusterlm-node` (`--cuda-device`,
  `--vram-reserve-mib`, `--strata-cpu-threads`, `--strata-ple-gguf FILE`, `--strata-mtp-dir DIR`, `--llama-gpu-layers`;
  `apps/common/backend_cli.hpp` parses them for both tools). Father's prefix/tail run on that backend and every spawned Node gets
  the same engine flags (Father-only options never reach a Node). A backend that is not built (or unknown) is a clear error
  before a model is written or a process spawned: exit 3 with the CMake option named (`built without
  CLUSTERLM_ENABLE_STRATA`), exit 2 for an unknown name. A Strata build without a usable CUDA device fails in `prepare`
  with `kHardwareUnavailable` (exit 3, recorded as the failed `cluster_pass_completed` check). The llama backend is
  Father-only (`--tier fast`, no Node stages, `--q 1`).
- `--model DIR` is a converted model directory instead of the generated fixture; Father reads it, Nodes receive only the
  plan-assigned objects from Father (a Node never takes a model path). `--no-reference` skips the Father-only reference run
  (needed when the model does not fit on Father alone); the reference runs happen before the cluster starts so a real backend
  never holds two models in device memory.
- `--node NAME=HOST:PORT@FINGERPRINT` (repeatable, with `--identity DIR` for Father's persistent identity, which the Nodes pin)
  uses Nodes that are already running on the other machines instead of spawning localhost processes: real links. Node memory
  series and the staging census are not available for them (their own release reports - resources released, storage cleaned,
  residual bytes - are recorded; run `storage-census` on that machine).
- Provenance: the result is `Measured` only when every input is real: `--model` (not the fixture), a real backend, `--on-target`,
  the default (not scripted) drafter, no `--impair`, no folded tokenizer ids, and no Node on loopback (`--node` endpoints on
  other machines, or a Father-only plan, which has no links). Anything simulated is listed under `simulated`
  (`localhost_cluster`, `fixture_model`, `reference_backend`, `scripted_drafter`, `network_preset`, `tokenizer_ids_folded`) and
  forces `Synthetic`; `--on-target` with localhost Nodes is not applied (`configuration.on_target_not_applied`). The tool never
  emits `Qualified`.
- `--corpus FILE|DIR --tokenizer-gguf FILE` tokenizes the corpus on Father with `father::GgufBpeTokenizer` (docs/tokenizer.md;
  the GGUF's metadata only). A directory gives one prompt per file; a file longer than twice the prompt length is cut into up
  to 8 consecutive windows. The text and token ids stay in the Father process: results carry counts only (`corpus`,
  `corpus_tokenizer`). A tokenizer larger than the model's vocabulary is refused for a real model; on the fixture model its ids
  fold into the 256-token range and the result says so (`tokenizer_ids_folded`, Synthetic). Without `--tokenizer-gguf` the
  bytes are the ids (fixture only). `placement-inputs` takes the same flags.
- Per-domain state (HQ-PERF-02): at release every domain reports the sequence state it actually had allocated (KV, recurrent,
  PLE) and its per-session window scratch, as the peak over the lease (`ReleaseComplete`, docs/protocol.md): metrics
  `state.father.stage<S>.bytes` / `.window_bytes`, `state.<node>.stage<S>.bytes` / `.window_bytes`, `state.total_bytes`,
  `state.total_window_bytes`, `state.max_context`. They are sized at the plan's `max_context` (the lease's allocation, not the
  prompt in use). The reference domain reports its real buffers; StrataDomain reports Strata's `session_bytes` and
  `Verifier::init_bytes` (the same carve the engine allocates with; it needs no device to compute).

- Provisioning: per Node `bytes`, `objects`, Father-view `prepare ms`, Node-side prepare time, wire bytes/s;
  `--phase prepare --repeat N` repeats cold prepare/release cycles.
- Per q (and per `--context`): prefill ms and effective prompt tok/s, decode tok/s, round time, cold and warm TTFT,
  accepted tokens per round, boundary bytes per emitted token.
- MTP: `mtp.proposed_positions` (q-1 per round), `mtp.accepted_positions`, `mtp.acceptance_rate`; draft, verify
  (prefix + remote + tail) and commit times.
- Per stage: Father compute (prefix + tail), each Node's compute (and CPU-expert time) from `StageTiming`, and remote
  wait = remote wall time minus the Nodes' reported compute; `stage.wait_fraction` is total wait over total round time.
- `--tier fast|strong|ultra` chooses a layer split (Ultra = Father + two Nodes), `--compare-routing` runs the plan
  with direct Node-to-Node forwarding and with Father relay, `--minutes` runs a sustained generation loop,
  `--corpus` as above.
- Correctness is checked against a Father-only reference on every run (unless `--no-reference`: then no
  `*_tokens_match_father_only_reference` check is made and `configuration.reference_check` is false); activation messages are
  checked to stay within the boundary ABI.
- `--contexts 8192,32768,...` runs a context sweep in one invocation with every result keyed `ctx<N>.q<q>.*` (even for
  one context); `--context` keeps the older keying (prefixed only for more than one). Per-context VRAM margins need the
  backend and are not recorded here.
- Inter-token gaps of the streamed output: `q<q>.inter_token_gap_ms` (per token; tokens of one `on_tokens` delivery are 0
  apart, so q > 1 shows a mass at 0) and `q<q>.inter_delivery_gap_ms` (between deliveries: the stall a reader sees), both
  with p50/p90/p99/max. Timestamps and counts only, never token values.
- Observers (OS and driver counters, never simulated; a source the OS cannot give is `unavailable: <reason>`, not 0):
  - `resources.<father|nodeN>.{rss_bytes,commit_bytes}` as distributions plus `.series` arrays (decimated beyond 2000
    points, `resources.series_stride` says by how much), `.rss_growth_bytes`/`.commit_growth_bytes`, sampled at
    generation start/end, every prepare cycle and (throttled by `--sample-ms`, default 200) every round;
    `--no-resources` disables it. Windows: `GetProcessMemoryInfo` (WorkingSetSize, PrivateUsage = commit charge); Linux:
    `/proc/<pid>/status` (VmRSS; VmData is the commit proxy, Linux has no per-process commit figure).
  - `nic.*` and `generation.nic.*` (with `--compare-routing`, prefixed `direct.`/`relay.`): rx/tx bytes, packets, errors
    and drops of Father's interface over the whole pass and over generation only, beside `boundary.payload_bytes_total`.
    The interface is `--nic NAME` or that of the route to the first Node (Linux `/proc/net/route`; Windows
    `GetBestInterfaceEx` + `GetIfEntry2`). On a localhost cluster this is `lo`, which counts both directions of every
    process on the host, so the wire/payload ratio is only meaningful on a real NIC.
  - `nvml.*`: see below. `census.*` (with `--census`): see `storage-census`.
- Provisioning breakdown per Node (HQ-PROV-01): Father `father_source_read_ms` (`stream_object` minus its callback),
  `father_chunk_digest_ms`, `father_send_ms` (includes transport backpressure, i.e. the transfer), their bytes/s, and
  Node `node_chunk_write_ms`, `node_seal_hash_ms`, `node_build_ms` from `PlanReady` (docs/protocol.md).
- `faults --only a,b` runs a subset: scenario names (`crash_<phase>`, `father_lost`, `local_activity`, `link_loss`,
  `stall`, `release_cycles`), a bare lifecycle phase (selects `crash_<phase>`) or `crash` (all of them); an unknown name
  is a usage error (exit 2) and the result records `scenarios_run`. `release_cycles` records the resource series per
  cycle (`release_cycles.resources.*`). `faults` takes `--backend` and its flags and `--model` like `cluster`.
- `faults --supervised` (or `--only supervised`) adds the HQ-REL-01 scenarios in which Node 0 runs under
  `clusterlm-node-service --console --simulate-activity` (its own `--helper-pipe` per instance), which supervises the
  worker as the Windows service does in a Job Object, and the bench counts the service's events:
  `supervised_forced_termination` (local activity while Ready, the worker hangs in its cleanup, misses the cooperative
  deadline `--deadline-ms`, default 1500, and is force-terminated and relaunched: `forced_terminations`,
  `activity_to_forced_termination_ms`, `residual_bytes_at_relaunch`) and `supervised_crash_recovery` (the worker dies in its
  cleanup after Father's release: `worker_relaunches`, `release_to_relaunch_ms`). In both, node 0 stages its objects on disk
  (`staged_bytes_before_fault` > 0), the relaunched worker's orphan recovery must leave nothing staged, the Node must be
  offered again (`offers`) and the next lease must generate correctly; two cycles each. The Job Object itself is Windows-only
  (docs/windows-service.md); here the supervisor is the same `NodeSupervisor` over a child process.

### Placement prediction vs measurement (`placement-validate`, HQ-PLACE-01)

`clusterlm-bench placement-validate [--top N] [--q Q] [--prompt-len L] [--max-new N] [--repeat R]` runs the placement search
(profiles as for `placement`, costed for the model that will actually run: `cost_inputs_from_manifest`), takes the top-N
ranked candidates, converts each to a `ClusterPlan` (`planning::to_cluster_plan`, including its GPU/CPU residency targets and
the VRAM each Node is planned to use) and runs it through the same cluster pass as `cluster` on a fresh set of Nodes. Per
candidate `cand<i>.predicted.{decode_tok_s,prefill_s,prepare_s,objective}` sit beside `cand<i>.measured.*` and
`cand<i>.measured_over_predicted.*`, the pass's own metrics are under `cand<i>.`, and `trace` has one `placement_candidate`
entry. `rank.{decode_tok_s,prefill_s,prepare_s}.*` says how well the predicted order matches the measured one: Spearman rank
correlation, the share of concordant candidate pairs and whether the predicted best is the measured best (`valid` is false
for a constant series or fewer than two candidates). On the fixture model with the Synthetic profiles this is Synthetic and
exercises the machinery only (localhost processes of one machine are not the target hardware, and the shipped profiles
describe quantizations the fixture does not use: the fixture's get a Synthetic stand-in throughput). The qualification run
uses `--model`, `--backend strata`, measured profiles and `--endpoint PROFILE_ID=HOST:PORT@FINGERPRINT` (repeatable, with
`--identity DIR`): each candidate then runs on the already-running Nodes of the profiles it places, in its own node order,
over real links, and the result is `Measured` only with `--on-target`, a real model and backend, no loopback endpoint and
no Synthetic profile field.

### GPU telemetry (`nvml`, also inside `cpu --sustained`, `calibrate --sustained-minutes`, `cluster --minutes`)

NVML is loaded at run time (`libnvidia-ml.so.1` / `nvml.dll`, override with `CLUSTERLM_NVML_LIBRARY`), with no link-time
dependency and no CUDA toolkit. Per GPU: SM and memory clocks, power, temperature, memory used (also as VRAM series in
`resources.gpu<N>.vram_used_bytes`), throttle reasons seen and the fraction of samples with an involuntary slowdown
(power cap, thermal, hardware). Without NVML the result carries `nvml.unavailable = "unavailable: <reason>"` and
`clusterlm-bench nvml` exits 3. `clusterlm-bench nvml --minutes 30 --sample-s 5` is the companion to run beside a
workload that has no sampler of its own (HQ-GPU-03).

### Power environment

`calibrate`, `profile` and `cpu --sustained` record the Windows active power plan (`power.plan_guid`, `power.plan_name`)
or the Linux cpufreq governor (`power.governor`); the calibrated profile's `note` says which plan it was measured under.

### Cache and temp census (`storage-census`, `faults` unless `--no-census`, `cluster --census`)

Before/after snapshots of the CUDA compute cache (`%APPDATA%\NVIDIA\ComputeCache`, `~/.nv/ComputeCache`,
`CUDA_CACHE_PATH`), NVIDIA shader caches, OS temp (shallow), ClusterLM per-user data and the Node staging roots. Names,
sizes and modification times only. The diff lists new/changed/removed files (`in_window` says whether a changed file's
mtime falls inside the run); a temp file another program wrote during the run is indistinguishable, and kernel file
tracing (ETW/procmon) is not collected. `storage-census --before --snapshot F`, then the workload, then
`--after --snapshot F` (or `--diff --snapshot A --against B`); `--root LABEL=PATH`, `--staging-root PATH` and `--exclude PATH`
adjust the locations.

## From results to placement inputs

`calibrate` runs CPU, memory, GPU (if present) and optionally network, then writes
`<out-dir>/profile-<machine-id>.json` and `network-<machine-id>.json`, plus the run's own result file.

| Profile field | Source |
|---|---|
| `cpu.expert_bytes_per_s[<rep>]` | `cpu.expert_bytes_per_s.<rep>` median |
| `cpu.usable_threads`, `cpu.q_scaling` | `cpu.usable_threads`, `cpu.q_scaling` |
| `cpu.sustained_factor` | sustained run of at least 30 minutes only |
| `cpu.features`, `cpu.arch` | CPUID + OS-enabled state |
| `memory.ram_total`, `memory.ram_bandwidth` | OS total; best read bandwidth |
| `memory.pinned_limit` | pinned probe (CUDA build) |
| `gpu.vram_total`, `gpu.vram_budget` | `cudaMemGetInfo` total; DXGI budget if known, else free VRAM |
| `gpu.pcie_h2d_bytes_per_s` | pinned H2D median at the 256 MiB ring |
| link `bandwidth_bytes_per_s` | the slower direction at the largest message |
| link `rtt_ms`, `jitter_ms` | p50 and stddev of the 64 B round trip |
| `father_egress_bytes_per_s` | concurrent egress with two or more real peers |
| everything else (safe allowance, dense layer times, GPU expert rate, prefill rate, overheads) | not measurable by the tool: Synthetic placeholder or the base file's value |

Placement loads these files unchanged (`clusterlm-bench placement --father profile-father.json --node ... --network
network-father.json`, or `placement::load_hardware_profile`). Each quantity keeps its provenance, a plan is only as
trustworthy as its weakest input, and the `placement` result is `Measured` only when every input quantity is.

### Provenance rules

1. Measured fields: `Measured`, source `bench:<run-id>@<machine>`.
2. Unmeasured fields: Synthetic placeholder, or the existing value.
3. Never Qualified, never promoted beyond Measured; `mark_qualified` fails while any field is Synthetic.
4. Updating an existing file (`--out-dir` file or `--base`): Synthetic is replaced by Measured; Measured only by a
   newer-or-equal run; Qualified never; Synthetic incoming never replaces anything; hand-entered Measured values are
   kept and reported. See `docs/adr/0151-profile-provenance-and-merge.md`.
5. Development-host and loopback results are labelled as such and must not be copied to a target machine.

## Running it

```
# development host smoke run (seconds)
clusterlm-bench calibrate --quick --machine-id dev-test --role father --out-dir results

# target machine (Windows, CUDA build): everything measurable, plus the links to both peers
clusterlm-bench calibrate --machine-id father --role father --on-target \
    --peer node-g14=HOST:7400 --peer node-3060=HOST:7400 --identity <dir> --trust <fp> --out results/calibrate-father.json

# sustained thermal run on the laptop (30 minutes)
clusterlm-bench cpu --provider strata-cpu --representations iq3_s --sustained --minutes 30 --on-target --machine-id g14
```

Each experiment's exact command is in the registry (`bench/qualification/experiments.json`), checked by the test suite
against the implemented command surface.
