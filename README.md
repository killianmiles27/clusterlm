# ClusterLM

Distributed heterogeneous local LLM inference for Windows.

| Component | Role |
|---|---|
| **ClusterLM Father** | User-facing coordinator and primary inference machine; the only permanent model store. |
| **ClusterLM Node** | Idle-machine compute worker. Holds only lease-scoped, plan-assigned model objects. |
| **ClusterLM Runtime** | Heterogeneous distributed inference runtime (execution domains, stage boundary, transport). |
| **ClusterLM Bench** | Hardware, transport and runtime qualification suite. |

- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- What is done and what waits for hardware: [docs/DEVELOPMENT-STATUS.md](docs/DEVELOPMENT-STATUS.md)
- Experiments that need the physical machines: [HARDWARE-QUALIFICATION.md](HARDWARE-QUALIFICATION.md)
- Backends: [docs/backends/strata-port.md](docs/backends/strata-port.md), [docs/backends/llama-local.md](docs/backends/llama-local.md), [docs/backends/llama-rpc.md](docs/backends/llama-rpc.md)
- Product: [docs/ui.md](docs/ui.md), [docs/pairing.md](docs/pairing.md), [docs/father-ipc.md](docs/father-ipc.md), [docs/windows-architecture.md](docs/windows-architecture.md), [docs/packaging.md](docs/packaging.md), [docs/tiers.md](docs/tiers.md)
- Runtime: [docs/protocol.md](docs/protocol.md), [docs/provisioning-lifecycle.md](docs/provisioning-lifecycle.md), [docs/model-manifest.md](docs/model-manifest.md), [docs/placement.md](docs/placement.md), [docs/benchmark-methodology.md](docs/benchmark-methodology.md), [docs/security/threat-model.md](docs/security/threat-model.md)

> **Status:** pre-hardware. Everything that can be built and verified without the three target PCs is implemented
> and tested on Linux (and type-checked or tested on Windows in CI): the distributed runtime, the Strata-derived
> backend (CPU kernels verified, CUDA path compiled but never run on a GPU), the llama.cpp Fast backend, the
> Windows service/helper/UI/installer, pairing, provisioning and the qualification tooling. **No performance number
> in this repository is a measurement of the target machines.** Fast/Strong/Ultra are not performance-qualified
> until the experiments in HARDWARE-QUALIFICATION.md run on the real hardware.

## Build

Requirements: CMake ≥ 3.24, a C++20 compiler (MSVC 2022 for the product; GCC 13 / Clang 17 for development),
OpenSSL 3, Python 3 (tooling only).

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build                 # unit + multi-process integration tests
ctest --test-dir build -LE integration # unit tests only
```

On Windows: `vcpkg install openssl:x64-windows`, then configure with
`-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake`.

Upstream sources (Strata, llama.cpp) are pinned in `third_party/upstream.json` and fetched on demand. They are
never committed:

```sh
python3 scripts/fetch_upstream.py --apply-patches   # pinned commits into third_party/upstream/, plus our patches
python3 scripts/fetch_upstream.py --check
```

Optional backends (off by default):

| Option | Builds |
|---|---|
| `-DCLUSTERLM_ENABLE_STRATA=ON` | Strata CUDA engine + CPU kernels (CUDA 12.x, sm_86/sm_89) |
| `-DCLUSTERLM_ENABLE_STRATA_CPU=ON` | Strata CPU expert kernels only (no CUDA) |
| `-DCLUSTERLM_ENABLE_LLAMA=ON` | llama.cpp Fast-tier backend and the P0-A RPC harness (`-DCLUSTERLM_LLAMA_CUDA=ON` for GPU) |

A backend that is not built is an error at `--backend` selection, never a silent fallback.

## Try it: a three-process cluster on one machine

```sh
B=build/bin
$B/clusterlm-fixture-model --out /tmp/clm/model          # deterministic generated model, no real weights

$B/clusterlm-node --name a --listen 127.0.0.1:7001 --staging /tmp/clm/a --insecure-loopback &
$B/clusterlm-node --name b --listen 127.0.0.1:7002 --staging /tmp/clm/b --insecure-loopback &

$B/clusterlm-father --model /tmp/clm/model --node a=127.0.0.1:7001 --node b=127.0.0.1:7002 \
    --plan 0-4@father,4-10@0,10-13@1,13-16@father --max-new 32 --q 4
```

Father keeps the prefix (embedding, PLE) and the tail (head, sampling, MTP). It provisions only layers 4–10 to
Node a and 10–13 to Node b, runs speculative windows across the Nodes, with direct Node→Node forwarding, and
releases both leases. Each Node then reports zero staged bytes.

- Mutual TLS: use `--identity DIR` and `--trust FINGERPRINT` instead of `--insecure-loopback`. The fingerprint is
  printed on the Node's first stdout line.
- Simulated 1GbE: add `--impair gige-simulated`. This is a simulation parameter, not a measurement.

## ClusterLM Bench

```sh
$B/clusterlm-bench cluster --q 1,2,4 --drafter scripted:0.3 --impair gige-simulated --out cluster.json
$B/clusterlm-bench faults --out faults.json         # crashes at every lifecycle phase, activity, link loss, stalls
$B/clusterlm-bench faults --supervised --out f.json # + Nodes under clusterlm-node-service: forced terminations, relaunches
$B/clusterlm-bench placement-validate --top 4       # top placement candidates executed: predicted vs measured, rank agreement
$B/clusterlm-bench transport --tls --impair gige-simulated
$B/clusterlm-bench placement --context 4096 --q 4   # placement over the synthetic target profiles
$B/clusterlm-bench profile                          # this host's CPU features, RAM, bandwidth
$B/clusterlm-bench qualification                    # the hardware qualification registry
```

`cluster`, `faults`, `placement-inputs` and `placement-validate` take `--backend reference|strata|llama` (with the engine flags of
`clusterlm-father`), `--model DIR` for a converted model and `--corpus FILE --tokenizer-gguf FILE` for real text; `cluster --node
NAME=HOST:PORT@FINGERPRINT` uses Nodes that are already running. A backend this build does not have fails before anything starts
(exit 3). Only a real model, a real backend, real links and `--on-target` give `Measured`
([docs/benchmark-methodology.md](docs/benchmark-methodology.md)).

Results follow [`bench/schema/benchmark-result.schema.json`](bench/schema/benchmark-result.schema.json).

- A run that used a localhost cluster, a fixture model, synthetic profiles or a simulated network is labelled
  `Synthetic`.
- Only real-link and real-hardware runs are `Measured`.
- The tool never emits `Qualified`.

## Product executables (Windows)

| Executable | What it is |
|---|---|
| `clusterlm-father-ui` | Father chat window: tiers, readiness, tok/s, TTFT, fallback, diagnostics, pairing |
| `clusterlm-father-agent` | Father background agent; the UI talks to it over a per-user named pipe |
| `clusterlm-node-service` | Node Windows service: supervises the worker in a Job Object, pairing, settings |
| `clusterlm-node-helper` | Per-session helper: reports local activity/lock to the service |
| `clusterlm-node-ui` | Node status window and tray icon |
| `clusterlm-node`, `clusterlm-father` | The worker and a command-line Father (development and qualification) |

Installers: `packaging/` (WiX MSIs; unsigned builds are marked `-UNSIGNED`, never silently signed).

## Repository layout

The module table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#modules) lists every directory and target.

## License

ClusterLM's own license has not been chosen yet; that is the owner's decision. Vendored third-party code keeps its
own license: `third_party/doctest` (MIT), `third_party/nlohmann` (MIT) and `third_party/imgui` (MIT). The pinned upstreams Strata and
llama.cpp are MIT.
