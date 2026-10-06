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
- Backend port analysis: [docs/backends/strata-port.md](docs/backends/strata-port.md), [docs/backends/llama-rpc.md](docs/backends/llama-rpc.md)

> **Status:** cloud-development phase. The distributed architecture runs end to end on localhost with the
> deterministic CPU reference backend and generated fixture models. **No performance number in this repository is a
> measurement of the target machines.** Fast/Strong/Ultra throughput is unqualified until the experiments in
> HARDWARE-QUALIFICATION.md run on the real hardware.

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
python3 scripts/fetch_upstream.py          # fetch the exact pinned commits into third_party/upstream/
python3 scripts/fetch_upstream.py --check
```

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
$B/clusterlm-bench transport --tls --impair gige-simulated
$B/clusterlm-bench placement --context 4096 --q 4   # placement over the synthetic target profiles
$B/clusterlm-bench profile                          # this host's CPU features, RAM, bandwidth
$B/clusterlm-bench qualification                    # the hardware qualification registry
```

Results follow [`bench/schema/benchmark-result.schema.json`](bench/schema/benchmark-result.schema.json).

- A run that used a localhost cluster, a fixture model, synthetic profiles or a simulated network is labelled
  `Synthetic`.
- Only real-link and real-hardware runs are `Measured`.
- The tool never emits `Qualified`.

## Repository layout

```
runtime/common        status, codec, digests, ids, logging
runtime/objects       geometry, manifest, canonical store, fixture models
runtime/domain        stage boundary ABI, ExecutionDomain, reference backend, drafters
runtime/windows       speculative-window ledger (windows = speculative windows)
runtime/transport     framed TCP, mutual TLS, network impairment, fault injection
runtime/protocol      Father/Node messages and channels
runtime/platform      file mapping, durable files, safe deletion, processes, Windows adapters
runtime/backends      Strata handoff layout; Strata/llama.cpp adapters (CUDA, optional)
node/lease-store      ephemeral lease object store, journal, orphan recovery
node/worker           Node service core
orchestrator/placement   hardware profiles, cost model, placement search
orchestrator/coordinator Father orchestration
orchestrator/planning    manifest -> cost inputs, placement plan -> executable plan
bench                 ClusterLM Bench and the LocalCluster harness
apps                  clusterlm-node, clusterlm-father
fixtures/profiles     SYNTHETIC development profiles of the three target machines
```

## License

ClusterLM's own license has not been chosen yet; that is the owner's decision. Vendored third-party code keeps its
own license: `third_party/doctest` (MIT) and `third_party/nlohmann` (MIT). The pinned upstreams Strata and
llama.cpp are MIT.
