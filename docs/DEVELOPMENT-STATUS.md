# Development status

This is the repository-wide review at the end of the pre-hardware phase. It records what is implemented, how it is
verified, what is simulated or reference-only, the known defects, and every remaining item classified as either
**Hardware Qualification Required** or **Still Implementable Here**. Nothing in this repository is a measurement of
the Father, G14 or 3060 machines.

## Verification that runs on every push (`.github/workflows/ci.yml`)

| Job | What it proves |
|---|---|
| `linux-gcc-release` | Full build + every ctest (unit, multi-process integration, fault suite, fuzz-corpus replay, tokenizer vocab tests against llama.cpp's own files) |
| `linux-clang-sanitizers` | The same suite under ASan + UBSan |
| `linux-clang-tidy` | Product code clean under `.clang-tidy` (bug patterns, Clang static analyzer, performance); zero findings enforced |
| `linux-strata-cuda-build` | Pinned Strata + ClusterLM patch series builds with CUDA 12.0 (sm_86/sm_89); CPU kernels tested; GPU tests registered as disabled (no GPU) |
| `linux-llama-cpu` | Pinned llama.cpp Fast backend: generation equals llama.cpp's own decode; RPC baseline harness on loopback |
| `windows-msvc` | MSVC build of everything + the full test suite on a Windows runner |
| `windows-mingw-syntax` | Every Windows-specific source compiled with `-Werror` (fast proxy) |
| `windows-packaging` | Static `/MT` build, no non-system DLL imports, WiX MSIs built, ICE-validated, installed, verified and uninstalled on a Windows runner |
| `qualification-doc` | `HARDWARE-QUALIFICATION.md` is generated from `bench/qualification/experiments.json` |

Both MSI smoke tests pass on a Windows runner (CI run 38, all nine jobs green). Node: install, LocalService
account, delayed auto start, recovery actions, SID type, preshutdown timeout, firewall rule scope (worker program,
TCP, no Public profile, no edge traversal), helper Run value, service start, staging ACL, orphan cleanup and complete
uninstall. Father: install, executables, catalog and licenses, agent Run value, PATH, the CLI starting with no runtime
DLL installed, and complete uninstall.

## Genuinely implemented

| Area | State and evidence |
|---|---|
| Distributed runtime | Father prefix → Node A → Node B → Father tail; token-free middle stages; fixed boundary ABI; direct Node↔Node or relay; mutual TLS 1.3 with pinned identities; three channels per Node. Bitwise-identical tokens versus Father-only execution for every plan, q and acceptance length (`test_transactions`, `integration_*`) |
| Speculative windows | `WindowLedger` (epochs, window ids, one outstanding window, commit 1..q, idempotent replay, `AbortWindow`, abort session); tested for every accept 0..q, aborted transport, worker cancel, stale epoch, duplicate commit, lost ack |
| MTP and sampling | Greedy and exact stochastic speculative sampling (accept min(1,P/Q), residual resample, bonus token), chi-square tested end to end; q 1–4 metrics (acceptance, tokens/round, draft/verify/commit time) |
| Prefill | Chunks pipelined across stages under bounded in-flight backpressure; per-stage local batches; cancellation drains in-flight chunks; per-stage instrumentation |
| Strata backend | `StrataDomain` over a `StrataEngine`; patches 0001–0006 (CUDA 12.0, token-free stages, `abort_window`, per-domain commit, memory-backed weights, init sizing); CPU expert kernels verified against ggml-cpu; model conversion; MTP drafter on the Father tail; `--backend strata` on Father, Nodes and bench. CUDA engine compiled, **never run on a GPU** |
| llama.cpp backend | Fast tier as a Father-only domain; logits bitwise equal to llama.cpp for the same batch shape; P0-A RPC baseline harness |
| Model ownership | GGUF v2/v3 parser, manifest builder (expert = 3 ranges), plan-scoped manifests, bounded streaming reads, chunk/object digests, RAM / GPU-target / temporary-file placement, no silent requantization (a conversion is refused unless pre-converted) |
| Provisioning | Resume after a broken bulk stream, cancel, temp-file fallback, no persistent cache, no LAN weight paging, crash-safe journal and orphan recovery, census = 0 bytes after every fault scenario |
| Placement | Provenance (Synthetic/Measured/Qualified, never auto-promoted), full candidate search, Pareto frontier, replan triggers, routing aggregates, `placement-validate` (predicted vs measured with rank correlation) |
| Father tokenizer | GGUF byte-level BPE (qwen2/qwen35 pre-tokenizers) matching llama.cpp's vocab tests exactly (46/46, 50/50); ChatML with Qwen3 thinking toggles; no fallback to a byte tokenizer |
| Windows | Service host, session helper, WTS/power notifications, Job Objects, named-pipe IPC with DACLs and peer checks, firewall rules, owner-only key and staging ACLs, `--apply-startup` |
| Product | Father chat UI (tiers, readiness, tok/s, TTFT, fallback, diagnostics, pairing), Node UI with tray; Father agent JSON API; SPAKE2 pairing bound to the TLS channel; versioned owner-only settings; unpair notification; provisioning progress |
| Packaging | Two WiX MSIs, static runtime, honest signing (unsigned builds are marked `-UNSIGNED`; nothing is signed without credentials) |
| Security and privacy | 13 fuzz targets with corpora replayed in CI; no generic execution path from Father to Node; token/text never on the wire or in logs (traffic and log tests); redacting diagnostics; threat model in `docs/security/threat-model.md` |
| Bench | Profiles, CPU/memory/GPU/PCIe probes, NVML telemetry (dynamically loaded), NIC counters, memory sampling, storage census, cluster/faults/supervised faults, baselines, placement inputs/validation; results labelled by real provenance |

## Simulated or reference behaviour (not the real backend)

- The **reference CPU backend** and **generated fixture models** drive all multi-process tests; the real model
  artifacts are not in this environment.
- **Strata CUDA execution** is compiled only. On a machine without a CUDA device `prepare` returns
  `kHardwareUnavailable`. End-to-end Strata runs in CI use a fake engine behind the real `StrataDomain`.
- **Network**: localhost with `transport::impair` presets; `gige-simulated` is a parameter, not a measurement.
- **Hardware profiles** in `fixtures/profiles/` are Synthetic; every result derived from them is labelled Synthetic.
- **Windows-only behaviour** that CI cannot exercise (interactive sessions, real lock/suspend, pairing over a LAN,
  UI rendering/DPI/tray) is type-checked and tested through mocks on Linux.

## Known defects and limitations

1. Strata MTP drafts are verified as one-hot proposals: Strata exposes only each draft's probability, not the full
   draft distribution. Sampling stays exact; acceptance is lower than optimal (ADR 0203).
2. Strata prefill goes through the Verifier in sub-batches of at most 8 positions, not Strata's batched prompt path;
   a window split into sub-batches can only be committed whole.
3. `describe_requirements` for Strata reports object sizes before binding; the exact on-device arena is known only
   after `prepare`.
4. llama.cpp prefill windows longer than 32 positions return only the last position's logits (ADR 0300).
5. The tokenizer supports the qwen2/qwen35 pre-tokenizers and ChatML only (no Jinja engine) and has not been run on
   a Qwen3.8 vocabulary.
6. The `PlanReady`/`ReleaseComplete` extensions assume Father and Node ship together (an old decoder rejects the
   longer body; docs/protocol.md "Extending messages").
7. Node settings saves restart the worker under the service lock (the pipe call blocks about a second); a failed
   restart after a successful save reports `kInternal` with the document already persisted.
8. On a localhost cluster the NIC counters read `lo`, so the wire/payload ratio is meaningful only on a real NIC.
9. The UI's bundled font is Latin-only; there is no screen-reader support (ADR 0320).

## Hardware Qualification Required

Every entry in [HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md) (41 entries, each with an exact command) is
pending. Grouped:

- **Throughput and tiers**: HQ-PERF-01..04, HQ-STRONG-01, HQ-FAST-01, HQ-TIER-01 — Fast/Strong/Ultra tok/s, the 20+
  tok/s Ultra target, long context, cold/warm TTFT, sustained runs.
- **Backend on GPU**: HQ-GPU-01..03, HQ-GPU-05, HQ-NUM-01, HQ-MTP-01..02, HQ-P0B-01, HQ-P0D-01 — Strata kernels,
  VRAM margins, abort on GPU, numerics on the real artifact, MTP acceptance.
- **CPU and memory**: HQ-PROF-01, HQ-CPU-01..02, HQ-PCIE-01 — expert throughput, thermals, pinned memory.
- **Network and provisioning**: HQ-NET-01..02, HQ-PROV-01, HQ-REL-01, HQ-STORE-01.
- **Placement**: HQ-PLACE-01..02 — calibration of the cost model against measured runs.
- **Baselines and alternatives**: HQ-P0A-01, HQ-P0C-01..02.
- **Windows and product**: HQ-WIN-01..04, HQ-SEC-01..03, HQ-INSTALL-01, HQ-PAIR-01, HQ-UI-01, HQ-MODEL-01.

Remaining tool gaps recorded in the registry are all of this kind (they need the real artifact, a CUDA GPU, the real
Windows service/SCM loop, ETW tracing on Windows, or interactive observation).

## Still Implementable Here

None of significance. Items deliberately left out because they cannot be verified without the hardware, or are not
needed by the selected models:

- Exposing Strata's full MTP draft distribution (a further Strata patch) — only verifiable on a GPU.
- Strata's batched prompt path for prefill — only verifiable on a GPU.
- A Jinja chat-template engine — the selected Qwen models use ChatML, which is implemented natively.
- clang-tidy over the test sources (product code is enforced).
