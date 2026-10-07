# Grouped expert-domain prototype (P0-C)

**Status: experimental, isolated, deletable.** This is the principal alternative to the layer-domain pipeline in
[ARCHITECTURE.md](../ARCHITECTURE.md). Nothing in the production runtime depends on it. Every number it produces
is Synthetic (fixture model, localhost, simulated links) except the analytic block, which is a labelled
calculation. It never emits `Qualified`. The hardware questions are `HQ-P0C-01` and `HQ-P0C-02` in
[HARDWARE-QUALIFICATION.md](../../HARDWARE-QUALIFICATION.md).

## Topology

Dense and common operations stay on Father: embedding, PLE, every layer's mixer (recurrent or attention), router,
shared expert, output head, and all sequence state. The 512 routed experts of each layer are partitioned across
**owners**: owner 0 is Father itself, owners 1..R are expert domains.

For each layer of a window of q positions, Father

1. runs the q mixers and routers,
2. sends **one** `ExpertBatch` to every remote domain that owns at least one selected expert: the q activations
   (H floats each) plus, per position, the (local expert id, weight) pairs for the experts that domain owns,
3. executes its own experts and the shared expert while the domains compute,
4. receives **one** `ExpertResult` per domain: one weighted partial sum (H floats) per position,
5. combines the partial sums in a deterministic order (below).

That is one request/response pair per participating domain per layer, never one per expert, and about 48 sequential
barriers per target pass for Flash-Next. Domains are stateless (no sequence state, no commit traffic); Father keeps
the recurrent state, KV and token history.

This topology sends selected expert ids and weights on the wire, which the addendum allows for this topology only.
It never sends token IDs, logits or text, and nothing logs routes. See ADR 0121.

## Code map

| Path | Role |
|---|---|
| `experimental/expert-domains/include/clusterlm/expert_domains/wire.hpp` | `ExpertBatch`, `ExpertResult`, `ExpertError`, bounded codecs (`ExpertDecodeLimits`) |
| `.../assignment.hpp` | `ExpertAssignment`: contiguous ranges, strided, or explicit owner vector |
| `.../domain_server.hpp` | `ExpertDomainServer`: owns experts for every layer, serves one connection on a thread |
| `.../father_executor.hpp` | `FatherExecutor`: whole-model executor with the split MoE exchange, per-layer metrics |
| `.../rig.hpp` | `GroupedRig`: Father + R domains over real loopback transport, optional network impairment |
| `.../analytic.hpp` | analytic cost model (a calculation) |
| `.../simulation.hpp` | the comparison bench driven by `clusterlm-expert-domain-bench` |
| `src/expert_kernel.hpp` | the SwiGLU expert kernel, same code as the reference backend |
| `tests/expert_domains/` | wire bounds, assignment, equivalence with the reference, one message per domain, failures |

The transport (frames, TLS pinning, `impair()`, `FaultInjector`) is `runtime/transport` unchanged. Domain experts
are loaded through the objects API: each domain's resolver is plan-scoped and holds only its routed-expert objects,
copied from Father's `CanonicalModelStore` (`provision_expert_objects`). A domain that is missing one of its experts
refuses to start; it never fetches anything else.

## Numerical equivalence

The expert kernel and Father's mixer/router/head are copies of the reference backend's math built on its `refmath`
header, compiled with `-ffp-contract=off`.

- **One owner executes every selected expert of a position: bitwise identical** to the unsplit reference. Asserted
  over a schedule of windows with q = 1..4 and every rejection length, comparing full logits.
- **Several owners: not bitwise.** The reference sums routed outputs in a single float chain. Each owner returns its
  own chain and Father adds the chains, so the association differs. The combination order depends only on the route
  (owners ordered by lowest selected expert id, then the shared expert), never on arrival timing, so results are
  reproducible: asserted bitwise between a raw run and a jittered, non-overlapped run.
- Measured on the fixture: worst relative logit difference about 5e-7; tolerance 1e-5 (ADR 0120). Argmax identical.

The difference cannot be removed without serializing the owners or returning one result per expert (ADR 0120).

## Running the bench

```
clusterlm-expert-domain-bench --out results/grouped.json
clusterlm-expert-domain-bench --q 1,2 --presets gige-simulated --windows 24 --skip-layer-domain
clusterlm-expert-domain-bench --analytic-only
```

Options: `--remote-domains N`, `--q`, `--presets`, `--windows`, `--prompt-len`, `--strided`, `--no-overlap`,
`--no-reference`, `--skip-layer-domain`, `--layers/--hidden/--experts/--active/--expert-ff/--shared-ff/--seed`
(scale the generated fixture), `--tolerance`, `--work`, `--quiet`. Presets are the transport presets (`unlimited`,
`gige-simulated`, `gige-degraded`, ...). Father's egress is one shared simulated NIC; each domain has its own.

For every (preset, q) it runs the grouped design in-process and, unless skipped, the layer-domain design through the
production Coordinator with two `clusterlm-node` processes (prefill rounds of exactly q positions stand in for
q-wide verification rounds, direct peer forwarding). It then prints both side by side, plus the analytic model for
the real geometry, and writes a BenchmarkResult-schema JSON (reusing `bench/src/result.hpp`).

Metric names (all `Synthetic`): `grouped.<preset>.q<q>.*` and `layer_domain.<preset>.q<q>.*` give per-window / per-round
critical path, messages and bytes per layer and window, per-layer exchange and barrier-wait distributions,
`expert_union_growth` (distinct experts / selections), `moe_cpu_concurrency` (Father's local expert work plus the
domains' compute divided by MoE wall time), `moe_overlap_fraction`; `compare.<preset>.q<q>.net_gain_ms` is the
layer-domain round minus the grouped window (positive favours grouped); `analytic_calc.*` are calculations.
`grouped.layer_barrier_ms` and `grouped.net_gain_ms` are the HQ-P0C-01 headline metrics at the first q under
`gige-simulated`.

### LAN peer mode and quantized kernels (HQ-P0C-02)

Domains can be separate processes or machines over the production transport (mutual TLS with pinned fingerprints;
`--insecure-loopback` is loopback-only, for tests). Every process derives the same deterministic fixture model (same
`--seed` and geometry flags) and the same expert ownership, so a domain loads only the experts it owns; there is no
provisioning protocol in the prototype and nothing but the grouped wire messages crosses the link.

```
# on each domain machine (i = 1..R); serves one Father at a time and outlives its Fathers
clusterlm-expert-domain-bench --listen 0.0.0.0:7500 --domain-index i --remote-domains R \
    --identity <dir> --trust <father-fingerprint> [--expert-kernel iq3_s]
# on Father
clusterlm-expert-domain-bench --peer d1=HOST1:7500@<d1-fingerprint> --peer d2=HOST2:7500@<d2-fingerprint> \
    --identity <dir> --trust <d1-fingerprint> --trust <d2-fingerprint> --presets unlimited --q 1,2,3,4
```

`--listen` prints `EXPERT_DOMAIN_LISTENING endpoint=... device_id=...`; `--exit-on-stdin-eof` and `--max-sessions N` end it.
In peer mode only the `unlimited` preset applies (the link is real; impairment presets are simulations) and the result
stays Synthetic because the model is the fixture; `simulated.localhost_cluster` is false when a peer is not loopback.
Peer mode returns bit-identical logits to the in-process domains for the same ownership (`test_peer`).

`--expert-kernel iq3_s|iq2_xs` runs the routed experts (Father's and the domains') through the Strata CPU IQ kernels
(`strata-cpu` provider of the bench; needs `-DCLUSTERLM_ENABLE_STRATA_CPU=ON`, and `--hidden`/`--expert-ff` multiples
of 256; without the kernels it fails with `kHardwareUnavailable`, never falling back to FP32). The expert blobs are
synthetic pseudo-random i-quant blocks seeded by (seed, layer, owner), not model data, so this mode measures kernel
cost, barrier time and message/byte counts, not accuracy: the FP32 reference comparison is skipped and the result says
so (ADR 0340).

## Synthetic results (this build, fixture model: 16 layers, H = 64, 32 experts, 4 active; 2 domains)

| preset | q | grouped ms/window | layer-domain ms/round | grouped msgs/layer | grouped B/layer |
|---|---|---|---|---|---|
| unlimited | 1 | 1.5 | 1.0 | 3.3 | 976 |
| unlimited | 4 | 4.0 | 2.0 | 4.0 | 4312 |
| gige-simulated | 1 | 10.2 | 2.7 | 3.3 | 976 |
| gige-simulated | 4 | 13.0 | 4.0 | 4.0 | 4312 |
| gige-degraded | 1 | 22.9 | 4.7 | 3.3 | 976 |
| gige-degraded | 4 | 24.8 | 7.0 | 4.0 | 4312 |

These time the software on a 4-core development container where the fixture's compute is microseconds; they show
the barrier cost of 16 sequential exchanges, not hardware behaviour. Read them as: grouped pays a per-layer
latency floor (about 0.6 ms per layer at `gige-simulated`) that the pipeline pays only per stage hop, and the
expert union grows with q (distinct experts per layer 4.0, 6.9, 9.5, 11.6 for q = 1..4 against 4, 8, 12, 16
selections) so remote compute grows slower than q.

## Analytic model (calculation, not a measurement)

For the real geometry (48 layers, H = 2560, 512 experts, 10 active, expert ff 640, 2 remote domains). Inputs are
Synthetic: link from the transport presets (mean one-way latency = latency + jitter/2), CPU expert streaming rates
from `fixtures/profiles` (`cpu.expert_bytes_per_s`: Father 18 GB/s, slower Node 11 GB/s), IQ3_S taken as 3.4375 bits
per weight. **The addendum's §7 text is not in this repository;** the formulas below are derived from the geometry
and the brief, so replace them if §7 differs.

With owners = R + 1, s = 1 / owners, K active, E experts, q positions:

- expected distinct experts per domain per layer: `U = E s (1 - (1 - K/E)^q)` (uniform, independent routing),
- request bytes per domain: `24 + 28 + 4qH + 2q + 8 qKs`; response bytes: `24 + 28 + 4qH + 12`,
- participation probability: `1 - (1 - s)^(qK)`,
- remote compute per layer: `U * (3 * ff * H * bpw / 8) / expert_rate`,
- barrier: Father's egress serializes the R requests FIFO, each domain computes, responses serialize on Father's
  ingress; network floor is the same with zero compute,
- exposed wait = `max(0, barrier - Father local work)`; per pass multiplies by the layer count,
- layer-domain: 3 hops (Father -> A -> B -> Father) of `24 + 128 + q * 51,216` bytes plus latency each.

Results at q = 1 under `gige-simulated` (110 MB/s, 0.15 ms + 0.025 ms jitter mean): 48 barriers, ~189 messages and
1.95 MB per pass, network floor 30.3 ms per pass, exposed wait 36.6 ms per pass, against 3 messages, 154 KB and
1.9 ms of network time for the pipeline. At q = 4: 7.9 MB per pass, floor 70.6 ms, exposed wait 111 ms against 6.1 ms.
The network floor alone bounds grouped decode at roughly 33 passes per second at q = 1 on that link, before any
compute, which is the number HQ-P0C-01 has to confirm or refute. Real routing is skewed and correlated, so the union
grows slower than the model says at q > 1 (favourable) while hot experts unbalance the domains (unfavourable);
HQ-P0C-02 measures both.

## Limitations

- Expert compute is the FP32 reference SwiGLU on CPU by default; `--expert-kernel` selects the Strata CPU IQ kernels on
  synthetic blobs (cost, not accuracy). No GPU kernels.
- By default domains run as threads over loopback TCP; `--listen` / `--peer` run them as separate processes or machines
  (tested over loopback only, with mutual TLS and with plain loopback). Coordinator integration does not exist.
- One session, one outstanding window. A lost domain fails the window and breaks the executor (rebuild it).
- Positions with no routes to a domain are still sent in the batch (bytes, not messages).
- Routing is the fixture's, not the model's.

## Removing it

Delete `experimental/expert-domains/`, `tests/expert_domains/`, this file, ADRs 0120-0122, the
`experimental/expert-domains` line in the top-level `CMakeLists.txt`, the `-I bench/src` addition in
`scripts/check_windows_compile.sh`, and `HQ-P0C-02` (and the P0C-01 command edit) in the qualification registry
(then `python3 scripts/gen_qualification_md.py`).
