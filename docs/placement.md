# Placement

The placement engine (`orchestrator/placement`, `orchestrator/planning`) decides which domain runs which
layers, which experts live on which GPU, how large each domain's local micro-batch is, and when a deployed
plan should be replaced. It consumes measured profiles and ranks candidates under an explicit cost model.
**It never measures anything and never claims optimality on real hardware.** Every number it reports carries
the provenance of the weakest input it used.

## Pipeline shape

`Father prefix -> Node 1 -> ... -> Node K -> Father tail`, K = 0..`max_remote_nodes` (default 3), plus the
Father-only plan. Father owns embedding + PLE (layers `[0, ple_layer]` must stay in the prefix) and
head + sampling + MTP; each Node holds exactly one contiguous range. A Node never appears twice.

## Cost model

Per layer `l` with routing frequency `f_e` (selection probability of expert `e` for one position) and `q`
verified positions: `p_q(e) = 1 - (1 - f_e)^q`; expected expert bytes touched per round
`U_l = sum_e bytes_l * p_q(e)`.

- **Residency.** A domain with a GPU reserves dense + state + scratch + local-batch workspace, then fills the
  rest with experts by critical-path time saved per VRAM byte: `p_q(e) * (1/cpu_eff_l - 1/gpu_bw)`,
  `cpu_eff_l = expert_bytes_per_s[quant_l] * sustained_factor`. GPU-resident union `G_l`; CPU-served
  `M_l = U_l - G_l`.
- **Stage time (decode round).** `dense = sum dense_layer_ms[kind] * (1 + dense_q_scaling*(q-1))`,
  `cpu = sum M_l / cpu_eff_l * (1 + q_scaling*(q-1))`, `gpu = sum G_l / gpu_bw`,
  `stage = dense + cpu + gpu - overlap_factor * min(cpu, dense + gpu)` (`overlap_factor = 0`: serial).
- **Transport.** `q * boundary_bytes / bw + rtt/2` per hop (including the last stage back to Father);
  `commit = max RTT(Father <-> node)`.
- **Round and decode.** `round = draft_ms + sum stage + sum transport + commit`;
  `decode_tok_s = 1000 * A / round`, A = acceptance (mean emitted tokens per round) for this q.
- **Prefill.** Per stage and chunk: `chunk / (prefill_rate * n_layers / stage_layers) * 1000 +
  ceil(chunk / local_batch) * (CPU-resident expert bytes touched by `local_batch` tokens) / cpu_eff * 1000`;
  links add `chunk * boundary / bw`. Chunks pipeline: `prefill_s = (first_chunk_path +
  (n_chunks - 1) * slowest_element) / 1000`, over `prompt_tokens`.
- **Preparation.** Provisioning shares Father's NIC: `provisioning_s = max(sum bytes / egress,
  max_node bytes_i / link_i)` (not parallel-additive); `upload_s` = slowest domain's `(dense +
  gpu_experts) / pcie_h2d`; `prepare_s = provisioning_s + upload_s`. Layers listed in `already_provisioned`
  cost no bytes.
- **Lease amortisation.** A request bears `share_i = min(1, output_tokens / expected_tokens_i)` of node `i`'s
  provisioning; a node whose own preparation exceeds `expected_lease_s_i` adds the excess to
  `lease_overrun_s` (flag `lease_overrun:<node>`).
- **Objective (lower is better).** `preparation * (effective_prepare_s + lease_overrun_penalty *
  lease_overrun_s) + throughput * (prefill_s + output_tokens / decode_tok_s)`.

Memory ledgers (bytes) per domain: VRAM = dense + gpu_experts + state + scratch + batch_workspace (a domain
without a usable GPU keeps dense + state in RAM); RAM = cpu_experts + staging + os_reserve (+ father_only on
Father). A plan is admitted only if every ledger fits `vram_budget` and `ram_safe_allowance`, every needed
throughput/latency quantity exists (missing measurements reject, never default), and every link exists.

### Local micro-batch (ADR 0250)

Each domain runs the cluster chunk in micro-batches sized by its own VRAM headroom (at most
`batch_headroom_fraction` of it, power of two). The smallest GPU shrinks only its own stage's batch and
therefore its own CPU-expert streaming in prefill, never the cluster chunk. `local_batch < max(min_local_batch,
q)` rejects the domain.

### Context, state and q

`context_tokens` is the capacity state is sized for (attention KV grows with it, recurrent state is fixed);
`prompt_tokens` is what is prefilled. Growing state displaces GPU experts and can make a profile infeasible.
`search_workload` evaluates profiles 4K/8K/16K/32K/64K/128K and q = 1..4 (acceptance per q is an input,
Synthetic unless measured) and merges per context (ADR 0253).

## Search space

- **Prefix sizes:** `prefix_layers_options` plus, with `sweep_prefix`, the smallest legal prefix
  (`ple_layer + 1`) and every multiple of `granularity` up to that + `prefix_sweep_span`.
- **Cuts (node range ends, hence Father tail size 0..N):** `c in (prefix, N)` with `(c-prefix) % g == 0` or
  `c % g == 0` or `(N-c) % g == 0`, plus `N` (empty tail).
- **Nodes:** every ordered subset of up to `max_remote_nodes` nodes (`allow_node_orders = false`: request
  order only), ranges strictly increasing.
- **Residency, batch, provisioning reuse:** computed per candidate from the above; GPU residency ties break to the
  lower layer then the lower expert id after snapping scores to 12 digits (deterministic, no hysteresis).

**Complexity.** `O(P * sum_{k<=K} P(N_nodes, k) * C(|cuts|, k))` stage assignments (P prefixes). 48 layers,
g = 4, 3 nodes, prefixes {3, 4, 8}: ~15,000 (the odd prefix 3 dominates). Each is costed in O(stages * layers) from memoised per-(domain, ranges)
admission/residency (`DomainEval`, computed once, including its digest), so the whole search takes well under
a second in RelWithDebInfo (about 0.25 s measured on the dev host for 3 nodes with the prefix sweep); the test
bound is generous for Debug CI.

## Pruning

Structural: a stage that fails admission rejects the plan before any cost model runs (reason recorded).
Dominance (`prune_dominated`): drop a candidate dominated on all of `prepare_s`, `effective_prepare_s`,
`lease_overrun_s`, `prefill_s`, `decode_tok_s` -- the exact inputs of the objective and the frontier, so the
recommended plan and the frontier are provably unchanged (ADR 0251). Two-phase costing means pruned candidates
never allocate ledgers/residency/hash.

## Pareto and recommendation

`PlacementResult::pareto` is the 3-D frontier (min `prepare_s`, max `decode_tok_s`, min `prefill_s`), also
reported per context in `WorkloadResult`. `recommended` is the lowest objective (ties: layer boundaries, then
hash -- never a domain name). For a workload (`WorkloadDescription`: expected prompt/output tokens, requests
per lease) the recommendation comes from the smallest profile covering prompt + output; lease expectation
becomes `requests_per_lease * output_tokens` tokens.

## Provenance

Every quantity is Synthetic, Measured or Qualified. A plan's provenance is the weakest of the profile fields
it **uses** (CPU throughput only for the quants and dense timings only for the layer kinds it hosts), the
links it traverses, Father egress (if it has nodes), model structure (bytes, routing), `draft_ms` and
acceptance. `non_qualified_inputs` names what holds it down (including the field paths). Placement tools
label results Synthetic or Measured; only `mark_qualified` (after an acceptance run) produces Qualified, and
`require_qualified` refuses anything else. Nothing here reads a device name.

## How measurements flow in

1. `clusterlm-bench calibrate` writes `HardwareProfile` / `NetworkProfile` JSON with per-field provenance
   (`load_hardware_profile`, `load_network_profile`); fields it did not measure stay Synthetic.
2. Routing statistics: `planning::load_routing_aggregates(path)` reads `clusterlm.routing_aggregates.v1`
   (aggregate counts or frequencies per layer x expert; a closed schema that rejects token, sequence, prompt or
   per-request fields; a file cannot claim Qualified). `apply_routing_aggregates` +
   `cost_inputs_from_manifest` put them into `ModelCostInputs` (provenance and source recorded).
3. Draft time and per-q acceptance are supplied as `Quantity` values; batch workspace per token is a model
   input (`ModelCostInputs::batch_scratch_bytes_per_token`, Synthetic in the estimate).
4. `planning::placement_report_json(request, result)` / `(request, workload_result)` render reports for
   Bench: provenance banner, input provenance, frontiers, recommendation, infeasible contexts.

## Replan triggers (ADR 0252)

`planning::should_replan(current_plan, ReplanInputs)` re-costs the deployed assignment in the new world and
compares it with the best candidate found with the current layers marked already provisioned. It reports
reasons (node unavailable, plan inadmissible with the admission reasons, context growth, decode/prefill
regression naming the slowed stage or link, preparation/lease-overrun regression, better plan available,
no feasible plan), whether to replan, and the bytes/seconds of reprovisioning. Replanning for quality needs
`min_objective_gain` (default 10%). Scenarios covered by tests (direction only): laptop CPU 30% slower,
3060 limited to 8.8 GiB, Father->G14 at 72 MB/s, G14 leased 25 minutes, a node needing 15 GB reprovisioning.

## Pending qualification

All of the above is a model. **HQ-PLACE-01** runs the top candidates on the real machines and compares
predicted with measured decode/prefill/prepare, node orders, prefix/tail sizes, per-context feasibility and local
batches; **HQ-PLACE-02** measures the model-side inputs (routing aggregates, batch workspace, draft time,
acceptance per q). Until they pass, no plan is Qualified and the 25% batch headroom cap, the 10% replan gain
and the overlap factor are policy defaults, not findings.
