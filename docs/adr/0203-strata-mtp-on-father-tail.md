# ADR 0203: Strata's MTP drafter on the Father tail, verified as a deterministic proposal

Status: accepted (WP6)

## Context

Strata's `MtpDrafter` consumes token IDs and the main model's final residual rows (`Verifier::final_R_all`) and
drafts with its own layer, K/V and a draft-vocabulary head. ClusterLM's speculative sampling
(`verify_speculative`, Leviathan/Chen) needs, for stochastic sampling, the full drafter distribution `Q_j` each draft
was sampled from. Strata reports only the scalar probability of each draft token, and its coupled draft sampling
(shared Philox draws with the target) is a different exactness mechanism that ClusterLM's Father-side sampler does
not use.

## Decision

* The drafter lives on the Father tail domain (`make_strata_mtp_drafter(tail)`), next to the head; nothing it reads
  or produces crosses the network. The tail engine implements `MtpEngine`; `StrataMtpDrafter` adapts it to
  `domain::Drafter`.
* A round catches the drafter up over the tail's last committed window (rows `R_{p+t}` paired with the token at
  `p+t+1`, taken from Father's committed history and the next token) and drafts greedily with Strata's own chain.
* `propose()` returns those drafts with `probs` empty: ClusterLM treats them as one-hot `Q`, which keeps stochastic
  speculative sampling exact (the emitted text is distributed as the target alone) at the cost of acceptance under
  sampling. The adapter accepts real distributions from an engine that supplies them (tested with a fake).
* Engine failures fall back to a valid deterministic proposal; generation never fails because of the drafter.

## Consequences

* Greedy decoding gets Strata's full MTP benefit; sampled decoding gets correct but lower acceptance until a
  sampled-chain variant (draft-head logits per step through `chain_launch` forcing) is qualified - HQ-MTP-02.
* Prompt cells before the first draft are caught up only for the last prefill window; the drafter's K/V for earlier
  prompt cells stays empty (lower early acceptance, never wrong tokens).
