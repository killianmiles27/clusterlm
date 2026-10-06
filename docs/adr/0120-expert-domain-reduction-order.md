# 0120. Reduction order of the grouped expert-domain topology

## Context

The reference domain computes the routed output of a position as one float chain in ascending expert id,
`y = (((0 + w1 d1) + w2 d2) + ... + wK dK)`, then adds the shared expert. The grouped topology (experimental,
`experimental/expert-domains/`) has each owner return ONE weighted partial sum per position, so the chain is cut
into per-owner pieces that Father adds together. Float addition is not associative, so the grouped result can differ
from the reference in the last bits. The brief asks for bitwise identity where it can be reproduced and a justified
tolerance otherwise.

## Decision

1. Each owner accumulates its selected experts in ascending (wire-enforced, strictly ascending) expert order starting
   from 0, with the reference kernel (`refmath`, `-ffp-contract=off`). Father then adds the owners' partial sums in an
   order that is a function of the route only: owners sorted by their lowest selected expert id (for contiguous range
   ownership this is simply owner order), then the shared expert. Arrival order never matters.
2. When ONE owner executes every selected expert of a position the computation is exactly the reference chain, and the
   tests assert BITWISE identical logits over multi-window q = 1..4 schedules with every rejection length.
3. With several owners the result is not bitwise identical. It cannot be made so while the owners run in parallel and
   return a single partial per position: the reference chain is sequential. Chaining the running accumulator through
   the owners would serialize the remote work and defeat the design. Per-expert results would meet the bitwise goal but
   break the "one weighted partial sum per position" contract and multiply the response bytes by about K.
4. The tolerance is `max |grouped - reference| / max |reference| <= 1e-5` on logits per window. Observed worst case
   on the 16-layer fixture over the q = 1..4 schedules is about 5e-7, i.e. a few float ulps of the logit scale; the bound leaves 20x
   margin. Argmax equality is asserted separately. Determinism independent of timing is asserted bitwise (jittered
   network and overlapped/non-overlapped local work give identical bits).

## Consequences

- Speculative acceptance (greedy argmax) can in principle flip on near-ties relative to the layer-domain design. The
  effect is of the order of the reference's own sensitivity to any reordering and must be re-evaluated on the real
  model (HQ-P0C-02) before the topology is adopted.
- The reduction order is fixed by the assignment and the route, so a given plan is reproducible run to run.
- Changing the ownership (ranges vs strided) changes the bits slightly; both are inside the tolerance.
