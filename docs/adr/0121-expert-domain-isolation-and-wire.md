# 0121. Grouped expert-domains stay isolated, with their own wire messages

## Context

The grouped expert-domain topology is the principal alternative to the layer-domain pipeline and may lose. It must
be deletable without touching production. It also needs messages that carry routing metadata (selected expert ids and
weights), which the production protocol deliberately cannot express: `runtime/protocol` has no field for routing,
tokens, logits or candidates, and tests rely on that.

## Decision

- All code lives in `experimental/expert-domains/` (library `clusterlm_expert_domains`, executable
  `clusterlm-expert-domain-bench`), tests in `tests/expert_domains/`. Nothing in `runtime/`, `node/`, `orchestrator/`
  or `apps/` depends on it. The only shared-file edits are one line in the top-level module list, one include path in
  `scripts/check_windows_compile.sh` and the qualification registry.
- Messages (`ExpertBatch`, `ExpertResult`, `ExpertError`) are defined in that module with their own type range
  (0x0E01..0x0E03). They reuse `runtime/transport` unchanged (framing, mutual TLS pinning, impairment, fault
  injection) by building `transport::Frame`s directly.
- The topology sends selected expert ids and weights on the wire. The addendum allows this for the experimental
  topology only. It still never sends token IDs, logits or text. Nothing logs routes: the server and executor emit no
  log lines containing route data, errors name the domain and layer only.
- Decoding is bounded before allocation (`ExpertDecodeLimits`): positions, hidden size, routes per position, local
  expert id, layer, error text, and a payload bound for the transport. Routes must be strictly ascending, finite.
- A domain is stateless between requests and holds only routed experts for its owned ids, resolved from a plan-scoped
  resolver; it refuses a plan that lacks one of its experts.
- The numerical kernel and the dense/mixer/router math are copies of the reference backend's code built on the same
  `refmath` header (included by relative path, no changes to `runtime/domain`). Equality with the reference is proven
  by tests rather than assumed.

## Consequences

- Deleting `experimental/expert-domains/`, `tests/expert_domains/`, the module-list line and the docs removes the
  topology completely.
- The duplicated layer math must be kept in step with the reference if the reference changes; the bitwise
  single-owner test fails if they drift.
