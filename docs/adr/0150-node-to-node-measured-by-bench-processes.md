# 0150 Node-to-Node links are measured by bench processes, not by `clusterlm-node`

## Context

Placement needs the Node A to Node B link (bandwidth, RTT, jitter) in addition to the Father to Node links. The
brief asked for a direct Node to Node mode "through LocalCluster": bench orchestrates, node A measures to node B.
`LocalCluster` launches real `clusterlm-node` processes, but a Node speaks only the lease protocol. It has no echo,
sink or source endpoint, and adding one would widen the Node-facing protocol with a message type whose only purpose
is measurement (the protocol is deliberately minimal and token-free; see `docs/ARCHITECTURE.md`).

## Decision

`clusterlm-bench transport --node-to-node` orchestrates two `clusterlm-bench` child processes (through
`platform::ChildProcess`): B runs `transport --serve`, A runs `transport --peer B`, each with its own device
identity and mutual TLS, and bench merges A's result under `node_to_node.*`. The serve side exits when its stdin is
closed (`--exit-on-stdin-eof`), so a dead orchestrator never leaves a listener behind.

On real machines the same two commands are run by hand: `--serve` on Node B and `--peer` on Node A. A localhost run
is always Synthetic (`simulated.localhost_cluster`). The result file is the only product; no Node binary changes.

## Consequences

- The Node protocol stays free of measurement messages.
- The measured path is the bench transport, not the Node's activation-forwarding code. It exercises the same framing,
  TLS and TCP stack, but not Node scheduling. Forwarding latency inside a real plan is measured by
  `cluster --compare-routing` (direct versus relay), which does go through the Nodes.
- No `LocalCluster` dependency for this mode, so it also works on a machine pair without the node binary.
