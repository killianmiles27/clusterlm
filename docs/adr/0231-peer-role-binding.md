# 0231 A direct peer Node can never claim the Father role

## Context

A Node serves three kinds of connections: Father's control, provision and activation channels, and, for one
lease, activation channels from an upstream peer Node. The `Hello` message carries a `role` field
(`kFather`/`kNode`/`kBench`) that the sender fills in. For the Node to treat a connection as Father, the code
only checked `hello.role == kFather`.

Transport authentication happens before `Hello`, and a peer Node authorized through `AuthorizePeer` is added to
the Node's runtime trust store so that its TLS handshake succeeds. That made the role claim the only thing
separating "the upstream Node I was told to accept activations from" from "Father". A compromised peer Node
could therefore open the control channel (when Father's was not attached), the provision channel or replace
Father's activation channel by saying `role = Father`, and then issue `ReleaseLease`, `AbortSession`,
`PreparePlan` and so on.

## Decision

The identity decides, the claim only has to agree with it. When a connection is authenticated and claims
`kFather`, the Node closes it if that identity is currently authorized as a direct peer (either
`inbound_allowed` or the `downstream` link) for this lease. A Father is never also a peer of a Node, so no
legitimate connection is affected, and the check needs no new configuration: Father's identity is added via
`--trust` exactly as before.

On activation channels the authenticated fingerprint (not `Hello.device_id`) is already what is matched against
`AuthorizePeer`. A paired-but-unauthorized identity that puts an authorized id in its Hello is refused.

Insecure loopback mode has no cryptographic identity; the claim stays self-asserted there, which is why that
mode refuses non-loopback endpoints and is test-only.

## Consequences

* A compromised Node cannot use its authorization as a stepping stone to Father-only operations on its neighbour.
* The rule depends on `AuthorizePeer` being the only source of dynamic peer trust. Any future dynamic trust
  source must register its identities in the same sets (or the check must move to an explicit "role by
  identity" table when pairing UX introduces roles).
* Pinned by `tests/security/test_peer_binding.cpp`.
