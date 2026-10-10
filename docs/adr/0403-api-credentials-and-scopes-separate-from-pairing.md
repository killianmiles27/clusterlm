# 0403: API credentials and scopes are a separate system from pairing

## Context
The Host will expose an HTTP API to unmodified clients and to MCP. Pairing (ADR 0270) is a SPAKE2/mTLS identity system for
Host↔Worker and must not be weakened or reused.

## Decision
Per-client bearer API keys (256-bit secret, SHA-256 stored, owner-only file, shown once) with additive scopes
`inference`, `status:read`, `profiles:prepare`, `profiles:admin`, `workers:pause`, `keys:admin`; loopback-only by default; LAN
only by an explicit local-user action and only with TLS (separate server key, not the device identity); no CORS by default;
rotation by overlap, revocation immediate; first key only via local IPC. No scope reaches the Worker control channel;
`workers:pause` is a Host-side scheduling flag only. Details: `docs/interfaces/auth-scopes-v1.md`.

## Consequences
B implements the listener and key store; G adds negative tests (API→Worker unreachable, secrets never logged). The certificate
source for LAN TLS (self-signed with shown fingerprint vs user-supplied) is fixed in the contract but its enrollment UX is E/B's.
Remote access beyond the LAN needs its own ADR.
