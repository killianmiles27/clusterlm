# API authentication and scopes v1

Frozen by ADR 0403. Applies to the Host's HTTP server (workstream B), management API, and MCP (workstream D). **Pairing is a
different system and is unchanged**: SPAKE2 pairing bound to mutual TLS 1.3 with pinned device fingerprints (ADR 0270). No API
key, scope or token can pair, unpair, address, or issue any command to a Worker; Workers accept control only from their paired
Host over the pinned channel (existing `peer role binding`, ADR 0231).

## 1. Network posture

| Setting | Default | Change requires |
|---|---|---|
| Listener | `127.0.0.1` (and `::1`) only | explicit "Allow LAN access" action by the **interactive local user** over local IPC/UI (never over HTTP, never via MCP) |
| LAN listener | off | the action above **and** TLS configured; plain HTTP is refused on any non-loopback address |
| TLS | server certificate = separate Host-generated self-signed key (not the device identity) with fingerprint shown in the UI, or a user-supplied cert/key | n/a |
| Internet exposure | not supported; binding to a public address requires `server.allow_non_private = true` and a warning; no UPnP, no tunnel | explicit setting; a separate design is needed for remote access |
| CORS | none (no CORS headers) | per-origin allowlist in settings; never `*` together with credentials |
| Request limits | body 4 MiB, header 16 KiB, 120 s idle, bounded concurrent connections | configurable within hard caps |

Loopback requests still require a key by default (local malware/other users on a shared machine); a "no-auth on loopback"
convenience is an explicit, visible setting for the Host's own UI, off for HTTP clients.

## 2. Credentials

* **API key**: `clk_<id>_<256-bit random, base32>`; `id` = `key_*` is public and appears in logs; the secret is shown **once**
  at creation. Stored as `SHA-256(secret)` with the key record in an owner-only file (same ACL rules as the identity key; ADR
  0133); compared in constant time. Records: `id, name, scopes[], allowed_models[] (api_model_ids/alias ids, or "*"), source
  (loopback|lan), created, expires?, last_used (coarse), revoked_at?, rate{rpm, concurrent, tokens_per_min}`.
* **Rotation**: create a new key (optionally "rotates: key_x"), both valid for an overlap window chosen by the owner, then
  revoke the old. **Revocation** is immediate and terminates the key's in-flight requests and streams.
* Keys are never accepted in URLs. `Authorization: Bearer …` only. Bootstrap: the first key can only be created through local IPC.
* Logs and diagnostics record key id, scope used, status, counts and timings; never the secret, prompt, completion, tool
  arguments or response bodies (redaction tested by G, extends the existing privacy tests).

## 3. Scopes (additive; none implies another)

| Scope | Grants | Never grants |
|---|---|---|
| `inference` | `GET /v1/models` (own visible models), `POST /v1/chat/completions` (and later `/v1/responses`, Anthropic adapter), `GET /ready`, own request cancel | any management read/write |
| `status:read` | read-only management: profiles & readiness, jobs, queue summary, machines (display name, state, capability; no fingerprint/address), benchmarks (with provenance), diagnostics bundle, active profile | prompts/completions of any client; key material |
| `profiles:prepare` | start/cancel prepare jobs, release leases for profiles the key may see | profile edits; forced release of another client's active request |
| `profiles:admin` | create/update/delete/import profiles & aliases, dry-run, `force` release | pairing, LAN-bind, key admin |
| `workers:pause` | mark a Worker "do not schedule" on the **Host** side (a Host policy flag; the Worker's own user can still override locally and resume is not remote-forceable against local use) | any command executed on a Worker; unpair |
| `keys:admin` | list/create/revoke keys with scopes ≤ the caller's own | granting `keys:admin` itself (only local IPC can) |

Separation rule: inference keys cannot call management; management keys do not get `inference` unless granted both. Default
for a new key is `inference` only. UI shows scopes in plain language and warns on `profiles:admin`/`keys:admin`.
`allowed_models` narrows visibility: a key never sees, and cannot probe the existence of, models outside its list
(`model_not_found`, same response as a nonexistent id).

## 4. MCP mapping (D)

The MCP server authenticates to the Host like any other client: stdio mode uses a key configured by the owner in the MCP
server's own config (owner-only file); Streamable HTTP mode uses bearer keys with the same scopes. Read-only tools need
`status:read`; privileged tools (`prepare_profile`, `release_session`, `pause_worker`, request profile change) need the scope
above **and** an explicit per-call confirmation channel defined by D (never auto-approved). There is no MCP tool that executes a
command, reads files, changes pairing, binds LAN or creates keys. MCP tool outputs follow the same privacy rules.

## 5. Per-client isolation

`client = key id` for HTTP, `local-ui` for the Host app, `mcp:<key id>` for MCP. Queue accounting, rate limits and any session
state are keyed by client. Cross-client reads (another client's conversation, queue contents, KV) do not exist as an operation;
`GET /queue` shows only the caller's own tickets unless the caller has `status:read` (then counts and positions without
content). Request-size, rate and token-budget limits apply per key and globally.

## 6. Threats this design addresses (feeds docs/security/threat-model.md; G extends it)

LAN neighbour using the API → keys + TLS + LAN opt-in; leaked inference key → cannot manage, cannot reach Workers, revocable,
rate-limited; browser CSRF/cross-origin → no CORS, bearer header required (not cookies); prompt logging → forbidden by
design and tested; key brute force → 256-bit secrets, constant-time compare, per-IP failure throttle; management abuse →
separate scopes, local-only bootstrap; Worker takeover via API → no code path from API to the Worker control channel
(extends the existing "no generic execution path from Father to Node" test).
