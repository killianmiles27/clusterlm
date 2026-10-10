# API authentication and scopes v1

Frozen v1.1: ADR 0403, revised by ADR 0407. Applies to the Host's HTTP server (workstream B), management API, and MCP
(workstream D). **Pairing is a different system and is unchanged**: SPAKE2 pairing bound to mutual TLS 1.3 with pinned device
fingerprints (ADR 0270). No API key, scope or token can pair, unpair, address, or issue any command to a Worker; Workers accept
control only from their paired Host over the pinned channel (existing `peer role binding`, ADR 0231).

## 1. Network posture

| Setting | Default | Change requires |
|---|---|---|
| Listener | `127.0.0.1` (and `::1`) only | explicit "Allow LAN access" action by the **interactive local user** over local IPC/UI (never over HTTP, never via MCP) |
| LAN listener | off | the action above **and** TLS configured; plain HTTP is refused on any non-loopback address |
| TLS | server certificate = separate Host-generated self-signed key (not the device identity) with fingerprint shown in the UI, or a user-supplied cert/key. The private key is stored owner-only (ADR 0133) and never exported, logged or put in a diagnostics bundle | n/a |
| Internet exposure | not supported; binding to a non-private address requires `server.allow_non_private = true` and a warning; no UPnP, no tunnel | explicit setting; a separate design is needed for remote access |
| CORS | none (no CORS headers) | per-origin allowlist in settings; never `*` |
| Request limits | body 4 MiB, header 16 KiB, 120 s idle, bounded concurrent connections | configurable within hard caps |

**Browser-borne attacks (CSRF, DNS rebinding) — mandatory checks on every request, before authentication:**

* **Host header allowlist.** The `Host` header must be `localhost`, `127.0.0.1` or `[::1]` (any port) on the loopback listener,
  or one of the configured LAN names/addresses on the LAN listener; anything else → 421 without processing. This defeats DNS
  rebinding (a page on `evil.example` rebound to 127.0.0.1 still sends `Host: evil.example`).
* **Origin.** A request carrying an `Origin` header is refused (403) unless the origin is in the CORS allowlist. Native clients
  and SDKs send no `Origin`.
* **Content type.** `POST` bodies must be `application/json` (415 otherwise), so a cross-site form/`text/plain` post cannot
  reach a handler without a CORS preflight.
* **No ambient credentials.** Cookies and HTTP auth other than `Authorization: Bearer` are never accepted.

Loopback requests require a key by default (local malware, other accounts on a shared machine). An owner may enable
**"no key on loopback"** (local IPC/UI only, visible in the UI): it then treats an unauthenticated loopback request as a client
`loopback-anon` with scope `inference` only, `allowed_models` = profiles/aliases with `exposure.api: true`, never any management
scope, and the browser checks above still apply. (v1.0 described this convenience ambiguously as being "for the Host's own UI";
the Host UI uses local IPC and never needs it.)

## 2. Credentials

* **API key format**: `clk_<id>_<secret>` where `<id>` is the key record id `key_[a-z0-9]{16}` (public, appears in logs) and
  `<secret>` is 256 random bits as 52 characters of lowercase RFC 4648 base32 without padding. Parsers split at the **last**
  `_`; anything malformed is `invalid_api_key`. The full key is shown **once** at creation.
* **Storage**: the record stores `H = SHA-256("clusterlm-api-key-v1" ‖ 0x00 ‖ id ‖ 0x00 ‖ secret)` (domain-separated and bound to
  the id, so records cannot be swapped between ids) in an owner-only file (same ACL rules as the identity key; ADR 0133). A fast
  hash is sufficient because the secret has 256 bits of entropy; there is no user-chosen password anywhere in this scheme.
* **Verification**: look up by id, compute `H`, compare in constant time. An unknown id runs the same computation against a
  fixed dummy record so unknown-id and wrong-secret take the same time and return the same 401 `invalid_api_key`.
  Authentication failures are throttled per source address (default: 10 per minute, then 429 with `Retry-After`, doubling up to
  15 min) and logged with the address and the presented key id only.
* **Records**: `id, name, scopes[], allowed_models[] (api_model_ids/alias ids, or "*"), network (loopback-only|lan), created,
  expires?, last_used (coarse, ≥ 1 min granularity), revoked_at?, rotates?, rate{rpm, concurrent, tokens_per_min}`.
  `network: loopback-only` (default) keys are refused on the LAN listener. (v1.0 called this field `source (loopback|lan)`
  without saying what it enforced.)
* **Rotation**: create a new key (optionally `rotates: key_x`), both valid for an overlap window chosen by the owner, then
  revoke the old. **Revocation** is immediate and terminates the key's queued and in-flight requests and streams
  (scheduler-admission-v1 §4). **Expiry** is checked at admission and at dequeue; a request already running when its key
  expires is allowed to finish.
* Keys are never accepted in URLs or query strings. `Authorization: Bearer …` only. Bootstrap: the first key can only be created
  through local IPC.
* Logs and diagnostics record key id, scope used, status, counts and timings; never the secret or `H`, prompt, completion, tool
  arguments or response bodies (redaction tested by G, extends the existing privacy tests).

## 3. Scopes (additive; none implies another)

| Scope | Grants | Never grants |
|---|---|---|
| `inference` | `GET /v1/models` (own visible models), `POST /v1/chat/completions` (and later `/v1/responses`, Anthropic adapter), `GET /ready`, own request cancel | any management read/write |
| `status:read` | read-only management: profiles & readiness, jobs, queue summary, machines (display name, state, capability; no fingerprint/address), benchmarks (with provenance), shareable diagnostics bundle, active profile | prompts/completions of any client; key material; paths |
| `profiles:prepare` | start/cancel prepare jobs, release leases for profiles the key may see | profile edits; forced release of another client's active request |
| `profiles:admin` | create/update/delete/import profiles & aliases, dry-run, `force` release | pairing, LAN-bind, key admin |
| `workers:pause` | mark a Worker "do not schedule" on the **Host** side (a Host policy flag; the Worker's own user can still override locally and resume is not remote-forceable against local use) | any command executed on a Worker; unpair |
| `keys:admin` | list/create/revoke keys within its own authority (below) | granting `keys:admin` itself (only local IPC can) |

`GET /health` needs no key and returns process liveness only (`{"status":"ok"}`; still subject to the §1 browser checks).

**No escalation through `keys:admin`.** A key created by a `keys:admin` key must have `scopes ⊆` the creator's scopes,
`allowed_models ⊆` the creator's, `network` no wider than the creator's, and `expires` no later than the creator's. A
`keys:admin` key may list and revoke only keys whose scopes and `allowed_models` are subsets of its own; it may not edit an
existing key's scopes (rotate instead). Only local IPC can create, list or revoke keys outside these limits.

**`allowed_models` applies to every scope**: inference, readiness, `GET /v1/models`, status reads, prepare/release and the
management listings are all filtered by it. A key with `profiles:admin` or `keys:admin` must have `allowed_models: ["*"]`
(enforced at creation) because those scopes can create new models. A key never sees, and cannot probe the existence of, models
outside its list (`model_not_found`, the same response as a nonexistent id, including on `GET /ready?model=`). A routing alias
never widens this: its candidates are filtered by the same visibility (profile-schema-v1 §6).

Separation rule: inference keys cannot call management; management keys do not get `inference` unless granted both. Default
for a new key is `inference` only, `network: loopback-only`. UI shows scopes in plain language and warns on
`profiles:admin`/`keys:admin` and on `network: lan`.

## 4. MCP mapping (D)

The MCP server authenticates to the Host like any other client: stdio mode uses a key configured by the owner in the MCP
server's own config (owner-only file); Streamable HTTP mode uses bearer keys with the same scopes and the same §1 checks.
Read-only tools need `status:read`; privileged tools (`prepare_profile`, `release_session`, `pause_worker`, request profile
change) need the scope above **and** an explicit per-call confirmation channel defined by D (never auto-approved). There is no
MCP tool that executes a command, reads files, changes pairing, binds LAN or creates keys. MCP tool outputs follow the same
privacy rules.

## 5. Per-client isolation

`client = key id` for HTTP, `local-ui` for the Host app, `mcp:<key id>` for MCP, `loopback-anon` for the optional unauthenticated
loopback client. Queue accounting, rate limits and any session state are keyed by client; the same key used over HTTP and MCP
shares one set of rate limits (keyed by key id). Cross-client reads (another client's conversation, queue contents, KV) do not
exist as an operation; `GET /queue` (`status:read`) shows the caller's own tickets and only anonymous positions of others.
Error `reasons[]` given to a key without `status:read` are generic and never name machines. Request-size, rate and
token-budget limits apply per key and globally.

## 6. Threats this design addresses (feeds docs/security/threat-model.md; G extends it)

LAN neighbour using the API → keys + TLS + LAN opt-in + `network: loopback-only` default; leaked inference key → cannot
manage, cannot reach Workers, revocable, rate-limited, limited to its models; browser CSRF/cross-origin and DNS rebinding →
Host allowlist, Origin refusal, JSON-only bodies, no CORS, bearer header only (not cookies); prompt logging → forbidden by
design and tested; key brute force/enumeration → 256-bit secrets, constant-time compare, uniform unknown-id path, per-address
failure throttle; scope escalation → subset rules for `keys:admin`, `allowed_models` on every scope, aliases cannot widen
visibility; management abuse → separate scopes, local-only bootstrap; Worker takeover via API → no code path from API to the
Worker control channel (extends the existing "no generic execution path from Father to Node" test).

## Amendments

* v1.1 (2026-10-10, ADR 0407): Host/Origin/content-type checks; scoped "no key on loopback"; exact key format and
  domain-separated hash; uniform unknown-id timing and failure throttle; `network` field semantics; expiry vs revocation;
  `keys:admin` subset rules; `allowed_models` on every scope; `/health` unauthenticated; generic reasons for inference-only
  keys; TLS key storage; shared rate limits across HTTP/MCP.
