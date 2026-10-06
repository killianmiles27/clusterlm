# ClusterLM threat model

Scope: the code in this repository as of the WP8 review (reference backend, localhost clusters, Linux CI,
MinGW syntax check for Windows). Hardware- and Windows-only properties that cannot be exercised here are listed
under [Verification status](#verification-status) with their `HQ-SEC-*` qualification entries.

Companion documents: [security-review.md](security-review.md) (findings and fixes),
[../ARCHITECTURE.md](../ARCHITECTURE.md) (what the pieces are), ADRs 0230-0233.

## 1. Assets

| ID | Asset | Where it lives | Why it matters |
|---|---|---|---|
| A-1 | Conversation content: chat roles, system prompts, user text, responses | Father process memory only (`FatherService`, `Conversation`) | The user's private data. It is the thing the privacy contract protects |
| A-2 | Token IDs, logits, sampling state | Father process only (prefix consumes tokens, tail produces logits) | Equivalent to A-1 |
| A-3 | Model weights (canonical copy) | Father's model directory | Large, licensed; the permanent store |
| A-4 | Model weights (lease copy) | A Node's RAM / lease-owned staging for one lease | Must be gone after release (storage-cleanup guarantee) |
| A-5 | Activations (`kResidualHandoffF32V1` records) | In flight Father -> Node A -> Node B -> Father; in a Node's RAM while executing | Hidden states of the conversation. Not claimed private from a Node (see 4) |
| A-6 | Device identities: private key + certificate per machine | `device_key.pem`, `device_cert.pem` in a per-device directory | Possession is the proof of being a paired device |
| A-7 | Pairing state: the set of trusted peer fingerprints | `SecurityConfig::trusted_peers`, runtime `TrustStore` | Decides who may talk to whom |
| A-8 | Node availability and local user priority | Node service policy (idle, lock, AC, power) | A Node's owner must never be harmed or blocked by Father |
| A-9 | Lease / epoch / window counters | Node and Father memory, content-free journal | Stale or replayed traffic must stay stale |
| A-10 | Diagnostics bundles and logs | User-chosen files, stderr | Leave the machine in bug reports |

## 2. Trust boundaries

```
 user ── Father process ──────────────┐   (B1: Father <-> local OS users / files)
        (A-1, A-2, A-3, prefix+tail)  │
                 │ mTLS TLS 1.3 over wired LAN   (B2: LAN)
                 ▼
        Node A process ── mTLS ──► Node B process   (B3: Node <-> Node, authorized per lease)
        (A-4, A-5 transient)
                 ▲
   Node service supervisor ── local process control, Job Object   (B4: supervisor <-> worker)
   local IPC (named pipe, future)                                 (B5: UI/helper <-> service)
   model download / manifest files                                (B6: untrusted artifact -> parser)
```

* **B1 Father host.** Father is the root of trust: it holds everything a user cares about. A compromised Father
  host is out of scope (it has A-1 by definition).
* **B2 LAN.** Untrusted. Every Father <-> Node and Node <-> Node connection is mutually authenticated TLS 1.3
  with pinned self-signed identities. `kInsecureLoopbackOnly` exists for tests and refuses non-loopback
  endpoints.
* **B3 Node <-> Node.** A direct link is trusted only for one lease and only for the identity Father named in
  `AuthorizePeer`; the trust is revoked on release. A peer is never trusted as Father (ADR 0231).
* **B4 supervisor <-> worker.** Cooperative revocation with a hard deadline; the worker is terminated through a
  Job Object on Windows. Neither side accepts commands from the network.
* **B5 local IPC.** Named pipes (Windows) / Unix sockets (dev) between the session helper, Node service, Father agent and UI: protected DACLs, remote clients rejected, peer pid/session/SID verified (`runtime/platform/ipc.hpp`, ADR 0130). Real-Windows ACL enforcement is HQ-SEC-02.
* **B6 artifacts.** Manifests, catalogs, profiles and (later) GGUF files are parsed with bounded, fuzzed decoders.

## 3. Adversaries

| ID | Adversary | Capability assumed | In scope |
|---|---|---|---|
| T-1 | LAN attacker | Sees and alters packets, connects to any listening port, replays captured traffic, runs a rogue "Node" or "Father" | Yes |
| T-2 | Malicious local non-admin user (on Father or on a Node) | Reads world-readable files, plants symlinks/junctions in directories it can write, connects to loopback ports and local pipes, races file operations | Yes |
| T-3 | Compromised Node (paired, authenticated) | Runs arbitrary code in the Node's worker; holds a valid identity; can send any protocol message to Father and to the peers it was authorized to reach | Yes (limited: see 4) |
| T-4 | Hostile Node administrator | Everything T-3 can do plus debugger / memory access to the Node's RAM and VRAM | Partly: activation privacy is **not** claimed |
| T-5 | Malicious or corrupt artifact | A manifest, catalog, profile or model file crafted to crash or exhaust a parser | Yes |
| T-6 | Compromised Father | Full control of the protocol toward Nodes | Bounded: Nodes execute no commands and enforce budgets, but Father decides what runs |

## 4. What mutual TLS protects, and what it does not

**Protects (T-1):**

* Confidentiality and integrity of every frame on the LAN (TLS 1.3 only, no session tickets, no resumption; a TLS
  1.2-only client cannot connect: `tests/security/test_tls_policy.cpp`).
* Mutual authentication by pinned SHA-256 certificate fingerprint. A client without a certificate, or with an
  unpaired one, is refused by the server; a client that does not trust the server refuses it.
* No downgrade to plaintext in `kMutualTls` mode: a plaintext peer cannot complete a handshake.
* Peer identity binding: the authenticated identity, not the self-asserted `Hello.device_id`, decides whether a
  direct peer is authorized (`tests/security/test_peer_binding.cpp`).

**Does not protect:**

* **Anything against the endpoint itself.** A Node decrypts and processes activations; T-3/T-4 see them.
  Hidden states are not proven non-invertible. **Activation privacy is not claimed.** The guarantee is narrower
  and structural: token IDs, text, logits and chat objects never leave Father (section 5).
* **Software integrity of a peer.** Pinning proves "this is the paired machine", not "this machine runs
  unmodified ClusterLM". `backend_build` is a compatibility string, not an attestation.
* **Pairing itself.** Fingerprint exchange (the trust-on-first-use or out-of-band step that fills
  `trusted_peers`) is a product surface that does not exist yet. Until it does, anything that can edit the trust
  list or the identity directory defeats the model (T-2 on the same machine).
* **Revocation and expiry.** Certificates are valid for ~10 years and expiry is deliberately not part of the
  decision; a lost device is revoked by removing its fingerprint from every trust list.
* **Traffic analysis.** Frame sizes, timing, window counts and `CommitWindow.accepted` (the accepted draft
  length) are visible to anyone who can observe the LAN or the Node. They reveal prompt length, generation
  length and speculation acceptance, not content.

## 5. Privacy contract

1. **Node-facing messages are structurally content-free.** `tests/privacy/test_message_schema.cpp` pins every
   message's exact encoded size, so a token array, text field or logit vector cannot be added to the wire
   without failing the test and forcing this document to be revisited. The only variable-length text on the wire is
   an allowlist of short strings (device ids, endpoint, backend build, resource summaries, error messages, release
   errors, abort reason); the only variable-length binary payloads are digest-checked model objects
   (`ProvisionChunk`) and fixed-ABI activations (`RunWindow`/`StageResult`: exactly header + positions x
   bytes-per-position).
2. **Real traffic confirms it.** `tests/privacy/test_traffic.cpp` runs Father + two real Nodes, records every
   frame on every connection, and proves that neither a 12-token sequence, any 3-token window of it, the chat
   text, the chat template, nor the generated continuation appears in any payload.
3. **Logs and diagnostics carry no content.** The logger has no API for tensors, token arrays or messages;
   `tests/privacy/test_logs.cpp` captures all levels over a full chat and checks for prompts, responses,
   token windows and float dumps. The diagnostics bundle (`orchestrator/diagnostics`) additionally redacts
   content-bearing keys, integer lists, float runs and blobs, and includes the conversation only on explicit
   opt-in, under a section marked `user_content`.
4. **Error replies never echo request bytes** (`tests/privacy/test_error_replies.cpp`).
5. **Nodes execute nothing from Father.** No message carries a command, path or executable (schema test);
   the source audit in `tests/security/test_hardening.cpp` rejects `system`, `popen`, `exec*`, `ShellExecute`,
   `cmd.exe`, `/bin/sh` and any process-launch API outside `ChildProcess`. Manifest object names are
   attacker-controlled text and never become paths: staging files are named `obj-<index>.part` by the store
   (test: "manifest object names never become staging paths").

## 6. Abuse cases and controls

| Case | Adversary | Control | Evidence |
|---|---|---|---|
| Connect to a Node port without credentials | T-1 | mTLS with client-cert requirement | `test_tls_policy` |
| Force a TLS downgrade | T-1 | min = max = TLS 1.3 | `test_tls_policy` |
| Replay a captured `RunWindow` | T-1/T-3 | lease generation + epoch + strictly increasing window id + state version | `test_replay`, `test_window_ledger` |
| Authorized peer Node claims to be Father | T-3 | role claim checked against the authenticated identity (ADR 0231) | `test_peer_binding` |
| Unauthorized paired device joins an activation channel | T-3 | `AuthorizePeer` identity must match; Hello claim ignored | `test_peer_binding` |
| Oversized or lying frame / message counts | T-1/T-3/T-6 | length checked before allocation; counts bounded by remaining input; receive buffer grows only with delivered bytes | fuzz targets, `test_hardening` |
| Hostile `PreparePlan` (huge `max_context`, wrapped sizes, duplicates, Father-only objects) | T-6 | admission against RAM allowance, saturating sums, role and object checks | `test_hardening` |
| `SealObject` for an object outside the plan | T-6/T-3 | bounds-checked (was an uncaught `std::out_of_range`) | `test_error_replies` |
| Path traversal through manifest names | T-6 | store-generated file names only; journal validates names | `test_hardening`, `fuzz_lease_journal` |
| Planted symlink in the staging root | T-2 | `remove_tree_no_follow`, `resolve_under_root`, link checks | `tests/platform/test_fs_safety.cpp`, `tests/lease_store` |
| Shell injection through a launched process | T-2/T-6 | no shell; one quoted argv per spawn | `test_hardening` (quoting round trip, source audit) |
| Conversation leaks into a bug report | any | redacting diagnostics, opt-in conversation | `test_diagnostics` |
| Crafted manifest/catalog/profile JSON | T-5 | bounded decoders, strict validation, no exceptions across modules | fuzz targets |

## 7. Residual risks

| ID | Risk | Mitigation / status |
|---|---|---|
| R-01 | Activations visible to a Node and its administrator | Accepted and documented. Mitigation is organizational (Nodes are the user's own machines) |
| R-02 | No pairing UX; trust lists are edited by hand | Product surface not built; model assumes the trust list is correct |
| R-03 | Father may name any `peer_endpoint` in `AuthorizePeer`; a Node connects there (limited by the pinned identity it must present) | Accepted: Father is trusted, and the connection fails unless the endpoint presents the authorized fingerprint |
| R-04 | Device key ACL on Windows (now written owner+SYSTEM only via `write_owner_only_file`, ADR 0133) is unverified on a real Windows host | HQ-SEC-01 |
| R-05 | Named-pipe DACL and client authorization are unverified on a real Windows host | HQ-SEC-02 |
| R-06 | NTFS reparse points / junctions / placeholders untested on a real Windows host | HQ-SEC-03; Linux symlink tests pass |
| R-07 | `kInsecureLoopbackOnly`: `Hello.device_id` is self-asserted | Loopback only; refuses non-loopback endpoints; for tests |
| R-08 | A dynamically trusted identity that is also Father would be refused Father role if it were also authorized as a peer | By design (ADR 0231); a Father is never also a peer |
| R-09 | GGUF parser not in this tree yet | Fuzz target and bounds review required at integration (security-review.md) |
| R-10 | Traffic-analysis metadata (sizes, timing, accepted lengths) | Accepted |
| R-12 | Who may replace the worker binary in the install directory (service launches it by path) | Installer ACL concern; not yet built |
| R-11 | Compromised Father controls what Nodes compute and can feed them chosen inputs | Accepted; Nodes enforce budgets and execute no commands |

## Verification status

| Property | Tested here | Needs real Windows / hardware |
|---|---|---|
| TLS 1.3 only, no tickets, client cert required | Yes (`test_tls_policy`, POSIX raw client) | Same code path on Windows via `test_transport`; no extra check |
| Pinned identity, wrong peer rejected, role escalation refused | Yes | - |
| Replay / stale lease / stale epoch | Yes | - |
| Frame and message bounds, allocation before check | Yes (10 fuzz targets, `test_hardening`) | - |
| Key file mode 0600 | Yes, POSIX (`test_security`) | **HQ-SEC-01**: Windows ACL |
| Local pipe ACL | Not implemented | **HQ-SEC-02** |
| Symlink-safe cleanup | Yes, POSIX | **HQ-SEC-03**: junctions, placeholders, sharing violations |
| Command-line quoting for `CreateProcessW` | Yes: the exact quoting function is tested against a reference CRT parser | **HQ-SEC-03**: real round trip through `CommandLineToArgvW` |
| No shell anywhere | Yes (source audit) | - |
| No content in traffic, logs, diagnostics, error replies | Yes (`tests/privacy`) | Real backend (Strata/llama.cpp) must be re-audited when integrated: its logging is not ours |
