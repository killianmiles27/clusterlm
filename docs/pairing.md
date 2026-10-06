# Pairing, persistent configuration and the Father agent

Product-integration notes for WP14. The Father UI protocol itself is in [father-ipc.md](father-ipc.md).

## Pairing a Node with Father

Nodes never trust an unpaired Father; Father only talks to Nodes whose fingerprints it pinned. Both are
established once, by pairing, when neither side knows the other's fingerprint yet.

1. **On the Node** start pairing mode locally: `clusterlm-node-service --pair` (console `pair` command with
   `--simulate-activity`). It prints, and in a UI would show, one line:
   `CLUSTERLM_NODE_PAIRING code=ABCD-EFGH endpoint=HOST:PORT fingerprint=ab12-cd34-ef56`.
   The listener (default port = data port + 1, `--pair-port`) stays open for 5 minutes (`--pair-window-seconds`).
2. **On Father** the user enters the Node's address (`HOST:PAIR_PORT`) and the code (IPC op `pairing.start`).
3. Both sides prove knowledge of the code; each stores the other's fingerprint, name and role. The Node records
   its Father in `node-settings.json` and restarts its worker with `--trust <father fingerprint>`; Father records
   the Node (name, `HOST:DATA_PORT`, fingerprint) in `father-settings.json`.
4. The user compares the short fingerprint printed by the Node with the one Father shows for it.

Protocol (code: `orchestrator/pairing`, rationale: ADR 0270):

| Step | Message | Notes |
|---|---|---|
| TLS 1.3, both certificates, **no pin** | | `SecurityConfig::pairing_channel`; nothing but pairing travels on it |
| 1 | Father -> Node `Hello{version, X}` | `X = x*G + w*M` (SPAKE2, P-256); `w = PBKDF2(code)` |
| 2 | Node -> Father `Share{Y}` | `Y = y*G + w*N` |
| 3 | Father -> Node `ConfirmA` | HMAC key from a transcript hash covering X, Y, Z, V, w, **both certificate fingerprints and the TLS exporter secret** |
| 4 | Node -> Father `ConfirmB` | sent only if `ConfirmA` verified |
| 5 | `Info` both ways | name, role, data port; only after both confirmations |

### What is and is not resisted

Resisted: eavesdropping; offline guessing from any recording (SPAKE2); a relay or spliced session with its own
certificate (the transcript binds the two real fingerprints and the exporter of the actual TLS session, so no
confirmation verifies, even when the relay forwards every message verbatim; tested); replay of recorded messages
(fresh ephemeral values and a fresh TLS session); an active attacker gets one guess per attempt and 3 attempts
per pairing mode. Rate limit: failed *confirmations* are counted; 3 lock the mode (listener closed, a new local
`--pair` is needed). Stalls, aborts and malformed frames are not counted (they test no guess). The code is single
use.

Not resisted: an attacker who sees the code before the user uses it can pair their own device (compare the
fingerprints in step 4); any LAN host can burn the 3 attempts (denial of pairing). The code is 40 bits, which is
adequate only because guessing is online and capped at 3.

### Unpair

* Node: `clusterlm-node-service --unpair` (service stopped) or `unpair` on the console of a running service. The
  trusted list is replaced and the worker is revoked (cooperative release, then forced) and restarted, so the
  Father loses its connections and leases immediately.
* Father: IPC op `pairing.unpair` removes the Node and its tier assignments and releases any session. The Node is
  not notified; it keeps trusting that Father until unpaired locally.

A Node holds exactly one paired Father; pairing a new one replaces the old.

### Known gaps (Windows)

* Pairing mode is started with the service's own CLI flag. A tray or helper-pipe command to start it while the
  service runs as LocalService is not implemented (tracked in HQ-PAIR-01).
* `--install` adds the firewall rule `ClusterLM Node pairing (TCP-In)` for the service executable (Private and
  Domain profiles, LocalSubnet). This is type-checked only; real behaviour is pending HQ-PAIR-01.
* Discovery is by typed address only.

## Persistent configuration (`orchestrator/config`, ADR 0271)

| Document | Location | Contents |
|---|---|---|
| `father-settings.json` | `default_paths().father_root` | paired Nodes, role -> machine assignments, selected tier, context preference, keep-ready policy, model directory per tier, confirmed model roots, advanced options |
| `node-settings.json` | `default_paths().node_root` | name, paired Father, allow-when-idle, idle seconds, AC only, temporary storage limit, paused, startup, caps (RAM/VRAM/threads) |

Atomic owner-only writes, strict validation, corrupt file moved aside (`.corrupt-<stamp>`) with defaults used,
newer-than-supported versions refused, older ones migrated through a hook with the original kept. Node caps map to
worker flags (`--ram-gib`, `--vram-gib`, `--disk-gib`); `caps.threads` is stored but not enforced by the worker yet.
CLI flags of `clusterlm-node-service` (`--idle-seconds`, `--allow-battery`, `--name`, `--ram-gib`, `--vram-gib`)
override the document for that run. `paused` or `allow_when_idle=false` start the service paused.

## Production providers (`orchestrator/father-service/production.hpp`)

* `ConfigDeploymentProvider`: tier + assignment + settings -> coordinator config (the assigned paired Nodes'
  endpoints with **pinned** fingerprints, mutual TLS, Father identity) and the `ClusterPlan` from the placement
  engine. Profiles: Measured files from `advanced.bench_results_dir` (`<fingerprint>.hardware.json`,
  `<name>.hardware.json`, `father.hardware.json`, `network.json`) when present, else the Synthetic fixtures in
  `--profiles-dir`. The plan's provenance is stated in the tier details ("placement used SYNTHETIC profiles ...").
* `LiveReadinessSource`: model directory + manifest, background verification of every object digest, user
  confirmation of the unpinned manifest root (`model.confirm`), Node state by probing the control channel (an
  offer means Available, silence means in use, no connection means offline; battery from the offer), Father power,
  placement feasibility. While a session is active nodes are reported from that session instead of being probed.
* No Strata/llama backend is built into this binary: every tier reports
  "backend not available in this build" and is never Ready. The only override is the explicit
  `--dev-fixture-model` flag (reference backend on the small fixture model): it is reported over IPC
  (`dev_fixture_model`) and written in every tier's details. Provisioning progress from Coordinator events is not
  wired (`ProductionOptions::provisioning` is an injection point; the Coordinator exposes no progress callback).

## Dev end to end

`tests/father_agent/test_dev_e2e.cpp` (labels `integration`, `slow`): two `clusterlm-node-service` processes, a
`clusterlm-father-agent --dev-fixture-model` and a scripted IPC client pair both Nodes (wrong code first), assign
roles, point the Ultra tier at the fixture model, prepare, chat (12 streamed tokens, `finished: completed`) and
unpair. A second test shows that without the override no tier is ever Ready. Linux/POSIX only in practice
(Unix sockets, XDG paths); the Windows build is type-checked by `scripts/check_windows_compile.sh`.

## HQ-PAIR-01

Manual procedure on the real LAN (Father, G14, 3060): see the experiment in
[HARDWARE-QUALIFICATION.md](../HARDWARE-QUALIFICATION.md#hq-pair-01). Run pairing for each Node, check
fingerprints, enter three wrong codes, run a TLS-terminating proxy as the relay, unpair on each side and verify
revocation. Record results with the bench result schema; nothing here is pre-measured.
