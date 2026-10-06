# Pairing, persistent configuration and the Father agent

Product-integration notes for WP14. The Father UI protocol itself is in [father-ipc.md](father-ipc.md).

## Pairing a Node with Father

Nodes never trust an unpaired Father; Father only talks to Nodes whose fingerprints it pinned. Both are
established once, by pairing, when neither side knows the other's fingerprint yet.

1. **On the Node** start pairing mode: the Node UI's **Pair with a Father...** button (helper-pipe
   `PairingModeRequest`: interactive user only, one request per 5 s), or `clusterlm-node-service --pair` (console `pair`
   command with `--simulate-activity`). Either way the service prints, and the UI shows, one line:
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
* Father: IPC op `pairing.unpair` removes the Node and its tier assignments, releases any session and then, if the
  Node is reachable, sends it one `UnpairNotice` on an ordinary control channel (mutual TLS, pinned to the Node's
  fingerprint). The Node accepts it only from a pinned (paired) Father: the worker releases its lease, stops trusting
  that identity at once, acknowledges, and tells the service, which clears `paired_father` in `node-settings.json` and
  restarts the worker trusting nobody (the same path as a local unpair). The service double-checks that the notifying
  Father is the one recorded as paired; on a mismatch it ignores the notice and restores the trust list. The reply says
  `node_notified` and, when false, why (`unreachable` / `refused`) with a note. **If the Node cannot be reached the
  unpair still succeeds and the Node keeps trusting that Father until it is unpaired locally** (the behaviour before
  the notice existed). The notice carries only a nonce; a stranger cannot send it (the TLS pin refuses the handshake),
  and a notice does not need the Node to be idle: it also ends an active lease.

A Node holds exactly one paired Father; pairing a new one replaces the old.

### Known gaps (Windows)

* Pairing mode can be started from the Node UI through the helper pipe, but it is the service process (LocalService)
  that opens the listener. Whether the firewall rule `--install` adds admits the Father without the CLI flag, and
  how the pipe's session check behaves for the real interactive user, are `HQ-PAIR-01` / `HQ-WIN-02` questions.
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
worker flags (`--ram-gib`, `--vram-gib`, `--disk-gib`, `--threads`). `caps.threads` is passed to the worker as a thread
cap (`NodeConfig::cpu_threads`): it is reported in the offer (`threads<=N`) and set on every `DomainSpec`
(`cpu_threads`). The reference backend is single-threaded by construction, so any cap of at least 1 already holds;
thread-pool backends map the field onto their pool when a domain is created (the Strata and llama options
`cpu_threads` / `n_threads` exist and are set from backend construction, which backend selection owns).
The Node UI changes these documents through the service (`SettingsUpdate` on the helper pipe, see `docs/ui.md`):
only the user-safe fields (idle policy, AC only, storage limit, start with system, RAM/VRAM/thread caps) can be
changed that way; the name, the paired Father, trust, paths and commands cannot, and are not on the wire. An update is
validated, persisted atomically, then applied: policy immediately, caps by restarting the worker after a cooperative
release (so no lease survives a cap change). Updates are limited to one per 0.5 s.
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
* `ProvisioningBoard`: the latest preparation progress per tier, fed by the service from the Coordinator's progress
  callback and read as `ProductionOptions::provisioning`. The rate is measured on this run (no rate, hence no ETA,
  before about half a second of data) and the entry is removed when the prepare ends, whatever the outcome, so a
  failed prepare cannot leave a tier stuck in Preparing.
* No Strata/llama backend is built into this binary: every tier reports
  "backend not available in this build" and is never Ready. The only override is the explicit
  `--dev-fixture-model` flag (reference backend on the small fixture model): it is reported over IPC
  (`dev_fixture_model`) and written in every tier's details.
* Provisioning progress: `Coordinator::prepare(plan, sink)` reports per Node bytes sent / total, objects sealed /
  total and a phase (`PreparePhase`); the service turns that into `prepare_progress` events with a `detail`
  (docs/father-ipc.md) and into the `ProvisioningBoard`.

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
