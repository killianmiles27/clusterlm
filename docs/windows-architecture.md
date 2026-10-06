# Windows product platform

How the ClusterLM Node and Father run as Windows software: processes, accounts, pipes, ACLs, startup, firewall,
paths, and what each piece verifies. Decisions are in `docs/adr/0130`-`0134`. Everything marked *unverified* needs
real Windows hardware: see `HARDWARE-QUALIFICATION.md` (`HQ-WIN-01`..`HQ-WIN-04`).

## Processes and accounts

| Process | Account | Session | Role |
|---|---|---|---|
| `clusterlm-node-service.exe` | `NT AUTHORITY\LocalService` (service SID `NT SERVICE\ClusterLMNode`) | 0 | SCM service. `ServiceCore`: helper pipe server, helper-fed activity, `NodeSupervisor`, power/session events |
| `clusterlm-node.exe` (worker) | same as the service (inherits the token) | 0 | Inference worker inside Job Object `ClusterLM-Node-Worker` (kill-on-close, memory limit). Listens on the Node data port |
| `clusterlm-node-helper.exe` | the logged-on user | each interactive session | Reports idle seconds + lock state at 1 Hz and immediately on lock/unlock/resume; no UI yet (tray later) |
| `clusterlm-father-agent.exe` | the logged-on user | the user's session | Hosts the Coordinator configuration, serves the Father UI over a per-user pipe |

```
 user session                        session 0 (service)
 ┌──────────────────┐  pipe  ┌─────────────────────────────────────────────┐
 │ node-helper      │───────▶│ ServiceCore ─ HelperActivityMonitor (fail   │
 │  GetLastInputInfo│ Ack/   │    │            closed on silence)           │
 │  WTS + power msgs│◀───────│    ▼                                         │
 └──────────────────┘ Status │ NodeSupervisor ──▶ worker (Job Object)       │
                             │    ▲                                         │
        SCM ── STOP/SHUTDOWN/PRESHUTDOWN/POWEREVENT/SESSIONCHANGE ──┘       │
                             └─────────────────────────────────────────────┘
 Father UI ◀── per-user pipe ──▶ father-agent ──▶ FatherApiHandler (other workstream)
```

Source map: `runtime/platform` (`ipc`, `ipc_messages`, `ipc_server_loop`, `helper_client`, `helper_activity`,
`helper_startup`, `platform_events`, `service_host`, `firewall`, `paths`, and the Win32 parts in `src/windows/`),
`node/service` (`NodeSupervisor::on_suspend/on_resume`, `ServiceCore`), `apps/node-service`, `apps/node-helper`,
`apps/father-agent` (`include/clusterlm/father/father_agent.hpp`).

## Pipes and ACLs

| Pipe | Created by | DACL (protected) | Peer check |
|---|---|---|---|
| `\\.\pipe\ClusterLM.Node.Helper` | service | deny network logon; allow SYSTEM, the service account (full), interactive users (`0x12019b`: read/write, no pipe-instance creation) | session must be a user session (`!= 0`); report's `session_id` must equal the peer's session; optional allowed SIDs |
| `\\.\pipe\ClusterLM.Father.UI.<user>` | father-agent | deny network logon; allow SYSTEM and the agent's own SID | peer SID must equal the agent's SID |

Both: byte mode, `PIPE_REJECT_REMOTE_CLIENTS`, first-instance flag, client `SECURITY_IDENTIFICATION`, frames <= 1 MiB.
POSIX development uses `<ipc-dir>/<name>.sock` (dir 0700, socket 0600, `SO_PEERCRED` uid check).
Messages: helper to service `ActivityReport`, `PauseRequest`, `ResumeRequest`, `StatusRequest`; service to helper
`StatusReply`, `Ack`; Father UI `kFatherRequest`/`kFatherReply`/`kFatherEvent` with opaque payloads. No message can carry tokens,
text, logits or prompts.

## Startup

- Service: automatic (delayed), recovery restart 5/30/60 s, preshutdown timeout 15 s (`node_service_install_spec()`;
  `clusterlm-node-service.exe --install --port N [--remote ADDR]`, `--uninstall`).
- Helper: started per ADR 0132. Default for the LocalService service is the machine Run key `ClusterLMNodeHelper`; if the
  service holds `SeTcbPrivilege` it launches the helper into each active session instead. A missing helper leaves the Node Busy.
- Father agent: ordinary per-user process (Run key or Startup shortcut; installer workstream).

## Power and sessions

- Service: `SERVICE_CONTROL_POWEREVENT` (`PBT_APMSUSPEND`, `PBT_APMRESUMEAUTOMATIC/SUSPEND`, `GUID_ACDC_POWER_SOURCE`,
  `GUID_POWER_SAVING_STATUS` via `SERVICE_ACCEPT_POWEREVENT`) and `SERVICE_CONTROL_SESSIONCHANGE` map to `PowerEvent` /
  `SessionEvent` and go to `ServiceCore`.
- Helper: a hidden top-level window (`WM_POWERBROADCAST`, `RegisterPowerSettingNotification`,
  `WTSRegisterSessionNotification`) feeds the same event types.
- Suspend: the supervisor revokes synchronously (no Father round trip) and stays suspended. Resume: Busy, pre-sleep reports are
  discarded, offering needs fresh helper reports that satisfy policy.

## Firewall

`firewall_rule_specs()` is the single source: inbound TCP, local port = Node data port, scoped to the `clusterlm-node.exe`
path, profiles Domain+Private (Public only by explicit opt-in), remote addresses `LocalSubnet` or the paired peers, edge
traversal off. `validate_firewall_rule` rejects anything broader. `make_windows_firewall_rules()` applies it with
`INetFwPolicy2`. `clusterlm-node-service --print-firewall-specs` emits the data as JSON for the installer.

## Paths

| | Windows | POSIX (dev) |
|---|---|---|
| Node root | `%ProgramData%\ClusterLM\Node` | `$XDG_STATE_HOME/clusterlm/node` |
| Node staging (ephemeral) | `...\Node\staging` | `.../node/staging` |
| Node identity / logs | `...\Node\identity`, `...\Node\logs` | `.../node/identity`, `.../node/logs` |
| Father root | `%LOCALAPPDATA%\ClusterLM` | `$XDG_DATA_HOME/clusterlm/father` |
| Father identity / models / logs | `...\identity`, `...\models`, `...\logs` | same under father |
| IPC | named pipes | `$XDG_RUNTIME_DIR/clusterlm` |

Node staging and identity are created owner+SYSTEM-only. The device key is written owner-only from its first byte (ADR 0133).

## What verifies what

| Property | Linux CI | Windows CI (MSVC) | Real hardware |
|---|---|---|---|
| IPC framing, bounds, garbage, auth logic, socket perms | `test_ipc` | compiles `ipc_win.cpp` | `HQ-WIN-02` (pipe ACL, session id, SID) |
| Helper staleness fail-closed, pause, sessions | `test_ipc` (`test_helper_activity`) | same tests | `HQ-WIN-02` |
| Helper -> service -> supervisor -> worker | `test_service_core` (real worker, real socket) | compiles | `HQ-WIN-02` |
| Suspend/resume supervisor behaviour | `test_node_service` | compiles | `HQ-WIN-02` |
| Helper startup decision | `test_platform` | same | `HQ-WIN-02` |
| Firewall specs and validation, mock apply | `test_platform` | same | `HQ-WIN-03` (COM, real rule scope) |
| Key and directory owner-only | `test_platform`, `test_ipc` (POSIX mode) | DACL inspection runs in the same tests | `HQ-WIN-04` (icacls) |
| SCM host, install spec | console mode + spec checks | compiles `service_host_win.cpp` | `HQ-WIN-01` |
| All Win32 sources type-check | `scripts/check_windows_compile.sh` (MinGW) | MSVC build | - |

## Manual procedures

The exact steps and pass criteria are the `measurements` and `acceptance` of each registry entry; the sections below are the anchors.

### HQ-WIN-01
Service install, start, stop, crash and shutdown recovery on both Nodes. See `HARDWARE-QUALIFICATION.md#hq-win-01`.

### HQ-WIN-02
Helper activity, lock, suspend/resume, AC/battery on the G14 and the 3060. See `HARDWARE-QUALIFICATION.md#hq-win-02`.

### HQ-WIN-03
Firewall rule scope across Father, a Node and a third machine. See `HARDWARE-QUALIFICATION.md#hq-win-03`.

### HQ-WIN-04
`icacls` verification of keys and staging on all three machines. See `HARDWARE-QUALIFICATION.md#hq-win-04`.

## Known limits

- Real Win32 code is type-checked only (MinGW syntax check on Linux; MSVC in CI). Nothing here has run on Windows yet.
- `OpenProcess` on the pipe server by a standard-user client (server-user pinning) may be denied; the helper does not pin by default.
- Modern standby and hibernate event sequences are not characterised.
- The tray icon, installer, pairing UX and the Father service API are other workstreams.
