# 0131 — Node service account, start type and recovery

## Context

The Node service supervises an inference worker that handles data from a remote peer and owns the GPU. It must
start without anyone logged on, survive crashes, release everything on stop/shutdown, and run with as little
privilege as possible.

## Decision

- Account: `NT AUTHORITY\LocalService`, with an unrestricted per-service SID (`NT SERVICE\ClusterLMNode`) available
  for ACLs. Not LocalSystem: the worker inherits the service's token, and nothing the service does needs SYSTEM.
- Start type: automatic (delayed), so the service does not compete with logon-critical services and the GPU driver
  is up before the worker starts.
- Recovery: restart after 5 s, 30 s, 60 s; failure counter resets after 24 h; actions also apply to non-zero exits.
- Controls accepted: STOP, SHUTDOWN, PRESHUTDOWN (15 s timeout; the worker is stopped cooperatively with the
  usual 2 s deadline then via the Job Object), POWEREVENT, SESSIONCHANGE.
- Suspend is handled synchronously inside the control handler: the supervisor revokes (cooperative, then forced)
  before returning, with no Father round trip. Resume starts Busy, discards pre-sleep helper reports and waits
  for fresh ones.
- `clusterlm-node-service --install/--uninstall` implements the declarative `ServiceInstallSpec` through
  `CreateServiceW`/`ChangeServiceConfig2W` and applies the firewall rule; the installer workstream consumes the same
  spec and `clusterlm-node-service --print-firewall-specs`.
- Console mode (`--console`) runs the identical `ServiceCore`; on non-Windows it is the only mode.

## Consequences

- LocalService has no `SeTcbPrivilege`, which shapes ADR 0132.
- The Node data directory must be creatable by LocalService: `%ProgramData%\ClusterLM\Node` is created by the
  service itself (owner+SYSTEM-only, ADR 0133); an installer-created directory must grant LocalService full control.
- Everything about real SCM behaviour (wait hints, preshutdown behaviour, recovery timing) is unverified until
  HQ-WIN-01 passes.
