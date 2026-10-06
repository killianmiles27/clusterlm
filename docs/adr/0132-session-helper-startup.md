# 0132 — How the session helper is started

## Context

Input activity can only be read inside the user's session, so a helper must run there. Two ways exist: the
service launches it (`WTSQueryUserToken` + `CreateProcessAsUserW`), or a machine-wide `HKLM\...\Run` value starts it at
every logon. The preference was the service-launched path because it survives user-level policy.

## Decision

`WTSQueryUserToken` requires `SeTcbPrivilege`, which only LocalSystem holds, and ADR 0131 keeps the service on
LocalService. Running the service as SYSTEM to start a helper would hand the worker's host a far larger blast radius
than the problem justifies. Therefore:

- The shipped default (`HelperLaunchStrategy::kAuto`) resolves by capability: with `SeTcbPrivilege` the service
  launches the helper into every active session; without it the installer registers the helper in the machine Run
  key (`ClusterLMNodeHelper`), and the service re-asserts the value when it can (it normally cannot write HKLM, so
  that failure is only a warning).
- Both paths are implemented (`WindowsHelperHost`), selected by `choose_helper_strategy`, and the reconciliation
  logic (`reconcile_helpers`: one helper per active non-zero session, forget ended sessions, remove the Run value
  when the service launches) is OS-independent and unit-tested with `MockHelperHost`.
- A helper takes a per-session named mutex, so a duplicate start is a no-op.
- A missing helper is safe, not a defect to hide: `HelperActivityMonitor` fails closed (no fresh report = in
  use), so the Node simply stays Busy. The service additionally tells the monitor when no user session exists at all
  (logon screen), so an unattended machine is not stuck Busy.

## Consequences

- The Run-key path depends on the machine policy allowing HKLM Run entries; if a site blocks them the Node stays
  Busy there until the administrator allows it or the service is installed as LocalSystem (supported, not default).
- HQ-WIN-02 verifies helper start per session, duplicates, and recovery after the helper is killed.
