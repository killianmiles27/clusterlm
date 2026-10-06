# 0290 The Node MSI delegates registration to the service executable; autostart is a machine Run value

## Context

The Node needs an SCM service (LocalService, delayed auto start, recovery actions, preshutdown timeout, service SID),
a program-scoped firewall rule and a per-session helper at logon. WiX can express the first two (`ServiceInstall`,
`ServiceConfig`, the Firewall extension), but `platform::node_service_install_spec()` and
`platform::firewall_rule_specs()` already are the single source of those values, verified by HQ-WIN-01 and HQ-WIN-03.
The Father needs a per-user background agent started at logon: an HKCU Run value, a scheduled logon task, or a
machine Run value.

## Decision

- The MSI runs `clusterlm-node-service --install --port N [--remote ADDR]` as a deferred, non-impersonated (SYSTEM)
  custom action with a rollback action, and `--uninstall` / `--cleanup` on removal. `--uninstall` now also removes the
  firewall rules. WiX therefore carries no copy of the service configuration, and the smoke test checks the result
  against what the code declares (`packaging/check_packaging.py` ties names and flags to the sources).
- Autostart uses the **machine Run key** (`HKLM\...\Run`) for both the Node helper (as ADR 0132 already decided) and
  the Father agent. An HKCU value cannot be written by a per-machine MSI for users other than the installer. A
  scheduled logon task needs XML plus a custom action, is invisible in the usual "Startup apps" UI, and adds a failure
  mode (task policy) without a benefit: the agent has no need for elevation, a delay or a restart policy. A Run value is
  removed with its WiX component, shows in Task Manager's Startup tab, and can be left out at install time
  (`ADDLOCAL=FatherCore`, feature `FatherAutostart`).
- The Node data directories under `%ProgramData%\ClusterLM\Node` are created by the service on first start, not by the
  MSI: the "owner" of an owner-only ACL is the account of the creating process (ADR 0133), and a directory created by
  the MSI would be owned by SYSTEM.

## Consequences

- Changing the service configuration needs no installer change.
- `--install` is a custom action, so it is not visible to MSI's own repair or ICE validation; the smoke test covers it.
- Every user of a Father machine gets the agent at logon. An administrator who wants a single user removes the
  feature or the Run value.
