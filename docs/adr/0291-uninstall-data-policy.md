# 0291 Uninstall removes model fragments always, pairing identity only on request

## Context

The product invariant is that a Node holds no model weights, packs or fragments once it stops participating. An
uninstall is the last chance to enforce that. Opposing needs: an upgrade (which is an uninstall of the old product) or a
reinstall should not force the user to pair again, and the Father's models are the user's own multi-gigabyte data.

## Decision

- **Node, always removed on uninstall (including upgrades):** the service and its firewall rules, the helper Run
  value, and the whole staging root. `clusterlm-node-service --cleanup` first runs `LeaseStore` recovery (so the lease
  journal accounts for what is deleted and crash orphans go through the same code as at service start), then deletes
  the staging root, and exits non-zero if anything survives. The uninstall custom action is `Return="ignore"` so a
  locked file cannot make the product un-uninstallable; the failure is in the verbose log and the smoke test and
  HQ-INSTALL-01 assert the directory is gone.
- **Node, kept by default:** the pairing identity (device key and certificate) and logs. They contain no model data, are
  owner-only (ADR 0133) and survive upgrade and reinstall. `msiexec /x ... CLUSTERLM_PURGE=1` also removes identity,
  logs and the Node directory (only if nothing unknown remains in it).
- **Father:** binaries, PATH entry and autostart are removed. `%LOCALAPPDATA%\ClusterLM` (identity, models, logs) is
  never touched: a machine-wide installer cannot enumerate other users' profiles and must not delete a model store.
  The documentation says how to delete it.

## Consequences

- A user who wants no trace runs the purge command; a stolen-laptop scenario is covered by ACLs, not by uninstall.
- A failed cleanup is visible only in logs unless someone checks; HQ-INSTALL-01 includes the check.
