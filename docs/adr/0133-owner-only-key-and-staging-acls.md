# 0133 — Owner-only ACLs for keys and staging; no DPAPI wrapping

## Context

The device private key was written with POSIX 0600 but inherited the directory ACL on Windows. The Node staging
root holds model objects that exist only for the life of a lease.

## Decision

- `platform::write_owner_only_file` creates the key with an explicit, protected DACL granting full control to the
  creating process's account and SYSTEM only, supplied in `SECURITY_ATTRIBUTES` so the file never exists with an
  inherited ACL. An existing file is tightened before it is overwritten. `DeviceIdentity::save` serialises the key
  into memory and writes it through this function on every OS (POSIX: `O_CREAT` 0600 plus `fchmod`), wiping the buffer.
- `platform::restrict_to_owner`, `create_owner_only_directory` and `is_owner_only` apply and inspect the same
  ACL for directories (entries inherited by children). `is_owner_only` checks that the DACL is protected and that
  every entry is an allow entry for the process account or SYSTEM.
- Node data lives under `%ProgramData%\ClusterLM\Node\{staging,identity,logs}`, Father's under
  `%LOCALAPPDATA%\ClusterLM\...` (`platform::default_paths`, `SHGetKnownFolderPath`); staging and identity are created
  owner-only by whichever account runs the service.
- DPAPI (`CryptProtectData`) is not used. User-scope DPAPI binds to a profile that does not exist for a
  LocalService daemon; machine-scope DPAPI is decryptable by any code running on the machine as any account, which
  the ACL already excludes, and it would add a second key format to a PEM identity that TLS loads directly. Offline
  theft of the disk is the threat DPAPI machine scope does not meaningfully address without BitLocker, which the
  installer should recommend instead.

## Consequences

- Administrators cannot read the key without taking ownership, which is deliberate and audited; there is no recovery
  path other than re-pairing, which is cheap.
- If HQ-WIN-04 or a threat review shows offline protection is required, the change is local to
  `DeviceIdentity::save/load` plus a migration of existing key files.
