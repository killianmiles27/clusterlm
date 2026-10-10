# Security policy

## Reporting a vulnerability

Report privately, not in a public issue. Use GitHub's private vulnerability reporting for this repository (Security tab,
"Report a vulnerability"). If that is unavailable to you, open an issue that says only "security contact requested" with no
details and the maintainer will arrange a private channel. There is no bug bounty and no guaranteed response time; this is a
pre-release project maintained by an individual.

Please include the affected component, version or commit, reproduction steps and the impact you believe it has. Do not
include real prompts or credentials in a report.

## Supported versions

There are no released versions yet. Only the current default branch is considered; fixes land there.

## What is and is not in scope

In scope: the Host (Father) and Worker (Node) programs, the pairing protocol, the Host-Worker transport, the Windows
service, helper and installers, local IPC, settings and key storage, and any client-facing API once it exists.

The security design, assets and trust boundaries are in [docs/security/threat-model.md](docs/security/threat-model.md); the
findings of the last internal review are in [docs/security/security-review.md](docs/security/security-review.md). Key facts:

* Host-Worker traffic is mutual TLS 1.3 with pinned identities established by SPAKE2 pairing ([docs/pairing.md](docs/pairing.md)).
* Token IDs and text never leave the Host; Workers receive activations for their assigned layers only, and this is **not** a
  claim that activations are private from a Worker you paired.
* A Worker is never given a generic execution path from the Host (no remote code execution surface).
* Planned client API credentials are separate from pairing identities and never grant Worker control
  ([auth scopes](docs/interfaces/auth-scopes-v1.md)). The client API is **not implemented yet**; until it is, there is no
  network listener for inference clients.

Out of scope: attacks that need an already-paired malicious Worker or a compromised Host account (see the threat model for
what a paired peer can and cannot do), physical access, and denial of service by someone who controls the LAN.

## Status of verification

Fuzz corpora, TLS policy, peer binding, replay and privacy (no-prompt-in-logs) tests run in CI (`tests/security`,
`tests/privacy`, `tests/fuzz`). Windows-only protections (service account, DACLs, Job Objects, firewall scoping) are verified on a
Windows CI runner for installation and by mocks for interactive behaviour; none has been exercised on the target machines
(`HQ-SEC-01..03`, `HQ-WIN-01..04`, [HARDWARE-QUALIFICATION.md](HARDWARE-QUALIFICATION.md)). No third-party audit has been done.
