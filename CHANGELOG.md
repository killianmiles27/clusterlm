# Changelog

All notable changes. No release has been cut: everything is under *Unreleased*. Format follows
[Keep a Changelog](https://keepachangelog.com/) loosely; versions are not yet assigned (see
[docs/release-procedure.md](docs/release-procedure.md)).

## Unreleased

### Added
- Generalization plan, frozen interface contracts v1 (profile schema, backend capability, scheduler/admission, API auth
  scopes, readiness), requirement ledger and ADRs 0400-0406 (workstream 0; documents only, no implementation yet).
- Verification and documentation (workstream G): `contracts-and-docs` CI job (schema validation of the interface examples,
  documentation links, licensing policy, offline guard), conformance scripts under `tests/conformance/`, contributing guide,
  security policy, issue and pull request templates, install guide, ordinary-user walkthrough, platform and model/backend
  matrices, testing guide, provenance guide, release procedure, license comparison (no license chosen).

### Pre-existing (before the generalization effort)
- Distributed heterogeneous runtime, Strata-derived backend (CPU kernels verified; CUDA never run on a GPU), llama.cpp Host
  backend, Windows service/helper/UI/installers, SPAKE2 pairing, provisioning, placement search, qualification tooling.
  See [docs/DEVELOPMENT-STATUS.md](docs/DEVELOPMENT-STATUS.md).
