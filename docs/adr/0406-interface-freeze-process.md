# 0406: Interface freeze, amendment path and display-name layer

## Context
Parallel threads (B, C, F) must not diverge on contracts; the contracts are still unexercised designs.

## Decision
- Everything under `docs/interfaces/` is frozen at v1 on 2026-10-10. A **breaking** change (removes/renames a field, changes
  semantics, tightens validation of existing data, alters a scope or error code) needs a new ADR (`docs/adr/`) that names the
  contract and version, plus a row in `docs/status.md`. A **compatible clarification or addition** (new optional field, new
  reason code, new example) may be made by the workstream that discovered it in the same commit as the implementation, with a
  one-line "Amendments" entry at the end of the document naming the thread and date; Architecture review (Z or a Thread 0
  successor) confirms.
- A thread that finds a defect in a frozen contract stops and files a cross-module request or ADR before deviating in code.
- Schemas in `docs/interfaces/schemas/` are normative for JSON shape; the Markdown is normative for semantics.
  `docs/interfaces/validate_examples.py` keeps examples honest.
- Display names: user-facing text says **Host** and **Worker**; code, IPC op names, logs and files keep Father/Node. No mass
  rename. E owns a single display-name table.
- Interface versions are semver-lite integers; a reader refuses a newer major version.

## Consequences
Cheap fixes stay cheap; contract churn stays visible. The cost is a small documentation obligation on each amendment.
