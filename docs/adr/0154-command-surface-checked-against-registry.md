# 0154 The accepted command line is declared once and checked against the qualification registry

## Context

`bench/qualification/experiments.json` holds a `command` for every hardware experiment, and
`HARDWARE-QUALIFICATION.md` is generated from it. Those command lines drifted from the tool (flags that were never
implemented), which is the worst failure mode for an experiment run on borrowed hardware.

## Decision

`bench/src/command_specs.cpp` declares each command, its subcommands and its accepted `--flags`. `main` rejects
anything else with exit code 2, and `test_bench` parses every registry command line against the same declaration
(placeholders such as `<machine>` are values). The registry was rewritten to implemented flags; what a command still
cannot measure is stated in an optional per-experiment `tool_gaps` list that the generator renders. Experiments that
need the real backends (`domain`, `baseline`, `numerics`) keep their command lines, resolve to the exit-3 stub and
list their pending items by reading the registry.

## Consequences

- Adding a flag means adding it to the spec, which is a one-line change next to the implementation.
- A registry entry can no longer name a flag that does not exist; a removed flag fails the test suite.
- `tool_gaps` makes the unmeasurable parts of an experiment explicit instead of implying full coverage.
