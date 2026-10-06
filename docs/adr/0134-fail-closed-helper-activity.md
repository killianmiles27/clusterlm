# 0134 — Helper-fed activity fails closed; suspend revokes locally

## Context

The service gets idle time and lock state only from helper reports. A dead, wedged or absent helper must never be
read as "the user is away", and a machine going to sleep must not wait for Father.

## Decision

- `HelperActivityMonitor` reports "in use" (idle 0, not locked) when there is no report, when any tracked
  session's last report is older than 5 s, when paused, and after `invalidate()` (called on resume). With several
  sessions the machine is idle only if all are, and locked only if all are. A session is dropped on logoff/disconnect
  events; a machine with no user session (reported by the service) counts as idle.
- `NodeSupervisor::on_suspend()` marks the supervisor suspended and revokes immediately (cooperative deadline, then
  forced termination); while suspended nothing is offered. `on_resume()` clears the flag, forces Busy and relies on the next
  policy evaluation with fresh reports. Both are idempotent (resume arrives twice on some machines).
- Power-source and Battery Saver notifications only wake the loop; the power state is always re-sampled by the
  existing `PowerMonitor`, so a lost notification cannot leave a stale decision.

## Consequences

- A busy machine with a flaky helper flaps offered/revoked at worst, never the reverse.
- The 5 s window and 1 s report interval are placeholders until HQ-WIN-02 measures report latency.
