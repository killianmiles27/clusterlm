# 0330 — Node product glue: helper protocol v2, Father unpair notice, settings through the service

## Context

WP13/WP14 left seams: the Node UI could not save settings (kUnimplemented), the helper pipe reported only the Node's own
state, a Father-side unpair did not reach the Node, pairing mode was a CLI flag, preparation progress was not wired, and
`caps.threads` was stored but unused.

## Decision

* **Helper protocol version 2.** `StatusReply` grew (worker lease state, parts sealed/planned) and four message kinds
  were added (settings get/update, pairing-mode request/reply). Rather than tolerate two layouts of one message, the
  version was bumped: a version-1 peer gets `kVersionMismatch`. Service, helper and Node UI ship in one installer.
* **Settings go through the service, which is the authority.** The wire form (`NodeSettingsView`) contains only the
  fields an interactive user may change; name, paired Father, trust, paths and commands are not representable. The
  service validates with `config::validate`, persists with `NodeSettingsStore`, then applies: policy at once, caps by
  restarting the worker after a cooperative release (a lease must not survive a cap change). Accepted updates are
  rate limited. The UI re-reads after saving; the CPU limit is a percentage in the UI and a thread count on the wire.
  `start_with_system` is persisted only: changing the SCM startup type from a user message is out of scope.
* **Lease state is polled, not asked per request.** The supervisor asks the worker for its state every 250 ms
  (`status` already existed) and caches it, so a helper-pipe status request never waits on the worker.
* **Father unpair is a control-channel message, not a new channel.** `UnpairNotice` is sent over the same pinned
  mutual-TLS control channel Father already uses. The worker accepts it only from a Father it was started trusting
  (peers authorized per lease are refused the Father role), releases the lease, revokes that identity, replies, and
  prints a line the supervisor reads; the service then clears `paired_father` and restarts the worker, after checking
  the notifier is the recorded Father. Best effort: an unreachable Node keeps the old behaviour (trusts until a local
  unpair), and the Father's answer says so.
* **Preparation progress is an optional sink on `Coordinator::prepare`.** Per Node counters (bytes, objects, phase),
  serialized and throttled, always ending before `prepare` returns. The service republishes it as events with a
  `detail`, and a `ProvisioningBoard` feeds readiness observation (measured rate only, cleared when a prepare ends).

## Consequences

* The helper pipe now carries user-initiated mutations; they require an interactive session and are rate limited, but
  the Windows behaviour of that check is unverified (`HQ-WIN-02`, `HQ-PAIR-01`, `HQ-UI-01`).
* A Father cannot guarantee the Node dropped trust; the UI says when it could not tell it.
* The reference backend ignores the thread cap (it is single-threaded by design); the field exists so thread-pool
  backends can honour it when backend selection wires their options.
