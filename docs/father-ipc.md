# Father UI <-> agent IPC (API version 1)

Transport: the local pipe/socket of `clusterlm-father-agent` (`ipc::Envelope`, version
`kFatherUiProtocolVersion` = 1; reachable only by the agent's own user). Payloads are UTF-8 JSON. The messages
carry the user's text because this is IPC between the user's own processes; the agent never logs them.

* UI -> agent `kFatherRequest`: `{"op": "...", "id": <number>, ...args}`.
* agent -> UI `kFatherReply`: `{"id", "ok": true, "result": {...}}` or `{"id", "ok": false, "error": {"code", "message"}}`
  (`code` is the `ErrorCode` name, e.g. `FAILED_PRECONDITION`). A payload that is not such an object gets an `Ack`
  with `kProtocolError` and no detail.
* agent -> UI `kFatherEvent`: `{"event": "...", ...}`, pushed on every connected UI pipe while jobs run.

## Operations

| op | args | result |
|---|---|---|
| `hello` | | `api_version`, `dev_fixture_model`, `device_fingerprint`, `device_short`, `settings_note` |
| `tiers.list` | `context_tokens?` | `tiers[]` {`tier_id`, `display_name`, `model_name`, `state` (Unavailable/Available/Preparing/Ready), `reasons[]`, `notes[]`, `details[]`, `progress?`, `suggested_fallback[]`, `roles[]`}, `selected`. `details` carry profile provenance, backend and dev-override notes |
| `tiers.select` | `tier_id` | `selected` |
| `tiers.prepare` | `tier_id`, `context_tokens?` | `accepted`; then `prepare_progress` ... `tier_ready` or `error` events |
| `chat.send` | `message`, `system_prompt?`, `max_new_tokens?`, `context_tokens?`, `q?` | `request_id`; then `tokens`, `fallback`, `finished` or `error` events |
| `chat.cancel` | `request_id` | |
| `session.release` | | releases every lease |
| `conversation.reset` | | fails while a job runs |
| `diagnostics.get` | `include_text?` | redacted counters and tier lines; conversation text only when `include_text` |
| `pairing.start` | `address` (`host:pairport`), `code`, `name?` | `device` {fingerprint, short, name, address, ...}; blocks up to 20 s |
| `pairing.list` | | `devices[]`, `assignments` |
| `pairing.unpair` | `fingerprint` | removes the device and its assignments; releases the session; then tells the Node (docs/pairing.md). Result: `node_notified` (bool), `notify_outcome` (`delivered` / `unreachable` / `refused`), and a `note` when the Node could not be told (it keeps trusting this Father until unpaired locally) |
| `assign.set` | `assignments` {role: fingerprint} | refused while a session is active |
| `model.confirm` | `tier_id` | records the inspected manifest root as the user's confirmation |
| `settings.get` / `settings.set` | `patch` over `selected_tier`, `context_tokens`, `keep_ready`, `model_dirs`, `advanced` | the document |

## Events

`tier_selected`, `prepare_progress` {`percent?`, `eta_seconds?` (an estimate), `message`, `detail?`}, `tier_ready`,
`tokens` {`request_id`, `tier_id`, `model_name` (the model that produced exactly these tokens), `token_count`, `text`},
`fallback` {`kind` retry/downgrade, `from_*`, `to_*`, `message` naming both models}, `finished` {`reason`,
`stats` incl. `answered_by[]`; numbers are observed on that run}, `error`, `released`. Token IDs are not sent.

`prepare_progress.detail` (present on events that come from the Coordinator's own progress): `phase` (overall:
`father-domains`, `provisioning`, `authorizing` or `done`), `bytes_sent`, `bytes_total`, `objects_sealed`,
`objects_total` and `nodes[]` {`name`, `phase` (`provisioning`, `node-preparing` or `node-ready`), `bytes_sent`,
`bytes_total`, `objects_sealed`, `objects_total`}. Counts and machine names only: no object names, no content. Events with a detail
come at phase changes and whenever an object is sealed, and at most every `progress_poll` (100 ms) otherwise; the
smooth `percent` events carry no detail, so a client keeps the last detail until a new one arrives.

## Behaviour notes

* Settings changes to assignments rebuild the service (releasing any session); model directories and advanced
  options apply at the next resolve.
* Keep-ready policy: when enabled an idle prepared session is released after `release_after_idle_minutes`
  (0 = never); when disabled after 60 s idle.
* The Coordinator and Nodes never see messages or roles; only token arrays reach the Coordinator.
