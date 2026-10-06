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
| `pairing.unpair` | `fingerprint` | removes the device and its assignments; releases the session |
| `assign.set` | `assignments` {role: fingerprint} | refused while a session is active |
| `model.confirm` | `tier_id` | records the inspected manifest root as the user's confirmation |
| `settings.get` / `settings.set` | `patch` over `selected_tier`, `context_tokens`, `keep_ready`, `model_dirs`, `advanced` | the document |

## Events

`tier_selected`, `prepare_progress` {`percent?`, `eta_seconds?` (an estimate), `message`}, `tier_ready`,
`tokens` {`request_id`, `tier_id`, `model_name` (the model that produced exactly these tokens), `token_count`, `text`},
`fallback` {`kind` retry/downgrade, `from_*`, `to_*`, `message` naming both models}, `finished` {`reason`,
`stats` incl. `answered_by[]`; numbers are observed on that run}, `error`, `released`. Token IDs are not sent.

## Behaviour notes

* Settings changes to assignments rebuild the service (releasing any session); model directories and advanced
  options apply at the next resolve.
* Keep-ready policy: when enabled an idle prepared session is released after `release_after_idle_minutes`
  (0 = never); when disabled after 60 s idle.
* The Coordinator and Nodes never see messages or roles; only token arrays reach the Coordinator.
