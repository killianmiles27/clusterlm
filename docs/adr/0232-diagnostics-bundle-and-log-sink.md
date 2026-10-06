# 0232 Redacting diagnostics bundle fed by a log sink

## Context

Bug reports need logs, node states, the plan and bench result locations. Users attach them to issues and emails,
so whatever the bundle contains leaves the machine. The privacy contract says conversation content exists only
on Father, and logs never contain prompts, responses, token arrays, routing or activations. The logger already
has no API for any of those, but a bundle also carries free text supplied by callers (status messages, plan
text), so relying on "nobody logs content" alone is a single point of failure.

## Decision

* `log::set_sink(std::function<void(std::string_view)>)` in `runtime/common/log`: an optional observer of every
  emitted line, called under the logger lock. It is how tests capture all log output (including debug) and how
  `diagnostics::LogRing` keeps a bounded ring of recent lines.
* `orchestrator/diagnostics` (`clusterlm_diagnostics`) builds a JSON bundle (`clusterlm.diagnostics.v1`) from build
  info, the ring, node status entries, the plan description, metrics (with provenance) and bench result *paths*.
* Redaction is applied to every caller-supplied string: values of content-bearing keys (`prompt`, `text`,
  `response`, `content`, `message`, `tokens`, ...) are replaced; runs of 4+ integers (token arrays) and 3+ floats
  (activation dumps) are replaced; hex/base64 blobs over 96/128 characters and any value over 256 characters are
  replaced; control characters are neutralized. 64-hex device ids and digests are identifiers and are kept.
* The conversation is excluded unless `BundleOptions::include_conversation` is set; then it appears only under
  `user_content` with an explicit warning, and `redaction.conversation_included` says so.
* `clusterlm-father diagnostics --out FILE [--model DIR --plan PLAN] [--bench-results FILE]...` writes a bundle for
  the installation (the CLI has no conversation, so nothing opt-in applies there).

## Consequences

* A bundle built after a real chat contains no conversation content (test: `tests/privacy/test_diagnostics.cpp`).
* Redaction is heuristic and deliberately aggressive (a list of four numbers in a log line is redacted even if
  it is not a token array). It is a second line of defence; the primary guarantee remains the logger API.
* The sink holds raw lines in memory only; redaction happens at bundle time, so improving the rules improves old
  lines too.
* Real backends (Strata, llama.cpp) write their own logs; those must be routed through the same sink or audited
  when integrated.
