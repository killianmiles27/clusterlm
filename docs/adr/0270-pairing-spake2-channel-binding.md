# 0270 — Pairing: SPAKE2 with TLS channel binding

## Context

Device identities are pinned (SHA-256 of the self-signed certificate). The pin has to come from somewhere: at
first contact neither Father nor Node knows the other's fingerprint, the LAN is untrusted, and the only shared
secret is a short code a human reads off one screen and types on the other. A hash of the code in a message (or
any commit-reveal over a low-entropy secret) lets an active attacker who plays one side test guesses offline.

## Decision

* Pairing runs over TLS 1.3 with both certificates but no pin (`SecurityConfig::pairing_channel`, the only place
  pinning is off). The channel carries nothing but the pairing exchange.
* The code is 8 RFC 4648 base32 characters (40 bits, CSPRNG), single use, valid for 5 minutes.
* Authentication is **SPAKE2 over P-256** (RFC 9382 symmetric form) implemented with OpenSSL `EC_POINT`/`BN`
  primitives; M and N are hash-to-curve points of fixed public strings; `w` = PBKDF2-HMAC-SHA256(code).
  An actual PAKE was practical, so commit-reveal was not used.
* The transcript hashed into the keys includes both certificate fingerprints and the TLS exporter secret
  (`SSL_export_keying_material`, label `EXPORTER-clusterlm-pairing-v1`). Key confirmation MACs are initiator-first;
  the Node sends its MAC only after verifying Father's. Device info (name, role, data port) is exchanged only
  after both confirmations.
* The Node serves one connection at a time, counts failed confirmations and **locks pairing mode after 3**
  (listener closed; a new local `--pair` is required). Stalls, malformed input and aborted runs are not counted
  (they test no guess), so a port scanner cannot lock the Node.
* On success each side stores the other's fingerprint (+ name, address, role) in its settings document; the Node's
  paired Father feeds `--trust` of its worker; unpair removes it and restarts the worker, dropping connections.

## Consequences

* Resisted: passive eavesdropping; offline guessing from any transcript; an active attacker gets one guess per
  run (3 per mode); a relay/splice with its own certificate (fingerprints and exporter differ per leg, so no
  confirmation verifies, even if relayed verbatim); replay of a recorded run (fresh shares and TLS session).
* Not resisted: someone who learns the code in time (shoulder surfing) can pair their own device — the Node shows
  its fingerprint short form and Father shows the stored one so the user can compare; a LAN host can burn the 3
  attempts (denial of pairing, not of security).
* The transport gained two small hooks (`pairing_channel`, `Connection::export_keying_material`); no behaviour
  changes for pinned channels.
* Real-LAN behaviour (firewall profile, discovery by typed address only) is pending HQ-PAIR-01.
