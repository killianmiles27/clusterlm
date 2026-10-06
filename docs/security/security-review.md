# Security review (WP8)

Method: read every decoder and every place a wire/file number reaches an allocation; fuzz ten decoders with
libFuzzer + ASan + UBSan (ADR 0230); write regression tests for each finding; document what cannot be tested
here. Threat model: [threat-model.md](threat-model.md).

## 1. Findings

Severity is for the product on a LAN with paired devices. "Test" names the regression that fails without the fix.

| ID | Sev | Finding | Location | Fix | Test |
|---|---|---|---|---|---|
| F-01 | High | A Node accepted `Hello.role == Father` from any authenticated identity, including a peer Node authorized for a direct link, which could then reach Father-only control/provision/activation paths of its neighbour | `node/worker/src/node_worker.cpp` `handle_connection` | Reject a Father claim from an identity that is an authorized peer (ADR 0231) | `test_peer_binding` |
| F-02 | High | `SealObject` with `object_index` outside the plan manifest called `vector::at` -> uncaught `std::out_of_range` on a Node thread -> `std::terminate` (remote crash of the Node service by an authenticated sender) | `node_worker.cpp` `seal_object` | Bounds check, `kOutOfRange` reply | `test_error_replies` (aborted before the fix) |
| F-03 | Med | `PreparePlan` was admitted without checking stage working memory (`max_context`, `max_window` size KV, snapshots, scratch), with unchecked `+=` of object sizes, duplicate assignments and unvalidated stage roles | `node_worker.cpp` `prepare_plan` | Saturating sums, duplicate/role checks, probe-domain `describe_requirements()` admission (ADR 0233) | `test_hardening` |
| F-04 | Med | `ModelManifest::validate` compared source ranges pairwise: O(n^2) in a wire-controlled count (65,536 ranges per object ~ 2e9 comparisons), and summed range lengths without overflow check | `runtime/objects/src/manifest.cpp` | Sorted sweep; overflow check | `test_hardening` |
| F-05 | Med | Frame reader sized the payload buffer from the header (`payload.assign(len, 0)`): 24 bytes pinned up to `max_payload` (64 MiB) of zero-filled memory per connection | `runtime/transport/src/framed.cpp` | Buffer grows 1 MiB at a time as bytes arrive | `test_hardening`, `fuzz_frame_stream` |
| F-06 | Med | Geometry had no upper bounds: a hostile manifest sized scratch/KV from raw u32 fields (Node OOM/`bad_alloc`) | `runtime/objects/src/geometry.cpp` | Sanity ceilings (ADR 0233) | `test_hardening` |
| F-07 | Low | `StageActivations`: `hc*H + H + hc` is a u32 expression; dimensions that pass the per-field bound wrap it, giving an inconsistent layout | `runtime/domain/src/boundary.cpp` | 64-bit bound on the record size in `decode` and `validate` | `test_hardening` |
| F-08 | Low | Several repeated fields (`assignments`, `sealed_objects`, `stages`, timings) were sized from the count before checking the payload could hold them (<= 8 MiB each) | `runtime/protocol/src/messages.cpp` | `remaining >= count * element_size` first | `test_hardening`, `fuzz_protocol_decode` |
| F-09 | Low | Catalog JSON depth guard let stray `}` bank negative depth | `orchestrator/catalog/src/catalog.cpp` | Depth never below 0 | `test_hardening` |
| F-10 | Low | `CanonicalModelStore` allocated `byte_size` of a converted object (not bounded by shard sizes) before refusing it; `manifest.json` read unbounded | `runtime/objects/src/canonical_store.cpp` | Check first; 512 MiB manifest cap | `test_hardening` |
| F-11 | Low | Reference domain sliced tensors by geometry-derived sizes without re-checking them against the object size (UB on a corrupt local manifest) | `runtime/domain/src/reference_domain.cpp` | Size check in `decode_f32` | covered by plan/manifest tests |
| F-12 | Low | Node kept every accepted-connection thread unjoined until shutdown (leak per connection) with no limit on live handlers | `node_worker.cpp` `accept_loop` | Join finished handlers, cap 64 live | full suite |
| F-13 | Info | Windows `CreateProcessW` quoting lived in an anonymous namespace and could not be tested | `runtime/platform/src/process.cpp` | Extracted to `platform/command_line.hpp`; also quotes newline/VT | `test_hardening` round trip |

Fuzzer-found defects are listed in section 3.

## 2. Allocation-overflow audit

Every `resize`, `reserve`, `assign`, `new`, sized `vector(n)` and `n * sizeof` in `runtime/`, `node/`,
`orchestrator/`, `bench/`, `experimental/` (plus the strata/llama adapters, which are skeletons) was reviewed.

| Site | Size comes from | Check before allocating | Verdict |
|---|---|---|---|
| `framed.cpp` payload | frame header `payload_len` | `<= max_payload` at header; buffer grows with delivered bytes | fixed (F-05) |
| `ByteReader::blob/str` | u32 length | `<= max_size` and `raw()` requires bytes present | ok |
| `ByteReader::f32_array` | u32 count | `<= max_count` and `remaining >= 4n` | ok |
| `StageActivations::decode` | positions, hc, H | `positions <= max`, dims bounded, 64-bit record bound, `remaining >= floats*4` | fixed (F-07) |
| `PreparePlan` stages/assignments | u32 counts | `<= kMaxStages / max_manifest_objects` and now `remaining` | fixed (F-08) |
| `RunWindow/StageResult` timings | u32 count | `<= 64` and `remaining` | fixed (F-08) |
| `ProvisionStatus.sealed_objects` | u32 count | `<= max_manifest_objects` and `remaining` | fixed (F-08) |
| `ProvisionChunk.data` | blob length | `<= max_provision_chunk` | ok |
| manifest `source_ranges`, deps, shards, objects | u32 counts | per-field caps; ranges need `remaining >= 20n`; each object read consumes input | ok |
| `ModelGeometry::decode` layer kinds | u32 | `<= 65536` and `remaining >= n` | ok |
| lease store `init_ram` / `init_disk` | object `byte_size` | `create_object` enforces the lease budget first | ok |
| `ReferenceDomain` scratch/KV/snapshots | geometry, `max_context`, `max_window` | geometry ceilings + Node admission | fixed (F-03, F-06) |
| `ReferenceDomain` prefix/tail objects | manifest vs geometry | size check in `decode_f32` | fixed (F-11) |
| `CanonicalModelStore::load_unlocked` | `byte_size` | unconverted: equals ranges <= shard size (validated); converted: refused first | fixed (F-10) |
| `stream_object` buffer | `provision_chunk_bytes` (Father config) | local config | ok |
| Coordinator `sealed` vector, ProvisionStatus indices | Node-supplied indices | `idx < size` | ok |
| expert-domain `decode_batch/result` | positions, hidden, routes | `ExpertDecodeLimits` before allocation; `remaining/4 >= n` | ok |
| `Journal::parse` | file text | line-wise; per-line token vector | ok (file is owner-only; unbounded size accepted, see section 4) |
| `bench` (`host_probe`, `cmd_transport`) | fixed constants / CLI flag | local tool | ok |
| `placement`, `planning` vectors | geometry / profile counts | geometry ceilings; local data | ok |
| CLI `std::stoi/stoull` | argv | not guarded (exceptions end the process) | informational |

## 3. Fuzzing

Targets (ADR 0230): `frame_stream` (framed reader + `MessageStream`), `protocol_decode` (all message types),
`stage_activations`, `manifest_decode`, `manifest_json`, `lease_journal`, `catalog_json`, `profile_json`,
`text_parsers` (endpoint, plan, fault rule, hex, presets), `expert_wire`. There is no IPC frame decoder and no
GGUF parser in this tree.

Runs (clang 18, libFuzzer, ASan+UBSan+leak detection, 65 s each, `-rss_limit_mb=2048 -timeout=10`, seeded from the
committed corpus; five at a time on a loaded 4-core host so exec/s is conservative):

| Target | Executions | Final cov / features | Findings |
|---|---|---|---|
| `frame_stream` | 104,249 | 1658 / 2993 | none |
| `protocol_decode` | 374,155 | 1694 / 3260 | none |
| `stage_activations` | 280,152 | 109 / 171 | none |
| `manifest_decode` | 206,373 | 655 / 1970 | none |
| `manifest_json` | 14,162 | 1992 / 6305 | none |
| `lease_journal` | 304,873 | 224 / 1206 | none |
| `catalog_json` | 28,966 | 1523 / 5149 | none |
| `profile_json` | 40,085 | 1796 / 5519 | none |
| `text_parsers` | 274,114 | 575 / 1532 | none |
| `expert_wire` | 1,300,014 | 350 / 567 | none |

Honest reading: the runs were made on the tree **after** the audit fixes F-04..F-10, which came from reading the
code (the fuzzers' default 4 KiB inputs cannot reach, for example, 65,536-range objects or 64 MiB frames), so
they report no further crash, leak, UB, timeout or oversized allocation. The defects those fixes remove are pinned
by deterministic regression tests instead, and by hand-built `crash-*` seeds in the replay corpus
(`stage_activations/crash-wrapped-record-size`, `protocol_decode/crash-provisionstatus-count`,
`protocol_decode/crash-prepareplan-stage-count`, `catalog_json/crash-stray-closers`,
`frame_stream/crash-hostile-length-max`). Longer runs with `-max_len` raised (frames, manifests) are the obvious
next step and cost nothing but CPU.

## 4. Review items

| Item | Result |
|---|---|
| mTLS configuration | TLS 1.3 min = max, no tickets, no session cache, `SSL_VERIFY_FAIL_IF_NO_PEER_CERT` on the server, chain validation replaced by fingerprint pinning (leaf only, depth 0 enforced). Verified from outside by a raw OpenSSL client: TLS 1.2-only is rejected, no `NewSessionTicket`, missing and unpaired client certificates are rejected (`test_tls_policy`) |
| Certificate / key storage | Key written `O_CREAT 0600` + `fchmod`; mode test exists (`tests/transport/test_security.cpp`). Windows inherits the directory ACL: HQ-SEC-01, residual R-04 |
| Pairing assumptions | No pairing flow exists. Trust lists are supplied by configuration (`--trust`, `trusted_peers`); anything that can edit them or the identity directory defeats pinning. Documented in the threat model (R-02) |
| Peer identity binding | Authenticated fingerprint, not `Hello.device_id`, is matched to `AuthorizePeer`; unauthorized paired identity refused even when claiming an authorized id; authorized peer cannot claim Father; trust revoked on release (`test_peer_binding`) |
| Stale / replayed epochs and leases | A `RunWindow` captured under lease N is answered `kStaleEpoch` after lease N+1 and counted; patched-lease replay is refused by the window ledger; a second control connection cannot inject commits (`test_replay`; ledger rules in `test_window_ledger`) |
| Frame bounds | Header validated (magic, version, reserved, length) before any allocation; fuzzed |
| Object / staging paths | File names are `obj-<index>.part`, generated by the store; the journal accepts only such names; manifest object names containing `..`, `/`, `\`, drive letters, device names and NUL staged on disk create exactly the generated files and nothing outside the root (`test_hardening`, `fuzz_lease_journal`) |
| Reparse points / links | `resolve_under_root`, `remove_tree_no_follow`, link refusal for root and leases dir; POSIX symlink tests exist. Junctions/placeholders need Windows: HQ-SEC-03 |
| Process launch inputs | `ChildProcess::spawn` is the only launch path (posix_spawn / `CreateProcessW`, one quoted command line, no shell). The quoting function is exposed and round-trips 22 hostile arguments through an independent reference of the CRT parsing rules. The Node service takes the worker path and arguments from its own command line/config, never from the network. Install-directory ACLs (who may replace the worker binary) are an installer concern (R-12) |
| No shell anywhere | Source audit test over `runtime node orchestrator apps bench experimental`: no `system/popen/exec*/fork/ShellExecute/WinExec`, no `cmd.exe`, `/bin/sh`, `powershell`, and `CreateProcess*`/`posix_spawn*` only in `process.cpp` |
| Nodes never execute Father commands | Schema test pins every message shape; the only strings on the wire are the allowlist, none named like a command/path. `AuthorizePeer.peer_endpoint` is the only destination field; the Node connects to it but must see the authorized fingerprint (R-03) |
| GGUF parser boundaries | Not present in this tree. When it lands: add a `gguf` fuzz target to `tests/fuzz` (the macro takes one line), bound tensor count, name length, dimension product and offsets against the file size before allocating, and re-run section 2 |
| Journal size | `Journal::replay` reads the whole file. It is in an owner-only directory and compacts to one record per release; a hostile owner can already delete or fill that directory. Accepted |
| Insecure loopback mode | `Hello.device_id` and role are self-asserted; non-loopback endpoints refused. Test-only |

## 5. Not covered here

Windows ACLs (keys, pipes), junction/placeholder behaviour, real GPU backends' own logging and allocation
behaviour, installer ACLs, pairing UX. See `HQ-SEC-01..03` in `bench/qualification/experiments.json` and the
residual-risk table in the threat model.
