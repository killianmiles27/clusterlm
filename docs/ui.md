# Father and Node UI

Two small Windows programs, both Dear ImGui over Win32 + Direct3D 11 (ADR 0320):

| Executable | Purpose |
|---|---|
| `clusterlm-father-ui` | Chat-first Father window. `--demo` runs a scripted client (no model, every number Synthetic). |
| `clusterlm-node-ui` | Intentionally boring Node window plus a real tray icon. `--tray` starts hidden. |

Both are single-instance per Windows session and are built only on WIN32 (`ui/*/win32`). Everything else builds
and is tested on Linux.

## Layout of the code

```
ui/common    FatherClient / NodeClient interfaces, InProcess / Ipc / Scripted clients, display helpers
             ui/common/win32: ImGuiHost (D3D11 device, swap chain, DPI)  [Windows only]
ui/father    FatherViewModel, draw_father_ui, win32/main.cpp
ui/node      NodeViewModel, draw_node_ui, win32/main.cpp (tray)
tests/ui     view-model tests, client tests, IPC end-to-end, headless draw smoke, in-process end-to-end
```

## Father window

* Left: tier cards (Fast / Strong / Ultra) with state (Unavailable, Available, Preparing x%, Ready), the model name,
  the service's own headline and notes, machines taking part, progress bar and an ETA always labelled *estimate*
  (no rate means no ETA). A tier is shown Ready only when the service lists it Ready; events alone never do it.
* Status: answering model and profile, context use (`812 / 4096 tokens`, or `at least N` when the client can only
  count generated tokens), participating machines, tok/s and time to first token of the last answer.
* Chat: streaming text from Tokens events only (nothing speculative can reach the screen), Enter sends,
  Ctrl+Enter adds a line, Cancel while an answer runs. Sending to an unavailable tier is blocked with the reason;
  an available but unprepared tier says it will be prepared first.
* While a tier is prepared the card lists, per machine, bytes sent of total and parts sealed of total ("G14: 1.2 GB
  of 4.5 GB (3 of 12 parts)"), then "received everything, loading it" and "ready". These lines come from the
  Coordinator's own counters (pushed `prepare_progress` events with a `detail`); the percentage and the ETA come from
  the readiness observation, and the ETA exists only once about half a second of transfer has been measured.
* Fallback and recovery are written into the transcript, for example "G14 became busy - switching from Strong (...)
  to Fast (...); your conversation is kept." The answer continues in a new block attributed "Continued by Fast (...)";
  the last block adds the whole-answer attribution. Errors appear inline and in a banner with the error code.
* Settings (context size, answer length, system prompt), paired machines (pair, unpair), New conversation, Release.
* Advanced diagnostics: last-answer rows with a provenance column (`Measured` observed on this run, `Synthetic` for the
  development fixture model or the demo; the UI never prints Qualified), the plan and per-stage timings (shown as
  "not reported by the service" because the API does not carry them), the redacted snapshot, and Export. The export
  contains no prompts or answers unless the box "Include my conversation text" is ticked for that export (default off,
  red warning when on, WARNING header in the file). Files go to `%LOCALAPPDATA%\ClusterLM\diagnostics`.

## Node window

State (Not running, Starting, Available, Preparing, Ready, In use, Busy, Paused, Cleanup needed, Stopping) with one
plain sentence, the paired Father fingerprint, temporary storage in use, Pause/Resume, a **Pair with a Father...**
button, and settings: help only when idle, AC power only, start with Windows, temporary storage limit (0 = no limit),
advanced CPU / graphics memory / memory limits.
No chat, no model selection, no GGUF vocabulary (checked by a test). Validation messages name the field.

* **Lease states.** Preparing, Ready, In use and Cleanup needed come from the worker's lease state, which the service
  reports in `StatusReply` (helper protocol version 2: state plus parts received and verified out of parts assigned,
  counts only). They show only while the Node is offering; a Node that is Busy, Paused, Suspended or Stopping shows
  that instead, whatever its worker still holds for a moment. The window adds one line ("Receiving temporary files:
  3 of 12", "Holding 12 temporary files", "Removing temporary files").
* **Settings.** Read from and saved through the service (`SettingsRequest` / `SettingsUpdate` on the helper pipe).
  "Saved." appears only after the service validated, persisted and applied the values; the window then re-reads what
  the service stored. A refusal (rate limit: one save per 0.5 s, validation, no settings store) is shown as "Settings
  not saved: ...". The CPU limit travels as a thread count (percent of this PC's logical processors; 100 = automatic);
  graphics memory and memory are allowances in GB. Changing a resource limit restarts the Node's worker after a
  cooperative release, so nothing is held across the change. The idle time is not edited here and survives a save.
* **Pairing.** The button sends `PairingModeRequest` (interactive user only, one per 5 s) and shows the line the service
  returns: the code, where to type it on the Father and the short fingerprint to compare. A refusal clears any old code.

Tray: left click toggles the window; right click menu Show/Hide, Pause/Resume, Quit; closing the window hides it; the
tooltip shows the state; the icon is re-added after Explorer restarts.

## Seams and what is real

| Piece | Status |
|---|---|
| `IpcFatherClient` | Real: JSON API of docs/father-ipc.md over the per-user pipe, lazy connect, reconnect, pushed events. Tested against a running `clusterlm-father-agent --dev-fixture-model`. The API gives no exact context use, plan or stage timings; the UI says so. Answer length and system prompt are UI-side (the agent only stores the context size). `stats_provenance()` is Synthetic when `hello` reports the dev fixture model. |
| `IpcNodeClient` | Real over the helper pipe: status (with the worker lease state and counts), pause, resume, settings read and save, pairing mode. Tested against a fake service on the real transport (`test_ui_clients`) and against the real `ServiceCore` with a real worker (`test_node_control`). |
| Node settings | Real: validated, persisted (`NodeSettingsStore`) and applied by `clusterlm-node-service`; see "Settings" above. `start_with_system` is persisted but not applied: changing the service's startup type belongs to the installer/SCM, not to a user-session message. The "cannot save yet" path is gone. |
| Pairing a Node from the Node UI | Real up to the Node: the button, the helper-pipe request and the service's pairing responder (`docs/pairing.md`). Whether the LocalService listener is reachable through the firewall without the CLI flag is `HQ-PAIR-01`. |
| Win32 shells | Compile-checked by `scripts/check_windows_compile.sh` (MinGW) and built by the MSVC CI job; never run on Windows by the authors. |
| Draw code | Runs headless in tests for every state and several window sizes; whether it looks right is HQ-UI-01. |
| Tray icon | Stock application icon; a branded icon is a packaging task. |

Known limits: `tick()` calls the client on the UI thread (a hung Node service stalls the Node window up to the IO
timeout, 2 s); the ImGui font is Latin only; no screen-reader support (ADR 0320).

## HQ-UI-01: manual walkthrough on Windows 11 {#hq-ui-01}

Registered in `bench/qualification/experiments.json`. Steps:

1. `clusterlm-father-ui --demo`: tier cards, Prepare animation, chat, a message containing the word "fallback" on
   Strong shows the inline switch message and attribution, Cancel, Advanced diagnostics (all rows Synthetic), Export.
2. Start `clusterlm-father-agent`, then `clusterlm-father-ui`: real tiers, pairing form, prepare, chat, fallback by
   using the G14, export (redacted and with the box ticked).
3. `clusterlm-node-ui` on a Node: tray menu, hide/show, Pause/Resume, settings save and reload (HQ-UI-01 lists the
   cases), the Pair button, lease states while a Father prepares, restart explorer.exe.
4. DPI 100/150/200 % and a drag between monitors of different DPI; an RDP session without a GPU.
Record results with the experiment's metrics; file blockers before release.
