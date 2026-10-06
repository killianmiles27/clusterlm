# 0320: UI toolkit is Dear ImGui, behind portable view-models

## Context
The product needs a simple Father chat window and a boring Node window/tray on Windows 11. Linux is the dev/CI host,
so UI behaviour must be testable without Windows, and the toolkit must not add a runtime dependency to install.

## Decision
* Dear ImGui v1.91.9 (MIT, release branch, not docking) is vendored unmodified under `third_party/imgui` with its
  LICENSE: core, `imgui_impl_win32`, `imgui_impl_dx11`, `imgui_stdlib`. No demo window.
* Rendering uses Win32 + Direct3D 11 (WARP fallback when no GPU). The shared host is `ui/common/win32/imgui_host.*`.
* All state and behaviour live in toolkit-free view-models (`FatherViewModel`, `NodeViewModel`) that depend only on
  the abstract `FatherClient` / `NodeClient`. The draw functions (`draw_father_ui`, `draw_node_ui`) are portable ImGui
  code with no backend, so Linux runs them headless (null renderer) for every view-model state.
* `IpcFatherClient` speaks the Father agent JSON API (docs/father-ipc.md); `IpcNodeClient` uses the helper-pipe
  messages. `InProcessFatherClient` and `ScriptedFatherClient` serve tests and `--demo`.
* DPI: per-monitor-v2 awareness, ImGui style rebuilt and `FontGlobalScale` set from the window DPI.

## Consequences
* Small binary, no extra runtime, builds with MSVC and MinGW; UI logic is unit-tested on Linux.
* ImGui is immediate-mode: no native accessibility (screen readers) and `FontGlobalScale` scaling is slightly soft.
  HQ-UI-01 records whether that is acceptable; the view-model split keeps a native-toolkit swap contained to the
  draw layer and the Win32 shells.
* The bundled font is Latin only, so service text is folded to ASCII punctuation before display.
