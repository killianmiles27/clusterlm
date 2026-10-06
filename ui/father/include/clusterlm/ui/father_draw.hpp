#pragma once
// Portable Dear ImGui drawing for the Father window. No backend, no platform headers: it needs only a current
// ImGui context in a frame (between NewFrame and Render). The Win32/D3D11 shell lives in ui/father/win32 and the
// Linux "headless frame" smoke test drives this same function with ImGui's null renderer.
#include <string>

#include "clusterlm/ui/father_viewmodel.hpp"

struct ImVec2;

namespace clusterlm::ui {

// Widget-local state that is not view-model state (text being typed, scroll tracking, open sections).
struct FatherDrawState {
  std::string input;
  std::string diagnostics_preview;  // copy so the read-only text box is selectable
  FatherSettings edit;              // settings form
  bool edit_loaded = false;
  std::size_t last_signature = 0;   // transcript length signature for auto-scroll
  double last_diag_refresh = -1e9;
  std::string settings_message;
  bool force_sections_open = false;  // smoke tests: draw the collapsible sections too
};

// Draws the whole window to fill `display_w` x `display_h` logical pixels. Calls vm.tick() first.
void draw_father_ui(FatherViewModel& vm, FatherDrawState& ui, float display_w, float display_h);

}  // namespace clusterlm::ui
