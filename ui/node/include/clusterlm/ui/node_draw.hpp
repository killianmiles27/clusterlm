#pragma once
// Portable Dear ImGui drawing for the Node window (see father_draw.hpp for the contract).
#include "clusterlm/ui/node_viewmodel.hpp"

namespace clusterlm::ui {

struct NodeDrawState {
  NodeSettings edit;  // what the controls currently show
  bool edit_loaded = false;
  std::uint32_t seen_revision = 0;   // NodeViewState::settings_revision the controls were last loaded from
  bool force_sections_open = false;  // smoke tests
};

void draw_node_ui(NodeViewModel& vm, NodeDrawState& ui, float display_w, float display_h);

}  // namespace clusterlm::ui
