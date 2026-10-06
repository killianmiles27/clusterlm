#include "clusterlm/ui/node_draw.hpp"

#include <algorithm>

#include <imgui/imgui.h>

namespace clusterlm::ui {

namespace {

ImVec4 state_colour(NodeUiState s) {
  switch (s) {
    case NodeUiState::kAvailable:
    case NodeUiState::kReady:
    case NodeUiState::kInUse: return ImVec4(0.35f, 0.80f, 0.45f, 1.0f);
    case NodeUiState::kPreparing:
    case NodeUiState::kStarting:
    case NodeUiState::kCleanupNeeded:
    case NodeUiState::kBusy: return ImVec4(0.95f, 0.75f, 0.30f, 1.0f);
    case NodeUiState::kPaused:
    case NodeUiState::kStopping: return ImVec4(0.60f, 0.60f, 0.62f, 1.0f);
    case NodeUiState::kUnreachable: return ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
  }
  return ImVec4(1, 1, 1, 1);
}

bool input_u32(const char* label, std::uint32_t* v, int step) {
  int i = static_cast<int>(std::min<std::uint32_t>(*v, 1u << 30));
  if (!ImGui::InputInt(label, &i, step, step * 10)) return false;
  *v = static_cast<std::uint32_t>(std::max(i, 0));
  return true;
}

}  // namespace

void draw_node_ui(NodeViewModel& vm, NodeDrawState& ui, float display_w, float display_h) {
  vm.tick();
  const NodeViewState& st = vm.state();
  if (!ui.edit_loaded) {
    ui.edit = st.settings;
    ui.edit_loaded = true;
  }

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(display_w, display_h));
  ImGui::Begin("ClusterLM Node", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoSavedSettings);

  ImGui::TextColored(state_colour(st.status.state), "%s", st.state_label.c_str());
  ImGui::TextWrapped("%s", st.description.c_str());
  ImGui::Spacing();
  ImGui::TextWrapped("%s", st.father_label.c_str());
  ImGui::TextWrapped("%s", st.storage_label.c_str());
  if (!st.banner.empty()) {
    ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.0f), "%s", st.banner.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Dismiss")) vm.dismiss_banner();
  }

  ImGui::BeginDisabled(!st.can_pause && !st.can_resume);
  if (st.can_resume) {
    if (ImGui::Button("Resume")) (void)vm.resume();
  } else if (ImGui::Button("Pause")) {
    (void)vm.pause();
  }
  ImGui::EndDisabled();

  ImGui::SeparatorText("Settings");
  ImGui::Checkbox("Help only when this PC is idle", &ui.edit.allow_when_idle);
  ImGui::Checkbox("Only on AC power", &ui.edit.ac_power_only);
  ImGui::Checkbox("Start with Windows", &ui.edit.start_with_windows);
  input_u32("Temporary storage limit (GB)", &ui.edit.temp_storage_limit_gb, 1);
  if (ui.force_sections_open) ImGui::SetNextItemOpen(true);
  if (ImGui::CollapsingHeader("Advanced resource limits")) {
    int cpu = static_cast<int>(ui.edit.cpu_cap_percent), gpu = static_cast<int>(ui.edit.gpu_memory_cap_percent);
    if (ImGui::SliderInt("CPU limit (%)", &cpu, 10, 100)) ui.edit.cpu_cap_percent = static_cast<std::uint32_t>(cpu);
    if (ImGui::SliderInt("Graphics memory limit (%)", &gpu, 10, 100)) ui.edit.gpu_memory_cap_percent = static_cast<std::uint32_t>(gpu);
    input_u32("Memory limit (GB, 0 = none)", &ui.edit.ram_cap_gb, 1);
  }
  if (ImGui::Button("Save settings")) (void)vm.apply_settings(ui.edit);
  if (!st.settings_error.empty()) ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.0f), "%s", st.settings_error.c_str());
  if (!st.settings_note.empty()) ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f), "%s", st.settings_note.c_str());
  else if (st.settings_persisted) ImGui::TextDisabled("Saved.");

  ImGui::End();
}

}  // namespace clusterlm::ui
