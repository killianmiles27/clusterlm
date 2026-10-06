#include "clusterlm/ui/father_draw.hpp"

#include <algorithm>
#include <array>

#include <imgui/imgui.h>
#include <imgui/misc/cpp/imgui_stdlib.h>

namespace clusterlm::ui {

namespace {

const ImVec4 kGreen{0.35f, 0.80f, 0.45f, 1.0f};
const ImVec4 kAmber{0.95f, 0.75f, 0.30f, 1.0f};
const ImVec4 kBlue{0.45f, 0.70f, 0.95f, 1.0f};
const ImVec4 kGrey{0.60f, 0.60f, 0.62f, 1.0f};
const ImVec4 kRed{0.95f, 0.40f, 0.40f, 1.0f};

ImVec4 tier_colour(catalog::TierState s) {
  switch (s) {
    case catalog::TierState::kReady: return kGreen;
    case catalog::TierState::kPreparing: return kAmber;
    case catalog::TierState::kAvailable: return kBlue;
    case catalog::TierState::kUnavailable: return kGrey;
  }
  return kGrey;
}

// Fixed pixel sizes are multiplied by the DPI scale the host put in FontGlobalScale (1.0 in headless runs).
float u() { return ImGui::GetIO().FontGlobalScale; }

void text_wrapped(const std::string& s, const ImVec4* colour = nullptr) {
  if (colour) ImGui::PushStyleColor(ImGuiCol_Text, *colour);
  ImGui::TextWrapped("%s", s.c_str());
  if (colour) ImGui::PopStyleColor();
}

void draw_tiers(FatherViewModel& vm, const FatherViewState& st) {
  ImGui::SeparatorText("Model");
  if (st.tiers.empty()) {
    text_wrapped(st.connection_message.empty() ? "No models listed yet." : st.connection_message, &kAmber);
    return;
  }
  for (const auto& t : st.tiers) {
    ImGui::PushID(t.id.c_str());
    ImGui::BeginGroup();
    if (ImGui::RadioButton(t.name.c_str(), t.selected)) (void)vm.select_tier(t.id);
    ImGui::SameLine();
    const ImVec4 col = tier_colour(t.state);
    ImGui::TextColored(col, "%s", t.state_label.c_str());
    ImGui::Indent();
    if (!t.model.empty()) text_wrapped(t.model, &kGrey);
    // The headline is the service's own words; Ready is never asserted by the UI.
    text_wrapped(t.headline);
    for (const auto& n : t.notes) text_wrapped(n, &kGrey);
    if (t.progress_percent) {
      ImGui::ProgressBar(static_cast<float>(std::clamp(*t.progress_percent, 0.0, 100.0) / 100.0), ImVec2(-1.0f, 0.0f));
    }
    if (!t.eta.empty()) ImGui::TextDisabled("Time left: %s", t.eta.c_str());
    if (!t.participants.empty()) {
      std::string m = "Machines: ";
      for (std::size_t i = 0; i < t.participants.size(); ++i) m += (i ? ", " : "") + t.participants[i];
      ImGui::TextDisabled("%s", m.c_str());
    }
    if (t.can_prepare && ImGui::SmallButton("Prepare")) (void)vm.prepare_tier(t.id);
    ImGui::Unindent();
    ImGui::EndGroup();
    ImGui::Spacing();
    ImGui::PopID();
  }
}

void draw_status(const FatherViewState& st) {
  ImGui::SeparatorText("This conversation");
  ImGui::TextDisabled("Answering model");
  text_wrapped(st.active_model.empty() ? "(none chosen)" : st.active_model);
  if (!st.active_tier.empty()) ImGui::TextDisabled("Profile: %s", st.active_tier.c_str());
  ImGui::TextDisabled("Context");
  ImGui::TextWrapped("%s", st.context.label.empty() ? "--" : st.context.label.c_str());
  if (st.context.context_tokens > 0) {
    const float f = std::clamp(static_cast<float>(st.context.tokens_used) / static_cast<float>(st.context.context_tokens), 0.0f, 1.0f);
    ImGui::ProgressBar(f, ImVec2(-1.0f, 6.0f * u()), "");
  }
  ImGui::TextDisabled("Machines in use");
  if (st.participants.empty()) {
    ImGui::TextUnformatted("--");
  } else {
    std::string m;
    for (std::size_t i = 0; i < st.participants.size(); ++i) m += (i ? ", " : "") + st.participants[i];
    text_wrapped(m);
  }
  ImGui::TextDisabled("Last answer");
  if (st.last_answer.valid) {
    ImGui::Text("%s   first token %s", st.last_answer.tok_s.c_str(), st.last_answer.ttft.c_str());
    if (st.last_answer.fallbacks > 0) ImGui::TextColored(kAmber, "Model changed %u time(s) during this answer", st.last_answer.fallbacks);
  } else {
    ImGui::TextUnformatted("--");
  }
  if (st.prepare.active) {
    ImGui::SeparatorText("Preparing");
    text_wrapped(st.prepare.model.empty() ? st.prepare.tier_id : st.prepare.model);
    if (st.prepare.percent) ImGui::ProgressBar(static_cast<float>(std::clamp(*st.prepare.percent, 0.0, 100.0) / 100.0), ImVec2(-1.0f, 0.0f));
    else ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-1.0f, 0.0f), "working");
    if (!st.prepare.eta.empty()) ImGui::TextDisabled("Time left: %s", st.prepare.eta.c_str());
    if (!st.prepare.message.empty()) text_wrapped(st.prepare.message, &kGrey);
  }
}

void draw_controls(FatherViewModel& vm, FatherDrawState& ui, const FatherViewState& st) {
  ImGui::SeparatorText("Actions");
  if (ImGui::Button("New conversation")) (void)vm.new_conversation();
  ImGui::SameLine();
  if (ImGui::Button("Release")) (void)vm.release();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stop and let the other PCs delete their temporary files");

  if (ui.force_sections_open) ImGui::SetNextItemOpen(true);
  if (ImGui::CollapsingHeader("Settings")) {
    if (!ui.edit_loaded) {
      ui.edit = st.settings;
      ui.edit_loaded = true;
    }
    static constexpr std::array<std::uint32_t, 6> kContexts{4096, 8192, 16384, 32768, 65536, 131072};
    const std::string cur = std::to_string(ui.edit.context_tokens) + " tokens";
    if (ImGui::BeginCombo("Context size", cur.c_str())) {
      for (auto c : kContexts)
        if (ImGui::Selectable((std::to_string(c) + " tokens").c_str(), c == ui.edit.context_tokens)) ui.edit.context_tokens = c;
      ImGui::EndCombo();
    }
    int n = static_cast<int>(ui.edit.max_new_tokens);
    if (ImGui::InputInt("Answer length", &n, 16, 128)) ui.edit.max_new_tokens = static_cast<std::uint32_t>(std::max(n, 0));
    ImGui::InputTextMultiline("System prompt", &ui.edit.system_prompt, ImVec2(-1.0f, 60.0f * u()));
    if (ImGui::Button("Apply settings")) {
      auto s = vm.apply_settings(ui.edit);
      ui.settings_message = s.is_ok() ? "Saved." : s.message();
    }
    if (!ui.settings_message.empty()) text_wrapped(ui.settings_message, &kGrey);
  }

  if (ui.force_sections_open) ImGui::SetNextItemOpen(true);
  if (ImGui::CollapsingHeader("Paired machines")) {
    if (st.pairing.machines.empty()) ImGui::TextDisabled("No paired machines.");
    for (const auto& m : st.pairing.machines) {
      ImGui::PushID(m.machine_id.c_str());
      ImGui::Text("%s (%s)", m.machine_id.c_str(), m.role_hint.c_str());
      ImGui::SameLine();
      if (ImGui::SmallButton("Unpair")) (void)vm.unpair(m.machine_id);
      ImGui::PopID();
    }
    ImGui::InputTextWithHint("##pair_addr", "Address shown on the Node (host:port)", &ui.pair.address);
    ImGui::InputTextWithHint("##pair_code", "Pairing code", &ui.pair.code);
    ImGui::InputTextWithHint("##pair_name", "Name (optional)", &ui.pair.name);
    if (ImGui::Button("Pair this machine")) {
      if (vm.start_pairing(ui.pair).is_ok()) ui.pair = PairingRequest{};
    }
    if (!st.pairing.message.empty()) text_wrapped(st.pairing.message, &kAmber);
  }

  bool open = st.diagnostics.open;
  if (ImGui::Checkbox("Advanced diagnostics", &open)) vm.set_diagnostics_open(open);
}

void draw_transcript(const FatherViewState& st, FatherDrawState& ui) {
  const float input_h = ImGui::GetTextLineHeightWithSpacing() * 5.5f;
  ImGui::BeginChild("transcript", ImVec2(0, -input_h), ImGuiChildFlags_Borders);
  if (st.entries.empty()) ImGui::TextDisabled("Ask something. Your conversation stays on this PC.");
  std::size_t sig = st.entries.size() * 1000003u;
  for (const auto& e : st.entries) {
    switch (e.kind) {
      case EntryKind::kUser:
        ImGui::TextColored(kBlue, "You");
        text_wrapped(e.text);
        break;
      case EntryKind::kAssistant: {
        const std::string body = e.full_text();
        sig += body.size();
        if (!e.attribution.empty()) ImGui::TextDisabled("%s", e.attribution.c_str());
        text_wrapped(body.empty() && e.streaming ? std::string("...") : body);
        if (e.streaming && !body.empty()) ImGui::TextDisabled("writing...");
        if (e.cancelled) ImGui::TextDisabled("(stopped)");
        break;
      }
      case EntryKind::kNotice: text_wrapped(e.text, &kAmber); break;
      case EntryKind::kError: text_wrapped(e.text, &kRed); break;
    }
    ImGui::Spacing();
  }
  if (sig != ui.last_signature) {
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f) ImGui::SetScrollHereY(1.0f);
    ui.last_signature = sig;
  }
  ImGui::EndChild();
}

void draw_input(FatherViewModel& vm, FatherDrawState& ui, const FatherViewState& st) {
  ImGui::BeginDisabled(!st.can_send);
  bool submit = ImGui::InputTextMultiline("##input", &ui.input, ImVec2(-90.0f * u(), ImGui::GetTextLineHeight() * 4.0f),
                                          ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine);
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::BeginDisabled(!st.can_send);
  if (ImGui::Button("Send", ImVec2(80.0f * u(), 0.0f))) submit = true;
  ImGui::EndDisabled();
  if (st.chat_busy && ImGui::Button("Cancel", ImVec2(80.0f * u(), 0.0f))) (void)vm.cancel();
  ImGui::EndGroup();
  if (submit && st.can_send) {
    if (vm.send(ui.input).is_ok()) {
      ui.input.clear();
      ImGui::SetKeyboardFocusHere(-1);
    }
  }
  if (!st.send_hint.empty()) text_wrapped(st.send_hint, st.can_send ? &kGrey : &kAmber);
  else ImGui::TextDisabled("Enter sends, Ctrl+Enter adds a line.");
}

void draw_diagnostics(FatherViewModel& vm, FatherDrawState& ui, const FatherViewState& st) {
  const double now = ImGui::GetTime();
  if (now - ui.last_diag_refresh > 1.0) {
    vm.refresh_diagnostics();
    ui.last_diag_refresh = now;
  }
  const auto& d = st.diagnostics;
  ImGui::BeginChild("diagnostics", ImVec2(0, 0), ImGuiChildFlags_Borders);
  ImGui::SeparatorText("Advanced diagnostics");
  if (ImGui::BeginTable("diag_rows", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Measure");
    ImGui::TableSetupColumn("Value");
    ImGui::TableSetupColumn("Source");
    ImGui::TableHeadersRow();
    for (const auto& r : d.rows) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(r.label.c_str());
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(r.value.c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(r.provenance == "Synthetic" ? kAmber : kGrey, "%s", r.provenance.c_str());
    }
    ImGui::EndTable();
  }
  if (d.rows.empty()) ImGui::TextDisabled("No finished answer yet.");
  ImGui::Text("Plan: %s", d.plan_summary.empty() ? "not reported by the service" : d.plan_summary.c_str());
  if (d.stage_timings.empty()) {
    ImGui::TextDisabled("Per-stage timings: not reported by the service");
  } else {
    for (const auto& s : d.stage_timings) ImGui::BulletText("%s: %.1f ms", s.stage.c_str(), s.ms);
  }
  ImGui::Separator();
  bool inc = d.include_conversation;
  if (ImGui::Checkbox("Include my conversation text in the export", &inc)) vm.set_include_conversation(inc);
  if (inc) text_wrapped("The saved file will contain your prompts and answers. Leave this off unless you need it.", &kRed);
  else ImGui::TextDisabled("Off: the file contains counts, ids and timings only.");
  if (ImGui::Button("Export diagnostics")) (void)vm.export_diagnostics();
  if (!d.export_message.empty()) text_wrapped(d.export_message, &kGrey);
  ui.diagnostics_preview = d.snapshot_text;
  ImGui::InputTextMultiline("##snapshot", &ui.diagnostics_preview, ImVec2(-1.0f, -1.0f), ImGuiInputTextFlags_ReadOnly);
  ImGui::EndChild();
}

}  // namespace

void draw_father_ui(FatherViewModel& vm, FatherDrawState& ui, float display_w, float display_h) {
  vm.tick();
  const FatherViewState st = vm.snapshot();

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(display_w, display_h));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;
  ImGui::Begin("ClusterLM", nullptr, flags);

  if (st.demo) text_wrapped("DEMO MODE: scripted events, no model is running. Every number shown is Synthetic.", &kAmber);
  if (st.banner.kind != Banner::Kind::kNone) {
    text_wrapped(st.banner.text, st.banner.kind == Banner::Kind::kError ? &kRed : &kGreen);
    ImGui::SameLine();
    if (ImGui::SmallButton("Dismiss")) vm.dismiss_banner();
  }

  const float side_w = std::max(240.0f * u(), display_w * 0.32f);
  ImGui::BeginChild("side", ImVec2(side_w, 0), ImGuiChildFlags_Borders);
  draw_tiers(vm, st);
  draw_status(st);
  draw_controls(vm, ui, st);
  ImGui::EndChild();
  ImGui::SameLine();
  ImGui::BeginChild("main", ImVec2(0, 0));
  const bool diag = st.diagnostics.open;
  ImGui::BeginChild("chat", ImVec2(0, diag ? display_h * 0.55f : 0));
  draw_transcript(st, ui);
  draw_input(vm, ui, st);
  ImGui::EndChild();
  if (diag) draw_diagnostics(vm, ui, st);
  ImGui::EndChild();

  ImGui::End();
}

}  // namespace clusterlm::ui
