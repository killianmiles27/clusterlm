// Headless ImGui frames: the portable draw code runs, without any rendering backend, for every view-model state
// and at awkward window sizes. This proves "does not crash and emits geometry"; it cannot prove it looks right
// (that is HQ-UI-01 on Windows).
#include <doctest/doctest.h>

#include <filesystem>

#include <imgui/imgui.h>

#include "clusterlm/ui/clients.hpp"
#include "clusterlm/ui/father_draw.hpp"
#include "clusterlm/ui/node_draw.hpp"

using namespace clusterlm;
using namespace clusterlm::ui;
using catalog::TierState;

namespace {

struct Headless {
  ImGuiContext* ctx;
  Headless() {
    IMGUI_CHECKVERSION();
    ctx = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 800);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* pixels = nullptr;
    int w = 0, h = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);  // no backend uploads it; the atlas just has to exist
  }
  ~Headless() { ImGui::DestroyContext(ctx); }
  Headless(const Headless&) = delete;
  Headless& operator=(const Headless&) = delete;

  template <typename Fn>
  int frame(float w, float h, Fn&& draw) {
    ImGui::GetIO().DisplaySize = ImVec2(w, h);
    ImGui::NewFrame();
    draw(w, h);
    ImGui::Render();
    return ImGui::GetDrawData()->TotalVtxCount;
  }
};

catalog::TierReadiness tier(const char* id, TierState st, const char* headline) {
  catalog::TierReadiness t;
  t.tier_id = id;
  t.model_name = std::string(id) + " model";
  t.state = st;
  t.reasons.push_back(headline);
  if (st == TierState::kPreparing) t.progress = catalog::PrepareProgress{55.0, 120.0, true};
  return t;
}

}  // namespace

TEST_CASE("father window draws in every view-model state at several sizes") {
  Headless h;
  ScriptedFatherClient c("Synthetic");
  c.set_tiers({tier("fast", TierState::kReady, "Fast is ready"), tier("strong", TierState::kPreparing, "Preparing Strong"),
               tier("ultra", TierState::kUnavailable, "The 3060 PC is offline")});
  c.set_participants("fast", {{"father", "This PC"}});
  c.set_paired({{"G14", "Node", "ab12"}});
  DiagnosticsExport d;
  d.text = "requests_started=1\nlast_error=\n";
  d.plan_summary = "plan";
  d.stage_timings = {{"G14 4-9", 5.0}};
  c.set_diagnostics(d);
  c.set_context_use(ContextUse{100, true});
  REQUIRE(c.select_tier("fast").is_ok());
  FatherViewModel vm(c, true);
  FatherDrawState ui;
  ui.force_sections_open = true;

  auto draw = [&](float w, float hh) { draw_father_ui(vm, ui, w, hh); };
  CHECK(h.frame(1280, 800, draw) > 0);  // idle

  vm.set_diagnostics_open(true);
  CHECK(h.frame(1280, 800, draw) > 0);  // with the advanced panel

  ui.input = "typed text";
  REQUIRE(vm.send("hello").is_ok());
  CHECK(h.frame(1280, 800, draw) > 0);  // busy, waiting for the first token
  father::TokensEvent t;
  t.request = 1;
  t.tier_id = "fast";
  t.model_name = "Fast model";
  t.text = "streaming ";
  t.tokens = {1};
  c.emit(t);
  CHECK(h.frame(1280, 800, draw) > 0);  // streaming
  father::FallbackEvent fb;
  fb.request = 1;
  fb.message = "G14 became busy - continuing on Fast; your conversation is kept";
  c.emit(fb);
  t.tier_id = "strong";
  t.model_name = "Strong model";
  t.text = std::string(3000, 'x');  // a very long unbroken word must not break layout or crash
  c.emit(t);
  c.emit(father::ErrorEvent{1, ErrorCode::kUnavailable, "node stopped answering"});
  c.emit(father::PrepareProgressEvent{"strong", "Strong model", 33.0, 60.0, "copying"});
  CHECK(h.frame(1280, 800, draw) > 0);  // fallback + error + preparing
  c.emit(father::FinishedEvent{1, father::FinishReason::kFailed, {}});
  CHECK(h.frame(800, 600, draw) > 0);
  CHECK(h.frame(320, 240, draw) > 0);  // tiny window
  CHECK(h.frame(1, 1, draw) >= 0);     // degenerate window: must not crash
  vm.set_include_conversation(true);
  CHECK(h.frame(1280, 800, draw) > 0);
}

TEST_CASE("father window draws when the Father side is unreachable and in demo mode") {
  Headless h;
  IpcFatherClient::Options o;
  o.endpoint.name = ipc::father_ui_pipe_name("smoke-absent");
  o.endpoint.socket_dir = std::filesystem::temp_directory_path() / "clm-ui-smoke-absent";
  IpcFatherClient c{o};
  FatherViewModel vm(c);
  FatherDrawState ui;
  ui.force_sections_open = true;
  auto draw = [&](float w, float hh) { draw_father_ui(vm, ui, w, hh); };
  CHECK(h.frame(1280, 800, draw) > 0);
  CHECK(h.frame(1280, 800, draw) > 0);
  CHECK(vm.snapshot().connection_message == IpcFatherClient::not_running_message());
}

TEST_CASE("node window draws in every state") {
  Headless h;
  struct Fake final : NodeClient {
    NodeStatus s;
    Result<NodeStatus> status() override { return s; }
    Status pause() override { return Status::ok(); }
    Status resume() override { return Status::ok(); }
    Result<NodeSettings> get_settings() override { return NodeSettings{}; }
    Status set_settings(const NodeSettings&) override { return make_error(ErrorCode::kUnavailable, "not saved"); }
    Result<NodePairingInfo> enter_pairing_mode() override { return NodePairingInfo{"ABCD-EFGH", "192.168.1.20:47601", "ab12-cd34-ef56", 300}; }
  } fake;
  NodeViewModel vm(fake);
  NodeDrawState ui;
  ui.force_sections_open = true;
  auto draw = [&](float w, float hh) { draw_node_ui(vm, ui, w, hh); };
  for (auto st : {NodeUiState::kUnreachable, NodeUiState::kStarting, NodeUiState::kAvailable, NodeUiState::kPreparing,
                  NodeUiState::kReady, NodeUiState::kInUse, NodeUiState::kBusy, NodeUiState::kPaused,
                  NodeUiState::kCleanupNeeded, NodeUiState::kStopping}) {
    fake.s.state = st;
    fake.s.paired_father = st == NodeUiState::kUnreachable ? "" : "ab12";
    fake.s.lease = st == NodeUiState::kPreparing ? ipc::LeaseState::kPreparing : ipc::LeaseState::kNone;
    fake.s.lease_parts_done = 3;
    fake.s.lease_parts_total = 12;
    vm.tick(true);
    CHECK(h.frame(440, 560, draw) > 0);
  }
  fake.s.state = NodeUiState::kAvailable;
  vm.tick(true);
  (void)vm.enter_pairing_mode();  // the pairing line is drawn too
  CHECK(h.frame(440, 560, draw) > 0);
  ui.edit.ram_gb = 0;  // an invalid edit still draws (the Save button is what reports it)
  CHECK(h.frame(440, 560, draw) > 0);
  CHECK(h.frame(200, 150, draw) > 0);
}
