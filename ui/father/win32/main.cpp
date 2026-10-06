// clusterlm-father-ui: the Father chat window (Win32 + Direct3D 11 + Dear ImGui). Windows only.
//
// All behaviour lives in FatherViewModel and the portable draw code; this file owns only the window, the message
// loop and the choice of client:
//   default  IpcFatherClient to the per-user Father agent pipe (the wire layout is WP14's: until it lands the
//            window opens and says "not available in this build" instead of pretending),
//   --demo   a scripted client whose every number is labelled Synthetic (no model runs): for the HQ-UI-01 walkthrough.
#include <windows.h>

#include <filesystem>
#include <memory>
#include <string>

#include "../../common/win32/imgui_host.hpp"
#include "clusterlm/platform/ipc_messages.hpp"
#include "clusterlm/platform/paths.hpp"
#include "clusterlm/ui/clients.hpp"
#include "clusterlm/ui/father_draw.hpp"
#include "clusterlm/ui/father_viewmodel.hpp"

namespace {

using clusterlm::ui::win32::ImGuiHost;

struct App {
  ImGuiHost host;
  std::unique_ptr<clusterlm::ui::FatherViewModel> vm;
  clusterlm::ui::FatherDrawState draw;
};
App* g_app = nullptr;

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  if (g_app && g_app->host.wnd_proc(hwnd, msg, wparam, lparam)) return TRUE;
  switch (msg) {
    case WM_SIZE:
      if (wparam == SIZE_MINIMIZED) return 0;
      if (g_app) g_app->host.on_resize(LOWORD(lparam), HIWORD(lparam));
      return 0;
    case WM_DPICHANGED: {
      const RECT* r = reinterpret_cast<const RECT*>(lparam);
      SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
      if (g_app) g_app->host.apply_dpi(hwnd);
      return 0;
    }
    case WM_SYSCOMMAND:
      if ((wparam & 0xfff0) == SC_KEYMENU) return 0;  // no ALT menu
      break;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wparam, lparam);
}

std::filesystem::path diagnostics_dir() {
  auto p = clusterlm::platform::default_paths();
  if (p.is_ok()) return p->father_root / "diagnostics";
  return std::filesystem::temp_directory_path() / "clusterlm-diagnostics";
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR cmd_line, int) {
  const bool demo = cmd_line && std::wstring(cmd_line).find(L"--demo") != std::wstring::npos;

  // One Father window per user session.
  HANDLE single = CreateMutexW(nullptr, FALSE, L"Local\\ClusterLM.Father.UI");
  if (single && GetLastError() == ERROR_ALREADY_EXISTS) {
    CloseHandle(single);
    return 0;
  }

  ImGuiHost::enable_dpi_awareness();

  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.style = CS_CLASSDC;
  wc.lpfnWndProc = wnd_proc;
  wc.hInstance = instance;
  wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
  wc.lpszClassName = L"ClusterLM.Father.UI";
  RegisterClassExW(&wc);
  HWND hwnd = CreateWindowExW(0, wc.lpszClassName, demo ? L"ClusterLM (demo)" : L"ClusterLM", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                              CW_USEDEFAULT, 1100, 720, nullptr, nullptr, instance, nullptr);
  if (!hwnd) return 1;

  App app;
  g_app = &app;
  if (!app.host.init(hwnd)) {
    MessageBoxW(hwnd, L"ClusterLM could not start Direct3D 11 graphics on this PC.", L"ClusterLM", MB_OK | MB_ICONERROR);
    DestroyWindow(hwnd);
    g_app = nullptr;
    return 1;
  }

  std::unique_ptr<clusterlm::ui::FatherClient> client;
  if (demo) {
    client = clusterlm::ui::make_demo_father_client();
  } else {
    auto uid = clusterlm::ipc::current_user_id();
    clusterlm::ui::IpcFatherClient::Options o;
    o.endpoint = {clusterlm::ipc::father_ui_pipe_name(uid.is_ok() ? uid.value() : std::string("user")), {}};
    client = std::make_unique<clusterlm::ui::IpcFatherClient>(std::move(o));
  }
  app.vm = std::make_unique<clusterlm::ui::FatherViewModel>(*client, demo);
  app.vm->set_exporter(clusterlm::ui::make_file_exporter(diagnostics_dir().string()));

  // Size the window for the monitor's DPI, then show it.
  RECT r{0, 0, static_cast<LONG>(1100.0f * app.host.dpi_scale()), static_cast<LONG>(720.0f * app.host.dpi_scale())};
  AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
  SetWindowPos(hwnd, nullptr, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER);
  ShowWindow(hwnd, SW_SHOWDEFAULT);
  UpdateWindow(hwnd);

  bool done = false;
  while (!done) {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
      if (msg.message == WM_QUIT) done = true;
    }
    if (done) break;
    if (!app.host.begin_frame()) {
      Sleep(20);
      continue;
    }
    float w = 0, h = 0;
    app.host.logical_size(&w, &h);
    clusterlm::ui::draw_father_ui(*app.vm, app.draw, w, h);
    app.host.end_frame();
  }

  // The view-model unsubscribes from the client; destroy it before the client and before ImGui goes away.
  app.vm.reset();
  client.reset();
  app.host.shutdown();
  g_app = nullptr;
  DestroyWindow(hwnd);
  UnregisterClassW(wc.lpszClassName, instance);
  if (single) CloseHandle(single);
  return 0;
}
