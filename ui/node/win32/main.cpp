// clusterlm-node-ui: the Node's small status window plus a real notification-area (tray) icon. Windows only.
//
//   * Tray icon via Shell_NotifyIconW: left click toggles the window, right click opens a menu with
//     Show window / Pause (or Resume) / Quit. The tooltip carries the Node state.
//   * Closing the window hides it to the tray; "Quit" is the only way to exit.
//   * `--tray` starts hidden (used by the per-user startup entry).
//   * The window is Dear ImGui over Direct3D 11 like the Father window; all behaviour is in NodeViewModel.
// The default icon is the stock application icon: a branded icon resource is a packaging task, not a UI one.
#include <windows.h>
#include <shellapi.h>

#include <cstddef>
#include <memory>
#include <string>

#include "../../common/win32/imgui_host.hpp"
#include "clusterlm/ui/clients.hpp"
#include "clusterlm/ui/node_draw.hpp"
#include "clusterlm/ui/node_viewmodel.hpp"

namespace {

using clusterlm::ui::win32::ImGuiHost;

constexpr UINT kTrayMessage = WM_APP + 1;
constexpr UINT kTrayId = 1;
constexpr UINT_PTR kMenuShow = 1001, kMenuPauseResume = 1002, kMenuQuit = 1003;

struct App {
  ImGuiHost host;
  std::unique_ptr<clusterlm::ui::NodeViewModel> vm;
  clusterlm::ui::NodeDrawState draw;
  UINT taskbar_created = 0;
};
App* g_app = nullptr;

std::wstring widen_ascii(const std::string& s) {
  std::wstring w;
  w.reserve(s.size());
  for (unsigned char c : s) w.push_back(c < 0x80 ? static_cast<wchar_t>(c) : L'?');
  return w;
}

void tray_add_or_update(HWND hwnd, bool add) {
  NOTIFYICONDATAW nid{};
  nid.cbSize = sizeof(nid);
  nid.hWnd = hwnd;
  nid.uID = kTrayId;
  nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
  nid.uCallbackMessage = kTrayMessage;
  nid.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
  const std::wstring tip = widen_ascii(g_app && g_app->vm ? g_app->vm->state().tooltip : std::string("ClusterLM Node"));
  const std::size_t n = tip.size() < 127 ? tip.size() : 127;  // szTip holds 128 wide chars including the NUL
  for (std::size_t i = 0; i < n; ++i) nid.szTip[i] = tip[i];
  Shell_NotifyIconW(add ? NIM_ADD : NIM_MODIFY, &nid);
}

void tray_remove(HWND hwnd) {
  NOTIFYICONDATAW nid{};
  nid.cbSize = sizeof(nid);
  nid.hWnd = hwnd;
  nid.uID = kTrayId;
  Shell_NotifyIconW(NIM_DELETE, &nid);
}

void toggle_window(HWND hwnd) {
  if (IsWindowVisible(hwnd)) {
    ShowWindow(hwnd, SW_HIDE);
  } else {
    ShowWindow(hwnd, SW_SHOW);
    SetForegroundWindow(hwnd);
  }
}

void show_tray_menu(HWND hwnd) {
  HMENU menu = CreatePopupMenu();
  if (!menu) return;
  const auto& st = g_app->vm->state();
  AppendMenuW(menu, MF_STRING, kMenuShow, IsWindowVisible(hwnd) ? L"Hide window" : L"Show window");
  const bool resume = st.can_resume;
  AppendMenuW(menu, MF_STRING | ((st.can_pause || st.can_resume) ? 0 : MF_GRAYED), kMenuPauseResume, resume ? L"Resume" : L"Pause");
  AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
  AppendMenuW(menu, MF_STRING, kMenuQuit, L"Quit ClusterLM Node");
  POINT p;
  GetCursorPos(&p);
  SetForegroundWindow(hwnd);  // required so the menu closes when the user clicks elsewhere
  const UINT cmd = static_cast<UINT>(TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON, p.x, p.y, 0, hwnd, nullptr));
  PostMessageW(hwnd, WM_NULL, 0, 0);
  DestroyMenu(menu);
  switch (cmd) {
    case kMenuShow: toggle_window(hwnd); break;
    case kMenuPauseResume:
      if (resume) (void)g_app->vm->resume();
      else (void)g_app->vm->pause();
      tray_add_or_update(hwnd, false);
      break;
    case kMenuQuit: DestroyWindow(hwnd); break;
    default: break;
  }
}

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  if (g_app && g_app->host.wnd_proc(hwnd, msg, wparam, lparam)) return TRUE;
  if (g_app && msg == g_app->taskbar_created && msg != 0) {  // Explorer restarted: the icon is gone
    tray_add_or_update(hwnd, true);
    return 0;
  }
  switch (msg) {
    case kTrayMessage:
      if (LOWORD(lparam) == WM_LBUTTONUP) toggle_window(hwnd);
      else if (LOWORD(lparam) == WM_RBUTTONUP || LOWORD(lparam) == WM_CONTEXTMENU) show_tray_menu(hwnd);
      return 0;
    case WM_CLOSE:
      ShowWindow(hwnd, SW_HIDE);  // to the tray; Quit is in the tray menu
      return 0;
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
      if ((wparam & 0xfff0) == SC_KEYMENU) return 0;
      break;
    case WM_DESTROY:
      tray_remove(hwnd);
      PostQuitMessage(0);
      return 0;
    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wparam, lparam);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR cmd_line, int) {
  const bool start_hidden = cmd_line && std::wstring(cmd_line).find(L"--tray") != std::wstring::npos;

  HANDLE single = CreateMutexW(nullptr, FALSE, L"Local\\ClusterLM.Node.UI");
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
  wc.lpszClassName = L"ClusterLM.Node.UI";
  RegisterClassExW(&wc);
  const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;  // small, fixed-size window
  HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"ClusterLM Node", style, CW_USEDEFAULT, CW_USEDEFAULT, 440, 560, nullptr,
                              nullptr, instance, nullptr);
  if (!hwnd) return 1;

  App app;
  g_app = &app;
  app.taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
  if (!app.host.init(hwnd)) {
    MessageBoxW(hwnd, L"ClusterLM Node could not start Direct3D 11 graphics on this PC.", L"ClusterLM Node", MB_OK | MB_ICONERROR);
    DestroyWindow(hwnd);
    g_app = nullptr;
    return 1;
  }

  clusterlm::ui::IpcNodeClient client{clusterlm::ui::IpcNodeClient::Options{}};
  app.vm = std::make_unique<clusterlm::ui::NodeViewModel>(client);
  app.vm->tick(true);
  tray_add_or_update(hwnd, true);

  RECT r{0, 0, static_cast<LONG>(440.0f * app.host.dpi_scale()), static_cast<LONG>(560.0f * app.host.dpi_scale())};
  AdjustWindowRect(&r, style, FALSE);
  SetWindowPos(hwnd, nullptr, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER);
  if (!start_hidden) {
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);
  }

  bool done = false;
  while (!done) {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
      if (msg.message == WM_QUIT) done = true;
    }
    if (done) break;
    if (!IsWindowVisible(hwnd)) {
      // Hidden in the tray: keep the state and tooltip fresh, otherwise sleep until input or the next poll.
      app.vm->tick();
      tray_add_or_update(hwnd, false);
      MsgWaitForMultipleObjects(0, nullptr, FALSE, 2000, QS_ALLINPUT);
      continue;
    }
    if (!app.host.begin_frame()) {
      Sleep(20);
      continue;
    }
    float w = 0, h = 0;
    app.host.logical_size(&w, &h);
    clusterlm::ui::draw_node_ui(*app.vm, app.draw, w, h);
    app.host.end_frame();
  }

  app.vm.reset();
  app.host.shutdown();
  g_app = nullptr;
  UnregisterClassW(wc.lpszClassName, instance);
  if (single) CloseHandle(single);
  return 0;
}
