#pragma once
// Win32 + Direct3D 11 host for Dear ImGui, shared by the Father and Node windows. Windows only.
// Owns the device, swap chain, render target and the ImGui context; the window itself (class, WndProc, tray icon)
// belongs to each executable. Everything here is verified only by compiling on Linux (MinGW) and by the manual
// walkthrough HQ-UI-01 on Windows 11.
#include <windows.h>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct IDXGISwapChain;
struct ID3D11RenderTargetView;

namespace clusterlm::ui::win32 {

class ImGuiHost {
 public:
  ImGuiHost() = default;
  ~ImGuiHost();
  ImGuiHost(const ImGuiHost&) = delete;
  ImGuiHost& operator=(const ImGuiHost&) = delete;

  // Call once, before creating any window: opts the process into per-monitor DPI awareness.
  static void enable_dpi_awareness();

  bool init(HWND hwnd);
  void shutdown();
  // Call from WM_SIZE (the host defers the buffer resize to the next frame).
  void on_resize(UINT width, UINT height);
  // Call from WM_DPICHANGED / after init: rescales fonts and spacing for the window's monitor.
  void apply_dpi(HWND hwnd);
  // Forward to ImGui's Win32 handler first thing in WndProc; true = handled.
  LRESULT wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

  // Starts a frame. False when the window is occluded/minimised (the caller should sleep instead of drawing).
  bool begin_frame();
  void end_frame();  // render + present
  float dpi_scale() const { return dpi_scale_; }
  // Logical (DPI-independent) size of the client area to hand to the draw functions.
  void logical_size(float* w, float* h) const;

 private:
  bool create_device(HWND hwnd);
  void destroy_device();
  void create_target();
  void destroy_target();

  HWND hwnd_ = nullptr;
  ID3D11Device* device_ = nullptr;
  ID3D11DeviceContext* context_ = nullptr;
  IDXGISwapChain* swap_ = nullptr;
  ID3D11RenderTargetView* target_ = nullptr;
  bool occluded_ = false;
  bool imgui_ready_ = false;
  UINT resize_w_ = 0, resize_h_ = 0;
  float dpi_scale_ = 1.0f;
};

}  // namespace clusterlm::ui::win32
