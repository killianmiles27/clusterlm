#include "imgui_host.hpp"

#include <d3d11.h>
#include <dxgi.h>

#include <imgui/backends/imgui_impl_dx11.h>
#include <imgui/backends/imgui_impl_win32.h>
#include <imgui/imgui.h>

// Declared in imgui_impl_win32.h only under #if 0 (the application is meant to forward-declare it).
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace clusterlm::ui::win32 {

ImGuiHost::~ImGuiHost() { shutdown(); }

void ImGuiHost::enable_dpi_awareness() { ImGui_ImplWin32_EnableDpiAwareness(); }

bool ImGuiHost::create_device(HWND hwnd) {
  DXGI_SWAP_CHAIN_DESC sd{};
  sd.BufferCount = 2;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferDesc.RefreshRate.Numerator = 60;
  sd.BufferDesc.RefreshRate.Denominator = 1;
  sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  sd.OutputWindow = hwnd;
  sd.SampleDesc.Count = 1;
  sd.Windowed = TRUE;
  sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

  D3D_FEATURE_LEVEL level;
  const D3D_FEATURE_LEVEL levels[2] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
  HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 2, D3D11_SDK_VERSION, &sd,
                                             &swap_, &device_, &level, &context_);
  if (hr == DXGI_ERROR_UNSUPPORTED)  // no usable GPU (remote session, VM): fall back to the software rasteriser
    hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, levels, 2, D3D11_SDK_VERSION, &sd, &swap_,
                                       &device_, &level, &context_);
  if (FAILED(hr)) return false;
  create_target();
  return true;
}

void ImGuiHost::destroy_device() {
  destroy_target();
  if (swap_) { swap_->Release(); swap_ = nullptr; }
  if (context_) { context_->Release(); context_ = nullptr; }
  if (device_) { device_->Release(); device_ = nullptr; }
}

void ImGuiHost::create_target() {
  ID3D11Texture2D* back = nullptr;
  if (FAILED(swap_->GetBuffer(0, IID_PPV_ARGS(&back))) || !back) return;
  device_->CreateRenderTargetView(back, nullptr, &target_);
  back->Release();
}

void ImGuiHost::destroy_target() {
  if (target_) { target_->Release(); target_ = nullptr; }
}

bool ImGuiHost::init(HWND hwnd) {
  hwnd_ = hwnd;
  if (!create_device(hwnd)) {
    destroy_device();
    return false;
  }
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;  // no imgui.ini next to the executable
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();
  if (!ImGui_ImplWin32_Init(hwnd) || !ImGui_ImplDX11_Init(device_, context_)) {
    ImGui::DestroyContext();
    destroy_device();
    return false;
  }
  imgui_ready_ = true;
  apply_dpi(hwnd);
  return true;
}

void ImGuiHost::shutdown() {
  if (imgui_ready_) {
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    imgui_ready_ = false;
  }
  destroy_device();
}

void ImGuiHost::on_resize(UINT width, UINT height) {
  resize_w_ = width;
  resize_h_ = height;
}

void ImGuiHost::apply_dpi(HWND hwnd) {
  if (!imgui_ready_) return;
  dpi_scale_ = ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd);
  if (dpi_scale_ <= 0.0f) dpi_scale_ = 1.0f;
  ImGuiStyle fresh;  // unscaled defaults, then dark colours, then scale: ScaleAllSizes is not idempotent
  ImGui::StyleColorsDark(&fresh);
  fresh.ScaleAllSizes(dpi_scale_);
  ImGui::GetStyle() = fresh;
  ImGui::GetIO().FontGlobalScale = dpi_scale_;
}

LRESULT ImGuiHost::wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  return imgui_ready_ ? ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam) : 0;
}

bool ImGuiHost::begin_frame() {
  if (occluded_ && swap_->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) return false;
  occluded_ = false;
  if (resize_w_ != 0 && resize_h_ != 0) {
    destroy_target();
    swap_->ResizeBuffers(0, resize_w_, resize_h_, DXGI_FORMAT_UNKNOWN, 0);
    resize_w_ = resize_h_ = 0;
    create_target();
  }
  ImGui_ImplDX11_NewFrame();
  ImGui_ImplWin32_NewFrame();
  ImGui::NewFrame();
  return true;
}

void ImGuiHost::end_frame() {
  ImGui::Render();
  const float clear[4] = {0.09f, 0.09f, 0.10f, 1.0f};
  if (target_) {
    context_->OMSetRenderTargets(1, &target_, nullptr);
    context_->ClearRenderTargetView(target_, clear);
  }
  ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
  const HRESULT hr = swap_->Present(1, 0);
  occluded_ = (hr == DXGI_STATUS_OCCLUDED);
}

void ImGuiHost::logical_size(float* w, float* h) const {
  const ImVec2 s = ImGui::GetIO().DisplaySize;
  *w = s.x;
  *h = s.y;
}

}  // namespace clusterlm::ui::win32
