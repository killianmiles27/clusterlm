// Hidden message window delivering power and session notifications to the per-user session helper.
#ifdef _WIN32

#include <windows.h>
#include <wtsapi32.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include "../internal_errors.hpp"
#include "clusterlm/platform/platform_events.hpp"
#include "win_events.hpp"

namespace clusterlm::platform {

namespace {

constexpr wchar_t kWindowClass[] = L"ClusterLM.EventWindow";

class WindowsEventWindow final : public EventWindow {
 public:
  WindowsEventWindow(PowerEventHub& power, SessionEventHub& session) : power_(power), session_(session) {}
  ~WindowsEventWindow() override { stop(); }

  Status start() override {
    std::unique_lock lock(mu_);
    if (thread_.joinable()) return make_error(ErrorCode::kFailedPrecondition, "event window already started");
    thread_ = std::thread([this] { thread_main(); });
    cv_.wait(lock, [this] { return ready_; });
    if (!start_status_.is_ok()) {
      lock.unlock();
      thread_.join();
    }
    return start_status_;
  }

  void stop() override {
    HWND w;
    {
      std::lock_guard lock(mu_);
      w = hwnd_;
    }
    if (w != nullptr) ::PostMessageW(w, WM_CLOSE, 0, 0);
    if (thread_.joinable()) thread_.join();
  }

 private:
  static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    auto* self = reinterpret_cast<WindowsEventWindow*>(::GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (msg == WM_NCCREATE) {
      const auto* cs = reinterpret_cast<const CREATESTRUCTW*>(lp);
      ::SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
      return ::DefWindowProcW(hwnd, msg, wp, lp);
    }
    if (self != nullptr) {
      if (msg == WM_POWERBROADCAST) {
        if (auto ev = detail::map_power_broadcast(static_cast<DWORD>(wp), reinterpret_cast<const void*>(lp)))
          self->power_.emit(*ev);
        return TRUE;
      }
      if (msg == WM_WTSSESSION_CHANGE) {
        if (auto ev = detail::map_session_change(static_cast<DWORD>(wp), static_cast<DWORD>(lp))) self->session_.emit(*ev);
        return 0;
      }
    }
    if (msg == WM_DESTROY) {
      ::PostQuitMessage(0);
      return 0;
    }
    return ::DefWindowProcW(hwnd, msg, wp, lp);
  }

  void thread_main() {
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof wc;
    wc.lpfnWndProc = &WindowsEventWindow::wnd_proc;
    wc.hInstance = ::GetModuleHandleW(nullptr);
    wc.lpszClassName = kWindowClass;
    ::RegisterClassExW(&wc);  // already-registered is fine
    // A top-level hidden window (not HWND_MESSAGE): message-only windows do not receive broadcast messages.
    HWND hwnd = ::CreateWindowExW(0, kWindowClass, L"ClusterLM", WS_OVERLAPPED, 0, 0, 0, 0, nullptr, nullptr, wc.hInstance, this);
    HPOWERNOTIFY h_ac = nullptr, h_saver = nullptr;
    bool wts = false;
    Status st = Status::ok();
    if (hwnd == nullptr) {
      st = detail::win_status("CreateWindowExW");
    } else {
      h_ac = ::RegisterPowerSettingNotification(hwnd, &detail::kGuidAcDcPowerSource, DEVICE_NOTIFY_WINDOW_HANDLE);
      h_saver = ::RegisterPowerSettingNotification(hwnd, &detail::kGuidPowerSavingStatus, DEVICE_NOTIFY_WINDOW_HANDLE);
      wts = ::WTSRegisterSessionNotification(hwnd, NOTIFY_FOR_THIS_SESSION) != FALSE;
      if (!wts) st = detail::win_status("WTSRegisterSessionNotification");
    }
    {
      std::lock_guard lock(mu_);
      hwnd_ = st.is_ok() ? hwnd : nullptr;
      start_status_ = st;
      ready_ = true;
    }
    cv_.notify_all();
    if (st.is_ok()) {
      MSG msg;
      while (::GetMessageW(&msg, nullptr, 0, 0) > 0) {
        ::TranslateMessage(&msg);
        ::DispatchMessageW(&msg);
      }
    }
    if (wts) ::WTSUnRegisterSessionNotification(hwnd);
    if (h_ac != nullptr) ::UnregisterPowerSettingNotification(h_ac);
    if (h_saver != nullptr) ::UnregisterPowerSettingNotification(h_saver);
    if (hwnd != nullptr) ::DestroyWindow(hwnd);
    std::lock_guard lock(mu_);
    hwnd_ = nullptr;
  }

  PowerEventHub& power_;
  SessionEventHub& session_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::thread thread_;
  HWND hwnd_ = nullptr;
  bool ready_ = false;
  Status start_status_;
};

}  // namespace

std::unique_ptr<EventWindow> make_windows_event_window(PowerEventHub& power, SessionEventHub& session) {
  return std::make_unique<WindowsEventWindow>(power, session);
}

}  // namespace clusterlm::platform

#endif  // _WIN32
