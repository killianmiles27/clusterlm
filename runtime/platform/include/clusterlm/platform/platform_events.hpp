#pragma once
// Power and session notifications as interfaces, so supervisor policy is testable without a desktop.
//
// Real sources (Windows only):
//   service   SERVICE_CONTROL_POWEREVENT / SERVICE_CONTROL_SESSIONCHANGE forwarded by the service host
//   helper    hidden message window: WM_POWERBROADCAST (+ RegisterPowerSettingNotification for
//             GUID_ACDC_POWER_SOURCE and GUID_POWER_SAVING_STATUS) and WM_WTSSESSION_CHANGE
// Both feed an EventHub; tests feed the same hub directly (the "mock").
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "clusterlm/common/status.hpp"

namespace clusterlm::platform {

enum class PowerEventKind : std::uint8_t {
  kSuspend,              // PBT_APMSUSPEND: the machine is about to sleep; handlers must be quick
  kResume,               // PBT_APMRESUMEAUTOMATIC / PBT_APMRESUMESUSPEND
  kPowerSourceChanged,   // GUID_ACDC_POWER_SOURCE
  kPowerSavingChanged,   // GUID_POWER_SAVING_STATUS
};

struct PowerEvent {
  PowerEventKind kind = PowerEventKind::kResume;
  bool on_ac_power = true;     // meaningful for kPowerSourceChanged
  bool battery_saver = false;  // meaningful for kPowerSavingChanged
};

enum class SessionEventKind : std::uint8_t { kLogon, kLogoff, kLock, kUnlock, kConnect, kDisconnect };

struct SessionEvent {
  SessionEventKind kind = SessionEventKind::kLogon;
  std::uint32_t session_id = 0;
};

using SubscriptionId = std::uint64_t;

template <typename Event>
class EventSource {
 public:
  using Handler = std::function<void(const Event&)>;
  virtual ~EventSource() = default;
  // Handlers run synchronously on the notifying thread (the SCM control thread / the helper's window thread).
  virtual SubscriptionId subscribe(Handler handler) = 0;
  virtual void unsubscribe(SubscriptionId id) = 0;
};

// Concrete source: the real adapters and tests call emit().
template <typename Event>
class EventHub final : public EventSource<Event> {
 public:
  using Handler = typename EventSource<Event>::Handler;
  SubscriptionId subscribe(Handler handler) override {
    std::lock_guard lock(mu_);
    handlers_.emplace_back(++next_, std::move(handler));
    return next_;
  }
  void unsubscribe(SubscriptionId id) override {
    std::lock_guard lock(mu_);
    handlers_.erase(std::remove_if(handlers_.begin(), handlers_.end(), [&](const auto& h) { return h.first == id; }),
                    handlers_.end());
  }
  void emit(const Event& e) {
    std::vector<std::pair<SubscriptionId, Handler>> copy;
    {
      std::lock_guard lock(mu_);
      copy = handlers_;
    }
    for (auto& h : copy) h.second(e);
  }

 private:
  std::mutex mu_;
  SubscriptionId next_ = 0;
  std::vector<std::pair<SubscriptionId, Handler>> handlers_;
};

using PowerEvents = EventSource<PowerEvent>;
using SessionEvents = EventSource<SessionEvent>;
using PowerEventHub = EventHub<PowerEvent>;
using SessionEventHub = EventHub<SessionEvent>;
using MockPowerEvents = PowerEventHub;      // tests call emit()
using MockSessionEvents = SessionEventHub;

// A hidden top-level message window on its own thread that turns WM_POWERBROADCAST / WM_WTSSESSION_CHANGE into
// hub events. Windows only (windows/event_window_win.cpp); used by the per-user session helper.
class EventWindow {
 public:
  virtual ~EventWindow() = default;
  virtual Status start() = 0;
  virtual void stop() = 0;
};

#ifdef _WIN32
std::unique_ptr<EventWindow> make_windows_event_window(PowerEventHub& power, SessionEventHub& session);
#endif

}  // namespace clusterlm::platform
